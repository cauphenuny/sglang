"""Sparse attention utilities for MiniCPM models.

This module provides sparse attention helpers and utilities for MiniCPM models,
combining both backend-agnostic sparse attention components and kernel utilities.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

import triton
from sgl_kernel import infllmv2_attn_stage1, max_pooling_1d_varlen

from sglang.srt.layers.attention.minicpm.sparse_kernels import (
    compress_k_complete_kernel_new,
)
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool


def batched_gather(a, cu_seqlen_q, select):
    select_bs = len(select)
    select = torch.tensor(select, device="cpu")
    starts = cu_seqlen_q[select]
    ends = cu_seqlen_q[select + 1]
    lengths = ends - starts

    max_len = lengths.max()
    local_offsets = torch.arange(max_len, device=a.device)[None, :]
    mask = local_offsets < lengths[:, None]

    local_offsets = local_offsets.expand(select_bs, -1)[mask]

    starts_expanded = starts.repeat_interleave(lengths)
    index = starts_expanded + local_offsets

    return a[index]


def compress_k_core_new(
    full_compressed_k,  # output
    batch,
    key_cache,
    token_table,
    compressed_k_table,
    cu_new_k_token_nums,
    history_compress_k_token_nums,
    cu_total_compress_k_token_nums,
    kernel_size,
    kernel_stride,
    max_context_length,
    padded=False,
):
    head_num_k = key_cache.shape[1]
    head_dim = key_cache.shape[2]

    # ==============================================================================
    # BUFFER ALLOCATION
    # ==============================================================================

    # Use provided explicit parameters for buffer allocation
    # max_chunks_per_seq is already the maximum possible chunks for any sequence
    # given max_context_length, kernel_size, and kernel_stride
    max_chunks_per_seq = (
        max_context_length // kernel_stride
        if padded
        else max(0, (max_context_length - kernel_size) // kernel_stride + 1)
    )

    # ==============================================================================
    # Launch kernel for ALL chunks (history + new)
    # ==============================================================================
    # Grid: (batch, max_chunks_per_seq, head_num_k)
    # - chunk_in_seq in [0, history_compress): process HISTORY chunks
    # - chunk_in_seq in [history_compress, total_chunks_in_seq): process NEW chunks
    #
    # max_chunks_per_seq is already the maximum possible chunks for any sequence,
    # so it's sufficient for both history and new chunks.
    #
    # All operations are in a single kernel, CUDA graph compatible.

    # Limit grid size to avoid too many thread blocks
    # If max_chunks_per_seq > max_grid_chunks, kernel will loop to handle remaining chunks
    MAX_GRID_CHUNKS = 1024  # Adjustable limit for grid dimension
    max_grid_chunks = min(max_chunks_per_seq, MAX_GRID_CHUNKS)

    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    # Grid size is now limited, kernel uses loop to handle all chunks
    grid = (batch, max_grid_chunks, head_num_k)

    compress_k_complete_kernel_new[grid](
        key_cache,
        token_table,
        cu_new_k_token_nums,
        history_compress_k_token_nums,
        compressed_k_table,
        cu_total_compress_k_token_nums,
        full_compressed_k,
        batch,
        max_chunks_per_seq,
        token_table.shape[1],
        compressed_k_table.shape[1],
        head_num_k,
        head_dim,
        kernel_size,
        kernel_stride,
        BLOCK_SIZE,
        max_grid_chunks,  # Pass the limit to kernel for loop control
        PADDED=padded,
    )

    return


def get_compress_k_v2(
    layer,
    forward_batch,
    metadata: MiniCPMSparseMetadata,
    full_compressed_k1,
    full_compressed_k2,
    max_context_length,
    k1_kernel_size,
    k1_kernel_stride,
    k2_kernel_size,
    k2_kernel_stride,
    padded=False,
):
    batch = len(forward_batch.req_pool_indices)
    key_cache = get_token_to_kv_pool().get_key_buffer(layer.layer_id)
    key_cache = key_cache.view(-1, layer.tp_k_head_num, layer.head_dim)

    for full_compressed_k, level, kernel_size, kernel_stride in (
        (
            full_compressed_k1,
            metadata.k1,
            k1_kernel_size,
            k1_kernel_stride,
        ),
        (
            full_compressed_k2,
            metadata.k2,
            k2_kernel_size,
            k2_kernel_stride,
        ),
    ):
        compress_k_core_new(
            full_compressed_k,
            batch,
            key_cache,
            metadata.base.page_table,
            level.table,
            level.cu_new_token_nums,
            level.history_compress_token_nums,
            level.cu_total_compress_token_nums,
            kernel_size,
            kernel_stride,
            max_context_length,
            padded=padded,
        )


def allocate_and_compress_keys(
    layer,
    forward_batch,
    metadata: MiniCPMSparseMetadata,
    k1_token_nums: int,
    k2_token_nums: int,
    k1_kernel_size: int,
    k1_kernel_stride: int,
    k2_kernel_size: int,
    k2_kernel_stride: int,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = None,
    max_context_length: int = 32768,
    minicpm_split_stage1: bool = False,
):
    """Allocate compressed key tensors and run compression.

    Args:
        layer: Model layer with head configuration
        forward_batch: Forward batch info
        metadata: MiniCPM sparse metadata
        k1_token_nums: Number of k1 tokens to allocate
        k2_token_nums: Number of k2 tokens to allocate
        k1_kernel_size: K1 compression window
        k1_kernel_stride: K1 compression stride
        k2_kernel_size: K2 compression window
        k2_kernel_stride: K2 compression stride
        dtype: Tensor data type (default: bfloat16)
        device: Tensor device (default: layer device)
        max_context_length: Maximum context length for the model (default: 32768)
        minicpm_split_stage1: If True, use padded kernel

    Returns:
        Tuple of (full_compressed_k1, full_compressed_k2)
    """
    if device is None:
        device = forward_batch.input_ids.device

    full_compressed_k1 = torch.full(
        (k1_token_nums, layer.tp_k_head_num, layer.head_dim),
        dtype=dtype,
        device=device,
        fill_value=float("-inf"),
    )
    full_compressed_k2 = torch.full(
        (k2_token_nums, layer.tp_k_head_num, layer.head_dim),
        dtype=dtype,
        device=device,
        fill_value=float("-inf"),
    )

    get_compress_k_v2(
        layer,
        forward_batch,
        metadata,
        full_compressed_k1,
        full_compressed_k2,
        max_context_length=max_context_length,
        k1_kernel_size=k1_kernel_size,
        k1_kernel_stride=k1_kernel_stride,
        k2_kernel_size=k2_kernel_size,
        k2_kernel_stride=k2_kernel_stride,
        padded=minicpm_split_stage1,
    )

    return full_compressed_k1, full_compressed_k2


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_context_len: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens: Optional[torch.Tensor] = None,
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None,
    max_seqlen_q_adjusted: Optional[int] = None,
    minicpm_split_stage1: bool = False,
) -> torch.Tensor:
    """Compressed attention computation for sparse attention.

    Computes attention scores between query and compressed keys (k and k2),
    then performs max pooling and selects top-k blocks.

    Args:
        q: Query tensor, shape (total_q_len, num_heads, head_dim)
        k: Compressed key tensor k1, shape (total_k_len, num_heads, head_dim)
        k2: Compressed key tensor k2, shape (total_k_len, num_heads, head_dim)
        kernel_size: Size of compression kernel
        kernel_stride: Stride of compression kernel
        block_size: Size of attention blocks
        topk: Number of top blocks to select
        cu_seqlens_q: Cumulative sequence lengths for query, shape (batch_size + 1)
        cu_seqlens_k: Cumulative sequence lengths for k, shape (batch_size + 1)
        cu_seqlens_k2: Cumulative sequence lengths for k2, shape (batch_size + 1)
        max_seqlen_q: Maximum sequence length in query
        init_blocks: Number of initial blocks to always attend to
        local_blocks: Number of local blocks to consider
        cache_lens: Cache lengths for each batch (optional)
        cu_seqlens_q_adjusted: Adjusted cumulative sequence lengths for query (for stage1 optimization)
        max_seqlen_q_adjusted: Adjusted maximum sequence length for query (for stage1 optimization)

    Returns:
        Top-k block indices, shape (num_heads, total_q_len, topk)
    """
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1

        current_ratio = q.shape[-2] // k.shape[-2]
        required_ratio = 16
        if current_ratio < required_ratio:
            repeat_times = required_ratio // current_ratio
            q = q.repeat_interleave(repeat_times, dim=-2)

        is_prefilling = max_seqlen_q > 1

        if is_prefilling:
            if cache_lens is None:
                cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

        if not is_prefilling and minicpm_split_stage1:
            batch_size = q.shape[0]
            k1_len = k.shape[0]
            q_head = q.shape[1]
            kv_head = k.shape[1]
            group_size = q_head // kv_head
            head_dim = k.shape[2]
            q_reshape = (
                q.reshape(batch_size, 1, q_head, head_dim)
                .transpose(1, 2)
                .reshape(batch_size, kv_head, group_size, head_dim)
                .transpose(0, 1)
                .reshape(-1, group_size, head_dim)
            )
            k_reshape = (
                k.reshape(batch_size, k1_len // batch_size, kv_head, head_dim)
                .transpose(1, 2)
                .transpose(-2, -1)
                .transpose(0, 1)
                .reshape(-1, head_dim, k1_len // batch_size)
            )

            scale = 1.0 / math.sqrt(head_dim)
            score = torch.bmm(q_reshape, k_reshape).mul_(scale)
            torch.nan_to_num(score, nan=float("-inf"), posinf=float("-inf"), out=score)
            torch.softmax(score, dim=-1, out=score)
            score = score.reshape(
                kv_head, batch_size, group_size, k1_len // batch_size
            ).sum(dim=2)
        else:
            score = infllmv2_attn_stage1(
                q.contiguous(),
                k.contiguous(),
                k2.contiguous(),
                cu_seqlens_q=cu_seqlens_q_adjusted,
                cu_seqlens_k=cu_seqlens_k,
                cu_seqlens_v=cu_seqlens_k2,
                max_seqlen_q=max_seqlen_q_adjusted,
                max_seqlen_k=max_context_len // kernel_stride,
                causal=is_prefilling,
            )

        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_context_len,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )

        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


def compressed_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    kernel_topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
    fused_kernel=None,
    max_cache_len=-1,
) -> torch.Tensor:
    """
    使用 tilelang online topk kernel 计算 compressed attention topk indices
    """
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1

        # Check if it's prefilling stage
        # Use max_seqlen_q > 1 to avoid .item() call for CUDA Graph compatibility
        is_prefilling = cache_lens is None or max_seqlen_q > 1

        total_q_len = q.shape[0]
        num_kv_heads = k.shape[1]
        head_dim = k.shape[2]

        num_heads = q.shape[1]
        groups = num_heads // num_kv_heads
        q_kernel = q.view(total_q_len, num_kv_heads, groups, head_dim)
        q_kernel = (
            q_kernel.transpose(1, 2)
            .reshape(total_q_len * groups, num_kv_heads, head_dim)
            .contiguous()
        )

        k_kernel = k.contiguous()

        pooled_k_len = (max_cache_len + block_size - 1) // block_size

        assert fused_kernel is not None, "fused_kernel is not initialized"

        # Compute actual output topk (same as original: min(topk, num_blocks))
        output_topk = min(topk, pooled_k_len)

        # Allocate output tensors
        topk_indices = torch.full(
            (num_kv_heads, total_q_len, kernel_topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        topk_values = torch.full(
            (num_kv_heads, total_q_len, kernel_topk),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        if is_prefilling:
            # =================================================================
            # PREFILL: Use bucketed max_seqlen_q and pooled_k_len
            # Compiles once per unique bucket combination
            # Supports chunk prefill with cache_lens tensor
            # =================================================================
            # Prepare cache_lens tensor for chunk prefill support
            # For standard prefill: cache_lens is None -> use zeros
            # For chunk prefill: cache_lens has values -> use as-is
            if cache_lens is None:
                cache_lens_tensor = torch.zeros(
                    batch_size, dtype=torch.int32, device=q.device
                )
            else:
                cache_lens_tensor = cache_lens.to(torch.int32)

            # Run prefill kernel with cache_lens for chunk prefill support
            fused_kernel(
                q_kernel,
                k_kernel,
                cu_seqlens_q,
                cu_seqlens_k,
                cache_lens_tensor,
                topk_indices,
                topk_values,
            )
        else:
            # =================================================================
            # DECODE: max_seqlen_q=1 (fixed), cache_lens passed as tensor
            # Compiles ONCE and reuses for all decode steps!
            # =================================================================
            # Prepare cache_lens as tensor (runtime value, not compile-time constant!)
            cache_lens_tensor = cache_lens.to(torch.int32)

            # Run decode kernel with cache_lens as tensor
            fused_kernel(
                q_kernel,
                k_kernel,
                cu_seqlens_q,
                cu_seqlens_k,
                cache_lens_tensor,
                topk_indices,
                topk_values,
            )

        # Note: q_idx masking is handled inside the kernel via causal_mask
        # which sets scores to -1e9 for K blocks beyond the causal boundary.
        # These blocks won't be selected in topk due to their low scores.

        # Sort with -1 values at the end (match original behavior)
        # Replace -1 with large value, sort, then replace back
        large_val = pooled_k_len + 1000  # Any value larger than max valid index
        topk_for_sort = topk_indices.clone()
        topk_for_sort[topk_for_sort == -1] = large_val
        topk_idx = topk_for_sort.sort(-1).values
        topk_idx[topk_idx == large_val] = -1

        # Truncate to output_topk (same as original: min(topk, num_blocks))
        topk_idx = topk_idx[:, :, :output_topk].contiguous()

        return topk_idx


class CompressionLevelMetadata(msgspec.Struct):
    """Metadata for a single compression level (k1 or k2).

    This struct groups all metadata fields for one compression level,
    reducing duplication and making the code more maintainable.
    """

    # Cumulative sequence lengths for compressed cache
    cu_seqlens: Optional[torch.Tensor] = None
    cu_seqlens_cpu: Optional[list[int]] = None

    # Token mapping table (request pool indices -> compressed cache tokens)
    table: Optional[torch.Tensor] = None

    # Compressed cache metadata
    history_compress_token_nums: Optional[torch.Tensor] = None
    cu_new_token_nums: Optional[torch.Tensor] = None
    cu_total_compress_token_nums: Optional[torch.Tensor] = None


class MiniCPMSparseMetadata(msgspec.Struct):
    base: FlashAttentionMetadata
    k1: Optional[CompressionLevelMetadata] = None
    k2: Optional[CompressionLevelMetadata] = None
    sparse_bs_list: Optional[list[int]] = None
    sparse_batch_size: int = 0
    sparse_idx: Optional[list[int]] = None
    seqlen_k_sparse_bs_tensor: Optional[torch.Tensor] = None
    token_to_bs: Optional[torch.Tensor] = None
    token_pos_in_bs: Optional[torch.Tensor] = None
    sparse_page_table: Optional[torch.Tensor] = None
    sparse_cache_seqlens_int32: Optional[torch.Tensor] = None
    sparse_cu_seqlens_q_cpu: Optional[torch.Tensor] = None
    sparse_cu_seqlens_q: Optional[torch.Tensor] = None
    sparse_cu_seqlens_k: Optional[torch.Tensor] = None
    sparse_max_seq_len_q: int = 1
    old_bs_to_new_bs_range: Optional[list[int]] = None
    cache_seqlens_int32_stage1: Optional[torch.Tensor] = None
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None
    max_seqlen_q_adjusted: int = 1


def _build_sequence_lengths(
    extend_seq_lens_cpu: list[int],
    seq_lens: torch.Tensor,
    sparse_bs_list: list[int],
) -> tuple[list[int], torch.Tensor]:
    return (
        [extend_seq_lens_cpu[index] for index in sparse_bs_list],
        seq_lens[sparse_bs_list].to(dtype=torch.int32),
    )


def _build_token_mappings(
    cu_seqlens_q_sparse_bs: torch.Tensor,
    extend_prefix_lens_sparse: torch.Tensor,
    seqlen_q_sparse_bs: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build token mapping tensors for sparse batches.

    Computes token_to_bs (which batch each token belongs to) and
    token_pos_in_bs (position of each token within its batch).

    Args:
        cu_seqlens_q_sparse_bs: Cumulative sequence lengths for sparse batches
        extend_prefix_lens_sparse: Extension prefix lengths for sparse batches (size: len(sparse_bs_list))
        seqlen_q_sparse_bs: Query sequence lengths for sparse batches

    Returns:
        Tuple of (token_to_bs, token_pos_in_bs)
        - token_to_bs: Tensor mapping each token to its batch index
        - token_pos_in_bs: Tensor mapping each token to its position within batch
    """
    # Total number of tokens in sparse batches
    q_shape_sparse_bs = cu_seqlens_q_sparse_bs[-1].item()

    # Build token_to_bs: which batch each token belongs to
    token_to_bs = torch.zeros(q_shape_sparse_bs, dtype=torch.int32, device="cpu")
    for i in range(len(seqlen_q_sparse_bs)):
        start = cu_seqlens_q_sparse_bs[i]
        end = cu_seqlens_q_sparse_bs[i + 1]
        token_to_bs[start:end] = i

    # Build token_pos_in_bs: position of each token within its batch
    token_pos_in_bs = torch.zeros(q_shape_sparse_bs, dtype=torch.int32, device="cpu")
    for i in range(len(seqlen_q_sparse_bs)):
        start = cu_seqlens_q_sparse_bs[i]
        end = cu_seqlens_q_sparse_bs[i + 1]
        token_pos_in_bs[start:end] = torch.tensor(
            [
                (idx + 1 + extend_prefix_lens_sparse[i].item())
                for idx in range(seqlen_q_sparse_bs[i])
            ],
            dtype=token_pos_in_bs.dtype,
            device=token_pos_in_bs.device,
        )

    return token_to_bs, token_pos_in_bs


def _compute_single_compression_metadata(
    forward_batch: ForwardBatch,
    base_metadata: FlashAttentionMetadata,
    req_to_sparse_token: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
) -> CompressionLevelMetadata:
    """Compute compression metadata for a single compression level (k1 or k2).

    Args:
        forward_batch: The forward batch to analyze
        base_metadata: Base metadata with cu_seqlens_q, cu_seqlens_k
        req_to_sparse_token: Mapping from request pool to sparse tokens
        kernel_size: Kernel size for compression
        kernel_stride: Kernel stride for compression
        cu_seqlens_q: Cumulative query sequence lengths

    Returns:
        Compression metadata for this level
    """
    bs = forward_batch.batch_size
    seq_lens_cpu = torch.as_tensor(
        forward_batch.seq_lens_cpu,
        dtype=base_metadata.cu_seqlens_q.dtype,
        device="cpu",
    )
    seqlen_cpu = torch.clamp(
        (seq_lens_cpu - kernel_size) // kernel_stride + 1,
        min=0,
    )

    cu_seqlens_cpu = F.pad(
        torch.cumsum(seqlen_cpu, dim=0, dtype=torch.int32), (1, 0)
    ).tolist()
    cu_seqlens = F.pad(
        torch.cumsum(
            seqlen_cpu.to(device=cu_seqlens_q.device), dim=0, dtype=torch.int32
        ),
        (1, 0),
    )
    token_table = req_to_sparse_token[forward_batch.req_pool_indices]

    # CUDA graph replay uses metadata buffers sized for the captured batch,
    # while ``forward_batch`` contains only the real (unpadded) requests.
    # Restrict the cumulative sequence-length views to the real batch so
    # all per-request compression metadata has exactly ``bs`` entries.
    token_nums = (
        base_metadata.cu_seqlens_k[1 : bs + 1] - base_metadata.cu_seqlens_k[:bs]
    )
    input_lens = cu_seqlens_q[1 : bs + 1] - cu_seqlens_q[:bs]
    history_lens = token_nums - input_lens

    history_compress_token_nums = torch.maximum(
        (history_lens - kernel_size) // kernel_stride + 1,
        torch.zeros(1, device=history_lens.device, dtype=torch.int32),
    )

    new_token_nums = token_nums - history_compress_token_nums * kernel_stride

    cu_new_token_nums = F.pad(
        torch.cumsum(new_token_nums, dim=0, dtype=torch.int32), (1, 0)
    )

    new_compress_token_nums = torch.maximum(
        (new_token_nums - kernel_size) // kernel_stride + 1,
        torch.zeros(1, device=new_token_nums.device, dtype=torch.int32),
    )

    total_compress_token_nums = history_compress_token_nums + new_compress_token_nums

    cu_total_compress_token_nums = F.pad(
        torch.cumsum(total_compress_token_nums, dim=0, dtype=torch.int32), (1, 0)
    )

    return CompressionLevelMetadata(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        table=token_table,
        history_compress_token_nums=history_compress_token_nums,
        cu_new_token_nums=cu_new_token_nums,
        cu_total_compress_token_nums=cu_total_compress_token_nums,
    )


def _build_k1_k2_compression_metadata(
    forward_batch: ForwardBatch,
    base_metadata: FlashAttentionMetadata,
    req_to_sparse_k1_token: torch.Tensor,
    req_to_sparse_k2_token: torch.Tensor,
    k1_kernel_size: int,
    k1_kernel_stride: int,
    k2_kernel_size: int,
    k2_kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
) -> dict[str, CompressionLevelMetadata]:
    """Build k1/k2 compression metadata.

    This method computes all k1/k2 compressed cache metadata needed for
    sparse attention by calling _compute_single_compression_metadata for each level.

    Args:
        forward_batch: The forward batch to analyze
        base_metadata: Base metadata with cu_seqlens_q, cu_seqlens_k
        req_to_sparse_k1_token: Mapping from request pool to sparse k1 tokens
        req_to_sparse_k2_token: Mapping from request pool to sparse k2 tokens
        k1_kernel_size: Kernel size for k1 compression
        k1_kernel_stride: Kernel stride for k1 compression
        k2_kernel_size: Kernel size for k2 compression
        k2_kernel_stride: Kernel stride for k2 compression
        cu_seqlens_q: Cumulative query sequence lengths

    Returns:
        Dictionary with 'k1' and 'k2' keys containing CompressionLevelMetadata
    """
    return {
        "k1": _compute_single_compression_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            req_to_sparse_token=req_to_sparse_k1_token,
            kernel_size=k1_kernel_size,
            kernel_stride=k1_kernel_stride,
            cu_seqlens_q=cu_seqlens_q,
        ),
        "k2": _compute_single_compression_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            req_to_sparse_token=req_to_sparse_k2_token,
            kernel_size=k2_kernel_size,
            kernel_stride=k2_kernel_stride,
            cu_seqlens_q=cu_seqlens_q,
        ),
    }


def _get_sparse_cache_len(
    seq_len: int,
    sparse_capacity: int,
    block_size: int,
) -> int:
    if seq_len <= sparse_capacity:
        return seq_len
    remainder = seq_len % block_size
    return (
        sparse_capacity if remainder == 0 else sparse_capacity - block_size + remainder
    )


def _build_sparse_prefill_metadata(
    forward_batch: ForwardBatch,
    sparse_bs_list: list[int],
    head_group_num: int,
    dense_len: int,
    sparse_topk: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    sparse_page_table_dtype: torch.dtype,
    sparse_page_table_device: torch.device,
) -> dict:
    """Build sparse prefill metadata.

    This method handles the complex page table and batch mapping logic
    for sparse prefill mode.

    Args:
        forward_batch: The forward batch to analyze
        sparse_bs_list: List of sparse batch indices
        head_group_num: Number of head groups
        dense_len: Dense length threshold for sparse activation
        sparse_topk: Top-K value for sparse attention
        block_size: Block size for sparse attention
        cu_seqlens_q: Cumulative query sequence lengths
        sparse_page_table_dtype: Data type for sparse page table
        sparse_page_table_device: Device for sparse page table

    Returns:
        Dictionary with prefill metadata
    """
    bs = forward_batch.batch_size

    max_sparse_cache_len = -1
    sparse_page_table_bs = 0
    old_bs_to_new_bs_range = [0 for _ in range(bs + 1)]
    sparse_max_seq_len_q = 1

    for i in range(bs):
        if forward_batch.seq_lens_cpu[i] >= dense_len:
            max_sparse_cache_len = max(max_sparse_cache_len, sparse_topk * block_size)
            sparse_page_table_bs += (
                forward_batch.extend_seq_lens_cpu[i] * head_group_num
            )
            old_bs_to_new_bs_range[i + 1] = (
                old_bs_to_new_bs_range[i]
                + head_group_num * forward_batch.extend_seq_lens_cpu[i]
            )
        else:
            max_sparse_cache_len = max(
                max_sparse_cache_len, forward_batch.seq_lens_cpu[i]
            )
            sparse_page_table_bs += head_group_num
            old_bs_to_new_bs_range[i + 1] = old_bs_to_new_bs_range[i] + head_group_num
            sparse_max_seq_len_q = max(
                sparse_max_seq_len_q, forward_batch.extend_seq_lens_cpu[i]
            )

    sparse_page_table = torch.zeros(
        (sparse_page_table_bs, max_sparse_cache_len),
        dtype=sparse_page_table_dtype,
        device=sparse_page_table_device,
    )
    sparse_cu_seqlens_q_cpu = torch.zeros(
        (sparse_page_table_bs + 1), dtype=cu_seqlens_q.dtype, device="cpu"
    )

    pt = 0
    for i in range(bs):
        if forward_batch.seq_lens_cpu[i] >= dense_len:
            for _ in range(forward_batch.extend_seq_lens_cpu[i] * head_group_num):
                sparse_cu_seqlens_q_cpu[pt + 1] = sparse_cu_seqlens_q_cpu[pt] + 1
                pt += 1
        else:
            for _ in range(head_group_num):
                sparse_cu_seqlens_q_cpu[pt + 1] = (
                    sparse_cu_seqlens_q_cpu[pt] + forward_batch.extend_seq_lens_cpu[i]
                )
                pt += 1

    assert (
        pt == sparse_page_table_bs
    ), f"sparse_page_table_bs {sparse_page_table_bs} vs pt {pt}"

    sparse_cu_seqlens_q = sparse_cu_seqlens_q_cpu.to(device=cu_seqlens_q.device)
    sparse_cache_seqlens = []
    sparse_capacity = sparse_topk * block_size
    for i in range(bs):
        seq_len = int(forward_batch.seq_lens_cpu[i])
        if seq_len >= dense_len:
            prefix_len = seq_len - forward_batch.extend_seq_lens_cpu[i]
            for token_offset in range(1, forward_batch.extend_seq_lens_cpu[i] + 1):
                sparse_cache_seqlens.extend(
                    [
                        _get_sparse_cache_len(
                            prefix_len + token_offset,
                            sparse_capacity,
                            block_size,
                        )
                    ]
                    * head_group_num
                )
        else:
            sparse_cache_seqlens.extend([seq_len] * head_group_num)

    sparse_cache_seqlens_int32 = torch.tensor(
        sparse_cache_seqlens,
        dtype=torch.int32,
        device=cu_seqlens_q.device,
    )
    sparse_cu_seqlens_k = F.pad(
        torch.cumsum(sparse_cache_seqlens_int32, dim=0, dtype=torch.int32),
        (1, 0),
    )

    sparse_idx = []
    for sparse_bs in sparse_bs_list:
        sparse_idx.extend(
            range(
                old_bs_to_new_bs_range[sparse_bs],
                old_bs_to_new_bs_range[sparse_bs + 1],
            )
        )

    return {
        "sparse_page_table": sparse_page_table,
        "sparse_cu_seqlens_q_cpu": sparse_cu_seqlens_q_cpu,
        "sparse_cu_seqlens_q": sparse_cu_seqlens_q,
        "old_bs_to_new_bs_range": old_bs_to_new_bs_range,
        "sparse_max_seq_len_q": sparse_max_seq_len_q,
        "sparse_idx": sparse_idx,
        "sparse_cache_seqlens_int32": sparse_cache_seqlens_int32,
        "sparse_cu_seqlens_k": sparse_cu_seqlens_k,
    }


def _build_sparse_decode_metadata(
    forward_batch: ForwardBatch,
    base_metadata: FlashAttentionMetadata,
    head_group_num: int,
    dense_len: int,
    sparse_topk: int,
    block_size: int,
) -> dict:
    """Build sparse decode metadata.

    This method handles sparse attention metadata for decode mode.

    Args:
        forward_batch: The forward batch to analyze
        base_metadata: Base metadata with cache_seqlens_int32, page_table
        head_group_num: Number of head groups
        dense_len: Dense length threshold
        sparse_topk: Top-K value for sparse attention
        block_size: Block size

    Returns:
        Dictionary with decode metadata
    """
    bs = forward_batch.batch_size
    cache_seqlens = base_metadata.cache_seqlens_int32
    page_table = base_metadata.page_table
    max_sparse_cache_len = 0

    sparse_cache_seqlens_cpu = torch.zeros(
        (bs * head_group_num,), dtype=cache_seqlens.dtype, device="cpu"
    )

    for b in range(bs):
        seq_len = int(forward_batch.seq_lens_cpu[b])
        if seq_len >= dense_len:
            sparse_cache_len = _get_sparse_cache_len(
                seq_len,
                sparse_topk * block_size,
                block_size,
            )

            if sparse_cache_len > max_sparse_cache_len:
                max_sparse_cache_len = sparse_cache_len

            sparse_cache_seqlens_cpu[b * head_group_num : (b + 1) * head_group_num] = (
                sparse_cache_len
            )
        else:
            if seq_len > max_sparse_cache_len:
                max_sparse_cache_len = seq_len

            sparse_cache_seqlens_cpu[b * head_group_num : (b + 1) * head_group_num] = (
                seq_len
            )

    sparse_cache_seqlens_int32 = sparse_cache_seqlens_cpu.to(
        device=cache_seqlens.device
    )
    sparse_cu_seqlens_k = F.pad(
        torch.cumsum(sparse_cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
    )
    sparse_cu_seqlens_q = torch.arange(
        0,
        bs * head_group_num + 1,
        dtype=torch.int32,
        device=base_metadata.cu_seqlens_q.device,
    )
    token_to_bs = torch.arange(0, bs, dtype=torch.int32, device=page_table.device)
    sparse_page_table = torch.zeros(
        (head_group_num * bs, max(dense_len, sparse_topk * block_size)),
        dtype=page_table.dtype,
        device=page_table.device,
    )

    return {
        "sparse_cache_seqlens_int32": sparse_cache_seqlens_int32,
        "sparse_cu_seqlens_k": sparse_cu_seqlens_k,
        "sparse_cu_seqlens_q": sparse_cu_seqlens_q,
        "sparse_page_table": sparse_page_table,
        "token_to_bs": token_to_bs,
    }


def _build_prefill_topk_metadata(
    forward_batch: ForwardBatch,
    query_states: torch.Tensor,
    tp_q_head_num: int,
    head_dim: int,
    dense_len: int,
    k1_metadata: CompressionLevelMetadata,
    k2_metadata: CompressionLevelMetadata,
) -> dict:
    """Build prefill TopK metadata.

    This method prepares all metadata needed for TopK computation in prefill mode,
    including sparse batch identification, sequence lengths, and query preparation.

    Args:
        forward_batch: The forward batch
        query_states: Query states from model layer
        tp_q_head_num: Number of query heads
        head_dim: Head dimension
        dense_len: Dense length threshold
        k1_metadata: K1 compression metadata
        k2_metadata: K2 compression metadata

    Returns:
        Dictionary with prefill TopK metadata
    """
    bs, seqlens_q, seqlens_k = (
        forward_batch.batch_size,
        forward_batch.extend_seq_lens_cpu,
        forward_batch.seq_lens_cpu,
    )

    k1_lens = [
        end - start
        for start, end in zip(
            k1_metadata.cu_seqlens_cpu,
            k1_metadata.cu_seqlens_cpu[1:],
        )
    ]
    k2_lens = [
        end - start
        for start, end in zip(
            k2_metadata.cu_seqlens_cpu,
            k2_metadata.cu_seqlens_cpu[1:],
        )
    ]

    sparse_bs = []
    seqlens_q_sparse_bs = []
    seqlens_k_sparse_bs = []

    for i in range(bs):
        if seqlens_k[i] >= dense_len:
            sparse_bs.append(i)
            seqlens_q_sparse_bs.append(seqlens_q[i])
            seqlens_k_sparse_bs.append(int(seqlens_k[i]))

    cu_seqlens_q = torch.cumsum(
        torch.tensor([0] + seqlens_q, dtype=torch.int32, device=query_states.device),
        dim=0,
        dtype=torch.int32,
    )

    query_states_reshaped = query_states.reshape(-1, tp_q_head_num, head_dim)

    query_states = batched_gather(query_states_reshaped, cu_seqlens_q, sparse_bs)

    cu_seqlens_q_sparse = torch.cumsum(
        torch.tensor(
            [0] + seqlens_q_sparse_bs, dtype=torch.int32, device=query_states.device
        ),
        dim=0,
        dtype=torch.int32,
    )
    cu_seqlens_k_sparse = torch.cumsum(
        torch.tensor(
            [0] + seqlens_k_sparse_bs, dtype=torch.int32, device=query_states.device
        ),
        dim=0,
        dtype=torch.int32,
    )

    return {
        "sparse_bs": sparse_bs,
        "k1_lens": k1_lens,
        "k2_lens": k2_lens,
        "cu_seqlens_q": cu_seqlens_q_sparse,
        "cu_seqlens_k": cu_seqlens_k_sparse,
        "max_seqlen_q": max(seqlens_q_sparse_bs),
        "max_seqlen_k": max(seqlens_k_sparse_bs),
        "query_states": query_states,
    }
