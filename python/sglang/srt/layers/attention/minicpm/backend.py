from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


import tilelang
import tilelang.math

from sglang.jit_kernel.minicpm_sala import (
    get_block_table_v2,
    get_block_table_v3,
)
from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    _bucket_size,
    fused_attn_pooling_online_topk_decode,
    fused_attn_pooling_online_topk_prefill,
)
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    CompressionLevelMetadata,
    SparseBatchAnalyzer,
    SparseConfig,
    SparseMetadataBuilder,
    allocate_and_compress_keys,
    compressed_attention,
    compressed_attention_tilelang,
    get_compress_k_v2,
    get_compress_k_v2_padded,
)


class MiniCPMSparseBackend(AttentionBackend):
    """MiniCPM sparse dispatch layered on the standard FlashAttention backend."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        fa_impl_ver=3,
    ):
        super().__init__()
        self.base_backend = FlashAttentionBackend(
            model_runner,
            skip_prefill=skip_prefill,
            fa_impl_ver=fa_impl_ver,
        )
        self.forward_metadata: Optional[FlashAttentionMetadata] = None
        self.max_context_len = self.base_backend.max_context_len
        self.device = self.base_backend.device
        self.enable_cuda_graph = not model_runner.server_args.disable_cuda_graph
        self.decode_cuda_graph_metadata = self.base_backend.decode_cuda_graph_metadata
        self.req_to_token_pool = self.base_backend.req_to_token_pool
        self.token_to_kv_pool = self.base_backend.token_to_kv_pool
        self.req_to_sparse_k1_token = (
            model_runner.req_to_token_pool.req_to_sparse_k1_token
        )
        self.req_to_sparse_k2_token = (
            model_runner.req_to_token_pool.req_to_sparse_k2_token
        )
        self.kv_cache_dtype = self.base_backend.kv_cache_dtype
        self.kv_cache_dtype_str = self.base_backend.kv_cache_dtype_str
        self.page_size = self.base_backend.page_size
        tp_size = get_parallel().attn_tp_size
        self.num_kv_heads = model_runner.model_config.num_key_value_heads // tp_size
        self.fa_impl_ver = self.base_backend.fa_impl_ver
        self.num_splits = self.base_backend.num_splits

        # Sparse attention configuration (required for MiniCPM)
        hf_config = model_runner.model_config.hf_config

        # MiniCPM must have sparse attention enabled
        if not isinstance(hf_config, MiniCPMHybridConfig) or not (
            hf_config.has_minicpm_sparse_attention
        ):
            raise ValueError(
                "MiniCPM model must have sparse attention enabled. "
                "Please ensure the model config has MiniCPM sparse attention enabled."
            )
        self.has_minicpm_sparse_attention = True

        self.kernel_size = hf_config.sparse_kernel_size
        self.kernel_stride = hf_config.sparse_kernel_stride
        self.init_blocks = hf_config.sparse_init_blocks
        self.block_size = hf_config.sparse_block_size
        self.window_size = hf_config.sparse_window_size
        self.minicpm_dense_as_sparse = model_runner.server_args.minicpm_dense_as_sparse
        self.dense_len = (
            0 if self.minicpm_dense_as_sparse else hf_config.sparse_dense_len
        )
        self.config_dense_len = hf_config.sparse_dense_len
        topk = hf_config.sparse_topk
        self.use_nope = hf_config.sparse_use_nope
        self.local_blocks = self.window_size // self.block_size  # local_blocks
        self.sparse_topk = topk + (self.window_size // self.block_size)
        self.num_sparse_topk_tokens = self.block_size * self.sparse_topk

        # Head group number derived from model configuration
        self.head_dim = model_runner.model_config.head_dim
        self.head_group_num = model_runner.model_config.num_key_value_heads
        self.heads_per_group = (
            model_runner.model_config.num_attention_heads // self.head_group_num
        )
        self.k1_kernel_size = self.kernel_size
        self.k1_kernel_stride = self.kernel_stride
        self.k2_kernel_size = self.kernel_size * 4
        self.k2_kernel_stride = self.kernel_stride * 4

        self.minicpm_fuse_topk = model_runner.server_args.minicpm_fuse_topk
        self.minicpm_split_stage1 = model_runner.server_args.minicpm_split_stage1

        max_cache_len = self.max_context_len
        pooled_k_len = (max_cache_len + self.block_size - 1) // self.block_size

        output_topk = min(self.sparse_topk, pooled_k_len)

        # For the kernel, we need power of 2 topk
        topk_power2 = tilelang.math.next_power_of_2(output_topk)
        kernel_topk = min(topk_power2, pooled_k_len)
        # Make sure it's still power of 2
        if kernel_topk != tilelang.math.next_power_of_2(kernel_topk):
            kernel_topk = tilelang.math.next_power_of_2(kernel_topk) // 2
        kernel_topk = max(8, kernel_topk)
        # FIXME: Read from model config
        dtype_str = "bfloat16"
        self.decode_fused_kernels = {}
        self.prefill_fused_kernels = {}
        bucketed_pooled_k_len = _bucket_size(pooled_k_len)

        pooling_block_stride = self.block_size // self.kernel_stride  # = 64 // 16 = 4
        pooling_pad_len = (
            self.kernel_size // self.kernel_stride - 1
        )  # = 32 // 16 - 1 = 1
        pooling_num_offs = (
            self.kernel_size // self.kernel_stride
            + self.block_size // self.kernel_stride
            - 1
        )

        if model_runner.server_args.minicpm_fuse_topk:
            bucketed_actual_max_seqlen_q = _bucket_size(
                model_runner.server_args.chunked_prefill_size
            )
            bucketed_actual_max_seqlen_k = _bucket_size(
                self.max_context_len // self.kernel_stride
            )
            for bs in range(1, model_runner.server_args.max_running_requests + 1):
                decode_kernel = fused_attn_pooling_online_topk_decode(
                    batch_size=bs,
                    groups=self.heads_per_group,
                    heads=model_runner.model_config.num_attention_heads,
                    dim=self.head_dim,
                    topk=kernel_topk,
                    pooled_k_len=bucketed_pooled_k_len,
                    m_block_dim=16,
                    block_stride=pooling_block_stride,
                    pad_len=pooling_pad_len,
                    num_offs=pooling_num_offs,
                    block_size=self.block_size,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    dtype_str=dtype_str,
                )
                self.decode_fused_kernels[bs] = decode_kernel
                prefill_kernel = fused_attn_pooling_online_topk_prefill(
                    batch_size=bs,
                    groups=self.heads_per_group,
                    heads=model_runner.model_config.num_attention_heads,
                    dim=self.head_dim,
                    topk=kernel_topk,
                    max_seqlen_q_grid=model_runner.server_args.chunked_prefill_size,  # Bucketed for grid
                    pooled_k_len=bucketed_pooled_k_len,
                    actual_max_seqlen_q=bucketed_actual_max_seqlen_q,  # Bucketed for causal mask
                    actual_max_seqlen_k=bucketed_actual_max_seqlen_k,  # Bucketed for causal mask
                    m_block_dim=16,
                    block_stride=pooling_block_stride,
                    pad_len=pooling_pad_len,
                    num_offs=pooling_num_offs,
                    block_size=self.block_size,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    dtype_str=dtype_str,
                )
                self.prefill_fused_kernels[bs] = prefill_kernel

        if model_runner.server_args.attention_backend != "minicpm_flashattn":
            raise ValueError(
                "MiniCPM sparse attention requires "
                "attention_backend='minicpm_flashattn'."
            )

        # Initialize sparse attention helpers (required for MiniCPM)
        sparse_config = SparseConfig.from_model_config(
            hf_config, model_runner.model_config
        )
        self.sparse_batch_analyzer = SparseBatchAnalyzer(sparse_config)
        self.sparse_metadata_builder = SparseMetadataBuilder(
            sparse_config,
            num_kv_heads=self.num_kv_heads,
            max_context_len=self.max_context_len,
        )

    def update_batch_for_sparse(
        self, forward_batch: ForwardBatch, metadata: FlashAttentionMetadata
    ):
        cu_seqlens_q = metadata.cu_seqlens_q

        compression_metadata = (
            self.sparse_metadata_builder.build_k1_k2_compression_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                req_to_sparse_k1_token=self.req_to_sparse_k1_token,
                req_to_sparse_k2_token=self.req_to_sparse_k2_token,
                k1_kernel_size=self.k1_kernel_size,
                k1_kernel_stride=self.k1_kernel_stride,
                k2_kernel_size=self.k2_kernel_size,
                k2_kernel_stride=self.k2_kernel_stride,
                cu_seqlens_q=cu_seqlens_q,
            )
        )

        # Map k1/k2 compression metadata objects
        metadata.k1 = compression_metadata["k1"]
        metadata.k2 = compression_metadata["k2"]

        if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.sparse_bs_list = (
                self.sparse_batch_analyzer.identify_sparse_batches(
                    forward_batch, self.minicpm_dense_as_sparse
                )
            )

            seqlen_q_sparse_bs, metadata.seqlen_k_sparse_bs_tensor = (
                self.sparse_metadata_builder.build_sequence_lengths(
                    cu_seqlens_q,
                    forward_batch.extend_prefix_lens,
                    metadata.sparse_bs_list,
                )
            )

            cu_seqlens_q_sparse_bs = torch.tensor(
                [0] + seqlen_q_sparse_bs, dtype=torch.int32, device=cu_seqlens_q.device
            ).cumsum(dtype=torch.int32, dim=0)

            extend_prefix_lens_sparse = torch.tensor(
                [
                    forward_batch.extend_prefix_lens_cpu[bs]
                    for bs in metadata.sparse_bs_list
                ],
                dtype=torch.long,
                device="cpu",
            )

            metadata.token_to_bs, metadata.token_pos_in_bs = (
                self.sparse_metadata_builder.build_token_mappings(
                    cu_seqlens_q_sparse_bs,
                    extend_prefix_lens_sparse,
                    seqlen_q_sparse_bs,
                )
            )
            metadata.token_to_bs = metadata.token_to_bs.to(
                device=metadata.cu_seqlens_q.device
            )
            metadata.token_pos_in_bs = metadata.token_pos_in_bs.to(
                device=metadata.cu_seqlens_q.device
            )

            prefill_metadata = (
                self.sparse_metadata_builder.build_sparse_prefill_metadata(
                    forward_batch=forward_batch,
                    base_metadata=metadata,
                    sparse_bs_list=metadata.sparse_bs_list,
                    head_group_num=self.head_group_num,
                    dense_len=self.dense_len,
                    sparse_topk=self.sparse_topk,
                    block_size=self.block_size,
                    cu_seqlens_q=cu_seqlens_q,
                    sparse_page_table_dtype=metadata.page_table.dtype,
                    sparse_page_table_device=metadata.page_table.device,
                )
            )

            metadata.sparse_page_table = prefill_metadata["sparse_page_table"]
            metadata.sparse_cu_seqlens_q_cpu = prefill_metadata[
                "sparse_cu_seqlens_q_cpu"
            ]
            metadata.sparse_cu_seqlens_q = prefill_metadata["sparse_cu_seqlens_q"]
            metadata.old_bs_to_new_bs_range = prefill_metadata["old_bs_to_new_bs_range"]
            metadata.sparse_max_seq_len_q = prefill_metadata["sparse_max_seq_len_q"]

            metadata.sparse_batch_size = len(metadata.sparse_bs_list)
            metadata.sparse_idx = prefill_metadata["sparse_idx"]

            # Stage1 optimization metadata for prefill mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            seqlens_q_sparse_list = []
            for i in range(forward_batch.batch_size):
                if forward_batch.seq_lens_cpu[i] >= self.dense_len:
                    seqlens_q_sparse_list.append(forward_batch.extend_seq_lens_cpu[i])

            if len(seqlens_q_sparse_list) > 0:
                seqlen_q_sparse_tensor = torch.tensor(
                    seqlens_q_sparse_list,
                    dtype=torch.int32,
                    device=metadata.cu_seqlens_q.device,
                )
                cu_seqlen_q_sparse_tensor = F.pad(
                    torch.cumsum(seqlen_q_sparse_tensor, dim=0, dtype=torch.int32),
                    (1, 0),
                )
                metadata.cu_seqlens_q_adjusted = (
                    cu_seqlen_q_sparse_tensor * self.heads_per_group
                )
                metadata.max_seqlen_q_adjusted = (
                    seqlen_q_sparse_tensor.max().item() * self.heads_per_group
                )
            else:
                metadata.cu_seqlens_q_adjusted = (
                    metadata.cu_seqlens_q * self.heads_per_group
                )
                metadata.max_seqlen_q_adjusted = (
                    metadata.max_seq_len_q * self.heads_per_group
                )
        else:
            decode_metadata = self.sparse_metadata_builder.build_sparse_decode_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                head_group_num=self.head_group_num,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                block_size=self.block_size,
            )

            metadata.sparse_cache_seqlens_int32 = decode_metadata[
                "sparse_cache_seqlens_int32"
            ]
            metadata.sparse_cu_seqlens_k = decode_metadata["sparse_cu_seqlens_k"]
            metadata.sparse_cu_seqlens_q = decode_metadata["sparse_cu_seqlens_q"]
            metadata.sparse_page_table = decode_metadata["sparse_page_table"]
            metadata.token_to_bs = decode_metadata["token_to_bs"]

            # Stage1 optimization metadata for decode mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            metadata.cu_seqlens_q_adjusted = (
                metadata.cu_seqlens_q * self.heads_per_group
            )
            metadata.max_seqlen_q_adjusted = (
                metadata.max_seq_len_q * self.heads_per_group
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_target_verify():
            raise NotImplementedError(
                "MiniCPM backend does not support speculative decoding (target verify)"
            )
        if forward_batch.forward_mode.is_draft_extend_v2():
            raise NotImplementedError(
                "MiniCPM backend does not support speculative decoding (draft extend)"
            )

        self.base_backend.init_forward_metadata(forward_batch)
        metadata = self.base_backend.forward_metadata
        self.update_batch_for_sparse(forward_batch, metadata)
        self.forward_metadata = metadata

    def get_topk_for_sparse(
        self,
        query_states,
        key_states,
        value_states,
        query_length,
        layer,
        forward_batch,
        is_prefill=True,
        dropout=0.0,
        softmax_scale=None,
        no_rope_param=None,
        past_key_value=None,
        decode_batch_id=0,
    ):
        if is_prefill:
            all_sparse = (
                self.forward_metadata.sparse_batch_size == forward_batch.batch_size
            )
            if all_sparse:
                # all batch is sparse
                metadata = self.forward_metadata
                compressed_k = torch.full(
                    (
                        forward_batch.batch_size
                        * self.max_context_len
                        // self.k1_kernel_stride,
                        self.head_group_num,
                        self.head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.device,
                    fill_value=float("-inf"),
                )
                compressed_k2 = torch.full(
                    (
                        forward_batch.batch_size
                        * self.max_context_len
                        // self.k2_kernel_stride,
                        self.head_group_num,
                        self.head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.device,
                    fill_value=float("-inf"),
                )

                get_compress_k_v2(
                    layer=layer,
                    forward_batch=forward_batch,
                    metadata=metadata,
                    full_compressed_k1=compressed_k,  # output
                    full_compressed_k2=compressed_k2,  # output
                    max_context_length=self.max_context_len,
                )

                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_in_batch_k = metadata.max_seq_len_k
                cu_seqlens_q = metadata.cu_seqlens_q
                max_seqlen_in_batch_q = metadata.max_seq_len_q

                ret = self.sparse_get_topk_impl(
                    query_states,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_in_batch_q,
                    max_seqlen_in_batch_k,
                    no_rope_param=no_rope_param,
                    compressed_k=compressed_k,
                    compressed_cu_seqlens=metadata.k1.cu_seqlens,
                    compressed_k2=compressed_k2,
                    compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                    fused_kernel=(
                        self.prefill_fused_kernels[forward_batch.batch_size]
                        if self.minicpm_fuse_topk
                        else None
                    ),
                )
                return ret

            topk_metadata = self.sparse_metadata_builder.build_prefill_topk_metadata(
                forward_batch=forward_batch,
                base_metadata=self.forward_metadata,
                key_states=key_states,
                query_states=query_states,
                tp_q_head_num=layer.tp_q_head_num,
                head_dim=layer.head_dim,
                compress_k1_kernel_size=self.k1_kernel_size,
                compress_k1_kernel_stride=self.k1_kernel_stride,
                compress_k2_kernel_size=self.k2_kernel_size,
                compress_k2_kernel_stride=self.k2_kernel_stride,
                dense_len=self.dense_len,
            )

            sparse_bs = topk_metadata["sparse_bs"]
            topk_metadata["seqlens_q_sparse_bs"]
            topk_metadata["seqlens_k_sparse_bs"]
            k1_lens = topk_metadata["k1_lens"]
            k2_lens = topk_metadata["k2_lens"]

            full_compressed_k1, full_compressed_k2 = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                k1_token_nums=sum(k1_lens),
                k2_token_nums=sum(k2_lens),
                dtype=key_states.dtype,
                device=key_states.device,
                max_context_length=self.max_context_len,
                minicpm_split_stage1=self.minicpm_split_stage1,
            )

            pt_k1, pt_k2 = 0, 0
            compressed_k = torch.zeros(
                (sum(k1_lens[sparse_bs]), layer.tp_k_head_num, layer.head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            compressed_k2 = torch.zeros(
                (sum(k2_lens[sparse_bs]), layer.tp_k_head_num, layer.head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )

            compressed_cu_seqlens, compressed_cu_seqlens2 = [0], [0]

            for sparse_bs_idx in sparse_bs:
                start = self.forward_metadata.k1.cu_seqlens[sparse_bs_idx]
                end = self.forward_metadata.k1.cu_seqlens[sparse_bs_idx + 1]
                compressed_k[pt_k1 : pt_k1 + (end - start), :, :] = full_compressed_k1[
                    start:end, :, :
                ]

                start2 = self.forward_metadata.k2.cu_seqlens[sparse_bs_idx]
                end2 = self.forward_metadata.k2.cu_seqlens[sparse_bs_idx + 1]
                compressed_k2[pt_k2 : pt_k2 + (end2 - start2), :, :] = (
                    full_compressed_k2[start2:end2, :, :]
                )

                pt_k1 += k1_lens[sparse_bs_idx]
                pt_k2 += k2_lens[sparse_bs_idx]
                compressed_cu_seqlens.append(
                    compressed_cu_seqlens[-1] + k1_lens[sparse_bs_idx]
                )
                compressed_cu_seqlens2.append(
                    compressed_cu_seqlens2[-1] + k2_lens[sparse_bs_idx]
                )

            compressed_cu_seqlens = torch.tensor(
                compressed_cu_seqlens, dtype=torch.int32, device=key_states.device
            )
            compressed_cu_seqlens2 = torch.tensor(
                compressed_cu_seqlens2, dtype=torch.int32, device=key_states.device
            )

            cu_seqlens_q = topk_metadata["cu_seqlens_q"]
            cu_seqlens_k = topk_metadata["cu_seqlens_k"]
            max_seqlen_in_batch_q = topk_metadata["max_seqlen_q"]
            max_seqlen_in_batch_k = topk_metadata["max_seqlen_k"]
            query_states = topk_metadata["query_states"]

            ret = self.sparse_get_topk_impl(
                query_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                no_rope_param=no_rope_param,
                compressed_k=compressed_k,
                compressed_cu_seqlens=compressed_cu_seqlens,
                compressed_k2=compressed_k2,
                compressed_cu_seqlens2=compressed_cu_seqlens2,
                fused_kernel=(
                    self.prefill_fused_kernels[forward_batch.batch_size]
                    if self.minicpm_fuse_topk
                    else None
                ),
            )
            return ret
        else:
            metadata = self.forward_metadata

            if self.enable_cuda_graph:
                if self.minicpm_split_stage1:
                    get_compress_k_v2_padded(
                        layer=layer,
                        forward_batch=forward_batch,
                        metadata=metadata,
                        full_compressed_k1=self.decode_cuda_graph_metadata[
                            "compress_k1"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k1_kernel_stride,
                            :,
                            :,
                        ],
                        full_compressed_k2=self.decode_cuda_graph_metadata[
                            "compress_k2"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k2_kernel_stride,
                            :,
                            :,
                        ],
                        max_context_length=self.max_context_len,
                    )
                else:
                    get_compress_k_v2(
                        layer=layer,
                        forward_batch=forward_batch,
                        metadata=metadata,
                        full_compressed_k1=self.decode_cuda_graph_metadata[
                            "compress_k1"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k1_kernel_stride,
                            :,
                            :,
                        ],
                        full_compressed_k2=self.decode_cuda_graph_metadata[
                            "compress_k2"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k2_kernel_stride,
                            :,
                            :,
                        ],
                        max_context_length=self.max_context_len,
                    )
            else:
                compressed_k, compressed_k2 = allocate_and_compress_keys(
                    layer=layer,
                    forward_batch=forward_batch,
                    metadata=metadata,
                    k1_token_nums=forward_batch.batch_size
                    * self.max_context_len
                    // self.k1_kernel_stride,
                    k2_token_nums=forward_batch.batch_size
                    * self.max_context_len
                    // self.k2_kernel_stride,
                    dtype=torch.bfloat16,
                    device=self.device,
                    max_context_length=self.max_context_len,
                    minicpm_split_stage1=self.minicpm_split_stage1,
                )

            topk_metadata = self.sparse_metadata_builder.build_decode_topk_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                query_states=query_states,
            )

            cu_seqlens_q = topk_metadata["cu_seqlens_q"]
            cu_seqlens_k = topk_metadata["cu_seqlens_k"]
            max_seqlen_in_batch_q = topk_metadata["max_seqlen_q"]
            max_seqlen_in_batch_k = topk_metadata["max_seqlen_k"]
            query_states = topk_metadata["query_states"]

            if self.enable_cuda_graph:
                ret = self.sparse_get_topk_impl(
                    query_states,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_in_batch_q,
                    max_seqlen_in_batch_k,
                    no_rope_param=no_rope_param,
                    compressed_k=self.decode_cuda_graph_metadata["compress_k1"][
                        : forward_batch.batch_size
                        * self.max_context_len
                        // self.k1_kernel_stride,
                        :,
                        :,
                    ],
                    compressed_cu_seqlens=metadata.k1.cu_seqlens,
                    compressed_k2=self.decode_cuda_graph_metadata["compress_k2"][
                        : forward_batch.batch_size
                        * self.max_context_len
                        // self.k2_kernel_stride,
                        :,
                        :,
                    ],
                    compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                    fused_kernel=(
                        self.decode_fused_kernels[forward_batch.batch_size]
                        if self.minicpm_fuse_topk
                        else None
                    ),
                )

            else:
                ret = self.sparse_get_topk_impl(
                    query_states,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_in_batch_q,
                    max_seqlen_in_batch_k,
                    no_rope_param=no_rope_param,
                    compressed_k=compressed_k,
                    compressed_cu_seqlens=metadata.k1.cu_seqlens,
                    compressed_k2=compressed_k2,
                    compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                    fused_kernel=(
                        self.decode_fused_kernels[forward_batch.batch_size]
                        if self.minicpm_fuse_topk
                        else None
                    ),
                )

        return ret

    def sparse_get_topk_impl(
        self,
        query_layer,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        #    max_seqlen_k1,
        no_rope_param=None,
        compressed_k=None,
        compressed_cu_seqlens=None,
        compressed_k2=None,
        compressed_cu_seqlens2=None,
        fused_kernel=None,
    ):
        cache_lens = None
        if max_seqlen_in_batch_k > max_seqlen_in_batch_q:
            if max_seqlen_in_batch_q == 1:
                cache_lens = self.forward_metadata.cache_seqlens_int32_stage1
            else:
                seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                cache_lens = seq_lens_k - seq_lens_q
        else:
            batch_size = cu_seqlens_q.shape[0] - 1
            cache_lens = torch.zeros(
                batch_size, dtype=torch.int32, device=cu_seqlens_q.device
            )

        if not self.minicpm_fuse_topk:
            topk_idx = compressed_attention(
                (
                    query_layer
                    if no_rope_param is None
                    else no_rope_param["query_states_no_rope"]
                ),
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.sparse_topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_cu_seqlens2,
                max_seqlen_in_batch_q,
                # self.max_context_len // self.kernel_size,
                self.max_context_len,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                cu_seqlens_q_adjusted=self.forward_metadata.cu_seqlens_q_adjusted,
                max_seqlen_q_adjusted=self.forward_metadata.max_seqlen_q_adjusted,
                # block_score_buffer=self.forward_metadata.block_score_buffer
                minicpm_split_stage1=self.minicpm_split_stage1,
            )
        else:
            topk_idx = compressed_attention_tilelang(
                (
                    query_layer
                    if no_rope_param is None
                    else no_rope_param["query_states_no_rope"]
                ),
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.sparse_topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_cu_seqlens2,
                max_seqlen_in_batch_q,
                self.forward_metadata.k1.max_seq_len,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                fused_kernel=fused_kernel,
                max_cache_len=self.max_context_len,
            )

        return topk_idx

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if layer.is_cross_attention:
            raise NotImplementedError(
                "MiniCPM backend does not support cross attention"
            )
        if forward_batch.forward_mode.is_draft_extend_v2():
            raise NotImplementedError(
                "MiniCPM backend does not support draft extend mode"
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                self.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case,
        # 4) fa_impl_ver != 4 since fa4 does not currently support fp8 queries and keys.
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        # MiniCPM backend does not support cross attention or encoder-only attention
        causal = True

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        # Get the appropriate page table (without local attention or SWA support)
        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens_k = metadata.cu_seqlens_k

        bs = forward_batch.batch_size
        if max(forward_batch.seq_lens_cpu) >= self.dense_len:
            q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            topk_idx = self.get_topk_for_sparse(
                q_reshaped, k, v, q.shape[0], layer, forward_batch
            )

            sparse_page_table_sparse_bs = get_block_table_v2(
                topk_idx,
                page_table,
                metadata.token_to_bs,
                metadata.token_pos_in_bs,
                metadata.seqlen_k_sparse_bs_tensor,
            ).reshape(-1, self.num_sparse_topk_tokens)

            # copy page table for sparse bs
            metadata.sparse_page_table[
                metadata.sparse_idx, : self.num_sparse_topk_tokens
            ] = sparse_page_table_sparse_bs
        else:
            total_k1 = self.forward_metadata.k1.cu_total_compress_token_nums[-1].item()
            total_k2 = self.forward_metadata.k2.cu_total_compress_token_nums[-1].item()

            full_compressed_k1_ext, full_compressed_k2_ext = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                k1_token_nums=total_k1,
                k2_token_nums=total_k2,
                dtype=k.dtype,
                device=k.device,
                max_context_length=self.max_context_len,
                minicpm_split_stage1=self.minicpm_split_stage1,
            )

        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim)
        if metadata.sparse_batch_size < bs:
            # copy dense page table for dense bs
            dense_bs_list = [i for i in range(bs) if i not in metadata.sparse_bs_list]
            for dense_bs in dense_bs_list:
                kv_len = forward_batch.seq_lens_cpu[dense_bs]
                sparse_page_table_idx_start = metadata.old_bs_to_new_bs_range[dense_bs]
                sparse_page_table_idx_end = metadata.old_bs_to_new_bs_range[
                    dense_bs + 1
                ]
                assert sparse_page_table_idx_end - sparse_page_table_idx_start == 2, (
                    "dense bs should have 2 head_group, but get {}".format(
                        sparse_page_table_idx_end - sparse_page_table_idx_start
                    )
                )

                ps = metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start]
                len_ = (
                    metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1]
                    - ps
                )
                assert len_ == forward_batch.extend_seq_lens_cpu[dense_bs], (
                    "dense bs seqlen mismatch {} vs {}".format(
                        len_, forward_batch.extend_seq_lens_cpu[dense_bs]
                    )
                )
                t = q_reshaped[ps : ps + 2 * len_, :, :].clone()
                q_reshaped[ps : ps + len_, :, :] = t[0::2, :, :]
                q_reshaped[ps + len_ : ps + 2 * len_, :, :] = t[1::2, :, :]

                metadata.sparse_page_table[sparse_page_table_idx_start, :kv_len] = (
                    page_table[dense_bs, :kv_len] * 2
                )
                metadata.sparse_page_table[sparse_page_table_idx_start + 1, :kv_len] = (
                    page_table[dense_bs, :kv_len] * 2 + 1
                )

        metadata.sparse_cache_seqlens_int32 = (
            (metadata.sparse_page_table != 0)
            .sum(dim=1)
            .to(dtype=cache_seqlens.dtype, device=cache_seqlens.device)
        )

        # this seem not necessary to update perlayer
        metadata.sparse_cu_seqlens_k = F.pad(
            torch.cumsum(
                metadata.sparse_cache_seqlens_int32, dim=0, dtype=cu_seqlens_k.dtype
            ),
            (1, 0),
        )

        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num // 2, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num // 2, layer.head_dim
        )

        result = flash_attn_with_kvcache(
            q=q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim),
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=metadata.sparse_page_table,
            cache_seqlens=metadata.sparse_cache_seqlens_int32,
            cu_seqlens_q=metadata.sparse_cu_seqlens_q,
            cu_seqlens_k_new=metadata.sparse_cu_seqlens_k,
            max_seqlen_q=metadata.sparse_max_seq_len_q,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            num_splits=self.num_splits,
            ver=self.fa_impl_ver,
            **kwargs,
        )

        if metadata.sparse_batch_size < bs:
            dense_bs_list = [i for i in range(bs) if i not in metadata.sparse_bs_list]
            for dense_bs in dense_bs_list:
                sparse_page_table_idx_start = metadata.old_bs_to_new_bs_range[dense_bs]
                sparse_page_table_idx_end = metadata.old_bs_to_new_bs_range[
                    dense_bs + 1
                ]
                assert sparse_page_table_idx_end - sparse_page_table_idx_start == 2, (
                    "dense bs should have 2 head_group"
                )

                ps = metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start]
                len_ = (
                    metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1]
                    - ps
                )
                assert len_ == forward_batch.extend_seq_lens_cpu[dense_bs], (
                    "dense bs seqlen mismatch {} vs {}".format(
                        len_, forward_batch.extend_seq_lens_cpu[dense_bs]
                    )
                )
                t = result[ps : ps + 2 * len_, :, :].clone()
                result[ps : ps + 2 * len_ : 2, :, :] = t[0:len_, :, :]
                result[ps + 1 : ps + 2 * len_ : 2, :, :] = t[len_ : 2 * len_, :, :]

        return result.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.fa_impl_ver in [3], "Only FA3 support decoding"

        # Check for unsupported features
        if layer.is_cross_attention:
            raise NotImplementedError(
                "MiniCPM backend does not support cross attention"
            )
        # MiniCPM does not support local attention

        bs = forward_batch.batch_size
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                self.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        # MiniCPM backend does not support cross attention or encoder-only attention
        causal = True

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        # Do multi-head attention (without cross-attention or local attention support)

        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        topk_idx = self.get_topk_for_sparse(
            q_reshaped.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            1,
            layer,
            forward_batch,
            False,
        )
        sparse_page_table = get_block_table_v3(
            topk_idx,
            page_table,
            metadata.token_to_bs,
            cache_seqlens,
            cache_seqlens,
        ).reshape(-1, self.num_sparse_topk_tokens)

        metadata.sparse_page_table[: 2 * bs, : self.num_sparse_topk_tokens] = (
            sparse_page_table[:, : self.num_sparse_topk_tokens]
        )

        q_reshaped_by_head_group = q_reshaped.reshape(
            -1, layer.tp_q_head_num // 2, layer.head_dim
        )
        assert self.page_size == 1
        key_cache_by_head_group = key_cache.reshape(
            -1, self.page_size, layer.tp_k_head_num // 2, layer.head_dim
        )
        value_cache_by_head_group = value_cache.reshape(
            -1, self.page_size, layer.tp_v_head_num // 2, layer.head_dim
        )

        sparse_cache_seqlens = metadata.sparse_cache_seqlens_int32
        sparse_cu_seqlens_k = metadata.sparse_cu_seqlens_k
        sparse_cu_seqlens_q = metadata.sparse_cu_seqlens_q

        result = flash_attn_with_kvcache(
            q=q_reshaped_by_head_group,
            k_cache=key_cache_by_head_group,
            v_cache=value_cache_by_head_group,
            page_table=metadata.sparse_page_table,
            cache_seqlens=sparse_cache_seqlens,
            cu_seqlens_q=sparse_cu_seqlens_q,
            cu_seqlens_k_new=sparse_cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            num_splits=self.num_splits,
            ver=self.fa_impl_ver,
            **kwargs,
        )

        return result.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.base_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        buffers = self.base_backend.decode_cuda_graph_metadata
        self.decode_cuda_graph_metadata = buffers
        sparse_max_num_pages = (
            self.num_sparse_topk_tokens + self.page_size - 1
        ) // self.page_size
        buffers.update(
            {
                "sparse_cache_seqlens": torch.full(
                    (max_bs * 2,),
                    self.num_sparse_topk_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "sparse_cu_seqlens_q": torch.arange(
                    0, max_bs * 2 + 1, dtype=torch.int32, device=self.device
                ),
                "sparse_cu_seqlens_k": torch.arange(
                    0,
                    (max_bs * 2 + 1) * self.num_sparse_topk_tokens,
                    self.num_sparse_topk_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "token_to_bs": torch.arange(
                    0, max_bs, dtype=torch.int32, device=self.device
                ),
                "token_pos_in_bs": torch.ones(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "sparse_page_table": torch.zeros(
                    max_bs * 2,
                    sparse_max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q_adjusted": torch.arange(
                    0, max_bs + 1, dtype=torch.int32, device=self.device
                )
                * self.heads_per_group,
                "cache_seqlens_int32_stage1": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
            }
        )

        for name, kernel_size, kernel_stride in (
            ("k1", self.k1_kernel_size, self.k1_kernel_stride),
            ("k2", self.k2_kernel_size, self.k2_kernel_stride),
        ):
            max_num_pages = (
                max(
                    0,
                    (self.max_context_len - kernel_size) // kernel_stride + 1,
                )
                + self.page_size
                - 1
            ) // self.page_size
            buffers[f"compress_{name}"] = torch.zeros(
                (
                    max_bs * self.max_context_len // kernel_stride,
                    self.head_group_num,
                    self.head_dim,
                ),
                dtype=torch.bfloat16,
                device=self.device,
            )
            buffers[f"{name}.table"] = torch.zeros(
                max_bs, max_num_pages, dtype=torch.int32, device=self.device
            )
            for field in (
                "history_compress_token_nums",
                "new_token_nums",
                "new_compress_token_nums",
                "total_compress_token_nums",
            ):
                buffers[f"{name}.{field}"] = torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                )
            for field in (
                "cu_seqlens",
                "cu_new_token_nums",
                "cu_new_compress_token_nums",
                "cu_total_compress_token_nums",
            ):
                buffers[f"{name}.{field}"] = torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        if not forward_batch.forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                "MiniCPM backend CUDA graph only supports decode/idle mode, "
                f"got {forward_batch.forward_mode}"
            )

        self.base_backend.init_forward_metadata_out_graph(forward_batch, in_capture)
        metadata = self.base_backend.forward_metadata
        if in_capture:
            self._bind_sparse_graph_metadata(forward_batch, metadata)
        else:
            self._replay_sparse_graph_metadata(forward_batch, metadata)
        self.forward_metadata = metadata

    def _build_sparse_decode_replay_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: FlashAttentionMetadata,
    ):
        decode_metadata = self.sparse_metadata_builder.build_sparse_decode_metadata(
            forward_batch=forward_batch,
            base_metadata=metadata,
            head_group_num=self.head_group_num,
            dense_len=self.dense_len,
            sparse_topk=self.sparse_topk,
            block_size=self.block_size,
        )
        compression_metadata = (
            self.sparse_metadata_builder.build_k1_k2_compression_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                req_to_sparse_k1_token=self.req_to_sparse_k1_token,
                req_to_sparse_k2_token=self.req_to_sparse_k2_token,
                k1_kernel_size=self.k1_kernel_size,
                k1_kernel_stride=self.k1_kernel_stride,
                k2_kernel_size=self.k2_kernel_size,
                k2_kernel_stride=self.k2_kernel_stride,
                cu_seqlens_q=metadata.cu_seqlens_q,
            )
        )
        return decode_metadata, compression_metadata

    def _bind_sparse_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: FlashAttentionMetadata,
    ):
        bs = forward_batch.batch_size
        buffers = self.decode_cuda_graph_metadata
        metadata.sparse_cache_seqlens_int32 = buffers["sparse_cache_seqlens"][: 2 * bs]
        metadata.sparse_cu_seqlens_q = buffers["sparse_cu_seqlens_q"][: 2 * bs + 1]
        metadata.sparse_cu_seqlens_k = buffers["sparse_cu_seqlens_k"][: 2 * bs + 1]
        metadata.token_to_bs = buffers["token_to_bs"][:bs]
        metadata.token_pos_in_bs = buffers["token_pos_in_bs"][:bs]
        metadata.sparse_page_table = buffers["sparse_page_table"][: 2 * bs]
        metadata.total_q = bs

        assume_kv_len = self.config_dense_len
        metadata.cu_seqlens_k.copy_(
            torch.arange(bs + 1, device=self.device, dtype=torch.int32) * assume_kv_len
        )
        metadata.max_seq_len_k = assume_kv_len

        for name, kernel_size, kernel_stride in (
            ("k1", self.k1_kernel_size, self.k1_kernel_stride),
            ("k2", self.k2_kernel_size, self.k2_kernel_stride),
        ):
            level = CompressionLevelMetadata()
            setattr(metadata, name, level)
            level_len = max(0, (assume_kv_len - kernel_size) // kernel_stride + 1)
            level.max_seq_len = level_len
            level.cu_seqlens = buffers[f"{name}.cu_seqlens"][: bs + 1]
            level.cu_seqlens.copy_(
                torch.arange(bs + 1, device=self.device, dtype=torch.int32) * level_len
            )
            level.table = buffers[f"{name}.table"][:bs]
            for field in (
                "history_compress_token_nums",
                "new_token_nums",
                "new_compress_token_nums",
                "total_compress_token_nums",
            ):
                setattr(level, field, buffers[f"{name}.{field}"][:bs])
            for field in (
                "cu_new_token_nums",
                "cu_new_compress_token_nums",
                "cu_total_compress_token_nums",
            ):
                setattr(level, field, buffers[f"{name}.{field}"][: bs + 1])

        metadata.cu_seqlens_q_adjusted = buffers["cu_seqlens_q_adjusted"][: bs + 1]
        metadata.cache_seqlens_int32_stage1 = buffers["cache_seqlens_int32_stage1"][:bs]
        metadata.max_seqlen_q_adjusted = metadata.max_seq_len_q * self.heads_per_group

    def _replay_sparse_graph_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: FlashAttentionMetadata,
    ):
        bs = forward_batch.batch_size
        real_bs = bs - forward_batch.num_padding
        if real_bs == 0:
            metadata.sparse_cache_seqlens_int32.zero_()
            metadata.sparse_cu_seqlens_k.zero_()
            metadata.cache_seqlens_int32_stage1.zero_()
            return

        sparse_forward_batch = SimpleNamespace(
            batch_size=real_bs,
            req_pool_indices=forward_batch.req_pool_indices[:real_bs],
            seq_lens_cpu=forward_batch.seq_lens_cpu[:real_bs],
        )
        decode_metadata, compression_metadata = (
            self._build_sparse_decode_replay_metadata(sparse_forward_batch, metadata)
        )
        metadata.sparse_cache_seqlens_int32[: 2 * real_bs].copy_(
            decode_metadata["sparse_cache_seqlens_int32"]
        )
        metadata.sparse_cu_seqlens_k[: 2 * real_bs + 1].copy_(
            decode_metadata["sparse_cu_seqlens_k"]
        )
        metadata.cache_seqlens_int32_stage1[:real_bs].copy_(
            metadata.cache_seqlens_int32[:real_bs] - 1
        )

        for name, kernel_stride, req_to_sparse in (
            ("k1", self.k1_kernel_stride, self.req_to_sparse_k1_token),
            ("k2", self.k2_kernel_stride, self.req_to_sparse_k2_token),
        ):
            self.decode_cuda_graph_metadata[f"compress_{name}"][
                : real_bs * self.max_context_len // kernel_stride
            ].fill_(float("-inf"))
            dst = getattr(metadata, name)
            src = compression_metadata[name]
            for field in (
                "history_compress_token_nums",
                "new_token_nums",
                "new_compress_token_nums",
                "total_compress_token_nums",
            ):
                getattr(dst, field)[:real_bs].copy_(getattr(src, field))
                if real_bs < bs:
                    getattr(dst, field)[real_bs:].zero_()
            for field in (
                "cu_seqlens",
                "cu_new_token_nums",
                "cu_new_compress_token_nums",
                "cu_total_compress_token_nums",
            ):
                dst_field = getattr(dst, field)
                src_field = getattr(src, field)
                dst_field[: real_bs + 1].copy_(src_field)
                if real_bs < bs:
                    dst_field[real_bs + 1 :].fill_(src_field[-1])
            dst.table.copy_(req_to_sparse[forward_batch.req_pool_indices])

        if real_bs < bs:
            metadata.sparse_cache_seqlens_int32[2 * real_bs :].zero_()
            metadata.sparse_cu_seqlens_k[2 * real_bs + 1 :].fill_(
                decode_metadata["sparse_cu_seqlens_k"][-1]
            )
            metadata.cache_seqlens_int32_stage1[real_bs:].zero_()

    def get_cuda_graph_seq_len_fill_value(self):
        return self.base_backend.get_cuda_graph_seq_len_fill_value()
