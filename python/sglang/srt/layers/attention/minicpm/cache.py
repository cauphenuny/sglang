from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.allocation import alloc_token_slots
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


class MiniCPMCompressedCache:
    def __init__(
        self,
        pool: ReqToTokenPool,
        *,
        kernel_size: int,
        kernel_stride: int,
        enable_memory_saver: bool,
    ):
        self.pool = pool
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        saver = TorchMemorySaverAdapter.create(enable=enable_memory_saver)
        with saver.region(GPU_MEMORY_TYPE_KV_CACHE):
            k1_size = (pool.max_context_len - kernel_size) // kernel_stride + 1
            k2_size = (pool.max_context_len - kernel_size * 4) // (
                kernel_stride * 4
            ) + 1
            pool.req_to_sparse_k1_token = torch.zeros(
                (pool._alloc_size, k1_size), dtype=torch.int32, device=pool.device
            )
            pool.req_to_sparse_k2_token = torch.zeros(
                (pool._alloc_size, k2_size), dtype=torch.int32, device=pool.device
            )
        self.allocated_lens = [
            [0] * pool._alloc_size,
            [0] * pool._alloc_size,
        ]
        self.allocator = None

    def _sparse_len(self, length: int, scale: int) -> int:
        kernel_size = self.kernel_size * scale
        if length < kernel_size:
            return 0
        return (length - kernel_size) // (self.kernel_stride * scale) + 1

    def tokens_needed(self, req_pool_idx: int | None, target_seq_len: int) -> int:
        current = (
            0
            if req_pool_idx is None
            else sum(lengths[req_pool_idx] for lengths in self.allocated_lens)
        )
        target = sum(self._sparse_len(target_seq_len, scale) for scale in (1, 4))
        return max(target - current, 0)

    def _allocate_to_lengths(
        self,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        req_indices = req_pool_indices_cpu.tolist()
        seq_lens = seq_lens_cpu.tolist()
        tables = (
            self.pool.req_to_sparse_k1_token,
            self.pool.req_to_sparse_k2_token,
        )
        plans = []
        for level, (table, scale) in enumerate(zip(tables, (1, 4))):
            targets = {
                req_idx: self._sparse_len(seq_len, scale)
                for req_idx, seq_len in zip(req_indices, seq_lens)
            }
            rows = [
                (req_idx, self.allocated_lens[level][req_idx], target)
                for req_idx, target in targets.items()
                if target > self.allocated_lens[level][req_idx]
            ]
            plans.append((table, rows, sum(end - start for _, start, end in rows)))

        allocator = tree_cache.token_to_kv_pool_allocator
        if self.allocator is not None and self.allocator is not allocator:
            raise RuntimeError("MiniCPM compressed cache allocator changed")

        allocated = []
        try:
            for _, _, size in plans:
                allocated.append(
                    alloc_token_slots(tree_cache, size) if size > 0 else None
                )

            for (table, rows, _), locs in zip(plans, allocated):
                if locs is None:
                    continue
                offset = 0
                for req_idx, start, end in rows:
                    count = end - start
                    table[req_idx, start:end] = locs[offset : offset + count].to(
                        torch.int32
                    )
                    offset += count
            for level, (_, rows, _) in enumerate(plans):
                for req_idx, _, end in rows:
                    self.allocated_lens[level][req_idx] = end
        except Exception:
            for locs in allocated:
                if locs is not None:
                    allocator.free(locs)
            raise

        if any(locs is not None for locs in allocated):
            self.allocator = allocator

    def alloc_for_extend(
        self,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        self._allocate_to_lengths(tree_cache, req_pool_indices_cpu, seq_lens_cpu)

    def alloc_for_decode(
        self,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        token_per_req: int,
    ) -> None:
        self._allocate_to_lengths(
            tree_cache,
            req_pool_indices_cpu,
            seq_lens_cpu + token_per_req,
        )

    def alloc_to_lengths(
        self,
        tree_cache: BasePrefixCache,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> None:
        self._allocate_to_lengths(tree_cache, req_pool_indices_cpu, seq_lens_cpu)

    def free(self, req_pool_idx: int) -> None:
        allocated = []
        for table, lengths in zip(
            (
                self.pool.req_to_sparse_k1_token,
                self.pool.req_to_sparse_k2_token,
            ),
            self.allocated_lens,
        ):
            length = lengths[req_pool_idx]
            if length > 0:
                allocated.append(table[req_pool_idx, :length].clone())
                table[req_pool_idx, :length].zero_()
                lengths[req_pool_idx] = 0

        if allocated:
            assert self.allocator is not None
            self.allocator.free(torch.cat(allocated))

    def clear(self) -> None:
        self.pool.req_to_sparse_k1_token.zero_()
        self.pool.req_to_sparse_k2_token.zero_()
        for lengths in self.allocated_lens:
            lengths[:] = [0] * len(lengths)
        self.allocator = None


def attach_compressed_cache(
    pool: ReqToTokenPool,
    *,
    kernel_size: int,
    kernel_stride: int,
    enable_memory_saver: bool,
) -> ReqToTokenPool:
    pool.attach_aux_cache(
        MiniCPMCompressedCache(
            pool,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            enable_memory_saver=enable_memory_saver,
        )
    )
    return pool


def create_req_to_token_pool(
    *,
    configurator: KVCacheConfigurator,
    size: int,
    max_context_len: int,
    enable_memory_saver: bool,
):
    config = configurator.model_config.hf_config
    sparse = config.has_minicpm_sparse_attention and (
        configurator.server_args.attention_backend
        in ("minicpm_flashattn", "minicpm_flashinfer")
    )
    cache_params = config.mamba2_cache_params
    extra_max_context_len = max_context_len - configurator.model_config.context_len
    if cache_params is None:
        pool = configurator._build_default_req_pool(
            max_num_reqs=size,
            extra_max_context_len=extra_max_context_len,
        )
    else:
        pool = configurator._build_hybrid_req_pool(
            max_num_reqs=size,
            extra_max_context_len=extra_max_context_len,
        )
    if not sparse:
        return pool
    return attach_compressed_cache(
        pool,
        kernel_size=config.sparse_kernel_size,
        kernel_stride=config.sparse_kernel_stride,
        enable_memory_saver=enable_memory_saver,
    )
