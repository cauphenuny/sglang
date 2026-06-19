import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.minicpm.sparse_utils import (
    SparseConfig,
    SparseMetadataBuilder,
)
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache


class TestMiniCPMSparseMetadata(unittest.TestCase):
    def test_chunked_prefill_saves_aux_prefix_indices(self):
        cache = MambaRadixCache.__new__(MambaRadixCache)
        aux_pool = Mock()
        cache.req_to_token_pool = SimpleNamespace(aux_pool=aux_pool)
        req = SimpleNamespace(prefix_indices=None)
        prefix_indices = torch.arange(8, dtype=torch.int64)

        cache._set_unfinished_req_prefix_indices(req, prefix_indices)

        self.assertIs(req.prefix_indices, prefix_indices)
        aux_pool.set_prefix_indices.assert_called_once_with(req, len(prefix_indices))

    def test_compression_metadata_ignores_cuda_graph_padding(self):
        config = SparseConfig(
            sparse_len=8192,
            sparse_topk=64,
            kernel_size=32,
            kernel_stride=16,
            block_size=64,
            window_size=2048,
            dense_len=8192,
            head_dim=128,
            num_kv_heads=8,
            head_group_num=8,
            k1_kernel_size=32,
            k1_kernel_stride=16,
            k2_kernel_size=128,
            k2_kernel_stride=64,
        )
        builder = SparseMetadataBuilder(config, num_kv_heads=8)

        # The graph was captured for batch size 4, but only the first three
        # requests are real during this replay.
        forward_batch = SimpleNamespace(
            batch_size=3,
            seq_lens_cpu=torch.tensor([100, 200, 300], dtype=torch.int32),
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        )
        base_metadata = SimpleNamespace(
            cu_seqlens_q=torch.arange(5, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 100, 200, 300, 400], dtype=torch.int32),
        )
        req_to_sparse_token = torch.arange(4 * 32, dtype=torch.int32).reshape(4, 32)

        metadata = builder.build_k1_k2_compression_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            req_to_sparse_k1_token=req_to_sparse_token,
            req_to_sparse_k2_token=req_to_sparse_token,
            k1_kernel_size=config.k1_kernel_size,
            k1_kernel_stride=config.k1_kernel_stride,
            k2_kernel_size=config.k2_kernel_size,
            k2_kernel_stride=config.k2_kernel_stride,
            cu_seqlens_q=base_metadata.cu_seqlens_q,
        )

        for level in (metadata["k1"], metadata["k2"]):
            self.assertEqual(level.table.shape[0], forward_batch.batch_size)
            self.assertEqual(
                level.history_compress_token_nums.numel(), forward_batch.batch_size
            )
            self.assertEqual(level.new_token_nums.numel(), forward_batch.batch_size)
            self.assertEqual(
                level.new_compress_token_nums.numel(), forward_batch.batch_size
            )
            self.assertEqual(
                level.total_compress_token_nums.numel(), forward_batch.batch_size
            )
            self.assertEqual(level.cu_new_token_nums.numel(), 4)
            self.assertEqual(level.cu_new_compress_token_nums.numel(), 4)
            self.assertEqual(level.cu_total_compress_token_nums.numel(), 4)


if __name__ == "__main__":
    unittest.main()
