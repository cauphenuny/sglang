import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm import sparse_utils
from sglang.srt.layers.attention.minicpm.backend import MiniCPMSparseBackend
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    CompressionLevelMetadata,
    SparseConfig,
    SparseMetadataBuilder,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sparse_config():
    return SparseConfig(
        sparse_len=8192,
        sparse_topk=64,
        kernel_size=32,
        kernel_stride=16,
        block_size=64,
        window_size=2048,
        dense_len=8192,
        head_dim=128,
        num_kv_heads=2,
        head_group_num=2,
        k1_kernel_size=32,
        k1_kernel_stride=16,
        k2_kernel_size=128,
        k2_kernel_stride=64,
    )


class TestMiniCPMSparseMetadata(unittest.TestCase):
    def test_dense_prefill_page_table_covers_total_sequence(self):
        builder = SparseMetadataBuilder(_sparse_config(), num_kv_heads=2)
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([7000], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([2904], dtype=torch.int32),
        )

        metadata = builder.build_sparse_prefill_metadata(
            forward_batch=forward_batch,
            base_metadata=None,
            sparse_bs_list=[],
            head_group_num=2,
            dense_len=8192,
            sparse_topk=96,
            block_size=64,
            cu_seqlens_q=torch.tensor([0, 2904], dtype=torch.int32),
            sparse_page_table_dtype=torch.int32,
            sparse_page_table_device=torch.device("cpu"),
        )

        self.assertEqual(metadata["sparse_page_table"].shape, (2, 7000))

    def test_dense_decode_page_table_covers_dense_threshold(self):
        builder = SparseMetadataBuilder(_sparse_config(), num_kv_heads=2)
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([7000], dtype=torch.int32),
        )
        base_metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([7000], dtype=torch.int32),
            page_table=torch.empty((1, 7000), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        )

        metadata = builder.build_sparse_decode_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            head_group_num=2,
            dense_len=8192,
            sparse_topk=96,
            block_size=64,
        )

        self.assertEqual(metadata["sparse_page_table"].shape, (2, 8192))

    def test_decode_metadata_supports_one_local_head_group(self):
        builder = SparseMetadataBuilder(_sparse_config(), num_kv_heads=1)
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
        )
        base_metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([10], dtype=torch.int32),
            page_table=torch.empty((1, 10), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        )

        metadata = builder.build_sparse_decode_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            head_group_num=1,
            dense_len=8192,
            sparse_topk=96,
            block_size=64,
        )

        self.assertEqual(metadata["sparse_cache_seqlens_int32"].tolist(), [10])
        self.assertEqual(metadata["sparse_page_table"].shape, (1, 8192))

    def test_compression_uses_configured_k1_k2_layout(self):
        layer = SimpleNamespace(layer_id=0, tp_k_head_num=1, head_dim=1)
        forward_batch = SimpleNamespace(req_pool_indices=[0])
        level = CompressionLevelMetadata(
            table=torch.empty(0),
            history_compress_token_nums=torch.empty(0),
            new_token_nums=torch.empty(0),
            cu_new_token_nums=torch.empty(0),
            new_compress_token_nums=torch.empty(0),
            cu_new_compress_token_nums=torch.empty(0),
            total_compress_token_nums=torch.empty(0),
            cu_total_compress_token_nums=torch.empty(0),
        )
        metadata = SimpleNamespace(
            page_table=torch.empty(0),
            k1=level,
            k2=level,
        )
        pool = SimpleNamespace(get_key_buffer=lambda _layer_id: torch.empty(1, 1, 1))

        with (
            patch.object(sparse_utils, "get_token_to_kv_pool", return_value=pool),
            patch.object(sparse_utils, "compress_k_core_new") as compress,
        ):
            sparse_utils.get_compress_k_v2(
                layer,
                forward_batch,
                metadata,
                torch.empty(0),
                torch.empty(0),
                max_context_length=256,
                k1_kernel_size=5,
                k1_kernel_stride=3,
                k2_kernel_size=13,
                k2_kernel_stride=7,
                padded=True,
            )

        self.assertEqual(
            [(call.args[14], call.args[15]) for call in compress.call_args_list],
            [(5, 3), (13, 7)],
        )
        self.assertTrue(all(call.kwargs["padded"] for call in compress.call_args_list))

    def test_fused_topk_kernels_compile_lazily_per_batch_size(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.minicpm_fuse_topk = True
        backend.decode_fused_kernels = {}
        backend.prefill_fused_kernels = {}
        backend.fused_kernel_kwargs = {"topk": 8}
        backend.prefill_kernel_max_seqlen_q_grid = 64

        with (
            patch.object(
                backend_module,
                "fused_attn_pooling_online_topk_prefill",
                return_value="prefill",
            ) as prefill,
            patch.object(
                backend_module,
                "fused_attn_pooling_online_topk_decode",
                return_value="decode",
            ) as decode,
        ):
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=True), "prefill"
            )
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=True), "prefill"
            )
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=False), "decode"
            )
            self.assertEqual(
                backend._get_fused_topk_kernel(3, is_prefill=False), "decode"
            )

        prefill.assert_called_once_with(
            topk=8,
            batch_size=3,
            max_seqlen_q_grid=64,
        )
        decode.assert_called_once_with(topk=8, batch_size=3)

    def test_compression_metadata_ignores_cuda_graph_padding(self):
        config = _sparse_config()
        builder = SparseMetadataBuilder(config, num_kv_heads=2)

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
