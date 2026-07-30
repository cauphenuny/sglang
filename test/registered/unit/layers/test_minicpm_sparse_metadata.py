import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.minicpm import backend as backend_module
from sglang.srt.layers.attention.minicpm import sparse_utils
from sglang.srt.layers.attention.minicpm.attention_adapter import (
    MiniCPMFlashAttentionAdapter,
)
from sglang.srt.layers.attention.minicpm.backend import (
    MiniCPMSparseBackend,
    _transpose_head_group_layout,
)
from sglang.srt.layers.attention.minicpm.sparse_utils import CompressionLevelMetadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _compression_layout():
    return SimpleNamespace(
        k1_kernel_size=32,
        k1_kernel_stride=16,
        k2_kernel_size=128,
        k2_kernel_stride=64,
    )


class _DeviceOffsetsMustNotBeRead:
    def __getitem__(self, _index):
        raise AssertionError("prefill layers must use scheduler-derived CPU offsets")


class TestMiniCPMSparseMetadata(CustomTestCase):
    def test_head_group_layout_round_trip(self):
        tensor = torch.arange(10).reshape(5, 2, 1)
        original = tensor.clone()

        _transpose_head_group_layout(
            tensor,
            [(1, 2)],
            head_group_num=2,
            heads_per_group=2,
            to_group_major=True,
        )
        self.assertEqual(
            tensor.squeeze(-1).tolist(),
            [[0, 1], [2, 3], [6, 7], [4, 5], [8, 9]],
        )

        _transpose_head_group_layout(
            tensor,
            [(1, 2)],
            head_group_num=2,
            heads_per_group=2,
            to_group_major=False,
        )
        self.assertTrue(torch.equal(tensor, original))

    def test_flashattn_variant_uses_fa4_on_blackwell(self):
        """Blackwell must select FA4 because FA3 binaries cannot execute there."""
        req_pool = SimpleNamespace(
            req_to_sparse_k1_token=torch.empty(0),
            req_to_sparse_k2_token=torch.empty(0),
        )
        flash_attn_backend = SimpleNamespace(
            max_context_len=256,
            device="cpu",
            decode_cuda_graph_metadata={},
            req_to_token_pool=req_pool,
            token_to_kv_pool=SimpleNamespace(),
            kv_cache_dtype=torch.bfloat16,
            kv_cache_dtype_str="bfloat16",
            page_size=1,
            fa_impl_ver=4,
            num_splits=1,
        )
        hf_config = SimpleNamespace(
            has_minicpm_sparse_attention=True,
            sparse_kernel_size=32,
            sparse_kernel_stride=16,
            sparse_init_blocks=1,
            sparse_block_size=64,
            sparse_window_size=64,
            sparse_dense_len=128,
            sparse_topk=1,
        )
        model_config = SimpleNamespace(
            hf_config=hf_config,
            num_attention_heads=16,
            head_dim=128,
            get_num_kv_heads=lambda _tp: 1,
        )
        model_runner = SimpleNamespace(
            dtype=torch.float16,
            token_to_kv_pool_allocator=SimpleNamespace(),
            server_args=SimpleNamespace(
                attention_backend="minicpm_flashattn",
                disable_cuda_graph=False,
                enable_memory_saver=False,
                chunked_prefill_size=64,
            ),
            model_config=model_config,
        )

        with (
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            patch.object(backend_module, "is_blackwell_supported", return_value=True),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ) as flash_attention,
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
        ):
            backend = MiniCPMSparseBackend(model_runner)

        flash_attention.assert_called_once_with(
            model_runner,
            skip_prefill=False,
            fa_impl_ver=4,
        )
        self.assertIs(backend.flash_attn_backend, flash_attn_backend)
        self.assertIs(
            backend.token_to_kv_pool,
            flash_attn_backend.token_to_kv_pool,
        )
        self.assertIsInstance(
            backend.attention_adapter,
            MiniCPMFlashAttentionAdapter,
        )
        self.assertEqual(backend.fused_kernel_kwargs["dtype_str"], "float16")
        self.assertEqual(backend.fused_kernel_kwargs["kernel_stride"], 16)

        model_runner.server_args.attention_backend = "minicpm_flashinfer"
        flashinfer_adapter = object()
        with (
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            patch.object(backend_module, "is_blackwell_supported", return_value=True),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ),
            patch.object(
                backend_module,
                "MiniCPMFlashInferAdapter",
                return_value=flashinfer_adapter,
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
        ):
            backend = MiniCPMSparseBackend(model_runner)

        self.assertIs(backend.flash_attn_backend, flash_attn_backend)
        self.assertIs(backend.attention_adapter, flashinfer_adapter)

        model_config.num_attention_heads = 8
        with (
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            patch.object(backend_module, "is_blackwell_supported", return_value=True),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
            self.assertRaisesRegex(ValueError, "16 query heads per KV head"),
        ):
            MiniCPMSparseBackend(model_runner)

    def test_dense_as_sparse_routes_short_prefill(self):
        req_pool = SimpleNamespace(
            req_to_sparse_k1_token=torch.empty(0),
            req_to_sparse_k2_token=torch.empty(0),
        )
        flash_attn_backend = SimpleNamespace(
            max_context_len=256,
            device="cpu",
            decode_cuda_graph_metadata={},
            req_to_token_pool=req_pool,
            token_to_kv_pool=SimpleNamespace(),
            page_size=1,
        )
        hf_config = SimpleNamespace(
            has_minicpm_sparse_attention=True,
            sparse_kernel_size=32,
            sparse_kernel_stride=16,
            sparse_init_blocks=1,
            sparse_block_size=64,
            sparse_window_size=64,
            sparse_dense_len=128,
            sparse_topk=1,
        )
        model_runner = SimpleNamespace(
            dtype=torch.float16,
            token_to_kv_pool_allocator=SimpleNamespace(),
            server_args=SimpleNamespace(
                attention_backend="minicpm_flashattn",
                disable_cuda_graph=False,
                enable_memory_saver=False,
                chunked_prefill_size=64,
            ),
            model_config=SimpleNamespace(
                hf_config=hf_config,
                num_attention_heads=16,
                head_dim=128,
                get_num_kv_heads=lambda _tp: 1,
            ),
        )

        with (
            backend_module.envs.SGLANG_MINICPM_DENSE_AS_SPARSE.override(True),
            patch.object(backend_module, "MiniCPMHybridConfig", SimpleNamespace),
            patch.object(backend_module, "is_blackwell_supported", return_value=False),
            patch.object(
                backend_module,
                "FlashAttentionBackend",
                return_value=flash_attn_backend,
            ),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1),
            ),
            patch.object(backend_module, "attach_compressed_cache"),
        ):
            backend = MiniCPMSparseBackend(model_runner)

        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            extend_seq_lens_cpu=[1],
            extend_prefix_lens_cpu=[0],
            forward_mode=SimpleNamespace(
                is_extend_or_draft_extend_or_mixed=lambda: True
            ),
        )
        metadata = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([1], dtype=torch.int32),
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            max_seq_len_q=1,
        )
        level = CompressionLevelMetadata()
        with patch.object(
            backend_module,
            "_build_k1_k2_compression_metadata",
            return_value={"k1": level, "k2": level},
        ):
            backend.update_batch_for_sparse(forward_batch, metadata)

        self.assertEqual(backend.dense_len, 0)
        self.assertEqual(metadata.sparse_bs_list, [0])

    def test_dense_prefill_page_table_covers_total_sequence(self):
        """Dense prefill must retain page-table coverage for the full sequence."""
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([7000], dtype=torch.int32),
            extend_seq_lens_cpu=torch.tensor([2904], dtype=torch.int32),
        )

        metadata = sparse_utils._build_sparse_prefill_metadata(
            forward_batch=forward_batch,
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

    def test_prefill_metadata_builds_layer_invariant_cache_lengths(self):
        """Sparse cache lengths must not be inferred from zero-valued table entries."""
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([200, 64], dtype=torch.int32),
            extend_seq_lens_cpu=[2, 3],
        )

        metadata = sparse_utils._build_sparse_prefill_metadata(
            forward_batch=forward_batch,
            sparse_bs_list=[0],
            head_group_num=2,
            dense_len=100,
            sparse_topk=2,
            block_size=64,
            cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
            sparse_page_table_dtype=torch.int32,
            sparse_page_table_device=torch.device("cpu"),
        )

        self.assertEqual(
            metadata["sparse_cache_seqlens_int32"].tolist(),
            [71, 71, 72, 72, 64, 64],
        )
        self.assertEqual(
            metadata["sparse_cu_seqlens_k"].tolist(),
            [0, 71, 142, 214, 286, 350, 414],
        )

    def test_dense_decode_page_table_covers_dense_threshold(self):
        """Dense decode must reserve page-table coverage through the dense threshold."""
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([7000], dtype=torch.int32),
        )
        base_metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([7000], dtype=torch.int32),
            page_table=torch.empty((1, 7000), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        )

        metadata = sparse_utils._build_sparse_decode_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            head_group_num=2,
            dense_len=8192,
            sparse_topk=96,
            block_size=64,
        )

        self.assertEqual(metadata["sparse_page_table"].shape, (2, 8192))

    def test_dense_decode_uses_sparse_topk(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        q = torch.ones(1, 1)
        k = torch.ones(1, 1, 1)
        v = torch.ones(1, 1, 1)
        key_cache = torch.ones(4, 1, 1, 1)
        value_cache = torch.ones(4, 1, 1, 1)
        backend.flash_attn_backend = SimpleNamespace(
            prepare_paged_mha_query=Mock(return_value=(q, None, None, None, None)),
            get_paged_mha_kv_cache=Mock(return_value=(key_cache, value_cache)),
            forward_decode=Mock(),
        )
        backend.token_to_kv_pool = SimpleNamespace(set_kv_buffer=Mock())
        backend.attention_adapter = SimpleNamespace(
            forward=Mock(return_value=torch.ones(1, 1, 1))
        )
        backend.forward_metadata = SimpleNamespace(
            page_table=torch.tensor([[0, 1, 0, 0]], dtype=torch.int32),
            cache_seqlens_int32=torch.tensor([2], dtype=torch.int32),
            sparse_page_table=torch.zeros((1, 4), dtype=torch.int32),
            sparse_cache_seqlens_int32=torch.tensor([2], dtype=torch.int32),
            sparse_cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            sparse_cu_seqlens_k=torch.tensor([0, 2], dtype=torch.int32),
            token_to_bs=torch.tensor([0], dtype=torch.int32),
            max_seq_len_q=1,
        )
        backend.head_group_num = 1
        backend.heads_per_group = 1
        backend.page_size = 1
        backend.block_size = 1
        backend.num_sparse_topk_tokens = 2
        backend.dense_len = 4
        backend._use_cuda_graph_buffers = False
        backend._compress_decode_keys = Mock()
        topk_idx = torch.tensor([[[0, 1]]], dtype=torch.int32)
        backend.get_topk_for_sparse = Mock(return_value=topk_idx)
        layer = SimpleNamespace(
            is_cross_attention=False,
            sliding_window_size=-1,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=1,
            v_head_dim=1,
            k_scale=None,
            v_scale=None,
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
            out_cache_loc=torch.tensor([1], dtype=torch.int64),
        )

        with patch.object(
            backend_module,
            "get_block_table_v3",
            return_value=torch.tensor([[3, 4]], dtype=torch.int32),
        ) as get_block_table:
            backend.forward_decode(q, k, v, layer, forward_batch)

        backend.get_topk_for_sparse.assert_called_once()
        get_block_table.assert_called_once()
        backend._compress_decode_keys.assert_not_called()
        backend.attention_adapter.forward.assert_called_once()
        backend.flash_attn_backend.forward_decode.assert_not_called()
        self.assertEqual(
            backend.forward_metadata.sparse_page_table[0, :2].tolist(),
            [3, 4],
        )

    def test_decode_metadata_supports_one_local_head_group(self):
        """Tensor parallelism may leave one local KV head without changing metadata."""
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
        )
        base_metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([10], dtype=torch.int32),
            page_table=torch.empty((1, 10), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        )

        metadata = sparse_utils._build_sparse_decode_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            head_group_num=1,
            dense_len=8192,
            sparse_topk=96,
            block_size=64,
        )

        self.assertEqual(metadata["sparse_cache_seqlens_int32"].tolist(), [10])
        self.assertEqual(metadata["sparse_page_table"].shape, (1, 8192))

    def test_decode_metadata_uses_scheduler_cpu_lengths(self):
        """Decode metadata must not synchronize device offsets to recover lengths."""
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([64, 200], dtype=torch.int32),
        )
        base_metadata = SimpleNamespace(
            cache_seqlens_int32=SimpleNamespace(
                dtype=torch.int32,
                device=torch.device("cpu"),
            ),
            page_table=torch.empty((2, 200), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        )

        metadata = sparse_utils._build_sparse_decode_metadata(
            forward_batch=forward_batch,
            base_metadata=base_metadata,
            head_group_num=2,
            dense_len=100,
            sparse_topk=2,
            block_size=64,
        )

        self.assertEqual(
            metadata["sparse_cache_seqlens_int32"].tolist(),
            [64, 64, 72, 72],
        )

    def test_cuda_graph_page_table_covers_dense_decode(self):
        """Captured dense decode must reserve a threshold-sized page table."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.flash_attn_backend = SimpleNamespace(
            decode_cuda_graph_metadata={},
            init_cuda_graph_state=lambda *_: None,
        )
        backend.attention_adapter = SimpleNamespace(
            init_cuda_graph_state=lambda *_: None,
        )
        backend.num_sparse_topk_tokens = 6144
        backend.page_size = 1
        backend.head_group_num = 2
        backend.device = "cpu"
        backend.model_dtype = torch.float16
        backend.heads_per_group = 16
        backend.max_context_len = 256
        backend.config_dense_len = 8192
        backend.dense_len = 8192
        backend.head_dim = 128
        backend.k1_kernel_size = 32
        backend.k1_kernel_stride = 16
        backend.k2_kernel_size = 128
        backend.k2_kernel_stride = 64

        backend.init_cuda_graph_state(max_bs=1, max_num_tokens=1)

        self.assertEqual(
            backend.decode_cuda_graph_metadata["sparse_page_table"].shape,
            (2, 8192),
        )
        self.assertEqual(
            backend.decode_cuda_graph_metadata["compress_k1"].dtype,
            torch.float16,
        )
        for level in ("k1", "k2"):
            for field in (
                "new_token_nums",
                "new_compress_token_nums",
                "cu_new_compress_token_nums",
                "total_compress_token_nums",
            ):
                self.assertNotIn(
                    f"{level}.{field}",
                    backend.decode_cuda_graph_metadata,
                )

    def test_compression_uses_configured_k1_k2_layout(self):
        """K1/K2 compression must honor checkpoint strides instead of fixed defaults."""
        layer = SimpleNamespace(layer_id=0, tp_k_head_num=1, head_dim=1)
        forward_batch = SimpleNamespace(req_pool_indices=[0])
        level = CompressionLevelMetadata(
            table=torch.empty(0),
            history_compress_token_nums=torch.empty(0),
            cu_new_token_nums=torch.empty(0),
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
            [(call.args[8], call.args[9]) for call in compress.call_args_list],
            [(5, 3), (13, 7)],
        )
        self.assertTrue(all(call.kwargs["padded"] for call in compress.call_args_list))

    def test_decode_compression_has_one_kernel_call_per_buffer_source(self):
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.forward_metadata = SimpleNamespace()
        backend.max_context_len = 8
        backend.k1_kernel_size = 2
        backend.k1_kernel_stride = 2
        backend.k2_kernel_size = 4
        backend.k2_kernel_stride = 4
        backend.minicpm_split_stage1 = True
        backend.device = torch.device("cpu")
        layer = SimpleNamespace(tp_k_head_num=1, head_dim=2)
        forward_batch = SimpleNamespace(batch_size=2)

        for use_graph_buffers in (False, True):
            with self.subTest(use_graph_buffers=use_graph_buffers):
                backend._use_cuda_graph_buffers = use_graph_buffers
                backend.decode_cuda_graph_metadata = {
                    "compress_k1": torch.empty(8, 1, 2),
                    "compress_k2": torch.empty(4, 1, 2),
                }
                with patch.object(backend_module, "get_compress_k_v2") as compress:
                    k1, k2 = backend._compress_decode_keys(
                        torch.empty(1, dtype=torch.float16),
                        layer,
                        forward_batch,
                    )

                self.assertEqual(k1.shape, (8, 1, 2))
                self.assertEqual(k2.shape, (4, 1, 2))
                compress.assert_called_once()
                self.assertTrue(compress.call_args.kwargs["padded"])

    def test_fused_topk_kernels_compile_lazily_per_batch_size(self):
        """Startup must not compile fused kernels for batch sizes that never run."""
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

    def test_forward_metadata_tracks_cuda_graph_buffer_ownership(self):
        """Only replay metadata may be marked as backed by CUDA graph buffers."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        metadata = SimpleNamespace()
        backend.flash_attn_backend = SimpleNamespace(
            forward_metadata=metadata,
            init_forward_metadata=lambda *_: None,
            init_forward_metadata_out_graph=lambda *_: None,
        )
        backend.update_batch_for_sparse = lambda *_: None
        backend._get_fused_topk_kernel = lambda *_args, **_kwargs: None
        backend._replay_sparse_graph_metadata = lambda *_: None
        backend.attention_adapter = SimpleNamespace(
            prepare_forward=lambda *_args, **_kwargs: None,
        )
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_idle=lambda: False,
            is_decode_or_idle=lambda: True,
        )
        forward_batch = SimpleNamespace(forward_mode=forward_mode, batch_size=1)

        backend._use_cuda_graph_buffers = True
        backend.init_forward_metadata(forward_batch)
        self.assertFalse(backend._use_cuda_graph_buffers)

        backend.init_forward_metadata_out_graph(forward_batch)
        self.assertTrue(backend._use_cuda_graph_buffers)

    def test_idle_batch_skips_sparse_metadata(self):
        """An idle DP rank must not attempt sparse metadata construction."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        metadata = SimpleNamespace()
        backend.flash_attn_backend = SimpleNamespace(
            forward_metadata=metadata,
            init_forward_metadata=lambda *_: None,
        )
        backend.update_batch_for_sparse = lambda *_: self.fail(
            "idle batches must not build sparse metadata"
        )
        forward_mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_idle=lambda: True,
        )

        backend.init_forward_metadata(SimpleNamespace(forward_mode=forward_mode))

        self.assertIs(backend.forward_metadata, metadata)

    def test_mixed_prefill_compiles_fused_topk_for_sparse_batch_only(self):
        """A mixed batch must compile fused top-k for its sparse sub-batch only."""
        backend = MiniCPMSparseBackend.__new__(MiniCPMSparseBackend)
        backend.forward_metadata = SimpleNamespace(
            sparse_batch_size=1,
            k1=SimpleNamespace(
                cu_seqlens=_DeviceOffsetsMustNotBeRead(),
                cu_seqlens_cpu=[0, 0, 1],
            ),
            k2=SimpleNamespace(
                cu_seqlens=_DeviceOffsetsMustNotBeRead(),
                cu_seqlens_cpu=[0, 0, 1],
            ),
        )
        backend.k1_kernel_size = 1
        backend.k1_kernel_stride = 1
        backend.k2_kernel_size = 1
        backend.k2_kernel_stride = 1
        backend.dense_len = 1
        backend.max_context_len = 1
        backend.minicpm_split_stage1 = True
        layer = SimpleNamespace(tp_q_head_num=1, tp_k_head_num=1, head_dim=1)
        forward_batch = SimpleNamespace(batch_size=2)

        with (
            patch.object(
                backend_module,
                "allocate_and_compress_keys",
                return_value=(torch.ones(1, 1, 1), torch.ones(1, 1, 1)),
            ) as allocate,
            patch.object(
                backend_module,
                "_build_prefill_topk_metadata",
                return_value={
                    "sparse_bs": [1],
                    "k1_lens": [0, 1],
                    "k2_lens": [0, 1],
                    "cu_seqlens_q": torch.tensor([0, 1], dtype=torch.int32),
                    "cu_seqlens_k": torch.tensor([0, 1], dtype=torch.int32),
                    "max_seqlen_q": 1,
                    "max_seqlen_k": 1,
                    "query_states": torch.empty(1, 1, 1),
                },
            ),
            patch.object(
                backend,
                "_get_fused_topk_kernel",
                return_value="sparse-kernel",
            ) as get_kernel,
            patch.object(
                backend,
                "sparse_get_topk_impl",
                side_effect=lambda *_args, **kwargs: kwargs["fused_kernel"],
            ),
        ):
            result = backend.get_topk_for_sparse(
                query_states=torch.empty(2, 1, 1),
                key_states=torch.empty(2, 1, 1),
                layer=layer,
                forward_batch=forward_batch,
            )

        self.assertEqual(result, "sparse-kernel")
        get_kernel.assert_called_once_with(1, is_prefill=True)
        self.assertFalse(allocate.call_args.kwargs["minicpm_split_stage1"])

    def test_compression_metadata_ignores_cuda_graph_padding(self):
        """CUDA graph padding rows must not alter offsets for real requests."""
        config = _compression_layout()

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

        metadata = sparse_utils._build_k1_k2_compression_metadata(
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

        self.assertEqual(metadata["k1"].cu_seqlens_cpu, [0, 5, 16, 33])
        self.assertEqual(metadata["k2"].cu_seqlens_cpu, [0, 0, 2, 5])
        for level in (metadata["k1"], metadata["k2"]):
            self.assertEqual(level.table.shape[0], forward_batch.batch_size)
            self.assertEqual(
                level.history_compress_token_nums.numel(), forward_batch.batch_size
            )
            self.assertEqual(level.cu_new_token_nums.numel(), 4)
            self.assertEqual(level.cu_total_compress_token_nums.numel(), 4)

    def test_sparse_sequence_lengths_use_scheduler_values(self):
        """Sparse query lengths must not be copied back from device offsets."""
        query_lengths, key_lengths = sparse_utils._build_sequence_lengths(
            extend_seq_lens_cpu=[3, 5],
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            sparse_bs_list=[1],
        )

        self.assertEqual(query_lengths, [5])
        self.assertEqual(key_lengths.tolist(), [20])


if __name__ == "__main__":
    unittest.main()
