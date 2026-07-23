from types import SimpleNamespace

import torch

from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.layers.attention.linear.lightning_backend import (
    LightningAttentionBackend,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_minicpm_lightning_config_defaults_are_complete():
    config = MiniCPMHybridConfig()

    assert config.lightning_use_rope is True
    assert config.use_output_gate is False
    assert config.attention_bias is False
    assert config.use_output_norm is False
    assert config.qk_norm is True


def test_minicpm_lightning_idle_batch_returns_empty_output():
    backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    backend._is_full_attn = lambda *_args: False
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: True)
    )
    layer = SimpleNamespace(tp_q_head_num=2, v_head_dim=4)

    output = backend.forward(
        q=torch.empty(0, 2, 4),
        layer=layer,
        forward_batch=forward_batch,
    )

    assert output.shape == (0, 8)


def test_minicpm_lightning_reuses_shared_backend_and_cache_shape():
    config = MiniCPMHybridConfig(
        num_hidden_layers=2,
        mixer_types=["lightning", "minicpm4"],
        lightning_nkv=4,
        lightning_head_dim=64,
    )

    spec, _ = get_linear_attn_config(config)
    assert spec.backend_class_name.endswith(".LightningAttentionBackend")

    with get_parallel().override(attn_tp_size=1):
        cache = config.mamba2_cache_params
    assert isinstance(cache, Mamba2CacheParams)
    assert cache.layers == [0]
    assert cache.shape.conv == [(0, 0)]
    assert cache.shape.temporal == (4, 64, 64)

    with get_parallel().override(attn_tp_size=1, attn_tp_rank=0):
        slopes = LightningAttentionBackend._build_slope_tensor(
            4, 2, device="cpu", layerwise_decay=False
        )
    assert len(slopes) == 2
    assert slopes[0].equal(slopes[1])
