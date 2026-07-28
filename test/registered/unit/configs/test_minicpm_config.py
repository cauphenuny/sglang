from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.layers.attention.linear.lightning_backend import (
    LightningAttentionBackend,
)
from sglang.srt.models.minicpm import MiniCPMLightningMixer
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_minicpm_lightning_config_defaults_are_complete():
    config = MiniCPMHybridConfig()

    assert config.scale_emb == 12
    assert config.scale_depth == 1.4
    assert config.dim_model_base == 256
    assert config.lightning_use_rope is True
    assert config.use_output_gate is False
    assert config.attention_bias is False
    assert config.use_output_norm is False
    assert config.qk_norm is True


def test_minicpm_empty_mixer_types_default_to_full_attention():
    config = MiniCPMHybridConfig(num_hidden_layers=3, mixer_types=[])

    assert config.mixer_types == ["minicpm4", "minicpm4", "minicpm4"]
    assert config.full_attention_layer_ids == [0, 1, 2]


def test_minicpm_short_mixer_pattern_repeats_to_layer_count():
    config = MiniCPMHybridConfig(
        num_hidden_layers=5,
        mixer_types=["minicpm4", "lightning-attn"],
    )

    assert config.mixer_types == [
        "minicpm4",
        "lightning-attn",
        "minicpm4",
        "lightning-attn",
        "minicpm4",
    ]
    assert config.full_attention_layer_ids == [0, 2, 4]
    assert config.lightning_layer_ids == [1, 3]


def test_minicpm_mixer_aliases_are_canonicalized():
    config = MiniCPMHybridConfig(
        num_hidden_layers=4,
        mixer_types=["attention", "lightning_attn"],
    )

    assert config.mixer_types == [
        "minicpm4",
        "lightning-attn",
        "minicpm4",
        "lightning-attn",
    ]


def test_minicpm_rejects_more_mixer_types_than_layers():
    with pytest.raises(ValueError, match="Invalid number of mixer types: 3"):
        MiniCPMHybridConfig(
            num_hidden_layers=2,
            mixer_types=["minicpm4", "lightning", "minicpm4"],
        )


def test_minicpm_lightning_dimensions_fall_back_to_base_attention():
    config = MiniCPMHybridConfig(
        hidden_size=96,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=None,
        lightning_nh=None,
        lightning_nkv=None,
        lightning_head_dim=None,
    )

    assert config.head_dim == 16
    assert config.lightning_nh == 6
    assert config.lightning_nkv == 3
    assert config.lightning_head_dim == 16


def test_minicpm_lightning_idle_batch_returns_empty_output():
    mixer = MiniCPMLightningMixer.__new__(MiniCPMLightningMixer)
    torch.nn.Module.__init__(mixer)
    mixer.hidden_size = 8
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: True)
    )

    output = mixer.forward(
        positions=torch.empty(0, dtype=torch.int64),
        hidden_states=torch.empty(0, 4),
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


def test_lightning_backend_uses_layer_scale(monkeypatch):
    captured = {}

    def fake_seg_la_fwd(**kwargs):
        captured.update(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(
        "sglang.srt.layers.attention.linear.lightning_backend.seg_la_fwd",
        fake_seg_la_fwd,
    )
    backend = LightningAttentionBackend.__new__(LightningAttentionBackend)
    backend.tp_slope = [torch.ones(1, 1, 1)]
    layer = SimpleNamespace(layer_id=0, scaling=0.25)
    metadata = SimpleNamespace(
        batch_size=1,
        query_start_loc=torch.tensor([0, 1]),
        has_initial_states=torch.tensor([False]),
    )
    q = torch.ones(1, 1, 1)

    backend._linear_attention_entry(
        q=q,
        k=q,
        v=q,
        kv_cache=torch.zeros(1, 1, 1, 1),
        state_indices_tensor=torch.tensor([0]),
        metadata=metadata,
        layer=layer,
    )

    assert captured["softmax_scale"] == 0.25
