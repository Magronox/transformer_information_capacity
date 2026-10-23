"""The OLMo-3 transformer used in place of GPT-2."""
from __future__ import annotations

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig
from torch import nn

from train import TrainConfig

SEQ_LEN = 2


def build_olmo3(cfg: TrainConfig, vocab_size: int) -> nn.Module:
    """OLMo-3 block without positional encoding or sliding window, SwiGLU hidden size round(8d/3)."""
    if cfg.dim % cfg.heads:
        raise ValueError(f"dim={cfg.dim} is not divisible by heads={cfg.heads}")
    config = TransformerConfig.olmo3_600M(
        vocab_size=vocab_size,
        d_model=cfg.dim,
        n_layers=cfg.layers,
        n_heads=cfg.heads,
        n_kv_heads=cfg.heads,
        head_dim=cfg.dim // cfg.heads,
        hidden_size_multiple_of=1,
        sliding_window=None,
        attn_backend=AttentionBackendName.torch,
        init_seed=cfg.seed,
    )
    config.block.sequence_mixer.rope = None
    config.block.feed_forward.hidden_size = max(1, round(8 * cfg.dim / 3))
    model = config.build(init_device="cpu")
    model.init_weights(max_seq_len=SEQ_LEN)
    # init_weights caches the CPU device; without this, forward ignores a later .cuda().
    model._device = None
    return model
