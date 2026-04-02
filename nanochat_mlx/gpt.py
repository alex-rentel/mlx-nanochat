"""
GPT model in MLX.
Port of Karpathy's nanochat GPT with all features:
- RoPE (rotary position embeddings)
- QK norm with sharpened attention
- GQA (grouped query attention)
- ReLU squared activation in MLP
- RMSNorm (parameter-free)
- Value Embeddings (ResFormer-style)
- Per-layer learnable resid_lambdas and x0_lambdas
- Smear (mix previous token embedding)
- Backout (subtract mid-layer residual before logit projection)
- Logit soft-capping at 15
- Untied embedding/unembedding weights

Note: mx.eval() calls in this file are used to materialize MLX lazy arrays,
not Python's eval() builtin.
"""

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def rms_norm(x):
    """Parameter-free RMS normalization."""
    return mx.fast.rms_norm(x, mx.ones(x.shape[-1]), eps=1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to x. x shape: (B, T, H, D)"""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=3)


def precompute_rotary_embeddings(seq_len, head_dim, base=100000):
    """Precompute cos/sin for RoPE."""
    channel_range = mx.arange(0, head_dim, 2).astype(mx.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = mx.arange(seq_len).astype(mx.float32)
    freqs = mx.outer(t, inv_freq)
    cos = mx.cos(freqs).astype(mx.float16)
    sin = mx.sin(freqs).astype(mx.float16)
    # Shape: (1, seq_len, 1, head_dim/2) for broadcasting
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.ve_gate_channels = 12
        self._has_ve = has_ve(layer_idx, config.n_layer)
        if self._has_ve:
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def __call__(self, x, ve, cos_sin, mask, cache=None):
        B, T, C = x.shape

        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer)
        if ve is not None and self._has_ve:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head)
            v = v + gate[..., None] * ve

        # Apply RoPE
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm + sharpened attention
        q = rms_norm(q) * 1.2
        k = rms_norm(k) * 1.2

        # KV cache for inference
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # GQA: expand kv heads to match query heads
        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = mx.repeat(k, n_rep, axis=2)
            v = mx.repeat(v, n_rep, axis=2)

        # Transpose to (B, H, T, D) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores = scores + mask

        weights = mx.softmax(scores, axis=-1)
        y = weights @ v  # (B, H, T, D)

        # Transpose back and project
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y), new_cache


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))  # ReLU squared
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, cos_sin, mask, cache=None):
        attn_out, new_cache = self.attn(rms_norm(x), ve, cos_sin, mask, cache)
        x = x + attn_out
        x = x + self.mlp(rms_norm(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = padded_vocab_size

        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # Per-layer learnable scalars
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.zeros((config.n_layer,))

        # Smear: mix previous token embedding into current
        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.smear_lambda = mx.zeros((1,))

        # Backout: subtract mid-layer residual before final logit projection
        self.backout_lambda = mx.array([0.2])

        # Value embeddings (ResFormer-style)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {}
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                self.value_embeds[str(i)] = nn.Embedding(padded_vocab_size, kv_dim)

        # Precompute rotary embeddings (10x overcompute like original)
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        self._cos, self._sin = precompute_rotary_embeddings(self.rotary_seq_len, head_dim)

    def init_weights(self):
        """Initialize weights following the original nanochat scheme."""
        n_embd = self.config.n_embd
        n_layer = self.config.n_layer
        s = 3**0.5 * n_embd**-0.5

        # Embedding and unembedding
        self.wte.weight = mx.random.normal(shape=self.wte.weight.shape) * 0.8
        self.lm_head.weight = mx.random.normal(shape=self.lm_head.weight.shape) * 0.001

        for block in self.blocks:
            # Attention weights: uniform init
            for proj in [block.attn.c_q, block.attn.c_k, block.attn.c_v]:
                proj.weight = mx.random.uniform(-s, s, shape=proj.weight.shape)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)

            # MLP weights
            block.mlp.c_fc.weight = mx.random.uniform(-s * 0.4, s * 0.4, shape=block.mlp.c_fc.weight.shape)
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)

        # Per-layer scalars
        resid = []
        x0 = []
        for i in range(n_layer):
            resid.append(1.15 - (0.10 * i / max(n_layer - 1, 1)))
            x0.append(0.20 - (0.15 * i / max(n_layer - 1, 1)))
        self.resid_lambdas = mx.array(resid)
        self.x0_lambdas = mx.array(x0)

        # Value embeddings
        for ve in self.value_embeds.values():
            ve.weight = mx.random.uniform(-s, s, shape=ve.weight.shape)

        # VE gate weights
        for block in self.blocks:
            if block.attn._has_ve:
                block.attn.ve_gate.weight = mx.random.uniform(0.0, 0.02, shape=block.attn.ve_gate.weight.shape)

        # Smear and backout
        self.smear_gate.weight = mx.zeros_like(self.smear_gate.weight)
        self.smear_lambda = mx.zeros((1,))
        self.backout_lambda = mx.array([0.2])

    def __call__(self, idx, targets=None, cache=None):
        B, T = idx.shape

        # Rotary embeddings
        T0 = 0
        if cache is not None and cache[0] is not None:
            # Get position from first layer's cache
            first_cache = cache[0]
            if first_cache is not None:
                T0 = first_cache[0].shape[1]
        cos = self._cos[:, T0:T0+T]
        sin = self._sin[:, T0:T0+T]
        cos_sin = (cos, sin)

        # Create causal mask
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(mx.float16)
            if T0 > 0:
                # During cache usage with T>1 (prefill), we need full mask
                prefix = mx.zeros((T, T0), dtype=mx.float16)
                mask = mx.concatenate([prefix, mask], axis=1)
            mask = mask[None, None, :, :]  # (1, 1, T, T+T0)
        else:
            mask = None

        # Embed tokens
        x = self.wte(idx)
        x = rms_norm(x)

        # Smear: mix previous token embedding into current
        if cache is None and T > 1:
            gate = self.smear_lambda * mx.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x_prev = x[:, :-1]
            x_curr = x[:, 1:] + gate * x_prev
            x = mx.concatenate([x[:, :1], x_curr], axis=1)

        # Forward through transformer blocks
        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        new_cache = []

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            # Value embedding for this layer
            ve = None
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)

            layer_cache = cache[i] if cache is not None else None
            x, kv = block(x, ve, cos_sin, mask, layer_cache)
            new_cache.append(kv)

            if i == backout_layer:
                x_backout = x

        # Subtract mid-layer residual
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = rms_norm(x)

        # Compute logits with soft-capping
        softcap = 15.0
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]  # Remove padding
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            # Training: compute cross-entropy loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)

            # Mask out -1 targets (padding)
            valid = targets_flat != -1
            if mx.any(valid):
                loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
                loss = mx.where(valid, loss, 0.0)
                loss = loss.sum() / valid.sum()
            else:
                loss = mx.array(0.0)
            return loss
        else:
            return logits, new_cache

    def num_params(self):
        """Count total parameters."""
        import mlx.utils
        return sum(p.size for _, p in mlx.utils.tree_flatten(self.parameters()))

    def estimate_flops(self):
        """Estimate FLOPs per token (forward + backward = 6x matmul params)."""
        nparams = 0
        for block in self.blocks:
            for _, p in block.parameters().items():
                if isinstance(p, mx.array):
                    nparams += p.size
        nparams += self.lm_head.weight.size

        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = self.config.n_layer * 12 * h * q * t
        return 6 * nparams + attn_flops


def build_model(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768):
    """Build a GPT model from depth (the one complexity dial)."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim

    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
    )

    model = GPT(config)
    model.init_weights()
    mx.eval(model.parameters())  # mx.eval materializes MLX lazy arrays
    return model
