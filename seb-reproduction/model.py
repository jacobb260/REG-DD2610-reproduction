"""
REPA DiT model in JAX/Flax NNX.

Architecture:
  - DiT-style transformer with adaLN-Zero conditioning
  - REPA: intermediate projector for alignment with frozen encoder features
  - CLS token injection from pretrained encoder
  - v-prediction (flow matching) output head
"""

import math
import jax
import jax.numpy as jnp
import flax.nnx as nnx

# ---------------------------------------------------------------------------
# Hyperparameters (model)
# ---------------------------------------------------------------------------
HIDDEN_SIZE   = 256
DEPTH         = 6
NUM_HEADS     = 4
PATCH_SIZE    = 4        # 32/4 = 8 → 64 patches
IMG_SIZE      = 32
IN_CHANNELS   = 3
NUM_CLASSES   = 10
ENC_FEAT_DIM  = 384      # DINOv2 ViT-S feature dim
ENCODER_DEPTH = 3        # after this many blocks, project for REPA
MLP_RATIO     = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sinusoidal_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Sinusoidal timestep embedding. t: (B,) → (B, dim)."""
    half = dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / (half - 1))
    args  = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> jnp.ndarray:
    """2D sin-cos positional embedding. Returns (grid_h*grid_w, embed_dim)."""
    assert embed_dim % 2 == 0
    h = jnp.arange(grid_h, dtype=jnp.float32)
    w = jnp.arange(grid_w, dtype=jnp.float32)
    grid_w, grid_h = jnp.meshgrid(w, h)                       # (H, W)
    grid = jnp.stack([grid_w.ravel(), grid_h.ravel()], axis=0) # (2, T)

    half = embed_dim // 2
    omega = 1.0 / (10000 ** (jnp.arange(half // 2, dtype=jnp.float32) / (half // 2)))
    # each spatial dim gets half//2 frequencies
    out_w = grid[0, :, None] * omega[None, :]  # (T, half//2)
    out_h = grid[1, :, None] * omega[None, :]
    emb = jnp.concatenate([jnp.sin(out_w), jnp.cos(out_w),
                            jnp.sin(out_h), jnp.cos(out_h)], axis=-1)  # (T, embed_dim)
    return emb  # (T, embed_dim)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------
class TimestepEmbedder(nnx.Module):
    """Sinusoidal → 2-layer MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256, rngs: nnx.Rngs = None):
        self.freq_dim = freq_dim
        self.fc1 = nnx.Linear(freq_dim, hidden_size, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        x = sinusoidal_embedding(t, self.freq_dim)
        x = nnx.silu(self.fc1(x))
        return self.fc2(x)


class LabelEmbedder(nnx.Module):
    """Class label embedding with dropout for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden_size: int, drop_prob: float = 0.1,
                 rngs: nnx.Rngs = None):
        self.drop_prob = drop_prob
        self.num_classes = num_classes
        # +1 for the unconditional (null) class
        self.embed = nnx.Embed(num_classes + 1, hidden_size, rngs=rngs)

    def __call__(self, y: jnp.ndarray, deterministic: bool = True,
                 rngs: nnx.Rngs = None) -> jnp.ndarray:
        if not deterministic and rngs is not None:
            drop_mask = jax.random.bernoulli(rngs.dropout(), self.drop_prob, y.shape)
            y = jnp.where(drop_mask, self.num_classes, y)   # replace with null class
        return self.embed(y)


class AdaLNZero(nnx.Module):
    """adaLN-Zero: predict scale/shift/gate from conditioning vector."""

    def __init__(self, hidden_size: int, rngs: nnx.Rngs):
        # 6 outputs: shift1, scale1, gate1, shift2, scale2, gate2
        self.linear = nnx.Linear(hidden_size, 6 * hidden_size,
                                 kernel_init=nnx.initializers.zeros,
                                 bias_init=nnx.initializers.zeros,
                                 rngs=rngs)

    def __call__(self, c: jnp.ndarray):
        return jnp.split(nnx.silu(self.linear(c)), 6, axis=-1)


class SelfAttention(nnx.Module):
    def __init__(self, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.qkv = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)
        self.proj = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, T, H, D)
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = jnp.einsum('bthd,bshd->bhts', q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out  = jnp.einsum('bhts,bshd->bthd', attn, v).reshape(B, T, C)
        return self.proj(out)


class DiTBlock(nnx.Module):
    """Single DiT block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 rngs: nnx.Rngs = None):
        self.norm1   = nnx.LayerNorm(hidden_size, use_bias=False, rngs=rngs)
        self.attn    = SelfAttention(hidden_size, num_heads, rngs=rngs)
        self.norm2   = nnx.LayerNorm(hidden_size, use_bias=False, rngs=rngs)
        mlp_dim      = int(hidden_size * mlp_ratio)
        self.mlp_fc1 = nnx.Linear(hidden_size, mlp_dim, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(mlp_dim, hidden_size, rngs=rngs)
        self.adaln   = AdaLNZero(hidden_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        s1, a1, g1, s2, a2, g2 = self.adaln(c)  # each (B, D)
        # Attention sub-block
        h = self.norm1(x) * (1 + a1[:, None, :]) + s1[:, None, :]
        x = x + g1[:, None, :] * self.attn(h)
        # MLP sub-block
        h = self.norm2(x) * (1 + a2[:, None, :]) + s2[:, None, :]
        h = nnx.gelu(self.mlp_fc1(h))
        h = self.mlp_fc2(h)
        x = x + g2[:, None, :] * h
        return x


class FinalLayer(nnx.Module):
    """Final adaLN + linear to predict patch velocity."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int,
                 rngs: nnx.Rngs):
        self.norm   = nnx.LayerNorm(hidden_size, use_bias=False, rngs=rngs)
        self.linear = nnx.Linear(hidden_size,
                                  patch_size * patch_size * out_channels, rngs=rngs)
        self.adaln  = nnx.Linear(hidden_size, 2 * hidden_size,
                                  kernel_init=nnx.initializers.zeros,
                                  bias_init=nnx.initializers.zeros,
                                  rngs=rngs)

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        params = nnx.silu(self.adaln(c))
        shift, scale = jnp.split(params, 2, axis=-1)
        x = self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class REPADiT(nnx.Module):
    """
    DiT with REPA:
      - Encoder CLS token injected via projection + LayerNorm
      - After encoder_depth blocks: projector MLP for alignment loss
      - adaLN-Zero conditioning on t + y
      - Final head: patch velocity prediction + CLS velocity prediction
    """

    def __init__(
        self,
        img_size:      int   = IMG_SIZE,
        patch_size:    int   = PATCH_SIZE,
        in_channels:   int   = IN_CHANNELS,
        hidden_size:   int   = HIDDEN_SIZE,
        depth:         int   = DEPTH,
        num_heads:     int   = NUM_HEADS,
        num_classes:   int   = NUM_CLASSES,
        enc_feat_dim:  int   = ENC_FEAT_DIM,
        encoder_depth: int   = ENCODER_DEPTH,
        mlp_ratio:     float = MLP_RATIO,
        rngs: nnx.Rngs = None,
    ):
        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.hidden_size   = hidden_size
        self.encoder_depth = encoder_depth
        self.enc_feat_dim  = enc_feat_dim

        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        # --- Patch embedding ---
        self.patch_embed = nnx.Linear(patch_size * patch_size * in_channels,
                                       hidden_size, rngs=rngs)

        # --- Frozen 2D sincos positional embedding ---
        pos_emb = get_2d_sincos_pos_embed(hidden_size, img_size // patch_size,
                                           img_size // patch_size)
        self.pos_embed = nnx.Variable(pos_emb[None])  # (1, T, D)

        # --- Conditioning ---
        self.t_embed = TimestepEmbedder(hidden_size, rngs=rngs)
        self.y_embed = LabelEmbedder(num_classes, hidden_size, rngs=rngs)

        # --- CLS token projection (encoder CLS → hidden_size) ---
        self.cls_proj  = nnx.Linear(enc_feat_dim, hidden_size, rngs=rngs)
        self.cls_norm  = nnx.LayerNorm(hidden_size, rngs=rngs)

        # --- Transformer blocks ---
        self.blocks = [DiTBlock(hidden_size, num_heads, mlp_ratio, rngs=rngs)
                       for _ in range(depth)]

        # --- REPA projector (after encoder_depth blocks) ---
        proj_hidden = hidden_size * 2
        self.proj_fc1 = nnx.Linear(hidden_size, proj_hidden, rngs=rngs)
        self.proj_fc2 = nnx.Linear(proj_hidden, enc_feat_dim, rngs=rngs)
        self.proj_norm = nnx.LayerNorm(enc_feat_dim, rngs=rngs)

        # --- Final layer ---
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels, rngs=rngs)

        # --- CLS velocity head (predict denoised CLS) ---
        self.cls_head = nnx.Linear(hidden_size, enc_feat_dim, rngs=rngs)

    def patchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """(B, H, W, C) → (B, T, patch_size^2 * C)."""
        B, H, W, C = x.shape
        p = self.patch_size
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)   # (B, gh, gw, p, p, C)
        x = x.reshape(B, -1, p * p * C)
        return x

    def unpatchify(self, x: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
        """(B, T, patch_size^2 * C) → (B, H, W, C)."""
        B, T, _ = x.shape
        p  = self.patch_size
        C  = self.in_channels
        gh = H // p
        gw = W // p
        x  = x.reshape(B, gh, gw, p, p, C)
        x  = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, H, W, C)

    def _project(self, z: jnp.ndarray) -> jnp.ndarray:
        """REPA projector: (B, T, D) → (B, T, enc_feat_dim)."""
        z = nnx.gelu(self.proj_fc1(z))
        z = self.proj_norm(self.proj_fc2(z))
        return z

    def __call__(
        self,
        x:         jnp.ndarray,  # noisy image (B, H, W, C)
        t:         jnp.ndarray,  # timestep scalar in [0,1] (B,)
        y:         jnp.ndarray,  # class labels (B,) int
        cls_token: jnp.ndarray,  # encoder CLS token (B, enc_feat_dim)
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
    ):
        B, H, W, C = x.shape

        # Patch embed + positional encoding
        tokens = self.patchify(x)                      # (B, T, p²C)
        tokens = self.patch_embed(tokens)              # (B, T, D)
        tokens = tokens + self.pos_embed.value         # broadcast (1, T, D)

        # Conditioning vector c = t_embed + y_embed
        t_emb = self.t_embed(t)                        # (B, D)
        y_emb = self.y_embed(y, deterministic=deterministic, rngs=rngs)
        c = t_emb + y_emb                              # (B, D)

        # Inject encoder CLS token as extra token at position 0
        cls_proj = self.cls_norm(self.cls_proj(cls_token))[:, None, :]  # (B, 1, D)
        tokens = jnp.concatenate([cls_proj, tokens], axis=1)            # (B, T+1, D)

        # Transformer blocks; collect intermediate features after encoder_depth
        z_tilde = None
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, c)
            if i == self.encoder_depth - 1:
                # Patch tokens only (skip CLS at index 0)
                z_tilde = self._project(tokens[:, 1:, :])  # (B, T, enc_feat_dim)

        # Predict patch velocities (excluding injected CLS at index 0)
        patch_tokens = tokens[:, 1:, :]                   # (B, T, D)
        v_pred = self.final_layer(patch_tokens, c)         # (B, T, p²C)
        v_pred = self.unpatchify(v_pred, H, W)             # (B, H, W, C)

        # Predict CLS velocity from the CLS position
        cls_v_pred = self.cls_head(tokens[:, 0, :])        # (B, enc_feat_dim)

        return v_pred, z_tilde, cls_v_pred
