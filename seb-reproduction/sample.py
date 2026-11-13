"""
REPA-DiT sampler: Euler ODE, 50 steps, generates a grid of CIFAR-10 images.

Usage:
  python sample.py [--step STEP_NUMBER]  (default: latest checkpoint)
"""

import os
import glob
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from PIL import Image

from model import REPADiT, IMG_SIZE, IN_CHANNELS, ENC_FEAT_DIM, NUM_CLASSES, PATCH_SIZE

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_STEPS    = 50        # Euler ODE steps
NUM_IMAGES   = 64        # grid of 64 images (8×8)
CFG_SCALE    = 2.0       # classifier-free guidance scale (1.0 = no guidance)
CKPT_DIR     = "checkpoints"
OUT_PATH     = "samples.png"
GRID_SIZE    = 8         # 8×8 grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_ema_model(ckpt_dir: str):
    # Find latest checkpoint
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.npz")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    latest = ckpts[-1]
    print(f"Loading checkpoint: {latest}")

    with open(os.path.join(ckpt_dir, "graphdef.pkl"), "rb") as f:
        graphdef = pickle.load(f)

    data = np.load(latest)
    ema_leaves = [jnp.array(data[k]) for k in sorted(data.files)]

    # Reconstruct model from graphdef + EMA state
    # Build a fresh model to get the state treedef
    dummy_rng = nnx.Rngs(params=0)
    dummy_model = REPADiT(rngs=dummy_rng)
    _, dummy_state = nnx.split(dummy_model)

    # Replace leaves with EMA values
    flat_state, treedef = jax.tree_util.tree_flatten(dummy_state)
    assert len(flat_state) == len(ema_leaves), (
        f"State leaf count mismatch: {len(flat_state)} vs {len(ema_leaves)}")
    ema_state = jax.tree_util.tree_unflatten(treedef, ema_leaves)

    model = nnx.merge(graphdef, ema_state)
    return model


def euler_sampler(model, x_T, y, cls_cond, num_steps: int = NUM_STEPS,
                  cfg_scale: float = CFG_SCALE):
    """
    Euler ODE sampler for flow matching.
    x_T:      (B, H, W, C) pure noise
    y:        (B,) class labels
    cls_cond: (B, D) encoder CLS features (use zeros for sampling without real encoder)
    Returns:  (B, H, W, C) clean images in [-1, 1]
    """
    B = x_T.shape[0]
    null_cls = jnp.zeros_like(cls_cond)
    null_y   = jnp.full((B,), NUM_CLASSES, dtype=jnp.int32)  # unconditional class

    @jax.jit
    def step_fn(x_t, t_scalar):
        t = jnp.full((B,), t_scalar)
        # Conditional
        v_cond, _, _ = model(x_t.astype(jnp.bfloat16), t, y,
                              cls_cond.astype(jnp.bfloat16))
        v_cond = v_cond.astype(jnp.float32)
        if cfg_scale > 1.0:
            v_uncond, _, _ = model(x_t.astype(jnp.bfloat16), t, null_y,
                                    null_cls.astype(jnp.bfloat16))
            v_uncond = v_uncond.astype(jnp.float32)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return v

    x = x_T
    dt = -1.0 / num_steps   # integrate from t=1 → t=0
    ts = np.linspace(1.0, 1.0 / num_steps, num_steps)
    for t_val in ts:
        v  = step_fn(x, float(t_val))
        x  = x + dt * v

    return jnp.clip(x, -1.0, 1.0)


def to_grid(imgs_np: np.ndarray, grid: int) -> Image.Image:
    """imgs_np: (N, H, W, C) float in [-1,1] → PIL image grid."""
    imgs = ((imgs_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    H, W, C = imgs.shape[1:]
    canvas = np.zeros((grid * H, grid * W, C), dtype=np.uint8)
    for idx in range(grid * grid):
        r, c = divmod(idx, grid)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = imgs[idx]
    return Image.fromarray(canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model = load_ema_model(CKPT_DIR)
    model.eval() if hasattr(model, 'eval') else None

    key = jax.random.PRNGKey(0)
    B   = NUM_IMAGES

    # Sample one image per class (cycling through 10 classes × 6 + 4)
    labels_np = np.array([i % NUM_CLASSES for i in range(B)], dtype=np.int32)
    y         = jnp.array(labels_np)

    # Start from pure noise
    x_T = jax.random.normal(key, (B, IMG_SIZE, IMG_SIZE, IN_CHANNELS))

    # Use zero CLS features for sampling (no real encoder at inference time)
    cls_cond = jnp.zeros((B, ENC_FEAT_DIM))

    print(f"Sampling {B} images with {NUM_STEPS} Euler steps (CFG scale={CFG_SCALE})...")
    samples = euler_sampler(model, x_T, y, cls_cond)
    samples_np = np.array(samples)

    grid_img = to_grid(samples_np, GRID_SIZE)
    grid_img.save(OUT_PATH)
    print(f"Saved sample grid → {OUT_PATH}")


if __name__ == "__main__":
    main()
