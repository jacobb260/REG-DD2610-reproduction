"""
REPA training script for CIFAR-10 on JAX/Flax NNX (TPU v5e1).

Steps:
  1. Pre-extract DINOv2 ViT-S/14 features for all 50k CIFAR-10 train images (CPU/torch).
  2. Train DiT with flow-matching + REPA alignment loss in JAX.
"""

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx
import tensorflow_datasets as tfds
import torch
import timm
from PIL import Image
from tqdm import tqdm

from model import REPADiT, HIDDEN_SIZE, DEPTH, NUM_HEADS, PATCH_SIZE, \
                  IMG_SIZE, IN_CHANNELS, NUM_CLASSES, ENC_FEAT_DIM, ENCODER_DEPTH

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE    = 256
TOTAL_STEPS   = 3000
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
EMA_DECAY     = 0.9999
PROJ_COEFF    = 0.5    # weight for REPA alignment loss
CLS_COEFF     = 0.1    # weight for CLS denoising loss
SAVE_EVERY    = 500
LOG_EVERY     = 50
SEED          = 42
CKPT_DIR      = "checkpoints"
# DINOv2 ViT-S/14 — timm model name
ENCODER_NAME  = "vit_small_patch14_dinov2.lvd142m"
ENCODER_IMG   = 224    # encoder expects 224×224
NUM_T_STEPS   = 50     # ODE steps for sampling (unused here, used in sample.py)

os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Pre-extract encoder features (runs on CPU/torch, once)
# ---------------------------------------------------------------------------
def preextract_features(encoder_name: str, batch_size: int = 512):
    """
    Returns:
      train_imgs      : (50000, 32, 32, 3)  uint8
      train_labels    : (50000,)             int32
      train_cls_feats : (50000, ENC_FEAT_DIM) float32
      train_patch_feats: (50000, 64, ENC_FEAT_DIM) float32  [64 = 16×16 patches at 224/14]
    """
    print("Loading CIFAR-10 via tensorflow_datasets...")
    ds = tfds.load("cifar10", split="train", as_supervised=True, shuffle_files=False)
    imgs_list, lbls_list = [], []
    for img, lbl in tfds.as_numpy(ds):
        imgs_list.append(img)
        lbls_list.append(int(lbl))
    train_imgs   = np.stack(imgs_list, axis=0)   # (50000, 32, 32, 3)
    train_labels = np.array(lbls_list, dtype=np.int32)

    print(f"Pre-extracting features with {encoder_name} on CPU...")
    device = torch.device("cpu")
    encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)
    encoder.eval().to(device)

    data_cfg  = timm.data.resolve_model_data_config(encoder)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    cls_feats_list   = []
    patch_feats_list = []

    n = len(train_imgs)
    for start in tqdm(range(0, n, batch_size), desc="Extracting features"):
        end   = min(start + batch_size, n)
        batch = train_imgs[start:end]  # (B, 32, 32, 3) uint8

        # Resize to 224 and apply transform
        tensors = []
        for img_np in batch:
            pil = Image.fromarray(img_np).resize((ENCODER_IMG, ENCODER_IMG), Image.BICUBIC)
            tensors.append(transform(pil))
        x = torch.stack(tensors).to(device)  # (B, 3, 224, 224)

        with torch.no_grad():
            # get_intermediate_layers returns list of (B, T, D) tensors
            feats = encoder.get_intermediate_layers(x, n=1, reshape=False)[0]  # (B, T+1, D)
        feats_np = feats.cpu().numpy().astype(np.float32)
        cls_feats_list.append(feats_np[:, 0, :])    # CLS token
        patch_feats_list.append(feats_np[:, 1:, :]) # patch tokens (B, 256, D) for 224/14

    cls_feats   = np.concatenate(cls_feats_list,   axis=0)  # (50000, D)
    patch_feats = np.concatenate(patch_feats_list, axis=0)  # (50000, 256, D)

    # For 32×32 images with patch_size=4 we have 64 patches in the DiT.
    # We need to match to 64 patch features from the encoder.
    # The encoder produces 256 patch tokens (224/14=16 → 16×16=256).
    # Reshape encoder patches to 16×16 and pool to 8×8 (4 → 1 avg) → 64 tokens.
    # 256 patches → (16, 16) → downsample to (8, 8) by average pooling 2×2 blocks
    B_total, _, D = patch_feats.shape
    pf = patch_feats.reshape(B_total, 16, 16, D)
    pf = pf.reshape(B_total, 8, 2, 8, 2, D).mean(axis=(2, 4))  # (B, 8, 8, D)
    patch_feats_64 = pf.reshape(B_total, 64, D)

    print(f"Features extracted. cls: {cls_feats.shape}, patches: {patch_feats_64.shape}")
    return train_imgs, train_labels, cls_feats, patch_feats_64


# ---------------------------------------------------------------------------
# 2. Loss function (pure JAX, jit-able)
# ---------------------------------------------------------------------------
def flow_matching_loss(v_pred, x0, eps, t):
    """
    Linear interpolant: x_t = (1-t)*x0 + t*eps
    Velocity target:    v   = eps - x0
    Loss: MSE(v_pred, v_target)
    """
    v_target = eps - x0
    return jnp.mean((v_pred - v_target) ** 2)


def repa_loss(z_tilde, z_enc):
    """
    REPA alignment loss (cosine similarity maximisation).
    z_tilde: (B, T, D_proj)  — DiT projected features
    z_enc:   (B, T, D_enc)   — encoder patch features
    """
    z_tilde = z_tilde / (jnp.linalg.norm(z_tilde, axis=-1, keepdims=True) + 1e-6)
    z_enc   = z_enc   / (jnp.linalg.norm(z_enc,   axis=-1, keepdims=True) + 1e-6)
    return -jnp.mean(jnp.sum(z_tilde * z_enc, axis=-1))


def cls_denoising_loss(cls_v_pred, cls_token_clean, cls_token_noisy, t):
    """
    CLS token is also noised along the same trajectory.
    cls_v_pred: (B, D)  — predicted velocity for CLS
    target velocity = cls_eps - cls_token_clean  (same as image)
    We use the noised CLS as: cls_noisy = (1-t)*cls_clean + t*cls_eps
    → cls_eps = (cls_noisy - (1-t)*cls_clean) / t  (numerically safe for t>0)
    """
    t_safe = jnp.clip(t, 1e-5, 1.0)[:, None]
    cls_eps = (cls_token_noisy - (1.0 - t[:, None]) * cls_token_clean) / t_safe
    v_target = cls_eps - cls_token_clean
    return jnp.mean((cls_v_pred - v_target) ** 2)


# ---------------------------------------------------------------------------
# 3. Training step
# ---------------------------------------------------------------------------
def make_train_step(model, optimizer):
    @nnx.jit
    def train_step(x0, y, cls_feats, patch_feats, t, eps, cls_eps, train_key):
        """
        x0:          (B, H, W, C) float32  clean images in [-1, 1]
        y:           (B,) int32
        cls_feats:   (B, D)
        patch_feats: (B, T, D)
        t:           (B,) float32 timestep in [0,1]
        eps:         (B, H, W, C) float32 noise
        cls_eps:     (B, D) float32 noise for CLS
        """
        # Noisy image and noisy CLS
        t4 = t[:, None, None, None]
        x_t       = (1.0 - t4) * x0 + t4 * eps
        cls_noisy = (1.0 - t[:, None]) * cls_feats + t[:, None] * cls_eps

        def loss_fn(model):
            rngs = nnx.Rngs(dropout=train_key)
            v_pred, z_tilde, cls_v_pred = model(
                x_t.astype(jnp.bfloat16),
                t,
                y,
                cls_noisy.astype(jnp.bfloat16),
                deterministic=False,
                rngs=rngs,
            )
            v_pred     = v_pred.astype(jnp.float32)
            z_tilde    = z_tilde.astype(jnp.float32)
            cls_v_pred = cls_v_pred.astype(jnp.float32)

            l_diff = flow_matching_loss(v_pred, x0, eps, t)
            l_repa = repa_loss(z_tilde, patch_feats)
            l_cls  = cls_denoising_loss(cls_v_pred, cls_feats, cls_noisy, t)
            loss   = l_diff + PROJ_COEFF * l_repa + CLS_COEFF * l_cls
            return loss, (l_diff, l_repa, l_cls)

        (loss, (l_diff, l_repa, l_cls)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True)(model)
        optimizer.update(grads)
        return loss, l_diff, l_repa, l_cls

    return train_step


# ---------------------------------------------------------------------------
# 4. Data utilities
# ---------------------------------------------------------------------------
def normalize(imgs):
    """uint8 [0,255] → float32 [-1, 1]."""
    return imgs.astype(np.float32) / 127.5 - 1.0


def make_batches(imgs, labels, cls_feats, patch_feats, batch_size, rng):
    n = len(imgs)
    idx = rng.permutation(n)
    imgs_s = imgs[idx]
    lbls_s = labels[idx]
    cls_s  = cls_feats[idx]
    pat_s  = patch_feats[idx]
    for start in range(0, n - batch_size + 1, batch_size):
        sl = slice(start, start + batch_size)
        yield (normalize(imgs_s[sl]), lbls_s[sl], cls_s[sl], pat_s[sl])


# ---------------------------------------------------------------------------
# 5. EMA helper (manual, in numpy/jax)
# ---------------------------------------------------------------------------
def ema_update(ema_params, model_params, decay):
    return jax.tree.map(lambda e, m: decay * e + (1 - decay) * m, ema_params, model_params)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    print(f"JAX devices: {jax.devices()}")

    # Pre-extract encoder features
    train_imgs, train_labels, cls_feats, patch_feats = preextract_features(ENCODER_NAME)

    # Build model
    rng = nnx.Rngs(params=SEED, dropout=SEED + 1)
    model = REPADiT(rngs=rng)

    # Optax optimizer (AdamW + cosine LR decay)
    schedule = optax.cosine_decay_schedule(LR, TOTAL_STEPS)
    tx = optax.adamw(schedule, weight_decay=WEIGHT_DECAY)
    optimizer = nnx.Optimizer(model, tx)

    # EMA: grab initial params
    graphdef, state = nnx.split(model)
    ema_state = jax.tree.map(lambda x: jnp.array(x), state)

    train_step = make_train_step(model, optimizer)

    np_rng     = np.random.default_rng(SEED)
    jax_key    = jax.random.PRNGKey(SEED)
    step_global = 0
    epoch       = 0

    print(f"Starting training: {TOTAL_STEPS} steps, batch_size={BATCH_SIZE}")
    t0 = time.time()

    while step_global < TOTAL_STEPS:
        epoch += 1
        batches = list(make_batches(train_imgs, train_labels, cls_feats, patch_feats,
                                     BATCH_SIZE, np_rng))
        for x0_np, y_np, cls_np, pat_np in batches:
            if step_global >= TOTAL_STEPS:
                break

            # Sample random timesteps and noise
            jax_key, sk1, sk2, sk3, sk4 = jax.random.split(jax_key, 5)
            t_np   = np.array(jax.random.uniform(sk1, (BATCH_SIZE,)))
            eps_np = np.array(jax.random.normal(sk2, x0_np.shape))
            cls_eps_np = np.array(jax.random.normal(sk3, cls_np.shape))

            # Cast to JAX arrays (bfloat16 for inputs, float32 for feats)
            x0_j   = jnp.array(x0_np,    dtype=jnp.float32)
            y_j    = jnp.array(y_np,     dtype=jnp.int32)
            cls_j  = jnp.array(cls_np,   dtype=jnp.float32)
            pat_j  = jnp.array(pat_np,   dtype=jnp.float32)
            t_j    = jnp.array(t_np,     dtype=jnp.float32)
            eps_j  = jnp.array(eps_np,   dtype=jnp.float32)
            cle_j  = jnp.array(cls_eps_np, dtype=jnp.float32)

            loss, l_diff, l_repa, l_cls = train_step(
                x0_j, y_j, cls_j, pat_j, t_j, eps_j, cle_j, sk4)

            # EMA update
            _, model_state = nnx.split(model)
            ema_state = ema_update(ema_state, model_state, EMA_DECAY)

            step_global += 1

            if step_global % LOG_EVERY == 0:
                elapsed = time.time() - t0
                print(f"step={step_global:4d} | loss={float(loss):.4f} "
                      f"| diff={float(l_diff):.4f} repa={float(l_repa):.4f} "
                      f"cls={float(l_cls):.4f} | {elapsed:.1f}s elapsed")

            if step_global % SAVE_EVERY == 0 or step_global == TOTAL_STEPS:
                ckpt_path = os.path.join(CKPT_DIR, f"step_{step_global:05d}.npz")
                ema_leaves = jax.tree_util.tree_leaves(ema_state)
                # Save graphdef structure separately via pickle for sampling
                import pickle
                with open(os.path.join(CKPT_DIR, "graphdef.pkl"), "wb") as f:
                    pickle.dump(graphdef, f)
                np.savez(ckpt_path,
                         **{f"leaf_{i}": np.array(v) for i, v in enumerate(ema_leaves)})
                print(f"  → checkpoint saved to {ckpt_path}")

    print(f"Training complete in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
