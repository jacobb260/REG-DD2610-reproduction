# REPA on CIFAR-10 — Minimal JAX/Flax Reproduction

A clean, self-contained reproduction of the core ideas from
**[REPA: Representation Alignment for Generation](https://arxiv.org/abs/2406.11838)**
targeting CIFAR-10 on a single **Google v5e1 TPU** (~30 min training).

> [!NOTE]
> This code is written from scratch and shares no code with the original repo.
> The original repo is provided for reference only.

---

## What's Implemented

### Core REPA ideas preserved

| Concept | Detail |
|---|---|
| **Flow matching (v-prediction)** | Linear interpolant `x_t = (1−t)·x0 + t·ε`, target `v = ε − x0` |
| **DiT with adaLN-Zero** | Conditioning `c = t_embed + y_embed` modulates every block via shift/scale/gate |
| **CLS token injection** | Encoder CLS token projected + LayerNorm, prepended as token 0 to the sequence |
| **REPA alignment loss** | After `ENCODER_DEPTH=3` blocks: MLP projects DiT features → cosine similarity with frozen encoder patch features |
| **CLS denoising loss** | CLS token also noised; model predicts its velocity via a head on token 0 |
| **EMA** | Exponential moving average of weights (`decay=0.9999`) |
| **Classifier-free guidance** | 10% label dropout during training; CFG at sampling time |

### Adaptations for CIFAR-10 / v5e1

| Original | This repo |
|---|---|
| ImageNet 256×256 via SD-VAE latents | CIFAR-10 32×32 raw pixels |
| SiT-XL/2 (~675M params) | Tiny DiT: `hidden=256, depth=6, heads=4` (~6M params) |
| DINOv2 ViT-B/14 (online) | DINOv2 ViT-S/14 via `timm` — **pre-extracted once on CPU** (~2 min) |
| PyTorch + Accelerate | JAX + Flax NNX, BF16 mixed precision |
| 1M steps on 8× A100 | ~3000 steps on 1× v5e1 |

---

## Files

```
seb-reproduction/
├── model.py          # REPADiT — full model (~280 lines)
├── train.py          # Training script (~310 lines)
├── sample.py         # Euler ODE sampler + PNG grid (~143 lines)
├── requirements.txt  # Dependencies
└── README.md
```

---

## Quick Start

```bash
cd seb-reproduction

# Install deps (TPU environment)
pip install -r requirements.txt

# Train (~30 min on v5e1)
python3 train.py

# Sample a grid of images after training
python3 sample.py
# → saves samples.png
```

---

## Architecture

```
Input: noisy image x_t (B, 32, 32, 3)
         ↓ patchify (patch_size=4) → 64 tokens
         ↓ patch_embed → (B, 64, 256)

Encoder CLS token (B, 384)
         ↓ cls_proj + LayerNorm → (B, 1, 256)
         ↓ prepend → (B, 65, 256)  [CLS + 64 patch tokens]

Conditioning:  c = TimestepEmbedder(t) + LabelEmbedder(y)  → (B, 256)

DiT blocks (×6, adaLN-Zero):
  blocks[0..2]  → after block 3: REPA projector → (B, 64, 384) = z_tilde
  blocks[3..5]

FinalLayer:
  patch tokens → v_pred (B, 32, 32, 3)   [velocity for denoising]
  CLS token    → cls_v_pred (B, 384)      [CLS velocity]

Loss = MSE(v_pred, v_target)                        # flow matching
     + 0.5 × (−cosine_sim(z_tilde, z_enc_patches))  # REPA alignment
     + 0.1 × MSE(cls_v_pred, cls_v_target)          # CLS denoising
```

---

## Sampling

The sampler uses an **Euler ODE integrator** integrating from `t=1` (noise) to `t=0` (image)
with **classifier-free guidance** (default scale `2.0`). The encoder is not needed at inference —
CLS conditioning is set to zeros.

