"""Single-device CIFAR-10/64 experiment: SiT, REPA and REG.

Uses fixed images and cached frozen-teacher features; see SMALL_EXPERIMENT.md.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from models.sit import SiT_models
from samplers import get_score_from_velocity

VAE_NAME = 'stabilityai/sd-vae-ft-mse'
SCALE = 0.18215


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    return torch.device(name if name != 'auto' else (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'))


def vae_for(device):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(VAE_NAME).eval().requires_grad_(False).to(device)


@torch.no_grad()
def prepare(args):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import functional as TF
    from torchvision.utils import save_image
    device = device_for(args.device)
    root = Path(args.data)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob('*.npy')) or (root / 'metadata.json').exists():
        raise ValueError('Cache directory already contains data; use a new directory.')
    dataset = CIFAR10(args.download_dir, train=True, download=True)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    # Loading torch.hub code is explicit; use the official DINOv2 repository.
    teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                             trust_repo=True).eval().requires_grad_(False).to(device)
    vae = vae_for(device)
    shapes = {'moments': (count, 8, 8, 8), 'patches': (count, 16, 768),
              'cls': (count, 768), 'labels': (count,)}
    arrays = {name: np.lib.format.open_memmap(root / f'{name}.npy', mode='w+',
              dtype=np.int64 if name == 'labels' else np.float16, shape=shape)
              for name, shape in shapes.items()}
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    for start in range(0, count, args.batch_size):
        items = [dataset[i] for i in range(start, min(start + args.batch_size, count))]
        x = torch.stack([TF.pil_to_tensor(image) for image, _ in items]).to(device).float() / 255
        x = F.interpolate(x, size=(64, 64), mode='bicubic', align_corners=False,
                          antialias=True).clamp(0, 1)
        posterior = vae.encode(x * 2 - 1).latent_dist
        moments = torch.cat([posterior.mean, posterior.std], dim=1)
        teacher_input = F.interpolate(x, size=(224, 224), mode='bicubic',
                                      align_corners=False, antialias=True)
        features = teacher.forward_features((teacher_input - mean) / std)
        # DINO's 16x16 patch grid -> SiT's 4x4 grid, before cosine normalization.
        patches = features['x_norm_patchtokens'].transpose(1, 2).reshape(-1, 768, 16, 16)
        patches = F.adaptive_avg_pool2d(patches, (4, 4)).flatten(2).transpose(1, 2)
        end = start + len(items)
        for name, value in [('moments', moments), ('patches', patches),
                            ('cls', features['x_norm_clstoken'])]:
            arrays[name][start:end] = value.cpu().numpy().astype(np.float16)
        arrays['labels'][start:end] = [label for _, label in items]
        if start == 0:
            reconstructed = (vae.decode(posterior.mean).sample + 1) / 2
            n = min(8, len(items))
            save_image(torch.cat([x[:n], reconstructed[:n].clamp(0, 1)]),
                       root / 'vae_reconstruction.png', nrow=n)
        print(f'Prepared {end}/{count}', flush=True)
    for array in arrays.values():
        array.flush()
    metadata = dict(version=1, dataset='CIFAR10', split='train', count=count,
                    resolution=64, num_classes=10, vae=VAE_NAME,
                    teacher='dinov2_vitb14', teacher_size=224,
                    alignment='adaptive_avg_pool_16x16_to_4x4', augmentation='none',
                    moments='unscaled mean and std', scale=SCALE)
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Cache complete: {root}. Inspect vae_reconstruction.png before training.')


def load_cache(root):
    root = Path(root)
    metadata = json.loads((root / 'metadata.json').read_text())
    expected = dict(version=1, resolution=64, num_classes=10, vae=VAE_NAME,
                    alignment='adaptive_avg_pool_16x16_to_4x4', scale=SCALE)
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('Unsupported cache metadata')
    arrays = {k: np.load(root / f'{k}.npy', mmap_mode='r')
              for k in ['moments', 'patches', 'cls', 'labels']}
    n = metadata['count']
    for key, shape in dict(moments=(n, 8, 8, 8), patches=(n, 16, 768),
                           cls=(n, 768), labels=(n,)).items():
        if arrays[key].shape != shape:
            raise ValueError(f'Invalid cache shape for {key}')
    if n < 1 or arrays['labels'].min() < 0 or arrays['labels'].max() > 9:
        raise ValueError('Invalid cache labels/count')
    digest = hashlib.sha256()
    for path in [root / 'metadata.json', *sorted(root.glob('*.npy'))]:
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
    return arrays, digest.hexdigest()


def make_model(method, depth=4):
    return SiT_models['SiT-S/2'](input_size=8, num_classes=10, method=method,
                                encoder_depth=depth, z_dims=[768], fused_attn=True)


def compute_loss(model, x, labels, patches=None, cls=None, proj_coeff=.5, cls_coeff=.03):
    b = x.shape[0]
    t = torch.rand(b, device=x.device)
    noise = torch.randn_like(x)
    noisy = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise
    cls_input = None
    cls_noise = torch.randn(b, 768, device=x.device)
    if model.method == 'reg':
        cls_input = (1 - t[:, None]) * cls + t[:, None] * cls_noise
    predicted, projected, cls_predicted = model(noisy, t, labels, cls_token=cls_input)
    denoise = F.mse_loss(predicted.float(), (noise - x).float())
    alignment = denoise.new_zeros(())
    cls_loss = denoise.new_zeros(())
    if model.method != 'sit':
        target = patches if model.method == 'repa' else torch.cat([cls[:, None], patches], dim=1)
        alignment = -(F.normalize(projected[0].float(), dim=-1) *
                      F.normalize(target.float(), dim=-1)).sum(-1).mean()
    if model.method == 'reg':
        cls_loss = F.mse_loss(cls_predicted.float(), (cls_noise - cls).float())
    total = denoise + proj_coeff * alignment + cls_coeff * cls_loss
    return total, dict(denoise=denoise.detach().item(), alignment=alignment.detach().item(),
                      cls=cls_loss.detach().item())


def train(args):
    device = device_for(args.device)
    seed_all(args.seed)
    arrays, fingerprint = load_cache(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume:
        raise ValueError('Output already has last.pt; use --resume or a new output directory.')
    model = make_model(args.method).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0,
                                  betas=(.9, .999))
    amp = args.precision != 'no'
    if amp and device.type != 'cuda':
        raise ValueError('Mixed precision in this runner requires CUDA; use --precision=no.')
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    rng = np.random.default_rng(args.seed)
    config = dict(method=args.method, model='SiT-S/2', resolution=64, num_classes=10,
                  encoder_depth=4, vae=VAE_NAME, seed=args.seed, batch_size=args.batch_size,
                  accumulation=args.accumulation, lr=args.lr, precision=args.precision,
                  proj_coeff=args.proj_coeff, cls_coeff=args.cls_coeff, cache=fingerprint)
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration/cache differs from checkpoint')
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        step = checkpoint['step']
    else:
        # Model initialization draws differ between methods; restart training RNG equally.
        seed_all(args.seed + 1)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model.train()
    start = time.monotonic()
    start_step = step
    print(f'{args.method}: {sum(p.numel() for p in model.parameters()):,} parameters; '
          f'effective batch {args.batch_size * args.accumulation}; {device}', flush=True)
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        totals = dict(denoise=0., alignment=0., cls=0.)
        for microstep in range(args.accumulation):
            seed_all(args.seed + 1 + step * args.accumulation + microstep)
            ids = rng.integers(len(arrays['labels']), size=args.batch_size)
            def batch(key):
                return torch.from_numpy(np.array(arrays[key][ids])).to(device)
            moments = batch('moments').float()
            mean, std = moments.chunk(2, dim=1)
            x = (mean + std * torch.randn_like(mean)) * SCALE
            labels = batch('labels').long()
            patches = batch('patches').float() if args.method != 'sit' else None
            cls = batch('cls').float() if args.method == 'reg' else None
            context = torch.autocast('cuda', dtype=dtype) if amp else nullcontext()
            with context:
                loss, metrics = compute_loss(model, x, labels, patches, cls,
                                              args.proj_coeff, args.cls_coeff)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            scaler.scale(loss / args.accumulation).backward()
            for k, v in metrics.items():
                totals[k] += v / args.accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            continue  # overflow: no optimizer/EMA step completed
        with torch.no_grad():
            for dst, src in zip(ema.parameters(), model.parameters()):
                dst.lerp_(src, 1 - .9999)
        step += 1
        if step % args.log_every == 0 or step == 1:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            record = dict(step=step, seconds=elapsed, seconds_per_step=elapsed / (step-start_step), **totals)
            with (output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            print(record, flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = dict(config=config, model=model.state_dict(), ema=ema.state_dict(),
                              optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), step=step,
                              numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            temporary = output / 'last.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(output / 'last.pt')
            print(f'Saved {output / "last.pt"} at step {step}', flush=True)


@torch.no_grad()
def sample_latents(model, batch_size, device, steps, offset=0):
    """Linear-path Euler-Maruyama; final step is deterministic, without CFG."""
    x = torch.randn(batch_size, 4, 8, 8, device=device)
    labels = (torch.arange(batch_size, device=device) + offset) % 10
    cls = torch.randn(batch_size, 768, device=device) if model.method == 'reg' else None
    times = torch.cat([torch.linspace(1., .04, steps, device=device), torch.zeros(1, device=device)])
    for i in range(steps):
        t, next_t = times[i], times[i+1]
        dt = next_t - t
        ts = t.expand(batch_size)
        velocity, _, cls_velocity = model(x, ts, labels, cls_token=cls)
        diffusion = 2 * t
        drift = velocity - .5 * diffusion * get_score_from_velocity(velocity, x, ts)
        x = x + dt * drift
        if cls is not None:
            cls_drift = cls_velocity - .5 * diffusion * get_score_from_velocity(cls_velocity, cls, ts)
            cls = cls + dt * cls_drift
        if i < steps - 1:
            x = x + torch.sqrt(diffusion * -dt) * torch.randn_like(x)
            if cls is not None:
                cls = cls + torch.sqrt(diffusion * -dt) * torch.randn_like(cls)
    return x, labels


@torch.no_grad()
def sample(args):
    from torchvision.utils import save_image
    seed_all(args.seed)
    device = device_for(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = make_model(config['method'], config['encoder_depth']).to(device).eval()
    model.load_state_dict(checkpoint['ema'])
    vae = vae_for(device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('*.png')):
        raise ValueError('Sample directory already contains PNG files; use a new directory.')
    previews, all_labels = [], []
    for start in range(0, args.count, args.batch_size):
        latents, labels = sample_latents(model, min(args.batch_size, args.count-start), device, args.steps, start)
        images = (vae.decode(latents / SCALE).sample + 1).div(2).clamp(0, 1)
        for j, image in enumerate(images):
            save_image(image, output / f'{start+j:06d}.png')
            if len(previews) < 64:
                previews.append(image.cpu())
        all_labels.extend(labels.cpu().tolist())
        print(f'Generated {start+len(images)}/{args.count}', flush=True)
    save_image(torch.stack(previews), output / 'preview.png', nrow=8)
    (output / 'labels.json').write_text(json.dumps(all_labels))
    (output / 'sampling.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        checkpoint_step=checkpoint['step'], seed=args.seed, steps=args.steps, cfg_scale=1.,
        count=args.count, config=config), indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data', default='data/cifar10-64')
    prep.add_argument('--download-dir', default='data/downloads')
    prep.add_argument('--limit', type=int, default=None, help='Pilot cache only; omit for all 50k images')
    prep.add_argument('--batch-size', type=int, default=32)
    training = sub.add_parser('train')
    training.add_argument('--method', choices=['sit', 'repa', 'reg'], required=True)
    training.add_argument('--data', default='data/cifar10-64')
    training.add_argument('--output', required=True)
    training.add_argument('--steps', type=int, default=20000)
    training.add_argument('--batch-size', type=int, default=32, help='Microbatch size')
    training.add_argument('--accumulation', type=int, default=4)
    training.add_argument('--precision', choices=['no', 'fp16', 'bf16'], default='no')
    training.add_argument('--lr', type=float, default=1e-4)
    training.add_argument('--proj-coeff', type=float, default=.5)
    training.add_argument('--cls-coeff', type=float, default=.03)
    training.add_argument('--seed', type=int, default=0)
    training.add_argument('--resume')
    training.add_argument('--log-every', type=int, default=100)
    training.add_argument('--save-every', type=int, default=1000)
    sampling = sub.add_parser('sample')
    sampling.add_argument('--checkpoint', required=True)
    sampling.add_argument('--output', required=True)
    sampling.add_argument('--count', type=int, default=64)
    sampling.add_argument('--batch-size', type=int, default=20)
    sampling.add_argument('--steps', type=int, default=250)
    sampling.add_argument('--seed', type=int, default=0)
    for child in (prep, training, sampling):
        child.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    args = parser.parse_args()
    for key in ('batch_size', 'accumulation', 'steps', 'count', 'log_every', 'save_every', 'limit'):
        if hasattr(args, key) and getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.command == 'sample' and args.steps < 2:
        parser.error('--steps must be at least 2')
    return args


if __name__ == '__main__':
    args = parse_args()
    {'prepare': prepare, 'train': train, 'sample': sample}[args.command](args)
