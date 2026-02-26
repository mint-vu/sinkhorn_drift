# sinkhorn_drift

Reproducing and extending **Generative Modeling via Drifting** (Kaiming He et al.) on class-conditional ImageNet 256x256 generation.

This codebase supports three feature encoder pipelines with a shared DiT-B/2 generator and implements multiple coupling algorithms (partial two-sided, row softmax, Sinkhorn) for drift field construction.

## Pipeline Overview

### SD-VAE Pipeline (MAE / MoCo-v2)

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `imagenet.encode_latents` | ImageNet RGB &rarr; SD-VAE latents `[N, 4, 32, 32]` |
| 2 | `imagenet.train_mae` | Pretrain ResNet-MAE feature encoder on SD-VAE latents (Appendix A.3) |
| 3 | `imagenet.train_drifting` | Train DiT-B/2 generator with drifting loss |
| Eval | `imagenet.eval_fid` | Generate 50K images, SD-VAE decode, compute FID |

### RAE Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `imagenet.encode_latents_rae` | ImageNet RGB &rarr; RAE latents `[N, 768, 16, 16]` |
| 2 | *(none)* | RAE latents are already semantic &mdash; no pretraining needed |
| 3 | `imagenet.train_drifting` | Train DiT-B/2 generator with drifting loss (direct latent features) |
| Eval | `imagenet.eval_fid_rae` | Generate 50K images, RAE decode, compute FID |

## Feature Encoder Modes

### 1. MAE &mdash; ResNet-MAE on SD-VAE Latents (default)

- **Tokenizer**: `stabilityai/sd-vae-ft-ema` (scaling factor 0.18215)
- **Latent shape**: `[4, 32, 32]`, dtype float16
- **Feature encoder**: Custom ResNet with GroupNorm (`imagenet/models/resnet_mae.py`), pretrained via masked autoencoder on SD-VAE latents
- **Feature extraction**: Multi-scale feature maps from every 2nd residual block (controlled by `--mae-every-n-blocks 2`), each flattened to `[N, 1, C*H*W]`
- **Drift loss**: Computed on multi-scale feature sets (Appendix A.5)

### 2. MoCo-v2 + ResNet-50 &mdash; Frozen MoCo Features on Decoded RGB

- **Tokenizer**: `stabilityai/sd-vae-ft-ema` (same as MAE)
- **Latent shape**: `[4, 32, 32]`, dtype float16
- **Feature encoder**: MoCo-v2 pretrained ResNet-50 (`imagenet/models/moco_resnet.py`)
  - Checkpoint format: loads `module.encoder_q.*` keys, strips FC head
  - Architecture: `torchvision.models.resnet50` backbone
- **Feature extraction**: Multi-scale feature maps from ResNet stages (layer1&ndash;layer4), each flattened to `[N, 1, C*H*W]`
- **Decode path**: SD-VAE latents &rarr; VAE decode to RGB &rarr; `Resize(256)` &rarr; `CenterCrop(224)` &rarr; ImageNet normalize (mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`) &rarr; MoCo-v2 ResNet-50
- **Stability defaults** (enabled by default):
  - `--moco-force-fp32`: feature extraction in float32
  - `--moco-gen-fp32`: generator forward in float32
  - `--moco-clamp-rgb`: decoded RGB clipped to `[0, 1]` before normalization

### 3. RAE &mdash; Representation Autoencoder (DINOv2 / SigLIP2 / MAE encoder)

- **Tokenizer**: RAE (replaces SD-VAE entirely)
  - Frozen pretrained vision encoder + trained ViT decoder
  - Available configs in `rae/configs/stage1/pretrained/`:
    - `DINOv2-B.yaml` &mdash; DINOv2 with registers, base (`facebook/dinov2-with-registers-base`), patch_size=14, hidden=768
    - `MAE.yaml` &mdash; MAE ViT encoder, patch_size=16, hidden=768
    - `SigLIP2.yaml` &mdash; SigLIP2 ViT encoder, patch_size=16, hidden=768
- **Latent shape**: `[768, 16, 16]`, dtype float16 (after channel normalization)
- **Feature encoder**: None needed &mdash; RAE latents are already semantically rich
- **Feature extraction**: Direct latent features via `flatten_latents_as_feature_set(x_lat)` &rarr; `[N, 1, 196608]` (768 &times; 16 &times; 16)
- **Decode path**: RAE decoder (ViT-XL) &rarr; RGB

RAE model weights required (obtain from collaborator):
- Decoder: `rae/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt`
- Normalization stats: `rae/models/stats/dinov2/wReg_base/imagenet1k/stat.pt`

## Generator Architecture: DiT-B/2

All three pipelines share the same DiT-B/2 generator (`imagenet/models/dit_b2.py`):

| Parameter | SD-VAE (MAE / MoCo) | RAE |
|-----------|---------------------|-----|
| `in_ch` / `out_ch` | 4 | 768 |
| `input_size` | 32 | 16 |
| `patch_size` | 2 | 2 |
| `hidden_dim` | 768 | 768 |
| `depth` | 12 | 12 |
| `n_heads` | 12 | 12 |

Architecture features: RoPE, RMSNorm, SwiGLU, QK-Norm, AdaLN-Zero, in-context conditioning (16 prepended learnable tokens, Appendix A.2).

## Drifting Loss (Algorithm 2)

Core implementation: `imagenet/drifting_loss.py`

Default settings match the paper (Table 8 / Appendix A.6&ndash;A.7):

| Parameter | Default | Paper Reference |
|-----------|---------|-----------------|
| `--drift-form` | `alg2_joint` | Algorithm 2 (joint drift field) |
| `--coupling` | `partial_two_sided` | A = sqrt(softmax\_row &middot; softmax\_col) |
| `--temps` | `0.02,0.05,0.2` | Multi-temperature (Eq. 22) |
| `--nc` | 64 | Classes per step |
| `--nneg` | 64 | Generated samples per class |
| `--npos` | 64 | Positive real samples per class |
| `--nuncond` | 16 | Unconditional (CFG) negatives per class |
| `--omega-min/max` | 1.0 / 4.0 | CFG omega range |
| `--omega-exp` | 3.0 | Power-law p(&omega;) &prop; &omega;^{-exp} |
| `--mask-self-neg` | True | Mask diagonal self-coupling |
| `--dist-metric` | `l2` | Pairwise distance &Vert;x&minus;y&Vert; |
| `--drift-theta-norm` | True | Per-temperature drift normalization (Eq. 25) |

Key formulas:
- **Feature normalization** (Eq. 20&ndash;21): S\_j = (1/&radic;C\_j) &middot; E[&Vert;&phi;(x)&minus;&phi;(y)&Vert;]
- **Temperature** (Eq. 22): &rho;\_j = &rho; &middot; &radic;C\_j
- **Drift normalization** (Eq. 24&ndash;25): &theta;\_j = &radic;(E[&Vert;V&Vert;&sup2;/C\_j])
- **Loss** (Eq. 26): L\_j = MSE(&phi;(x), sg(&phi;(x) + V))
- **CFG weight** (A.7): w = (N\_neg &minus; 1)(&omega; &minus; 1) / N\_unc

## Optimizer Defaults (Table 8)

| Parameter | Default |
|-----------|---------|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Betas | (0.9, 0.95) |
| Warmup steps | 5000 |
| Gradient clip | 2.0 |
| EMA decay | 0.999 |
| Total steps | 30,000 |
| AMP dtype | fp16 (bf16 also supported, no GradScaler) |

## Environment Setup

Run everything from the repo root:

```bash
cd /path/to/sinkhorn_drift
python -m imagenet.encode_latents --help
```

### Python Dependencies

Core:
- Python 3.10+, PyTorch >= 2.0, torchvision
- `numpy`, `tqdm`, `Pillow`
- `diffusers`, `transformers`, `accelerate` (for SD-VAE)
- `cleanfid` (for FID computation)

```bash
pip install numpy tqdm pillow
pip install diffusers transformers accelerate
pip install cleanfid
```

RAE-specific (see `rae/requirements.txt`):
- `transformers==4.56.2` (DINOv2/SigLIP2 pretrained encoders)
- `omegaconf==2.3.0`
- `timm==0.9.16`
- `einops`

```bash
pip install -r rae/requirements.txt
```

Tips:
- If HuggingFace downloads are slow, set `HF_TOKEN` (and optionally `HF_HOME` to a shared cache).
- If you see "`accelerate` was not found", install `accelerate`.

## Data

### ImageNet JPEGs

Scripts expect the standard ImageNet folder layout:

```
IMAGENET_ROOT/
  train/<class_name>/*.JPEG
  val/<class_name>/*.JPEG
```

Default `--imagenet-root` is `/home/public/imagenet`; override on your machine.

### Shared Artifacts (Google Drive)

For collaborators, large gitignored artifacts are uploaded here:

- Google Drive: https://drive.google.com/drive/folders/1mQyHTG-W7BeNKhluPIaKko7kyo9UM_vk?usp=sharing
- Contains: `data/` (pre-encoded SD-VAE latents), `runs/imagenet_mae/` (trained MAE checkpoints)

Download (rclone recommended) and place under the repo root.

### Latent Files Format

Stage 1 produces, for each split:

| File | Shape | Dtype |
|------|-------|-------|
| `<split>_latents.npy` | `[N, C, H, W]` | float16 |
| `<split>_labels.npy` | `[N]` | int64 |
| `<split>_meta.json` | &mdash; | JSON |

SD-VAE: `C,H,W = 4,32,32`. RAE (DINOv2-B): `C,H,W = 768,16,16`.

## Stage 1 &mdash; Encode Latents

### SD-VAE Latents (for MAE / MoCo-v2)

```bash
# Multi-GPU with merge
torchrun --standalone --nproc_per_node=4 -m imagenet.encode_latents \
  --imagenet-root /path/to/imagenet --split train \
  --out-dir data/imagenet256_latents \
  --batch-size 64 --num-workers 8 --pin-memory --merge

# With random-resized-crop augmentation (for latent-MAE pretraining, Appendix A.3)
torchrun --standalone --nproc_per_node=4 -m imagenet.encode_latents \
  --imagenet-root /path/to/imagenet --split train \
  --out-dir data/imagenet256_latents_rrc \
  --random-resized-crop --rrc-scale-min 0.2 --rrc-scale-max 1.0 \
  --hflip-prob 0.5 \
  --batch-size 64 --num-workers 8 --pin-memory --merge
```

### RAE Latents (for RAE pipeline)

```bash
torchrun --standalone --nproc_per_node=8 -m imagenet.encode_latents_rae \
  --rae-config rae/configs/stage1/pretrained/DINOv2-B.yaml \
  --imagenet-root /path/to/imagenet --split train \
  --out-dir data/imagenet256_rae_latents \
  --batch-size 64 --num-workers 8 --pin-memory --merge
```

### Reconstruction Check

```bash
python -m imagenet.inspect_latents \
  --latents-dir data/imagenet256_latents_rrc --split train \
  --num 64 --batch-size 16 --device cuda --save-grid
```

## Stage 2 &mdash; Pretrain ResNet-MAE (only for MAE mode)

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_mae \
  --latents-dir data/imagenet256_latents_rrc --latents-split train \
  --batch-size 512 --global-batch 8192 \
  --amp --amp-dtype fp16 --fused-adamw \
  --num-workers 8 --pin-memory \
  --run-name latent_mae_w256
```

Outputs: `runs/imagenet_mae/<timestamp>_<run-name>_<id>/checkpoints/ckpt_final.pt`

Reconstruction Check check:

```bash
python -m imagenet.inspect_mae \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt --mae-use-ema \
  --latents-dir data/imagenet256_latents_rrc --split train \
  --num 32 --batch-size 16 --mask-ratio 0.5 --device cuda --amp --save-grid
```

## Stage 3 &mdash; Train Drifting Generator

Each feature encoder mode supports two configurations:
- **Baseline** (Kaiming's Drift [2]): `--drift-form alg2_joint --coupling partial_two_sided` &mdash; Algorithm 2 with geometric-mean double-softmax coupling
- **Sinkhorn** (our paper): `--drift-form split --coupling sinkhorn --no-mask-self-neg` &mdash; Algorithm 1, cross-minus-self with doubly-stochastic Sinkhorn couplings

Our Sinkhorn drifting field (Algorithm 1 in our ECCV paper) follows the structure V(X) = P\_cross Y &minus; P\_self X:
- Separate Sinkhorn plans: &Pi;⁺ = Sinkhorn(X, Y⁺), &Pi;⁻ = Sinkhorn(X, Y⁻)
- Row-normalize to barycentric weights: P⁺ = RowNorm(&Pi;⁺), P⁻ = RowNorm(&Pi;⁻)
- V = P⁺Y⁺ &minus; P⁻Y⁻
- **No diagonal masking** (`--no-mask-self-neg`): Sinkhorn's doubly-stochastic constraint prevents mass from concentrating on diagonal, unlike softmax where masking is needed (see paper &sect;5.1)

### 3.1 MAE &mdash; ResNet-MAE Feature Encoder

#### MAE + Baseline (paper reproduction)

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --feature-encoder mae \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt --mae-use-ema \
  --amp --online-sample-every 1000 \
  --run-name mae_baseline
```

#### MAE + Sinkhorn

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --feature-encoder mae \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt --mae-use-ema \
  --drift-form split --coupling sinkhorn --sinkhorn-marginal weighted_cols --no-mask-self-neg \
  --amp --online-sample-every 1000 \
  --run-name mae_sinkhorn
```

### 3.2 MoCo-v2 + ResNet-50 Feature Encoder

Download MoCo-v2 checkpoint (ResNet-50, 800 epochs): https://dl.fbaipublicfiles.com/moco-v2/checkpoint_0199.pth.tar

MoCo mode decodes SD-VAE latents to RGB, then extracts frozen MoCo-v2 ResNet-50 features. Requires `--imagenet-root` and `--moco-ckpt`.

#### MoCo-v2 + Baseline (paper reproduction)

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --feature-encoder moco \
  --imagenet-root /home/public/imagenet \
  --moco-ckpt /path/to/moco_v2_800ep/checkpoint_0199.pth.tar \
  --amp --online-sample-every 1000 \
  --run-name moco_baseline
```

#### MoCo-v2 + Sinkhorn

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --feature-encoder moco \
  --imagenet-root /home/public/imagenet \
  --moco-ckpt /path/to/moco_v2_800ep/checkpoint_0199.pth.tar \
  --drift-form split --coupling sinkhorn --sinkhorn-marginal weighted_cols --no-mask-self-neg \
  --amp --online-sample-every 1000 \
  --run-name moco_sinkhorn
```

### 3.3 RAE &mdash; DINOv2-B Direct Latent Features

RAE latents are already semantically rich, so no separate feature encoder is needed. The drift loss is computed directly on the `[768, 16, 16]` latents via `flatten_latents_as_feature_set`.

RAE model weights required (obtain from collaborator):
- Decoder: `rae/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt`
- Stats: `rae/models/stats/dinov2/wReg_base/imagenet1k/stat.pt`

#### RAE + Baseline (paper reproduction)

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --feature-encoder rae \
  --rae-config rae/configs/stage1/pretrained/DINOv2-B.yaml \
  --latents-dir data/imagenet256_rae_latents \
  --amp --online-sample-every 1000 \
  --run-name rae_dinov2b_baseline
```

#### RAE + Sinkhorn

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --feature-encoder rae \
  --rae-config rae/configs/stage1/pretrained/DINOv2-B.yaml \
  --latents-dir data/imagenet256_rae_latents \
  --drift-form split --coupling sinkhorn --sinkhorn-marginal weighted_cols --no-mask-self-neg \
  --amp --online-sample-every 1000 \
  --run-name rae_dinov2b_sinkhorn
```

### Sinkhorn Notes (Alignment with Our ECCV Paper)

Key flags for Sinkhorn (differences from baseline):
- `--drift-form split`: separate Sinkhorn plans for cross and self coupling (paper Algorithm 1)
- `--coupling sinkhorn`: full doubly-stochastic Sinkhorn iterations (default 20 iters)
- `--no-mask-self-neg`: Sinkhorn's doubly-stochastic constraint prevents diagonal degeneracy &mdash; no masking needed (paper &sect;5.1; toy impl `compute_drift` line 508: `dtype != "sinkhorn"`)
- `--sinkhorn-marginal weighted_cols`: encodes CFG weight w in column marginals for the Y⁻ plan (works with any omega including 1.0)

Contrast with baseline: Kaiming's Drift uses `--mask-self-neg` because row-softmax concentrates all mass on zero-cost diagonal entries, making repulsion vanish. Sinkhorn enforces global column marginals, so each column receives bounded mass regardless of cost.

### Debug / Quick Test

```bash
# Quick test (20 steps, MAE mode)
python -m imagenet.train_drifting \
  --feature-encoder mae \
  --latents-dir data/imagenet256_latents --split train \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt --mae-use-ema \
  --amp --debug

# Overfit on one batch (quick overfit check: loss should decrease steadily)
torchrun --nproc_per_node=1 -m imagenet.train_drifting \
  --feature-encoder mae \
  --latents-dir data/imagenet256_latents --split train \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt --mae-use-ema \
  --overfit-one-batch --fail-on-nan --log-drift-stats --log-every 1 \
  --max-items 256 --steps 500 --amp
```

### Common Gotchas

- `--nc` must be divisible by `world_size` (DDP). Default `--nc 64` works for 1/2/4/8 GPUs.
- With `--max-items` for debugging, you may hit "Not enough classes with >= Npos samples". Fix: increase `--max-items` or reduce `--nc/--npos`.
- Progress bar: only rank 0 shows tqdm. Check `logs.jsonl` for progress.

## Evaluation

### SD-VAE Models (MAE / MoCo-v2)

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.eval_fid \
  --ckpt runs/imagenet_drift/<RUN>/checkpoints/ckpt_step_30000.pt --use-ema \
  --num-gen 50000 --batch-size 64 \
  --omegas 1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --real-dir /path/to/imagenet/val
```

### RAE Models

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.eval_fid_rae \
  --ckpt runs/imagenet_drift/<RUN>/checkpoints/ckpt_step_30000.pt --use-ema \
  --rae-config rae/configs/stage1/pretrained/DINOv2-B.yaml \
  --num-gen 50000 --batch-size 64 \
  --omegas 1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --real-dir /path/to/imagenet/val
```

Both scripts sweep over `--omegas` and report per-omega FID plus best-omega summary.


## Project Structure

```
sinkhorn_drift/
├── imagenet/
│   ├── data/
│   │   ├── imagenet_folders.py     # ImageNet dataset builder
│   │   └── latents_memmap.py       # Shape-agnostic memmap I/O ([4,32,32] and [768,16,16])
│   ├── models/
│   │   ├── dit_b2.py               # DiT-B/2 generator
│   │   ├── ema.py                  # Exponential moving average
│   │   ├── moco_resnet.py          # MoCo-v2 ResNet-50 feature encoder
│   │   └── resnet_mae.py           # ResNet-MAE feature encoder (GroupNorm)
│   ├── utils/
│   │   ├── dist.py                 # DDP / distributed utilities
│   │   ├── misc.py                 # Seeding, timing helpers
│   │   └── runs.py                 # Run directory management
│   ├── tests/                      # Unit tests (pytest)
│   ├── drifting_loss.py            # Core drift field + coupling algorithms
│   ├── drift_field.py              # Drift field helpers
│   ├── encode_latents.py           # Stage 1: RGB → SD-VAE latents
│   ├── encode_latents_rae.py       # Stage 1: RGB → RAE latents
│   ├── train_mae.py                # Stage 2: Pretrain ResNet-MAE
│   ├── train_drifting.py           # Stage 3: Train drifting generator
│   ├── eval_fid.py                 # Eval: FID for SD-VAE models
│   ├── eval_fid_rae.py             # Eval: FID for RAE models
│   ├── inspect_latents.py          # Visualize decoded latents
│   ├── inspect_mae.py              # Visualize MAE reconstructions
│   └── vae_sd.py                   # SD-VAE load / encode / decode
├── rae/                            # RAE (Representation Autoencoder) module
│   ├── src/
│   │   ├── stage1/
│   │   │   ├── rae.py              # RAE model: encode() / decode()
│   │   │   ├── encoders/           # DINOv2, MAE, SigLIP2 frozen encoders
│   │   │   └── decoders/           # ViT decoder
│   │   └── utils/model_utils.py    # instantiate_from_config()
│   ├── configs/
│   │   ├── stage1/pretrained/      # DINOv2-B.yaml, MAE.yaml, SigLIP2.yaml
│   │   └── decoder/                # ViTB, ViTL, ViTXL configs
│   ├── models/                     # Pretrained weights (not in git)
│   └── requirements.txt
├── weights/                        # Checkpoints (not in git)
├── data/                           # Pre-encoded latent datasets
├── runs/                           # Training outputs
├── scripts/                        # Shell scripts
├── paper/                          # Reference papers
└── toyExp/                         # Toy experiments (not needed for ImageNet)
```

## Known Issues

### Temperature Bug (Fixed)

The original implementation had a bug in the temperature formula at `drifting_loss.py`:

```python
# Bug:  temp_eff = rho * c        (c = feature dimension)
# Fix:  temp_eff = rho * sqrt(c)  (Eq. 22: rho_tilde = rho * sqrt(C))
```

Impact by feature encoder:

| Mode | C (feature dim) | Bug temp (rho=0.02) | Correct temp | Effect |
|------|-----------------|---------------------|--------------|--------|
| MAE | varies per scale | moderate | moderate | Degraded but may train |
| MoCo-v2 | varies per scale | moderate | moderate | Degraded but may train |
| RAE | 196,608 | 3,932 | 8.86 | **NaN** (coupling &rarr; uniform &rarr; zero drift &rarr; AMP overflow) |

**Status**: Fixed in this codebase. The collaborator's branch (`sinkhorn_rae`) still has this bug at `drifting_loss.py:564`.

## References

- K. He et al., *Generative Modeling via Drifting*, 2025.
- X. Chen et al., *Improved Baselines with Momentum Contrastive Learning* (MoCo-v2), 2020.
- M. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
