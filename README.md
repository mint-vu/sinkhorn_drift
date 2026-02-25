# sinkhorn_drift (Focus on ImageNet256)

This repo contains an **ImageNet256** pipeline for the paper **Generative Modeling via Drifting** (baseline) plus our follow-up ablations (different couplings / drift forms).
If you only care about ImageNet experiments, focus on `imagenet/`.

Large files (`data/`, `runs/`, `outputs/`, `paper/`) are shared them separately (e.g. Drive/rclone).

## Repo layout

- `imagenet/`: Stage 1–3 pipeline + evaluation
  - `encode_latents.py`: **Stage 1** (ImageNet JPEG → SD‑VAE latents)
  - `train_mae.py`: **Stage 2** (latent‑MAE pretrain; ResNet‑style MAE, Kaiming's Appendix A.3)
  - `train_drifting.py`: **Stage 3** (drifting generator training; Kaiming's paper §5.2)
  - `drifting_loss.py`: drifting loss math core (Kaiming's Appendix A.5–A.7 + our ideas)
  - `eval_fid.py`: decode generated latents → RGB and compute FID (Kaiming's paper §5.2)
  - `inspect_latents.py` / `inspect_mae.py`: verify SD-VAE and latent-MAE works or not before full training
- `toyExp/`: toy derivations / small experiments (not required for ImageNet runs)

## Environment setup

You can run everything from the repo root:

```bash
cd /path/to/drift
python -m imagenet.encode_latents --help
```

### Python deps (minimal)

- Python 3.10+ recommended
- PyTorch + CUDA (install the correct build for your machine)
- Core: `numpy`, `tqdm`, `Pillow`
- Stage 1/inspect/eval needs SD‑VAE from HuggingFace:
  - `diffusers`, `transformers` (and `accelerate` recommended)
- FID needs: `cleanfid`

Example (after installing PyTorch):

```bash
pip install numpy tqdm pillow
pip install diffusers transformers accelerate
pip install cleanfid
```

Tips:
- If HuggingFace downloads are slow / rate-limited, set `HF_TOKEN` (and optionally `HF_HOME` to a shared cache).
- If you see “`accelerate` was not found…”, install `accelerate` (it speeds up and reduces RAM during model loading).

## Data assumptions

### Shared artifacts (Google Drive)

For collaborators, we uploaded the large, gitignored artifacts here:

- Google Drive folder: https://drive.google.com/drive/folders/1mQyHTG-W7BeNKhluPIaKko7kyo9UM_vk?usp=sharing
- Contains (at least):
  - `data/` (pre-encoded SD-VAE latents; e.g. `data/imagenet256_latents_rrc/`)
  - `runs/imagenet_mae/` (trained latent-MAE checkpoints used by Stage 3)

Download (I recommend use rclone, it is so fast) and place them under the repo root so the paths match the example commands.

### ImageNet JPEGs

Scripts expect the standard ImageNet folder layout:

```
IMAGENET_ROOT/
  train/<class_name>/*.JPEG
  val/<class_name>/*.JPEG
```

Default `--imagenet-root` in scripts is `/home/public/imagenet` (cluster-specific); override on your machine.

### Latents files (Stage 1 output)

Stage 1 produces, for each `split in {train,val}`:

- `<out-dir>/<split>_latents.npy`  (shape `[N,4,32,32]`, dtype `float16`)
- `<out-dir>/<split>_labels.npy`   (shape `[N]`, dtype `int64`)
- `<out-dir>/<split>_meta.json`    (preprocess + VAE config)

These are used by Stage 2 (recommended) and Stage 3 (required).

p.s. I only did train folder.

## Stage 1 — Encode ImageNet images to SD‑VAE latents

Single GPU:

```bash
python -m imagenet.encode_latents \
  --imagenet-root /path/to/imagenet \
  --split train \
  --out-dir data/imagenet256_latents \
  --batch-size 64 --num-workers 8 --pin-memory
```

Multi-GPU (writes per-rank shards, then merges on rank0 if `--merge`):

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.encode_latents \
  --imagenet-root /path/to/imagenet \
  --split train \
  --out-dir data/imagenet256_latents \
  --batch-size 64 --num-workers 8 --pin-memory \
  --merge
```

Paper-style augmentation **for latent‑MAE pretraining** (Appendix A.3) is supported here:

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.encode_latents \
  --imagenet-root /path/to/imagenet \
  --split train \
  --out-dir data/imagenet256_latents_rrc \
  --random-resized-crop --rrc-scale-min 0.2 --rrc-scale-max 1.0 \
  --hflip-prob 0.5 \
  --batch-size 64 --num-workers 8 --pin-memory --merge
```

Notes:
- By default, Stage 1 uses `Resize(256)+CenterCrop(256)` and outputs latents scaled by `0.18215` (diffusers convention).
- If you use `--random-resized-crop/--hflip-prob`, the decoded images will **not** pixel-align with a deterministic center-cropped JPEG view; that’s expected.

### Stage 1 sanity check (recommended)

Decode a few random latents back to PNGs:

```bash
python -m imagenet.inspect_latents \
  --latents-dir data/imagenet256_latents_rrc \
  --split train --num 64 --batch-size 16 \
  --device cuda --save-grid
```

Optional: compare against original JPEGs (side-by-side grid):

```bash
python -m imagenet.inspect_latents \
  --latents-dir data/imagenet256_latents_rrc \
  --split train --num 64 --batch-size 16 \
  --compare-jpeg --imagenet-root /path/to/imagenet \
  --device cuda --save-grid
```

Outputs go to `outputs/inspect_latents/<timestamp>_<split>/` and include `meta.json` + `grid.png` (and `grid_pairs.png` if `--compare-jpeg`).

## Stage 2 — Train latent‑MAE (ResNet‑style MAE; Kaiming's Appendix A.3)

Recommended: **train from offline latents** (fast; avoids JPEG decode + VAE encode).

Example (4 GPUs):

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_mae \
  --latents-dir data/imagenet256_latents_rrc --latents-split train \
  --batch-size 512 --global-batch 8192 \
  --amp --amp-dtype fp16 --fused-adamw \
  --num-workers 8 --pin-memory \
  --run-name latent_mae_w256
```

Outputs:
- Run dir: `runs/imagenet_mae/<timestamp>_<run-name>_<id>/`
- Final checkpoint: `.../checkpoints/ckpt_final.pt` (contains both raw + EMA weights)

Resume:

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_mae \
  --resume runs/imagenet_mae/<RUN>/checkpoints/ckpt_optstep_<K>.pt \
  --latents-dir data/imagenet256_latents_rrc --latents-split train \
  --batch-size 512 --global-batch 8192 \
  --amp --amp-dtype fp16 --fused-adamw \
  --num-workers 8 --pin-memory
```

### Stage 2 sanity check (recommended)

Visualize MAE reconstructions (decoded to RGB via SD‑VAE):

```bash
python -m imagenet.inspect_mae \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt \
  --mae-use-ema \
  --latents-dir data/imagenet256_latents_rrc --split train \
  --num 32 --batch-size 16 --mask-ratio 0.5 \
  --device cuda --amp --amp-dtype fp16 \
  --save-grid
```

Optional: also dump the center-cropped JPEG view for side-by-side comparison:

```bash
python -m imagenet.inspect_mae \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt \
  --mae-use-ema \
  --latents-dir data/imagenet256_latents_rrc --split train \
  --num 32 --batch-size 16 --mask-ratio 0.5 \
  --compare-jpeg --imagenet-root /path/to/imagenet \
  --device cuda --amp --amp-dtype fp16 \
  --save-grid
```

If you want to verify the **exact feature sets** used in Stage 3 (Kaiming's Appendix A.5), add `--check-feature-sets`.

Outputs go to `outputs/inspect_mae/<timestamp>_<split>/` and include `meta.json` + `grid_triplets.png`.

## Stage 3 — Train drifting generator on latents (Kaiming'spaper §5.2)

This trains a DiT‑B/2‑like generator that maps:

`z ~ N(0,I)  + class label c + CFG strength ω  →  x_latent ∈ R^{4×32×32}`.

Feature encoder choices:

- `--feature-encoder mae` (default): drifting loss in **latent‑MAE feature space** (Kaiming's Appendix A.5–A.7).
- `--feature-encoder moco`: drifting loss in **MoCo-v2 ResNet feature space**. This path decodes generated latents to RGB via SD‑VAE, applies ImageNet normalization, then extracts frozen MoCo features.

### Baseline (What I am doing now)

The baseline is the default behavior:

- `--drift-form alg2_joint`
- `--coupling partial_two_sided`
- `--alg2-impl logspace`
- `--mask-self-neg` enabled
- `--dist-metric l2` (‖x−y‖)

'logspace' to stable numerical

Example (4 GPUs):

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --latents-dir data/imagenet256_latents --split train \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt \
  --mae-use-ema \
  --amp \
  --run-name stage3_baseline
```

Debug smoke test (small sizes, 20 steps):

```bash
python -m imagenet.train_drifting \
  --latents-dir data/imagenet256_latents --split train \
  --mae-ckpt runs/imagenet_mae/<RUN>/checkpoints/ckpt_final.pt \
  --mae-use-ema --amp \
  --debug
```

Outputs:
- Run dir: `runs/imagenet_drift/<timestamp>_<run-name>_<id>/`
- Checkpoints: `.../checkpoints/ckpt_step_<K>.pt` and `ckpt_final.pt`
- Online sample grids (optional): `.../samples/step_<K>_omega_<W>_grid.png` + `..._meta.json`
- Logs: `.../logs.jsonl` (JSONL, one record per `--log-every` steps)
  - Precision fields are explicit: `precision_mode`, `amp_requested`, `amp_active`,
    `gen_forward_dtype`, `feature_forward_dtype`, `grad_scaler_enabled`, `tf32_effective`.
  - `drift_norm` monitoring is enabled by default (`--log-drift-stats`); disable with `--no-log-drift-stats`.
  - Loss interpretation fields: `feature_set_count`, `temps_count`, `loss_per_feature_set`,
    `loss_shape_norm` (normalizes by `temps_count^2`).
  - If online sampling is enabled, logs include `online_sample_grid` and `online_sample_meta`.

### Stage 3 with MoCo-v2 features (skip Stage 2 training)

Download official MoCo-v2 weights (ResNet-50, 800ep) from:
- https://dl.fbaipublicfiles.com/moco-v2/checkpoint_0199.pth.tar

Then run:

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.train_drifting \
  --feature-encoder moco \
  --imagenet-root /path/to/imagenet \
  --moco-ckpt /path/to/checkpoint_0199.pth.tar \
  --vae-id stabilityai/sd-vae-ft-ema \
  --vae-dtype fp32 \
  --amp \
  --save-every 1000 \
  --online-sample-every 1000 \
  --online-sample-count 16 \
  --online-sample-omega 1.0 \
  --run-name stage3_moco
```

Stability defaults in MoCo mode:
- `--moco-force-fp32` enabled (feature extraction in float32)
- `--moco-gen-fp32` enabled (generator forward in float32 in MoCo mode)
- `--moco-clamp-rgb` enabled (decoded RGB clipped to `[0,1]` before normalization)
- `--fail-on-nan` enabled and mandatory (fail fast on non-finite decode/features/loss/grad/params)

### Follow-up knobs (for ablations)

These are **not** the original Drift paper baseline, but are used for follow-up experiments:

- `--drift-form split` (cross-minus-self form `V=Pxy@y_pos - Pxneg@y_neg`)
- `--coupling row` (row-softmax coupling)
- `--coupling sinkhorn --sinkhorn-iters 20 --sinkhorn-marginal weighted_cols`
- `--dist-metric l2_sq` (‖x−y‖² ablation)

Important constraint:
- If you run `--coupling sinkhorn` with `--sinkhorn-marginal none` and `omega<=1` is possible, the CFG weight can be non-positive (`w<=0`) and Sinkhorn marginals become infeasible.
  Use `--sinkhorn-marginal weighted_cols` (recommended) or set `--omega-min > 1`.

### Common gotchas

- `--nc` is **global** and must be divisible by `world_size` (DDP). Default `--nc 64` works for 1/2/4/8 GPUs.
- If you use `--max-items` for debugging, you may hit:
  “Not enough classes with >= Npos samples …”.
  Fix: increase `--max-items`, or reduce `--nc/--npos`.
- Progress bar: only rank0 shows tqdm. Even if the terminal looks quiet, `logs.jsonl` should update.

## Monitoring progress (steps / speed / ETA)

You can always inspect the latest record:

```bash
RUN=runs/imagenet_drift/<RUN_DIR>
tail -n 1 "$RUN/logs.jsonl"
```

For a rolling “step + avg speed + ETA” (Ctrl+C to stop):

```bash
RUN=runs/imagenet_drift/<RUN_DIR> python - <<'PY'
import os, json, time, datetime, re
from pathlib import Path

run = Path(os.environ["RUN"])
cfg = json.load(open(run / "config.json", "r", encoding="utf-8"))
steps_total = int(cfg["config"]["args"]["steps"])
log_path = run / "logs.jsonl"

m = re.match(r"(\\d{8}_\\d{6})_", run.name)
start = datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp() if m else (run / "config.json").stat().st_mtime

prev_step, prev_t = None, None

def last_nonempty_line(path: Path, max_bytes: int = 1 << 16) -> bytes:
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        n = min(size, max_bytes)
        f.seek(-n, 2)
        data = f.read(n)
    for ln in reversed(data.splitlines()):
        if ln.strip():
            return ln
    raise RuntimeError("log file empty")

while True:
    last = json.loads(last_nonempty_line(log_path))
    step = int(last.get("step", 0))
    now = time.time()
    elapsed = max(now - start, 1e-9)
    avg_sps = step / elapsed
    inst_sps = float("nan") if prev_step is None else (step - prev_step) / max(now - prev_t, 1e-9)
    eta_h = (steps_total - step) / max(avg_sps, 1e-9) / 3600
    pct = 100.0 * step / max(steps_total, 1)
    print(f\"{time.strftime('%F %T')}  {step}/{steps_total} ({pct:5.1f}%)  avg {avg_sps:6.3f}  inst {inst_sps:6.3f}  ETA {eta_h:6.2f} h\", flush=True)
    prev_step, prev_t = step, now
    time.sleep(30)
PY
```

## Evaluation — generate images + compute FID

FID follows the paper protocol: generate latents → decode to RGB (SD‑VAE) → compute FID against ImageNet val.

Example (single GPU, omega sweep):

```bash
python -m imagenet.eval_fid \
  --ckpt runs/imagenet_drift/<RUN>/checkpoints/ckpt_step_30000.pt \
  --use-ema \
  --num-gen 50000 --batch-size 64 \
  --omega-min 1.0 --omega-max 4.0 --omega-num 7 \
  --real-dir /path/to/imagenet/val
```

Multi-GPU generation (recommended for 50k samples):

```bash
torchrun --standalone --nproc_per_node=4 -m imagenet.eval_fid \
  --ckpt runs/imagenet_drift/<RUN>/checkpoints/ckpt_step_30000.pt \
  --use-ema \
  --num-gen 50000 --batch-size 64 \
  --omegas 1.0,1.5,2.0,2.5,3.0,3.5,4.0 \
  --real-dir /path/to/imagenet/val
```

Outputs (under `.../eval_fid/` by default):
- `gen_omega_<...>/` folders with PNGs + `fid.json`
- `fid_sweep.json` and `fid_sweep.csv` summary

## Paper alignment （To exactly reproduce）

Key mapping:
- Stage 1: `imagenet/encode_latents.py`
- Stage 2 (Kaiming's Appendix A.3): `imagenet/train_mae.py`, `imagenet/models/resnet_mae.py`
- Stage 3 (Kaiming's paper §5.2 + Kaiming's Appendix A.2/A.5–A.8): `imagenet/train_drifting.py`, `imagenet/drifting_loss.py`
