# Depth-Anything-3_training

DA3 provides a great code for testing, this repository provides the training pipeline.

The model architecture and inference code (`src/depth_anything_3/{model,api.py,...}`) are
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)'s own, unmodified. What
this repo adds on top is `src/depth_anything_3/train/`: everything needed to fine-tune a
released DA3 monocular-depth checkpoint (e.g. `depth-anything/DA3MONO-LARGE`) on your own
image + depth data. This README only covers that training pipeline.

## What's in `train/`

- `dataset.py` -- filelist-based dataset (`image_path depth_path` pairs, one per line; images
  via `cv2.imread`, depth as `.npy`, `.exr`, or 16-bit PNG).
- `transform.py` -- Resize -> NormalizeImage -> PrepareForNet -> Crop pipeline, matched to DA3's
  own inference preprocessing (longest-side resize, not DA2's shortest-side + square crop).
- `losses.py` -- `SiLogLoss` (always on, ported from DA2) plus two optional, opt-in-by-weight
  regularizers: `GradientLoss` (multi-scale gradient matching) and `GlobalLocalLoss` (a
  scale-and-shift-invariant loss, global + per-patch, approximating the DA3 paper's `L_gl` term
  -- see the docstring in `losses.py` for what's exact vs. approximated).
- `train.py` -- the training script (single or multi-GPU via `torchrun`, backbone frozen by
  default with optional partial unfreezing).
- `configs/default.yaml` -- every setting above lives here (OmegaConf), no argparse flags.
- `README.md` -- the detailed reference: full data format spec, preprocessing internals,
  loss/schedule rationale, and an explicit list of how this diverges from the DA2 training
  recipe it's structured after.

## Quickstart

```bash
cd Depth-Anything-3_training
PYTHONPATH=src python -m depth_anything_3.train.train \
    data.train_filelist=path/to/train.txt \
    data.val_filelist=path/to/val.txt \
    train.save_path=workspace/finetune_mono
```

Multi-GPU: prefix with `torchrun --nproc_per_node=<N>`. Every setting can also be edited
directly in `configs/default.yaml`, pointed at via `--config path/to/your.yaml`, or overridden
inline with dotlist notation (`train.epochs=10 train.lr=1e-5`) -- all three combine, inline
overrides win. See [`src/depth_anything_3/train/README.md`](src/depth_anything_3/train/README.md)
for the full option list and data format.

## Logging

- **TensorBoard**: written to `train.save_path` automatically, no config needed.
- **Weights & Biases**: on by default (`wandb.enabled: true`); set `wandb.enabled=false` to turn
  it off. Logs `train/loss`, `train/lr` per iteration and the full `eval/*` metric set per epoch,
  plus one RGB | GT-depth | predicted-depth panel from validation each epoch for a quick visual
  sanity check. Project/run name/entity are set under `wandb:` in the config.
