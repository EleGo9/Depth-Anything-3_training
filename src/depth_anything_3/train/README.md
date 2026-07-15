# Fine-tuning DA3's monocular depth head

Fine-tunes the depth head of a released DA3 monocular checkpoint (e.g.
`depth-anything/DA3MONO-LARGE`) on your own image + GT-depth pairs. The DINOv2 backbone is
frozen by default (optionally unfreeze the last N transformer blocks); only the DPT depth head
is trained. Structured after Depth Anything V2's `metric_depth` training code — see the "How
this differs from the DA2 reference" section below for the handful of deliberate deviations.

## Data format

A **filelist**: a plain text file with one `image_path depth_path` pair per line
(space-separated), e.g.:

```
/data/scene01/rgb/0001.jpg /data/scene01/depth/0001.png
/data/scene01/rgb/0002.jpg /data/scene01/depth/0002.png
...
```

- Images: anything `cv2.imread` supports.
- Depth: `.npy` (raw float array), `.exr` (EXR support is force-enabled at import time via
  `OPENCV_IO_ENABLE_OPENEXR`, since OpenCV disables it by default), or a 16-bit PNG (loaded via
  `cv2.IMREAD_ANYDEPTH`). Depth values are multiplied by `data.depth_scale` to convert to meters
  (e.g. `0.01` if your PNG stores centimeters, `0.001` for millimeters). Pixels with depth
  `<= 0`, `> data.max_depth`, or non-finite are treated as invalid and excluded from the
  loss/metrics.

You need two filelists, one for `data.train_filelist` and one for `data.val_filelist`. A quick
way to generate one from parallel image/depth directories with matching filenames:

```python
import glob, os
imgs = sorted(glob.glob("data/rgb/*.jpg"))
with open("train.txt", "w") as f:
    for img in imgs:
        depth = img.replace("rgb", "depth").replace(".jpg", ".png")
        if os.path.exists(depth):
            f.write(f"{img} {depth}\n")
```

## Launching

**Environment**: the `da3` conda env's editable install of `depth_anything_3` currently points
at `/home/elena/repos/Depth-Anything-3` (the other, non-`_training` repo), not this one. Since
the DA3 model code itself is the same between the two repos, that env works fine for training —
but it doesn't know about this repo's new `depth_anything_3.train` subpackage, so point
`PYTHONPATH` at *this* repo's `src/` when launching (as done below), or `pip install -e .` from
this repo's root into that env if you'd rather not repeat `PYTHONPATH` every time.

**Config**: all settings live in [`train/configs/default.yaml`](configs/default.yaml) (an
OmegaConf YAML, same mechanism used elsewhere in this repo — see `depth_anything_3.cfg`). Either
edit a copy of it and pass `--config path/to/your.yaml`, or override individual keys inline via
dotlist notation (`section.key=value`) on the command line — both can be combined, and inline
overrides win. `data.train_filelist`, `data.val_filelist`, and `train.save_path` have no default
(`???` in the YAML) and must be set one way or the other.

Single node, multiple GPUs (via `torchrun`):

```bash
cd /home/elena/repos/Depth-Anything-3_training
PYTHONPATH=src torchrun --nproc_per_node=4 -m depth_anything_3.train.train \
    data.train_filelist=path/to/train.txt \
    data.val_filelist=path/to/val.txt \
    train.save_path=workspace/finetune_mono \
    train.epochs=20 train.bs=4 train.lr=5e-5
```

Single GPU / CPU debugging (no `torchrun` needed, falls back to a single process):

```bash
cd /home/elena/repos/Depth-Anything-3_training
PYTHONPATH=src python -m depth_anything_3.train.train \
    data.train_filelist=path/to/train.txt data.val_filelist=path/to/val.txt \
    train.save_path=workspace/debug_run train.epochs=1 train.bs=1
```

Or with your own config file:

```bash
PYTHONPATH=src python -m depth_anything_3.train.train --config path/to/your.yaml
```

`model.pretrained` accepts an HF Hub repo id or a local directory containing `config.json` +
`model.safetensors` (same convention as `DepthAnything3.from_pretrained` used everywhere else
in this repo).

Checkpoints (`latest.pth`: unwrapped model state dict + optimizer + epoch + best metrics so
far) are written to `train.save_path` every epoch by rank 0. Resume with
`train.resume=workspace/finetune_mono/latest.pth`.

## Preprocessing

Images are resized with the **same algorithm DA3 uses at inference**
(`InputProcessor._resize_longest_side` + `_make_divisible_by_resize`): longest side scaled to
`data.img_size` (default 504) preserving aspect ratio, cubic/area interpolation depending on
up/downscale, then rounded to a multiple of 14. This is *not* the same as Depth Anything V2's
own shortest-side-resize-then-square-crop convention — DA3's own preprocessing doesn't force a
square image. Because batched training still needs a fixed tensor shape, a crop to
`(img_size, img_size)` is applied afterwards (random crop for train, none for val — val uses
batch size 1 so it can keep the native aspect ratio). Depth maps are resized with
`cv2.INTER_NEAREST` regardless of what the image used, to avoid smearing depth discontinuities.
Normalization uses the exact same ImageNet mean/std as DA3's `InputProcessor` (imported, not
duplicated).

## Loss / metrics / schedule

- Loss: `SiLogLoss` (scale-invariant log loss), same as DA2's `metric_depth` training. An
  optional multi-scale gradient-matching term is available via `train.grad_weight` but is `0`
  (off) by default to match the reference recipe.
- Validation metrics: `d1, d2, d3, abs_rel, sq_rel, rmse, rmse_log, log10, silog` — same set DA2
  reports.
- Optimizer: `AdamW`, two param groups (unfrozen-backbone params at `train.lr`, head params at
  `train.lr * 10`), poly LR decay `lr * (1 - iter/total_iters)**0.9`, matching DA2's schedule.
- Backbone unfreezing: `train.fine_tune_layers="20,21,22,23"` unfreezes those exact DINOv2 block
  indices (0-indexed; `da3-large`'s backbone has 24 blocks, indices 0-23) in addition to the
  always-trainable head. `train.unfreeze_last_n_blocks=N` is a contiguous-tail shorthand for the
  same thing, used only when `train.fine_tune_layers` isn't set. Default: fully frozen backbone.
- Precision: **fp32 by default** (no autocast), matching the DA2 reference. Set
  `train.amp_bf16=true` to enable bf16 autocast on the forward pass for lower memory use (master
  weights stay fp32).

## How this differs from the DA2 reference script you'd recognize

- **Checkpoint loading**: DA2's script loads only backbone (`'pretrained' in name`) weights from
  a raw `.pth`, to bootstrap a fresh head from an ImageNet-pretrained backbone. Here we load the
  **full** released DA3 checkpoint (backbone *and* head) via `DepthAnything3.from_pretrained`,
  since the goal is fine-tuning DA3's own pretrained depth head, not training a new one.
- **The DPT sky head needs a zero-weighted dummy loss term under DDP** (found by actually
  running this against the real `DA3MONO-LARGE` checkpoint under DDP, not just reasoned about):
  DA3's DPT head always produces an auxiliary `sky` output (`use_sky_head=True`), and this
  training recipe has no sky supervision. `find_unused_parameters=True` is necessary but *not
  sufficient* here — DDP traces "used" params from the tensors `forward()` *returns* (the whole
  `output` dict, which includes `sky`), so it sees the sky head as used and expects a gradient
  for it every iteration; since our loss never actually touches `output.sky`, that expectation
  is never met, backward leaves the reducer's buckets incomplete, and the *next* iteration's
  forward crashes with "Expected to have finished reduction in the prior iteration" (confirmed:
  the real repro shows iteration 0 succeeding and iteration 1 crashing, naming the sky head's 4
  conv params as the ones that never received a gradient). `train.py` fixes this by adding
  `loss = loss + 0.0 * output.sky.sum()` whenever `sky` is present, so every parameter that
  contributed to the module's returned outputs actually gets a (zero) gradient every iteration.
- **Known interaction to be aware of**: `DepthAnything3Net.forward` unconditionally
  post-processes its own `depth` output using its own `sky` prediction (clipping "sky" pixels,
  per the model's *own* classification, to a constant depth) whenever a `sky` head is present —
  this runs during training too, not just inference. For datasets where GT depth is already
  invalid/missing over sky (most driving/outdoor depth datasets, e.g. KITTI-style LiDAR with no
  return in the sky), this has no effect since those pixels are already excluded from the loss.
  If your dataset has valid GT depth in regions the (initially imperfectly calibrated) sky head
  misclassifies as sky, those pixels' supervision will be corrupted by this constant-value
  overwrite — worth spot-checking training predictions/losses early on if your data isn't a
  standard driving/outdoor sky-free-GT dataset.
- **Resize geometry**: matches DA3's own aspect-preserving longest-side resize instead of DA2's
  shortest-side + square-crop convention (see "Preprocessing" above).
