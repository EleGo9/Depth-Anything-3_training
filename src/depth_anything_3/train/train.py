"""
Fine-tune DA3's monocular depth head (DPT) on image + GT-depth pairs, DINOv2 backbone frozen
by default. Structured after Depth Anything V2's metric_depth train.py; see
src/depth_anything_3/train/README.md for the preprocessing/loss/schedule choices and how they
map onto (or deliberately diverge from) that reference.

All settings live in train/configs/default.yaml. Point --config at your own copy, and/or
override individual values inline via dotlist notation:

    torchrun --nproc_per_node=<N> -m depth_anything_3.train.train \
        --config path/to/my_run.yaml \
        data.train_filelist=path/to/train.txt data.val_filelist=path/to/val.txt \
        train.save_path=workspace/finetune_mono
"""

from __future__ import annotations

import logging
import os
import pprint
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import load_config, to_dict_recursive
from depth_anything_3.train.dataset import MonocularDepthDataset
from depth_anything_3.train.dist_helper import setup_distributed
from depth_anything_3.train.losses import DepthLoss
from depth_anything_3.train.metrics import eval_depth
from depth_anything_3.train.transform import IMAGENET_MEAN, IMAGENET_STD
from depth_anything_3.train.utils import init_log
from depth_anything_3.utils.visualize import visualize_depth
import cv2

_default_config = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")


def _log_val_sample_to_wandb(img: torch.Tensor, gt_depth: torch.Tensor, pred_depth: torch.Tensor,
                              mask: torch.Tensor, step: int) -> None:
    """Logs one RGB | GT depth | predicted depth panel (same color scale for GT/pred) to wandb."""
    img_np = img.detach().float().cpu().permute(1, 2, 0).numpy()
    img_np = ((img_np * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1) * 255).astype(np.uint8)

    mask_np = mask.detach().cpu().numpy()
    gt_np = gt_depth.detach().float().cpu().numpy().copy()
    pred_np = pred_depth.detach().float().cpu().numpy().copy()
    gt_np[~mask_np] = 0
    pred_np[~mask_np] = 0

    # In val mode, transform.Resize only resizes the image (resize_target=False), so depth/pred
    # stay at the GT's native resolution while img is still at the (smaller) model input
    # resolution -- resize it up to match before building the side-by-side panel.
    h, w = gt_np.shape[-2:]
    if img_np.shape[:2] != (h, w):
        img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)
    gt_np[~mask_np] = 0
    pred_np[~mask_np] = 0

    gt_vis, dmin, dmax = visualize_depth(gt_np, ret_minmax=True)
    pred_vis = visualize_depth(pred_np, depth_min=dmin, depth_max=dmax)

    panel = np.concatenate([img_np, gt_vis, pred_vis], axis=1)
    wandb.log({"val/sample": wandb.Image(panel, caption="RGB | GT depth | Pred depth")}, step=step)


def parse_args():
    argv = sys.argv[1:]

    config_path = _default_config
    if "--config" in argv:
        config_idx = argv.index("--config")
        config_path = argv[config_idx + 1]
        argv = argv[:config_idx] + argv[config_idx + 2 :]

    return load_config(config_path, argv=argv)


def freeze_backbone(net, unfreeze_last_n_blocks: int = 0, fine_tune_layers: str | None = None) -> None:
    """
    Freezes net.backbone entirely, then optionally unfreezes specific DINOv2 blocks
    (net.backbone.pretrained.blocks[i]). `fine_tune_layers`, a comma-separated list of explicit
    block indices (e.g. "20,21,22,23"), takes precedence over `unfreeze_last_n_blocks` (a
    contiguous-tail shorthand) when both are given. net.head is left untouched (always
    trainable) -- freezing/unfreezing it is the caller's responsibility.
    """
    for p in net.backbone.parameters():
        p.requires_grad = False

    blocks = net.backbone.pretrained.blocks
    if fine_tune_layers:
        indices = [int(x.strip()) for x in fine_tune_layers.split(",") if x.strip()]
    elif unfreeze_last_n_blocks > 0:
        indices = list(range(len(blocks) - unfreeze_last_n_blocks, len(blocks)))
    else:
        indices = []

    for idx in indices:
        for p in blocks[idx].parameters():
            p.requires_grad = True


def build_param_groups(model, lr: float, head_lr_mult: float = 10.0):
    """
    Two groups by name, exactly like the DA2 reference (util-free inline version): backbone
    params at `lr`, everything else (the depth head) at `lr * head_lr_mult`. Frozen backbone
    params are still included (with requires_grad=False) rather than filtered out -- AdamW
    simply skips params with no gradient during `.step()`, and this keeps `optimizer.param_groups`
    a fixed [backbone, head] pair regardless of how many blocks are unfrozen, so the per-iteration
    LR update below can index into it directly instead of matching by role.
    """
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    return [
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_mult},
    ]


def main():
    config = parse_args()

    logger = init_log("global", logging.INFO)
    logger.propagate = False

    rank, world_size = setup_distributed(port=config.train.port)
    local_rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        os.makedirs(config.train.save_path, exist_ok=True)
        all_args = {**to_dict_recursive(config), "ngpus": world_size}
        logger.info("\n%s", pprint.pformat(all_args))
        writer = SummaryWriter(config.train.save_path)
        if config.wandb.enabled:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.run_name,
                config=all_args,
            )

    cudnn.enabled = True
    cudnn.benchmark = True

    size = config.data.img_size
    trainset = MonocularDepthDataset(
        config.data.train_filelist, "train", size=size,
        min_depth=config.data.min_depth, max_depth=config.data.max_depth, depth_scale=config.data.depth_scale,
    )
    trainsampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=config.train.bs, pin_memory=True, num_workers=4,
                              drop_last=True, sampler=trainsampler)

    valset = MonocularDepthDataset(
        config.data.val_filelist, "val", size=size,
        min_depth=config.data.min_depth, max_depth=config.data.max_depth, depth_scale=config.data.depth_scale,
    )
    valsampler = DistributedSampler(valset, shuffle=False)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                            drop_last=False, sampler=valsampler)

    api_model = DepthAnything3.from_pretrained(config.model.pretrained)
    net = api_model.model.float()  # fp32 master weights; amp_bf16 only affects the forward autocast
    freeze_backbone(net, config.train.unfreeze_last_n_blocks, config.train.fine_tune_layers)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # no-op for DA3 (no BatchNorm layers), kept for parity

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    net.to(device)
    ddp_kwargs = dict(find_unused_parameters=True, broadcast_buffers=False)
    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank, **ddp_kwargs
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(net, **ddp_kwargs)

    criterion = DepthLoss(
        grad_weight=config.train.grad_weight,
        gl_weight=config.train.gl_weight,
        gl_grid_size=config.train.gl_grid_size,
    ).to(device)

    head_lr_mult = 10.0
    param_groups = build_param_groups(model, config.train.lr, head_lr_mult=head_lr_mult)
    optimizer = AdamW(param_groups, lr=config.train.lr, betas=(0.9, 0.999), weight_decay=0.01)

    start_epoch = 0
    previous_best = {"d1": 0, "d2": 0, "d3": 0, "abs_err": 20, "abs_rel": 100, "sq_rel": 100,
                      "rmse": 100, "rmse_log": 100, "log10": 100, "silog": 100}
    if config.train.resume:
        ckpt = torch.load(config.train.resume, map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        previous_best = ckpt.get("previous_best", previous_best)

    total_iters = config.train.epochs * len(trainloader)

    for epoch in range(start_epoch, config.train.epochs):
        if rank == 0:
            logger.info(
                "===========> Epoch: %d/%d, d1: %.3f, d2: %.3f, d3: %.3f",
                epoch, config.train.epochs, previous_best["d1"], previous_best["d2"], previous_best["d3"],
            )
            logger.info(
                "===========> Epoch: %d/%d, abs_rel: %.3f, abs_err: %.3f, sq_rel: %.3f, rmse: %.3f, "
                "rmse_log: %.3f, log10: %.3f, silog: %.3f",
                epoch, config.train.epochs, previous_best["abs_rel"], previous_best["abs_err"],
                previous_best["sq_rel"], previous_best["rmse"], previous_best["rmse_log"],
                previous_best["log10"], previous_best["silog"],
            )

        trainsampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img = sample["image"].to(device)
            depth = sample["depth"].to(device)
            valid_mask = sample["valid_mask"].to(device)

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.train.amp_bf16):
                output = model(img.unsqueeze(1))
                pred = output.depth[:, 0]

            mask = (valid_mask == 1) & (depth >= config.data.min_depth) & (depth <= config.data.max_depth)
            loss = criterion(pred.float(), depth, mask)
            if "sky" in output:
                # DPT's sky head always runs and its output is part of forward()'s return value,
                # so DDP (find_unused_parameters=True) traces it as "used" and expects a gradient
                # for it every iteration. We have no sky supervision, so without this zero-weighted
                # term the sky head's params never actually get a gradient and DDP's reducer never
                # completes its buckets -- it then errors on the *next* iteration's forward with
                # "Expected to have finished reduction in the prior iteration".
                loss = loss + 0.0 * output.sky.float().sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            iters = epoch * len(trainloader) + i
            lr = config.train.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * head_lr_mult

            if rank == 0:
                writer.add_scalar("train/loss", loss.item(), iters)
                if config.wandb.enabled:
                    wandb.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[1]["lr"]}, step=iters)
                if i % 100 == 0:
                    logger.info(
                        "Iter: %d/%d, LR: %.7f, Loss: %.3f",
                        i, len(trainloader), optimizer.param_groups[1]["lr"], loss.item(),
                    )

        model.eval()
        results = {k: torch.zeros(1, device=device) for k in previous_best.keys()}
        nsamples = torch.zeros(1, device=device)

        val_sample_logged = False
        with torch.no_grad():
            for sample in valloader:
                img = sample["image"].to(device).float()
                depth = sample["depth"].to(device)[0]
                valid_mask = sample["valid_mask"].to(device)[0]

                output = model(img.unsqueeze(1))
                pred = output.depth[:, 0]
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True)[0, 0]

                mask = (valid_mask == 1) & (depth >= config.data.min_depth) & (depth <= config.data.max_depth)
                if mask.sum() < 10:
                    continue

                cur_results = eval_depth(pred[mask], depth[mask])
                for k in results:
                    results[k] += cur_results[k]
                nsamples += 1

                if rank == 0 and config.wandb.enabled and not val_sample_logged:
                    _log_val_sample_to_wandb(img[0], depth, pred, mask, step=iters)
                    val_sample_logged = True

        dist.barrier()
        for k in results:
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            nsamples_val = max(nsamples.item(), 1.0)
            logger.info("=" * 90)
            logger.info(", ".join(f"{k:>8}" for k in results.keys()))
            logger.info(", ".join(f"{(v / nsamples_val).item():8.3f}" for v in results.values()))
            logger.info("=" * 90)
            for name, metric in results.items():
                writer.add_scalar(f"eval/{name}", (metric / nsamples_val).item(), epoch)
            if config.wandb.enabled:
                wandb.log(
                    {f"eval/{name}": (metric / nsamples_val).item() for name, metric in results.items()},
                    step=iters,
                )

        for k in results:
            val = (results[k] / max(nsamples.item(), 1.0)).item()
            if k in ("d1", "d2", "d3"):
                previous_best[k] = max(previous_best[k], val)
            else:
                previous_best[k] = min(previous_best[k], val)

        if rank == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(config.train.save_path, "latest.pth"))

    if rank == 0 and config.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
