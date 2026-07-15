"""
Dict-based transforms for monocular depth fine-tuning, mirroring Depth Anything V2's
training transform pipeline (Resize -> NormalizeImage -> PrepareForNet -> Crop) but with
the resize geometry matched to Depth Anything 3's own inference preprocessing
(depth_anything_3.utils.io.input_processor.InputProcessor): longest-side resize preserving
aspect ratio, rounded to a multiple of PATCH_SIZE, instead of DA2's shortest-side + square
crop convention. See src/depth_anything_3/train/README.md for details.

Each transform operates on a dict sample:
    {"image": HWC float32 in [0,1], "depth": HW float32, "valid_mask": HW bool}
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from depth_anything_3.utils.io.input_processor import InputProcessor

IMAGENET_MEAN = InputProcessor.NORMALIZE.mean
IMAGENET_STD = InputProcessor.NORMALIZE.std
PATCH_SIZE = InputProcessor.PATCH_SIZE


def _nearest_multiple(x: int, p: int) -> int:
    down = (x // p) * p
    up = down + p
    return max(p, up if abs(up - x) <= abs(x - down) else down)


class Resize:
    """
    Resize the longest side of the image to `size`, preserving aspect ratio, then round both
    dimensions to the nearest multiple of `ensure_multiple_of`. This replicates
    InputProcessor._resize_longest_side + InputProcessor._make_divisible_by_resize exactly
    (same interpolation rule: CUBIC when upscaling, AREA when downscaling), which is what the
    pretrained DA3 backbone/head were run with at inference time.

    Depth (and valid_mask) are resized with INTER_NEAREST regardless of the image's
    interpolation choice, to avoid smearing depth discontinuities across object boundaries.
    """

    def __init__(self, size: int = 504, resize_target: bool = True, ensure_multiple_of: int = PATCH_SIZE):
        self.size = size
        self.resize_target = resize_target
        self.ensure_multiple_of = ensure_multiple_of

    def _target_hw(self, h: int, w: int) -> tuple[int, int]:
        longest = max(h, w)
        scale = self.size / float(longest) if longest != self.size else 1.0
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        new_h = _nearest_multiple(new_h, self.ensure_multiple_of)
        new_w = _nearest_multiple(new_w, self.ensure_multiple_of)
        return new_h, new_w

    def __call__(self, sample: dict) -> dict:
        h, w = sample["image"].shape[:2]
        new_h, new_w = self._target_hw(h, w)
        if (new_h, new_w) != (h, w):
            scale = self.size / float(max(h, w))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            sample["image"] = cv2.resize(sample["image"], (new_w, new_h), interpolation=interp)
            if self.resize_target and "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
                sample["valid_mask"] = cv2.resize(
                    sample["valid_mask"].astype(np.uint8),
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        return sample


class NormalizeImage:
    """ImageNet normalization, reusing the exact mean/std DA3's InputProcessor uses at inference."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, sample: dict) -> dict:
        sample["image"] = (sample["image"] - self.mean) / self.std
        return sample


class PrepareForNet:
    """HWC -> CHW, contiguous float32."""

    def __call__(self, sample: dict) -> dict:
        sample["image"] = np.ascontiguousarray(sample["image"].transpose(2, 0, 1)).astype(np.float32)
        if "depth" in sample:
            sample["depth"] = sample["depth"].astype(np.float32)
        return sample


class Crop:
    """
    Crop to a fixed (size, size) shape divisible by PATCH_SIZE, needed only so samples within a
    training batch share a common tensor shape (DA3's own crop-free inference pipeline handles
    each image at its native aspect ratio). Random crop in train mode, center crop otherwise.
    """

    def __init__(self, size: int, mode: str = "train"):
        self.size = size
        self.mode = mode

    def __call__(self, sample: dict) -> dict:
        _, h, w = sample["image"].shape
        size = min(self.size, h, w)
        size = (size // PATCH_SIZE) * PATCH_SIZE
        size = max(size, PATCH_SIZE)
        if self.mode == "train":
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
        else:
            top = (h - size) // 2
            left = (w - size) // 2
        sample["image"] = sample["image"][:, top : top + size, left : left + size]
        if "depth" in sample:
            sample["depth"] = sample["depth"][top : top + size, left : left + size]
            sample["valid_mask"] = sample["valid_mask"][top : top + size, left : left + size]
        return sample
