"""
Monocular depth dataset for fine-tuning DA3's depth head, structured the same way as
Depth Anything V2's per-dataset classes (e.g. VKITTI2): a plain-text filelist of
"image_path depth_path" pairs, loaded with cv2, and a Resize -> NormalizeImage ->
PrepareForNet -> [Crop] transform pipeline.
"""

from __future__ import annotations

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from depth_anything_3.train.transform import IMAGENET_MEAN, IMAGENET_STD, Crop, NormalizeImage, PrepareForNet, Resize


def _load_depth(depth_path: str) -> np.ndarray:
    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth file: {depth_path}")
        depth = depth.astype(np.float32)
    return depth


class MonocularDepthDataset(Dataset):
    def __init__(
        self,
        filelist_path: str,
        mode: str,
        size: int = 504,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        depth_scale: float = 1.0,
    ):
        assert mode in ("train", "val")
        self.mode = mode
        self.size = size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale

        with open(filelist_path, "r") as f:
            self.filelist = [line.strip() for line in f.read().splitlines() if line.strip()]

        self.transform = Compose(
            [
                Resize(size=size, resize_target=(mode == "train")),
                NormalizeImage(),
                PrepareForNet(),
            ]
            + ([Crop(size, mode="train")] if mode == "train" else [])
        )

    def __getitem__(self, item):
        img_path, depth_path = self.filelist[item].split(" ")[:2]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = _load_depth(depth_path) * self.depth_scale
        valid_mask = (depth > self.min_depth) & (depth <= self.max_depth) & np.isfinite(depth)

        sample = self.transform({"image": image, "depth": depth, "valid_mask": valid_mask})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(sample["valid_mask"])
        sample["image_path"] = img_path
        # if item % 100:
        #     img_vis = sample["image"].numpy().transpose(1, 2, 0)
        #     img_vis = ((img_vis * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1) * 255).astype(np.uint8)
        #     img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

        #     depth_vis = sample["depth"].numpy().copy()
        #     valid = sample["valid_mask"].numpy()
        #     depth_vis[~valid] = 0
        #     if valid.any():
        #         dmin, dmax = depth_vis[valid].min(), depth_vis[valid].max()
        #         depth_vis = (depth_vis - dmin) / max(dmax - dmin, 1e-6)
        #     depth_vis = depth_vis.clip(0, 1)

        #     cv2.imshow('sample["image"]', img_vis)
        #     cv2.imshow('sample["depth"]', depth_vis)
        #     cv2.waitKey(1)

        return sample

    def __len__(self):
        return len(self.filelist)
