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
import yaml
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


def _load_fisheye_calib(path: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Same calibration YAML schema as test_finetune.py's _load_fisheye_calib (width, height,
    K.value0..8, a >=4-coeff D, e.g. as dumped by a Kalibr-style calibration tool)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    k = data["K"]
    K = np.array(
        [[k["value0"], k["value1"], k["value2"]],
         [k["value3"], k["value4"], k["value5"]],
         [k["value6"], k["value7"], k["value8"]]],
        dtype=np.float64,
    )
    D = np.array(data["D"][:4], dtype=np.float64).reshape(4, 1)  # cv2.fisheye uses exactly 4 coeffs
    return K, D, int(data["width"]), int(data["height"])


class MonocularDepthDataset(Dataset):
    def __init__(
        self,
        filelist_path: str,
        mode: str,
        size: int = 504,
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        depth_scale: float = 1.0,
        rect: bool = False,
        calib_path: str | None = None,
    ):
        assert mode in ("train", "val")
        self.mode = mode
        self.size = size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.rect = rect

        with open(filelist_path, "r") as f:
            self.filelist = [line.strip() for line in f.read().splitlines() if line.strip()]

        # Fisheye rectification: image + depth are undistorted onto a virtual pinhole camera
        # (cv2.fisheye) before the Resize/Crop transform pipeline below, using the same
        # calibration YAML schema as test_finetune.py. One calib_path applies to the whole
        # dataset -- fine as long as every sample was captured by the same physical camera.
        self._undistort_maps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        if self.rect:
            assert calib_path, "data.calib_path must be set when data.rect=true"
            self._calib = _load_fisheye_calib(calib_path)

        self.transform = Compose(
            [
                Resize(size=size, resize_target=(mode == "train")),
                NormalizeImage(),
                PrepareForNet(),
            ]
            + ([Crop(size, mode="train")] if mode == "train" else [])
        )

    def _rectify(self, image: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Undistorts image (linear) + depth (nearest, to avoid smearing depth discontinuities)
        from self._calib's fisheye calibration onto a virtual pinhole camera. Pixels remapped
        from outside the original image become 0 in both, which reads as invalid once
        valid_mask is computed downstream -- no separate mask remapping needed."""
        K, D, orig_w, orig_h = self._calib
        h, w = image.shape[:2]
        cache_key = (w, h)
        if cache_key not in self._undistort_maps:
            K_scaled = K.copy()
            K_scaled[0, :] *= w / orig_w
            K_scaled[1, :] *= h / orig_h
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K_scaled, D, (w, h), np.eye(3), balance=0.0
            )
            self._undistort_maps[cache_key] = cv2.fisheye.initUndistortRectifyMap(
                K_scaled, D, np.eye(3), K_new, (w, h), cv2.CV_32FC1
            )
        map1, map2 = self._undistort_maps[cache_key]
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        depth = cv2.remap(depth, map1, map2, interpolation=cv2.INTER_NEAREST)
        return image, depth

    def __getitem__(self, item):
        img_path, depth_path = self.filelist[item].split(" ")[:2]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = _load_depth(depth_path) * self.depth_scale

        if self.rect:
            image, depth = self._rectify(image, depth)

        valid_mask = (depth > self.min_depth) & (depth <= self.max_depth) & np.isfinite(depth)

        sample = self.transform({"image": image, "depth": depth, "valid_mask": valid_mask})

        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = torch.from_numpy(sample["valid_mask"])
        sample["image_path"] = img_path
        if item % 100:
            img_vis = sample["image"].numpy().transpose(1, 2, 0)
            img_vis = ((img_vis * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1) * 255).astype(np.uint8)
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

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
