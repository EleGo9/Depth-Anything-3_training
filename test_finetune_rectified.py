"""
Quick inference check for a fine-tuned DA3 monocular depth checkpoint (see
src/depth_anything_3/train/train.py): loads the base DA3 architecture, overwrites its weights
with a fine-tuned latest.pth/best.pth, runs depth inference on one image, and exports the
resulting point cloud as a .glb you can open in a viewer (e.g. Blender, or
https://gltf-viewer.donmccurdy.com/) to eyeball the reconstruction.

DA3MONO is depth-only: its raw forward output is just {depth, sky, aux} -- no camera pose or
intrinsics estimation branch (unlike the multi-view DA3 presets), so export_to_glb (which always
needs a K/W2C to unproject depth -> 3D) has nothing to work with by default. If you have real
calibration for the input image, pass it via --intrinsics/--extrinsics; otherwise a reasonable
default is assumed (identity extrinsics, fx=fy=max(H,W) intrinsics) -- good enough to look at the
reconstructed shape, not a metrically-correct point cloud.

export_to_glb itself always unprojects with a plain pinhole model (K^-1 @ [u,v,1] * depth), which
is wrong for a fisheye source image -- it warps the reconstruction increasingly towards the
edges/corners where fisheye distortion is strongest. If --input actually came from a fisheye lens,
pass --fisheye-calib (a YAML with width/height/K/D, e.g. as dumped by a Kalibr-style calibration
tool) and the RGB image is undistorted onto a virtual pinhole camera via OpenCV's cv2.fisheye model
*before* inference -- so the model itself sees rectified content (fisheye distortion is generally
out-of-distribution for it) and the predicted depth is already consistent with a pinhole camera,
with no need to warp the depth map after the fact.

Usage:
    python test_finetune.py --input path/to/image.jpg --checkpoint workspace/finetune_mono/best.pth

    # Also export the original (non-fine-tuned) pretrained model's GLB for comparison, into
    # <output>/pretrained/scene.glb and <output>/finetuned/scene.glb:
    python test_finetune.py --input path/to/image.jpg --checkpoint workspace/finetune_mono/best.pth \
        --compare-pretrained

    # Input image came from a fisheye lens: undistort before building the point cloud.
    python test_finetune.py --input path/to/fisheye.jpg --checkpoint workspace/finetune_mono/best.pth \
        --fisheye-calib path/to/cam_surround_b.yaml
"""

import argparse
import os

import cv2
import numpy as np
import torch
import yaml

from depth_anything_3.api import DepthAnything3
from depth_anything_3.train.inspect_model import print_model_summary
from depth_anything_3.utils.export.glb import export_to_glb

parser = argparse.ArgumentParser(
    description="Run a fine-tuned DA3 monocular depth checkpoint on one image and export a GLB point cloud"
)
parser.add_argument("--input", type=str, required=True, help="Path to the input image")
parser.add_argument("--checkpoint", type=str, required=True,
                     help="Path to a fine-tuned latest.pth/best.pth from train.py")
parser.add_argument("--pretrained", type=str, default="depth-anything/DA3MONO-LARGE",
                     help="Base DA3 preset the checkpoint was fine-tuned from (must match its architecture)")
parser.add_argument("--output", type=str, default="output", help="Directory to write scene.glb into")
parser.add_argument("--compare-pretrained", action="store_true",
                     help="Also run the original (non-fine-tuned) --pretrained checkpoint on the same image, "
                          "for a side-by-side comparison. Writes <output>/pretrained/scene.glb and "
                          "<output>/finetuned/scene.glb instead of <output>/scene.glb.")
parser.add_argument("--intrinsics", type=str, default=None,
                     help="Optional path to a 3x3 camera intrinsics matrix (.txt or .npy). Default: "
                          "fx=fy=max(H,W), principal point centered. Ignored if --fisheye-calib is set "
                          "(the undistorted virtual pinhole intrinsics are used instead).")
parser.add_argument("--extrinsics", type=str, default=None,
                     help="Optional path to a 4x4 world-to-camera matrix (.txt or .npy). Default: identity "
                          "(camera at the world origin).")
parser.add_argument("--fisheye-calib", type=str, default=None,
                     help="Path to a fisheye calibration YAML for the REAL camera that captured --input "
                          "(top-level keys: width, height, K.value0..8, D (>=4 coeffs), fisheye: true). When "
                          "set, the RGB image is undistorted onto a virtual pinhole camera (cv2.fisheye) before "
                          "inference -- required for a geometrically correct cloud from a fisheye source image.")
parser.add_argument("--conf-thresh-percentile", type=float, default=0.001,
                     help="[GLB] Lower percentile for the adaptive confidence threshold (lower = keep more points)")
parser.add_argument("--num-max-points", type=int, default=1_000_000, help="[GLB] Max points in the point cloud")


def _load_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path) if path.endswith(".txt") else np.load(path)


def _load_fisheye_calib(path: str) -> tuple[np.ndarray, np.ndarray, int, int]:
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


def _undistort_fisheye_image(
    image: np.ndarray, K: np.ndarray, D: np.ndarray, orig_w: int, orig_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remaps a fisheye-distorted RGB image onto a virtual pinhole camera via OpenCV's cv2.fisheye
    model, so it can be handed to inference already rectified. K/D are the real calibration at
    orig_w x orig_h; image may be at a different resolution (e.g. if it was saved/loaded at a
    different size than it was calibrated at) -- K is scaled proportionally to match (D is
    resolution-independent for the fisheye model, no rescaling needed there). Returns the
    undistorted image plus the virtual pinhole K it was rectified onto, both at image's resolution.
    """
    h, w = image.shape[:2]
    K_scaled = K.copy()
    K_scaled[0, :] *= w / orig_w
    K_scaled[1, :] *= h / orig_h

    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_scaled, D, (w, h), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_scaled, D, np.eye(3), K_new, (w, h), cv2.CV_32FC1)

    image_u = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return image_u, K_new


def run_and_export(api_model, args, output_dir: str, image, fisheye_K_new: np.ndarray | None):
    """Runs inference on `image` with api_model's currently-loaded weights and exports a GLB.

    `image` is either args.input (a path, when no fisheye rectification is needed) or an
    already-undistorted RGB array. `fisheye_K_new` is the virtual pinhole K that array was
    rectified onto (at its resolution before inference resizes it), or None if not rectifying.
    """
    print(f"Running inference on {args.input}...")
    prediction = api_model.inference(
        [image],
        export_dir=None,  # export manually below, once intrinsics/extrinsics/conf are filled in
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
    )

    if fisheye_K_new is not None:
        # Image was already rectified before inference; just rescale K to the (possibly resized)
        # processed resolution -- depth/processed_images need no further warping.
        h, w = prediction.depth.shape[-2:]
        orig_h, orig_w = image.shape[:2]
        K_scaled = fisheye_K_new.copy()
        K_scaled[0, :] *= w / orig_w
        K_scaled[1, :] *= h / orig_h
        prediction.intrinsics = K_scaled[None].astype(np.float32)
        print(f"Rectified with fisheye calib {args.fisheye_calib} -> virtual pinhole K:\n{K_scaled}")
    elif args.intrinsics:
        prediction.intrinsics = _load_matrix(args.intrinsics).astype(np.float32)[None]
    elif prediction.intrinsics is None:
        h, w = prediction.depth.shape[-2:]
        focal = max(h, w)
        prediction.intrinsics = np.array(
            [[[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]], dtype=np.float32
        )

    if args.extrinsics:
        prediction.extrinsics = _load_matrix(args.extrinsics).astype(np.float32)[None]
    elif prediction.extrinsics is None:
        prediction.extrinsics = np.eye(4, dtype=np.float32)[None]

    if prediction.conf is None:
        prediction.conf = np.ones_like(prediction.depth)
    print(args.input.split('/')[-1].split('.')[0])

    export_to_glb(
        prediction=prediction,
        export_dir=output_dir,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
        scene_name=args.input.split('/')[-1].split('.')[0]
    )

    print(f"GLB exported to {output_dir}/scene.glb")
    print(f"depth: {prediction.depth.shape}  range [{prediction.depth.min():.3f}, {prediction.depth.max():.3f}]")


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.fisheye_calib:
        K, D, orig_w, orig_h = _load_fisheye_calib(args.fisheye_calib)
        raw_image = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
        image, fisheye_K_new = _undistort_fisheye_image(raw_image, K, D, orig_w, orig_h)
        print(f"Rectified {args.input} with fisheye calib {args.fisheye_calib} before inference")
    else:
        image, fisheye_K_new = args.input, None

    print(f"Loading base architecture {args.pretrained}...")
    api_model = DepthAnything3.from_pretrained(args.pretrained)
    api_model.to(device)

    if args.compare_pretrained:
        print("\n=== Original pretrained model ===")
        run_and_export(api_model, args, os.path.join(args.output, "pretrained"), image, fisheye_K_new)

    print(f"\nLoading fine-tuned weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    api_model.model.load_state_dict(ckpt["model"])
    api_model.model.eval()
    api_model.to(device)

    print_model_summary(api_model.model)

    print("\n=== Fine-tuned model ===" if args.compare_pretrained else "")
    run_and_export(
        api_model, args, os.path.join(args.output, "finetuned") if args.compare_pretrained else args.output,
        image, fisheye_K_new,
    )


if __name__ == "__main__":
    main()
