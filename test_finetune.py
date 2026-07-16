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

Usage:
    python test_finetune.py --input path/to/image.jpg --checkpoint workspace/finetune_mono/best.pth

    # Also export the original (non-fine-tuned) pretrained model's GLB for comparison, into
    # <output>/pretrained/scene.glb and <output>/finetuned/scene.glb:
    python test_finetune.py --input path/to/image.jpg --checkpoint workspace/finetune_mono/best.pth \
        --compare-pretrained
"""

import argparse
import os

import numpy as np
import torch

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
                          "fx=fy=max(H,W), principal point centered.")
parser.add_argument("--extrinsics", type=str, default=None,
                     help="Optional path to a 4x4 world-to-camera matrix (.txt or .npy). Default: identity "
                          "(camera at the world origin).")
parser.add_argument("--conf-thresh-percentile", type=float, default=0.001,
                     help="[GLB] Lower percentile for the adaptive confidence threshold (lower = keep more points)")
parser.add_argument("--num-max-points", type=int, default=1_000_000, help="[GLB] Max points in the point cloud")


def _load_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path) if path.endswith(".txt") else np.load(path)


def run_and_export(api_model, args, output_dir: str):
    """Runs inference on args.input with api_model's currently-loaded weights and exports a GLB."""
    print(f"Running inference on {args.input}...")
    prediction = api_model.inference(
        [args.input],
        export_dir=None,  # export manually below, once intrinsics/extrinsics/conf are filled in
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
    )

    h, w = prediction.depth.shape[-2:]
    if args.intrinsics:
        prediction.intrinsics = _load_matrix(args.intrinsics).astype(np.float32)[None]
    elif prediction.intrinsics is None:
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

    export_to_glb(
        prediction=prediction,
        export_dir=output_dir,
        conf_thresh_percentile=args.conf_thresh_percentile,
        num_max_points=args.num_max_points,
    )

    print(f"GLB exported to {output_dir}/scene.glb")
    print(f"depth: {prediction.depth.shape}  range [{prediction.depth.min():.3f}, {prediction.depth.max():.3f}]")


def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading base architecture {args.pretrained}...")
    api_model = DepthAnything3.from_pretrained(args.pretrained)
    api_model.to(device)

    if args.compare_pretrained:
        print("\n=== Original pretrained model ===")
        run_and_export(api_model, args, os.path.join(args.output, "pretrained"))

    print(f"\nLoading fine-tuned weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    api_model.model.load_state_dict(ckpt["model"])
    api_model.model.eval()
    api_model.to(device)

    print_model_summary(api_model.model)

    print("\n=== Fine-tuned model ===" if args.compare_pretrained else "")
    run_and_export(
        api_model, args, os.path.join(args.output, "finetuned") if args.compare_pretrained else args.output
    )


if __name__ == "__main__":
    main()
