"""
Print a structural summary of a DA3 model: top-level modules, DINOv2 backbone blocks, and DPT
head submodules, each with parameter counts and trainable/frozen status. Handy for deciding
`train.unfreeze_last_n_blocks` / `train.fine_tune_layers` in configs/default.yaml before
launching a real run -- it applies the exact same freeze_backbone() used by train.py, so the
trainable/frozen status shown here matches what a real training run would do.

Usage:
    PYTHONPATH=src python -m depth_anything_3.train.inspect_model
    PYTHONPATH=src python -m depth_anything_3.train.inspect_model \
        --pretrained depth-anything/DA3MONO-LARGE --unfreeze-last-n-blocks 4
"""

from __future__ import annotations

import argparse

from depth_anything_3.api import DepthAnything3
from depth_anything_3.train.train import freeze_backbone

parser = argparse.ArgumentParser(description="Print a layer/parameter summary of a DA3 model")
parser.add_argument("--pretrained", type=str, default="depth-anything/DA3MONO-LARGE")
parser.add_argument("--unfreeze-last-n-blocks", type=int, default=0)
parser.add_argument("--fine-tune-layers", type=str, default=None)


def _fmt(n: int) -> str:
    return f"{n:,}"


def _status(module) -> str:
    n_params = sum(p.numel() for p in module.parameters())
    n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if n_params == 0:
        return "--"
    if n_trainable == n_params:
        return "trainable"
    if n_trainable == 0:
        return "frozen"
    return "mixed"


def _line(name: str, module, indent: int = 0) -> str:
    n_params = sum(p.numel() for p in module.parameters())
    return (
        f"{'  ' * indent}{name:<24s} {type(module).__name__:<20s} "
        f"params={_fmt(n_params):>14s}  ({_status(module)})"
    )


def print_model_summary(net) -> None:
    """Prints a structural summary of a DA3 net (the model wrapped inside DepthAnything3.model)."""
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total else 0.0

    print("=" * 88)
    print(f"Model summary -- total params: {_fmt(total)}  |  trainable: {_fmt(trainable)} ({pct:.2f}%)")
    print("=" * 88)

    for top_name, top_module in net.named_children():
        print(f"\n[{top_name}] ({type(top_module).__name__})")

        pretrained = getattr(top_module, "pretrained", None)
        blocks = getattr(pretrained, "blocks", None) if pretrained is not None else None
        if blocks is not None:
            # DINOv2-style backbone: patch_embed -> blocks[i] -> norm
            for name, module in pretrained.named_children():
                if name == "blocks":
                    print(f"  blocks ({len(blocks)} total):")
                    for i, block in enumerate(blocks):
                        print(_line(f"block[{i}]", block, indent=2))
                else:
                    print(_line(name, module, indent=1))
        else:
            for name, module in top_module.named_children():
                print(_line(name, module, indent=1))
    print()


def main():
    args = parser.parse_args()
    api_model = DepthAnything3.from_pretrained(args.pretrained)
    net = api_model.model
    freeze_backbone(net, args.unfreeze_last_n_blocks, args.fine_tune_layers)
    print_model_summary(net)


if __name__ == "__main__":
    main()
