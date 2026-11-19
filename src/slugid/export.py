"""Export a trained checkpoint to ONNX for CPU serving.

Writes model.onnx (image -> species/genus/family logits) and labels.json (class names and
preprocessing constants) so the serving side needs only onnxruntime, numpy and Pillow.

Usage:
    python -m slugid.export --checkpoint data/runs/bioclip_e12/best.pt --out data/export
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from slugid.model import LEVELS, SlugNet


def bioclip_architecture() -> tuple[nn.Module, int]:
    """BioCLIP's vision tower is a plain ViT-B/16; build it without downloading weights."""
    import open_clip

    visual = open_clip.create_model("ViT-B-16").visual
    visual.proj = None
    return visual, visual.transformer.width


class _Tuple(nn.Module):
    """ONNX outputs must be tensors, not a dict."""

    def __init__(self, model: SlugNet):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return tuple(out[lvl] for lvl in LEVELS)


def load(checkpoint: Path, backbone_fn=bioclip_architecture) -> tuple[SlugNet, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    backbone, dim = backbone_fn()
    n_classes = {lvl: len(meta["taxonomy"][lvl]) for lvl in LEVELS}
    model = SlugNet(backbone, dim, n_classes)
    model.load_state_dict(ckpt["model"])
    return model.eval(), meta


def labels(meta: dict) -> dict:
    tax = meta["taxonomy"]
    return {
        "species": tax["species"],
        "genus": tax["genus"],
        "family": tax["family"],
        "img_size": meta["config"]["img_size"],
        "crop_pct": 0.875,
        "mean": meta["mean"],
        "std": meta["std"],
    }


def export(model: SlugNet, meta: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    size = meta["config"]["img_size"]
    path = out / "model.onnx"
    torch.onnx.export(
        _Tuple(model),
        (torch.zeros(1, 3, size, size),),
        path,
        input_names=["image"],
        output_names=list(LEVELS),
        dynamic_axes={"image": {0: "batch"}, **{lvl: {0: "batch"} for lvl in LEVELS}},
        dynamo=False,
    )
    (out / "labels.json").write_text(json.dumps(labels(meta)))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, meta = load(args.checkpoint)
    print(f"Wrote {export(model, meta, args.out)}")


if __name__ == "__main__":
    main()
