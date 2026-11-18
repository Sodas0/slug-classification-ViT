"""Optional location prior: p(class | lat, lon), combined with image logits at inference.

Location is never required. Without it, predictions come from the image model alone.

Usage:
    python -m slugid.geo --manifest data/manifest_clean.parquet \
        --classes-from runs/bioclip/best.pt --val-logits runs/bioclip/val_logits.pt \
        --out runs/geo
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def encode(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Wrap-around encoding of coordinates in degrees (SINR-style)."""
    x = torch.stack([lon / 180.0, lat / 90.0], dim=-1) * math.pi
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.ReLU()
        )

    def forward(self, x):
        return x + self.net(x)


class GeoNet(nn.Module):
    def __init__(self, n_classes: int, dim: int = 256, depth: int = 4, dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4, dim), nn.ReLU(), *[ResBlock(dim, dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, n_classes)

    def forward(self, lat, lon):
        return self.head(self.body(encode(lat, lon)))


def combine(image_logits, geo_log_probs=None, weight: float = 1.0, floor: float = -8.0):
    """Image logits plus a weighted, floored location log-prior. No location -> image only.

    The image model is trained with logit adjustment, so its logits are prior-free and the
    location model supplies the (local) class prior. The floor keeps a species seen outside
    its known range from being ruled out entirely.
    """
    if geo_log_probs is None:
        return image_logits
    return image_logits + weight * geo_log_probs.clamp(min=floor)


def observations(manifest: pd.DataFrame, split: str, classes: list[str]) -> pd.DataFrame:
    """One row per observation with coordinates and a known class."""
    rows = manifest[(manifest["split"] == split) & manifest["label"].isin(classes)]
    rows = rows.dropna(subset=["latitude", "longitude"])
    return rows.drop_duplicates("observation_uuid")


def train_geo(
    obs: pd.DataFrame,
    classes: list[str],
    epochs=30,
    batch_size=2048,
    lr=1e-3,
    jitter_deg=0.1,
    seed=0,
    device=None,
) -> GeoNet:
    """Fit p(class | location) with cross-entropy. Coordinates get jitter so coarse input works."""
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = {c: i for i, c in enumerate(classes)}
    lat = torch.tensor(obs["latitude"].to_numpy(), dtype=torch.float32, device=device)
    lon = torch.tensor(obs["longitude"].to_numpy(), dtype=torch.float32, device=device)
    y = torch.tensor(obs["label"].map(index).to_numpy(), device=device)
    model = GeoNet(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(y) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(y), device=device).split(batch_size):
            la = (lat[idx] + jitter_deg * torch.randn_like(lat[idx])).clamp(-90, 90)
            lo = lon[idx] + jitter_deg * torch.randn_like(lon[idx])
            loss = nn.functional.cross_entropy(model(la, lo), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def geo_log_probs(model: GeoNet, lat, lon, device=None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    lat = torch.as_tensor(np.asarray(lat, dtype=np.float32), device=device)
    lon = torch.as_tensor(np.asarray(lon, dtype=np.float32), device=device)
    return model(lat, lon).log_softmax(-1).cpu()


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> dict:
    ok = (logits.argmax(1) == target).float()
    per_class = [ok[target == c].mean().item() for c in target.unique()]
    return {"top1": ok.mean().item(), "macro_top1": float(np.mean(per_class))}


def tune(image_logits, geo_lp, has_loc, target, weights, floors):
    """Grid-search weight/floor on val macro top-1. Rows without a location stay image-only."""
    best = None
    for w in weights:
        for f in floors:
            logits = image_logits.clone()
            logits[has_loc] = combine(image_logits[has_loc], geo_lp[has_loc], w, f)
            acc = accuracy(logits, target)
            if best is None or acc["macro_top1"] > best[2]["macro_top1"]:
                best = (w, f, acc)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classes-from", type=Path, required=True, help="image model best.pt")
    parser.add_argument("--val-logits", type=Path, required=True, help="saved val image logits")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--floor",
        type=float,
        default=-8.0,
        help="min location log-prob; fixed, not tuned, since val can't show range expansion",
    )
    args = parser.parse_args()

    classes = torch.load(args.classes_from, weights_only=True, map_location="cpu")["meta"][
        "classes"
    ]
    manifest = pd.read_parquet(args.manifest)
    model = train_geo(observations(manifest, "train", classes), classes, epochs=args.epochs)

    saved = torch.load(args.val_logits, weights_only=True)
    val = manifest.set_index("photo_id").loc[saved["photo_id"]]
    index = {c: i for i, c in enumerate(classes)}
    target = torch.tensor(val["label"].map(index).to_numpy())
    has_loc = torch.tensor(val[["latitude", "longitude"]].notna().all(axis=1).to_numpy())
    geo_lp = torch.zeros_like(saved["logits"])
    geo_lp[has_loc] = geo_log_probs(
        model, val.loc[has_loc.numpy(), "latitude"], val.loc[has_loc.numpy(), "longitude"]
    )

    image_only = accuracy(saved["logits"], target)
    geo_only = accuracy(geo_lp[has_loc], target[has_loc])
    w, f, combined = tune(
        saved["logits"],
        geo_lp,
        has_loc,
        target,
        weights=[0.25, 0.5, 0.75, 1.0, 1.5],
        floors=[args.floor],
    )
    report = {
        "val_image_only": image_only,
        "val_location_only": geo_only,
        "val_image_plus_location": combined,
        "weight": w,
        "floor": f,
        "val_rows_with_location": int(has_loc.sum()),
        "val_rows": len(target),
    }
    print(json.dumps(report, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "classes": classes, "weight": w, "floor": f},
        args.out / "geo.pt",
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
