"""Fine-tune a backbone with hierarchical heads on the cleaned manifest.

Usage:
    python -m slugid.train --config configs/train_bioclip.yaml \
        --manifest data/manifest_clean.parquet --image-dir data/images --out runs/bioclip
Re-running with the same --out resumes from last.pt.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from slugid.data.dataset import SlugDataset, class_names
from slugid.data.transforms import eval_transform, train_transform
from slugid.model import (
    LEVELS,
    SlugNet,
    build_backbone,
    log_prior,
    logit_adjusted_loss,
    param_groups,
    taxonomy,
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def level_targets(species: torch.Tensor, maps: dict) -> dict:
    return {lvl: species if lvl == "species" else maps[lvl][species] for lvl in LEVELS}


@torch.no_grad()
def evaluate(models: dict, loader, maps: dict, n_species: int, device) -> dict:
    """Top-1/top-5 (micro), macro top-1 over species, and genus/family top-1, per model."""
    stats = {
        name: {
            "correct": torch.zeros(n_species, device=device),
            "top5": 0,
            "genus": 0,
            "family": 0,
        }
        for name in models
    }
    counts = torch.zeros(n_species, device=device)
    for m in models.values():
        m.eval()
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        counts += torch.bincount(target, minlength=n_species)
        tgt = level_targets(target, maps)
        for name, m in models.items():
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = m(images)
            s, logits = stats[name], out["species"].float()
            hit = logits.argmax(1) == target
            s["correct"] += torch.bincount(target[hit], minlength=n_species)
            s["top5"] += (
                (logits.topk(min(5, n_species), 1).indices == target[:, None]).any(1).sum().item()
            )
            for lvl in ("genus", "family"):
                s[lvl] += (out[lvl].argmax(1) == tgt[lvl]).sum().item()
    n, present = counts.sum().item(), counts > 0
    return {
        name: {
            "top1": s["correct"].sum().item() / n,
            "top5": s["top5"] / n,
            "macro_top1": (s["correct"][present] / counts[present]).mean().item(),
            "genus_top1": s["genus"] / n,
            "family_top1": s["family"] / n,
        }
        for name, s in stats.items()
    }


def make_loaders(manifest, image_dir, classes, cfg, mean, std, limit=None):
    trainable = manifest[manifest["label"].isin(classes)]
    train = trainable[trainable["split"] == "train"]
    val = trainable[trainable["split"] == "val"]
    if limit:
        train = train.sample(min(limit, len(train)), random_state=cfg["seed"])
        val = val.sample(min(limit, len(val)), random_state=cfg["seed"])
    size = cfg["img_size"]
    train_ds = SlugDataset(train, image_dir, classes, train_transform(size, mean=mean, std=std))
    val_ds = SlugDataset(val, image_dir, classes, eval_transform(size, mean=mean, std=std))
    workers = cfg.get("num_workers") or len(os.sched_getaffinity(0))
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, cfg["batch_size"], shuffle=False, **common)
    return train_dl, val_dl, torch.tensor(train_ds.targets)


def train(
    cfg: dict,
    manifest: pd.DataFrame,
    image_dir: Path,
    out: Path,
    model=None,
    mean=None,
    std=None,
    limit=None,
    device=None,
):
    """Run training. `model`/`mean`/`std` override the configured backbone (used by tests)."""
    seed_everything(cfg["seed"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)

    classes = class_names(manifest)
    tax = taxonomy(manifest, classes)
    n_classes = {lvl: len(tax[lvl]) for lvl in LEVELS}
    maps = {
        lvl: torch.tensor(tax[f"{lvl}_of_species"], device=device) for lvl in ("genus", "family")
    }
    if model is None:
        backbone, dim, mean, std = build_backbone(cfg["backbone"])
        model = SlugNet(backbone, dim, n_classes)
    model.to(device)
    train_dl, val_dl, train_targets = make_loaders(
        manifest, image_dir, classes, cfg, mean, std, limit
    )
    priors = {
        lvl: log_prior(level_targets(train_targets.to(device), maps)[lvl], n_classes[lvl])
        for lvl in LEVELS
    }

    opt = torch.optim.AdamW(
        param_groups(model, cfg["lr"], cfg["weight_decay"], cfg["layer_decay"]),
        betas=(0.9, 0.999),
    )
    total = cfg["epochs"] * len(train_dl)
    warmup = int(cfg["warmup_epochs"] * len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, total))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    ema = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(cfg["ema_decay"]), use_buffers=True
    )

    start, best = 0, -1.0
    meta = {
        "classes": classes,
        "taxonomy": tax,
        "config": cfg,
        "mean": list(mean),
        "std": list(std),
    }
    last = out / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start, best = ckpt["epoch"] + 1, ckpt["best"]
        print(f"Resumed from epoch {ckpt['epoch']} (best macro top-1 {best:.4f})")

    weights = cfg["level_weights"]
    for epoch in range(start, cfg["epochs"]):
        model.train()
        t0, running = time.time(), 0.0
        bar = tqdm(train_dl, desc=f"epoch {epoch}", mininterval=30)
        for step, (images, target) in enumerate(bar):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            tgt = level_targets(target, maps)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out_logits = model(images)
            loss = sum(
                weights[lvl]
                * logit_adjusted_loss(
                    out_logits[lvl].float(),
                    tgt[lvl],
                    priors[lvl],
                    cfg["logit_adjust_tau"],
                    cfg["label_smoothing"],
                )
                for lvl in LEVELS
            )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
            running += loss.item()
            if step % 50 == 0:
                bar.set_postfix(loss=f"{running / (step + 1):.3f}")

        metrics = evaluate({"model": model, "ema": ema.module}, val_dl, maps, len(classes), device)
        pick = max(metrics, key=lambda k: metrics[k]["macro_top1"])
        record = {
            "epoch": epoch,
            "train_loss": running / len(train_dl),
            "minutes": (time.time() - t0) / 60,
            **{f"{k}/{m}": v for k, d in metrics.items() for m, v in d.items()},
        }
        print(json.dumps(record))
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

        if metrics[pick]["macro_top1"] > best:
            best = metrics[pick]["macro_top1"]
            weights_src = model if pick == "model" else ema.module
            torch.save(
                {
                    "model": weights_src.state_dict(),
                    "meta": meta,
                    "epoch": epoch,
                    "source": pick,
                    "val": metrics[pick],
                },
                out / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
            },
            out / "last.pt",
        )
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="use at most N train and N val images (debug)")
    parser.add_argument("--epochs", type=int, help="override config epochs (debug)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs:
        cfg["epochs"] = args.epochs
    best = train(cfg, pd.read_parquet(args.manifest), args.image_dir, args.out, limit=args.limit)
    print(f"Best val macro top-1: {best:.4f}")


if __name__ == "__main__":
    main()
