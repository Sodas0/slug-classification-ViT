"""Backbone + hierarchical (species / genus / family) linear heads."""

import re

import pandas as pd
import torch
from torch import nn

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
LEVELS = ("species", "genus", "family")


def build_backbone(name: str):
    """Return (backbone, feature_dim, mean, std). Backbone maps images to pooled features."""
    if name == "bioclip":
        import open_clip

        visual = open_clip.create_model("hf-hub:imageomics/bioclip").visual
        visual.proj = None  # fine-tune on the 768-d pooled token, not the CLIP projection
        return visual, visual.transformer.width, CLIP_MEAN, CLIP_STD
    raise ValueError(f"unknown backbone {name!r}")


class SlugNet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, n_classes: dict[str, int]):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({lvl: nn.Linear(feat_dim, n_classes[lvl]) for lvl in LEVELS})

    def forward(self, x):
        feats = self.backbone(x)
        return {lvl: head(feats) for lvl, head in self.heads.items()}


def taxonomy(manifest: pd.DataFrame, classes: list[str]) -> dict:
    """Map each species class to a genus and family index. Non-slug labels map to themselves."""
    train = manifest[manifest["split"] == "train"]
    ranks = train.groupby("label")[["genus", "family"]].agg(
        lambda s: s.mode().iat[0] if s.notna().any() else None
    )
    out = {"species": list(classes)}
    for rank in ("genus", "family"):
        per_class = [
            c if train.loc[train["label"] == c, "group"].iat[0] != "slug" else ranks.at[c, rank]
            for c in classes
        ]
        names = sorted(set(per_class))
        index = {n: i for i, n in enumerate(names)}
        out[rank] = names
        out[f"{rank}_of_species"] = [index[n] for n in per_class]
    return out


def param_groups(model: nn.Module, lr: float, weight_decay: float, layer_decay: float):
    """AdamW groups with layer-wise LR decay; no weight decay on biases, norms and embeddings."""
    blocks = {
        int(m.group(1))
        for n, _ in model.named_parameters()
        if (m := re.search(r"resblocks\.(\d+)\.", n))
    }
    top = len(blocks) + 1  # 0 = patch/pos embeddings, 1..L = blocks, L+1 = final norm + heads

    def layer_id(name):
        if m := re.search(r"resblocks\.(\d+)\.", name):
            return int(m.group(1)) + 1
        stem = ("class_embedding", "positional_embedding", "conv1", "ln_pre")
        if name.startswith("backbone.") and name.split(".")[1] in stem:
            return 0
        return top

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        no_wd = p.ndim <= 1 or "embedding" in name
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr * layer_decay ** (top - lid),
                "weight_decay": 0.0 if no_wd else weight_decay,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def logit_adjusted_loss(logits, target, log_prior, tau=1.0, label_smoothing=0.0):
    """Cross-entropy on logits shifted by tau * log class prior (Menon et al., 2021)."""
    return nn.functional.cross_entropy(
        logits + tau * log_prior, target, label_smoothing=label_smoothing
    )


def log_prior(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=n_classes).float().clamp(min=1)
    return (counts / counts.sum()).log()
