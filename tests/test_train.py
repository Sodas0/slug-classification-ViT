import json

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from slugid.model import LEVELS, SlugNet, logit_adjusted_loss, param_groups, taxonomy
from slugid.train import lr_lambda, train

ROWS = [  # label, genus, family, group
    ("Arion ater agg.", "Arion", "Arionidae", "slug"),
    ("Arion hortensis", "Arion", "Arionidae", "slug"),
    ("Limax maximus", "Limax", "Limacidae", "slug"),
    ("other", "Cornu", "Helicidae", "other"),
]
CFG = {
    "img_size": 32,
    "epochs": 2,
    "batch_size": 8,
    "num_workers": 0,
    "lr": 1e-3,
    "layer_decay": 0.65,
    "weight_decay": 0.05,
    "warmup_epochs": 1,
    "label_smoothing": 0.1,
    "logit_adjust_tau": 1.0,
    "level_weights": {"species": 1.0, "genus": 0.5, "family": 0.25},
    "ema_decay": 0.9,
    "grad_clip": 1.0,
    "seed": 0,
}


@pytest.fixture
def data(tmp_path):
    rows, pid = [], 0
    for split, n in [("train", 6), ("val", 3)]:
        for i, (label, genus, family, group) in enumerate(ROWS):
            for _ in range(n):
                pid += 1
                Image.new("RGB", (48, 40), (i * 60, 80, 160)).save(tmp_path / f"{pid}.jpg")
                rows.append((str(pid), "jpg", label, genus, family, group, split))
    cols = ["photo_id", "extension", "label", "genus", "family", "group", "split"]
    return pd.DataFrame(rows, columns=cols), tmp_path


def tiny_model(manifest):
    classes = sorted(manifest.loc[manifest["split"] == "train", "label"].unique())
    tax = taxonomy(manifest, classes)
    backbone = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    return SlugNet(backbone, 8, {lvl: len(tax[lvl]) for lvl in LEVELS})


def test_taxonomy_maps_species_to_genus_and_family(data):
    manifest, _ = data
    classes = sorted(manifest["label"].unique())
    tax = taxonomy(manifest, classes)
    assert tax["genus"] == ["Arion", "Limax", "other"]
    assert tax["family"] == ["Arionidae", "Limacidae", "other"]
    genus = [tax["genus"][i] for i in tax["genus_of_species"]]
    assert genus == ["Arion", "Arion", "Limax", "other"]


def test_layer_decay_and_no_weight_decay_on_norms():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc, self.ln = nn.Linear(4, 4), nn.LayerNorm(4)

    class FakeViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.transformer = nn.Module()
            self.transformer.resblocks = nn.ModuleList([Block(), Block()])
            self.ln_post = nn.LayerNorm(4)

    model = SlugNet(FakeViT(), 4, {"species": 3, "genus": 2, "family": 2})
    groups = param_groups(model, lr=1.0, weight_decay=0.1, layer_decay=0.5)
    lr_of = {id(p): g["lr"] for g in groups for p in g["params"]}
    wd_of = {id(p): g["weight_decay"] for g in groups for p in g["params"]}
    vit = model.backbone
    assert lr_of[id(model.heads["species"].weight)] == 1.0
    assert lr_of[id(vit.transformer.resblocks[1].fc.weight)] == 0.5
    assert lr_of[id(vit.transformer.resblocks[0].fc.weight)] == 0.25
    assert lr_of[id(vit.conv1.weight)] == 0.125
    assert wd_of[id(vit.transformer.resblocks[0].fc.weight)] == 0.1
    assert wd_of[id(vit.transformer.resblocks[0].ln.weight)] == 0.0
    assert wd_of[id(model.heads["species"].bias)] == 0.0


def test_logit_adjustment_is_plain_ce_at_tau_zero():
    logits, target = torch.randn(5, 3), torch.tensor([0, 1, 2, 0, 1])
    prior = torch.tensor([0.7, 0.2, 0.1]).log()
    plain = nn.functional.cross_entropy(logits, target)
    assert torch.isclose(logit_adjusted_loss(logits, target, prior, tau=0.0), plain)
    assert not torch.isclose(logit_adjusted_loss(logits, target, prior, tau=1.0), plain)


def test_lr_schedule_warms_up_then_decays_to_zero():
    assert lr_lambda(0, 10, 100) == pytest.approx(0.1)
    assert lr_lambda(9, 10, 100) == pytest.approx(1.0)
    assert lr_lambda(100, 10, 100) == pytest.approx(0.0)


def test_train_writes_checkpoints_and_resumes(data, tmp_path):
    manifest, image_dir = data
    out = tmp_path / "run"
    kw = dict(mean=(0.5,) * 3, std=(0.5,) * 3, device=torch.device("cpu"))
    best = train(CFG, manifest, image_dir, out, model=tiny_model(manifest), **kw)
    assert 0.0 <= best <= 1.0
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in records] == [0, 1]
    assert {"ema/macro_top1", "model/top5", "model/genus_top1"} <= set(records[0])

    ckpt = torch.load(out / "best.pt", weights_only=True)
    assert ckpt["meta"]["classes"] == sorted(r[0] for r in ROWS)

    # Same output dir with more epochs picks up after the last finished epoch.
    train({**CFG, "epochs": 3}, manifest, image_dir, out, model=tiny_model(manifest), **kw)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in records] == [0, 1, 2]
