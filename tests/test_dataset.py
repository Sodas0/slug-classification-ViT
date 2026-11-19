import pandas as pd
import pytest
import torch
from PIL import Image

from slugid.data.dataset import OOD_TARGET, SlugDataset, class_names
from slugid.data.transforms import eval_transform, train_transform


@pytest.fixture
def setup(tmp_path):
    rows = [
        ("1", "jpg", "Limax maximus", "train"),
        ("2", "jpg", "Arion ater agg.", "train"),
        ("3", "jpg", "other", "val"),
        ("4", "jpg", "ood", "ood"),
    ]
    manifest = pd.DataFrame(rows, columns=["photo_id", "extension", "label", "split"])
    for i, (pid, ext, *_) in enumerate(rows):
        Image.new("RGB", (500, 250), (i * 60, 100, 200)).save(tmp_path / f"{pid}.{ext}")
    return manifest, tmp_path


def test_class_names_come_from_train_split_sorted(setup):
    manifest, _ = setup
    assert class_names(manifest) == ["Arion ater agg.", "Limax maximus"]


def test_dataset_targets_and_shapes(setup):
    manifest, image_dir = setup
    classes = class_names(manifest) + ["other"]
    ds = SlugDataset(manifest, image_dir, classes, eval_transform(224))
    assert len(ds) == 4
    assert [ds[i][1] for i in range(4)] == [1, 0, 2, OOD_TARGET]
    image, _ = ds[0]
    assert image.shape == (3, 224, 224) and image.dtype == torch.float32


def test_eval_transform_is_deterministic_and_train_is_not(setup):
    _, image_dir = setup
    img = Image.open(image_dir / "1.jpg").convert("RGB")
    ev = eval_transform(224)
    assert torch.equal(ev(img), ev(img))
    tr = train_transform(224)
    torch.manual_seed(0)
    a = tr(img)
    b = tr(img)
    assert a.shape == (3, 224, 224) and not torch.equal(a, b)
