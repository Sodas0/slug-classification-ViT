"""PyTorch dataset over a cleaned manifest."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

OOD_TARGET = -1


def class_names(manifest: pd.DataFrame) -> list[str]:
    """Sorted training labels. This order defines the model's output indices."""
    return sorted(manifest.loc[manifest["split"] == "train", "label"].unique())


class SlugDataset(Dataset):
    """Yields (image, target). Labels outside `classes` (the held-out set) get target -1."""

    def __init__(self, manifest: pd.DataFrame, image_dir: Path, classes: list[str], transform):
        index = {name: i for i, name in enumerate(classes)}
        self.paths = [
            Path(image_dir) / f"{p}.{e}"
            for p, e in zip(manifest["photo_id"], manifest["extension"], strict=True)
        ]
        self.targets = [index.get(label, OOD_TARGET) for label in manifest["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.targets[i]
