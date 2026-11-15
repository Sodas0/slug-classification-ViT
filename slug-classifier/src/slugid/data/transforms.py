"""Image transforms. Augmentation happens on the fly during training, never on disk."""

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.35, 1.0)),
            T.RandomHorizontalFlip(),
            # Mild, hue-preserving: color is often what distinguishes slug species.
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def eval_transform(size=224, crop_pct=0.875, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Resize the short side, then center crop, so aspect ratio is preserved."""
    return T.Compose(
        [
            T.Resize(int(round(size / crop_pct))),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
