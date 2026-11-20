"""CPU inference: preprocessing and the ONNX Predictor. No web code; slugid.api wraps it.

Depends only on numpy, Pillow and onnxruntime so the serving image stays small. Expects
model.onnx and labels.json from slugid.export in the model directory.
"""

import io
import json
import os

import numpy as np
from PIL import Image, ImageOps

LEVELS = ("species", "genus", "family")


def preprocess(img: Image.Image, size: int, crop_pct: float, mean, std) -> np.ndarray:
    """Same as the torchvision eval transform: resize short side, center crop, normalize."""
    img = img.convert("RGB")
    short = int(round(size / crop_pct))
    w, h = img.size
    if w <= h:
        new = (short, int(short * h / w))
    else:
        new = (int(short * w / h), short)
    img = img.resize(new, Image.BILINEAR)
    w, h = img.size
    left, top = int(round((w - size) / 2.0)), int(round((h - size) / 2.0))
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return x.transpose(2, 0, 1)[None]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class Predictor:
    def __init__(self, model_dir: str):
        import onnxruntime as ort

        self.labels = json.loads(open(os.path.join(model_dir, "labels.json")).read())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = len(os.sched_getaffinity(0))  # CPUs we may use, not the host's
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )

    def __call__(self, image_bytes: bytes, k: int = 5) -> dict:
        lab = self.labels
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        x = preprocess(img, lab["img_size"], lab["crop_pct"], lab["mean"], lab["std"])
        outs = dict(zip(LEVELS, self.session.run(list(LEVELS), {"image": x}), strict=True))
        result = {}
        for lvl in LEVELS:
            p = softmax(outs[lvl][0])
            top = np.argsort(-p)[:k]
            result[lvl] = [{"name": lab[lvl][i], "prob": round(float(p[i]), 4)} for i in top]
        ex = lab.get("examples", {})
        names = [s["name"] for s in result["species"]]
        result["examples"] = {n: ex[n] for n in names if n in ex}
        return result
