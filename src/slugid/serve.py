"""CPU inference and the AWS Lambda handler, which also serves an upload page (index.html).

Depends only on numpy, Pillow and onnxruntime (plus boto3, which Lambda provides) so the
serving image stays small. Expects model.onnx and labels.json from slugid.export in
MODEL_DIR. Each prediction is logged as one JSON object to s3://LOG_BUCKET/predictions/
when LOG_BUCKET is set, partitioned by date for Athena.
"""

import base64
import io
import json
import os
import time
import uuid
from datetime import UTC, datetime

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


_predictor = None
_s3 = None


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def _page() -> dict:
    html = open(os.path.join(os.path.dirname(__file__), "index.html")).read()
    return {
        "statusCode": 200,
        "headers": {"content-type": "text/html; charset=utf-8"},
        "body": html,
    }


def _log(record: dict, bucket: str):
    global _s3
    if _s3 is None:
        import boto3

        _s3 = boto3.client("s3")
    key = f"predictions/dt={record['ts'][:10]}/{record['id']}.json"
    _s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(record).encode())


def handler(event, context):
    """Lambda Function URL handler. GET / serves the web page, GET /health a health check,
    and POST takes raw image bytes and returns predictions."""
    global _predictor
    cold = _predictor is None
    if cold:
        _predictor = Predictor(os.environ.get("MODEL_DIR", "/opt/model"))

    method = event.get("requestContext", {}).get("http", {}).get("method", "POST")
    if method == "GET":
        if event.get("rawPath", "/") == "/health":
            return _response(200, {"status": "ok", "classes": len(_predictor.labels["species"])})
        return _page()

    body = event.get("body") or ""
    data = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("latin-1")
    if not data:
        return _response(400, {"error": "POST an image as the request body"})

    start = time.perf_counter()
    try:
        pred = _predictor(data)
    except Exception as e:  # unreadable image
        return _response(400, {"error": f"could not read image: {e}"})
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    record = {
        "id": getattr(context, "aws_request_id", None) or str(uuid.uuid4()),
        "ts": datetime.now(UTC).isoformat(),
        "model_version": os.environ.get("MODEL_VERSION", "dev"),
        "species": pred["species"][0]["name"],
        "species_prob": pred["species"][0]["prob"],
        "genus": pred["genus"][0]["name"],
        "family": pred["family"][0]["name"],
        "top5": pred["species"],
        "latency_ms": latency_ms,
        "cold_start": cold,
        "image_bytes": len(data),
    }
    if bucket := os.environ.get("LOG_BUCKET"):
        try:
            _log(record, bucket)
        except Exception as e:  # logging must never break a prediction
            print(f"prediction log failed: {e}")
    return _response(200, {"id": record["id"], "latency_ms": latency_ms, **pred})
