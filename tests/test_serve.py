import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from slugid import serve
from slugid.data.transforms import eval_transform
from slugid.export import export, labels
from slugid.model import LEVELS, SlugNet

TAX = {
    "species": ["Arion ater agg.", "Limax maximus", "other"],
    "genus": ["Arion", "Limax", "other"],
    "family": ["Arionidae", "Limacidae", "other"],
}
MEAN, STD = [0.48, 0.46, 0.41], [0.27, 0.26, 0.28]
META = {"taxonomy": TAX, "config": {"img_size": 32}, "mean": MEAN, "std": STD}


def jpeg(size=(57, 40), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (*size[::-1], 3), dtype=np.uint8))
    buf = __import__("io").BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def model_dir(tmp_path):
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    model = SlugNet(backbone, 8, {lvl: len(TAX[lvl]) for lvl in LEVELS}).eval()
    export(model, META, tmp_path)
    return tmp_path, model


@pytest.mark.parametrize("size", [(57, 40), (40, 57), (36, 36)])
def test_preprocess_matches_torchvision(size):
    from io import BytesIO

    img = Image.open(BytesIO(jpeg(size)))
    expected = eval_transform(32, mean=MEAN, std=STD)(img.convert("RGB"))[None].numpy()
    got = serve.preprocess(img, 32, 0.875, MEAN, STD)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_labels_carry_preprocessing():
    lab = labels(META)
    assert lab["species"] == TAX["species"] and lab["img_size"] == 32 and lab["mean"] == MEAN


def test_onnx_matches_torch(model_dir):
    path, model = model_dir
    pred = serve.Predictor(str(path))
    x = np.random.default_rng(0).standard_normal((2, 3, 32, 32)).astype(np.float32)
    onnx_out = pred.session.run(list(LEVELS), {"image": x})
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x))
    for lvl, o in zip(LEVELS, onnx_out, strict=True):
        np.testing.assert_allclose(o, torch_out[lvl].numpy(), atol=1e-4)


def test_predictor_returns_ranked_probs(model_dir):
    result = serve.Predictor(str(model_dir[0]))(jpeg(), k=2)
    for lvl in LEVELS:
        probs = [r["prob"] for r in result[lvl]]
        assert len(probs) == 2 and probs == sorted(probs, reverse=True)
        assert {r["name"] for r in result[lvl]} <= set(TAX[lvl])


def call(event):
    return serve.handler(event, SimpleNamespace(aws_request_id="req-1"))


def test_handler(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir[0]))
    monkeypatch.delenv("LOG_BUCKET", raising=False)
    monkeypatch.setattr(serve, "_predictor", None)
    logged = []
    monkeypatch.setattr(serve, "_log", lambda record, bucket: logged.append(record))

    get = {"requestContext": {"http": {"method": "GET"}}}
    health = call({**get, "rawPath": "/health"})
    assert health["statusCode"] == 200 and json.loads(health["body"])["classes"] == 3
    page = call({**get, "rawPath": "/"})
    assert page["headers"]["content-type"].startswith("text/html")
    assert "Slug Identifier" in page["body"]

    event = {"body": base64.b64encode(jpeg()).decode(), "isBase64Encoded": True}
    ok = call(event)
    body = json.loads(ok["body"])
    assert ok["statusCode"] == 200 and body["id"] == "req-1" and len(body["species"]) == 3
    assert not logged  # no bucket configured

    monkeypatch.setenv("LOG_BUCKET", "bucket")
    call(event)
    assert logged[0]["species"] == body["species"][0]["name"] and logged[0]["cold_start"] is False

    bad = call({"body": base64.b64encode(b"not an image").decode(), "isBase64Encoded": True})
    assert bad["statusCode"] == 400
    assert call({"body": ""})["statusCode"] == 400
