import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from slugid.export import export
from slugid.model import LEVELS, SlugNet

TAX = {
    "species": ["Arion ater agg.", "Limax maximus", "other"],
    "genus": ["Arion", "Limax", "other"],
    "family": ["Arionidae", "Limacidae", "other"],
}
META = {
    "taxonomy": TAX,
    "config": {"img_size": 32},
    "mean": [0.48, 0.46, 0.41],
    "std": [0.27, 0.26, 0.28],
}
EXAMPLES = {"Limax maximus": [{"url": "https://x/photos/1/small.jpg", "obs": "u1"}]}


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (57, 40), (90, 120, 60)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    model = SlugNet(backbone, 8, {lvl: len(TAX[lvl]) for lvl in LEVELS}).eval()
    export(model, META, tmp_path, EXAMPLES)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from slugid.api import app

    with TestClient(app) as c:  # the context manager runs the lifespan (model load)
        yield c


def test_page_and_health(client):
    page = client.get("/")
    assert page.headers["content-type"].startswith("text/html") and "Slug Identifier" in page.text
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["classes"] == 3
    assert health["uptime_s"] >= 0


def test_predict(client):
    r = client.post("/v1/predict", files={"file": ("slug.jpg", jpeg(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and len(body["species"]) == 3  # only 3 classes exist
    probs = [c["prob"] for c in body["species"]]
    assert probs == sorted(probs, reverse=True)
    assert body["examples"].keys() <= set(EXAMPLES)


def test_predict_rejects_bad_input(client):
    bad = client.post("/v1/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 400
    empty = client.post("/v1/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert empty.status_code == 400
    assert client.post("/v1/predict").status_code == 422  # missing field: FastAPI validates


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/v1/predict"} <= set(paths)


def test_predict_logs_prediction(client, monkeypatch):
    from slugid import api

    logged = []
    monkeypatch.setattr(api, "log_prediction", lambda app, record, bucket: logged.append(record))
    files = {"file": ("slug.jpg", jpeg(), "image/jpeg")}

    client.post("/v1/predict", files=files)
    assert not logged  # no LOG_BUCKET configured

    monkeypatch.setenv("LOG_BUCKET", "bucket")
    r = client.post("/v1/predict", files=files, headers={"x-amzn-request-id": "req-1"})
    assert r.json()["id"] == "req-1"
    assert logged[0]["id"] == "req-1" and logged[0]["cold_start"] is False
    assert logged[0]["species"] == r.json()["species"][0]["name"]

    def boom(app, record, bucket):
        raise RuntimeError("s3 down")

    monkeypatch.setattr(api, "log_prediction", boom)
    assert client.post("/v1/predict", files=files).status_code == 200  # logging never breaks it
