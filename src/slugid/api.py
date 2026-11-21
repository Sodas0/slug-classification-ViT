"""FastAPI service: the upload page, a health check and POST /v1/predict.

On AWS it runs inside Lambda behind the Lambda Web Adapter (see Dockerfile), which turns
Function URL requests into plain HTTP calls to uvicorn. Locally:
    MODEL_DIR=model uvicorn slugid.api:app

Each prediction is logged as one JSON object to s3://LOG_BUCKET/predictions/ when LOG_BUCKET
is set, partitioned by date for Athena.
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from slugid.serve import Predictor

MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")
PAGE = (Path(__file__).parent / "index.html").read_text()


class Candidate(BaseModel):
    name: str
    prob: float


class Example(BaseModel):
    url: str
    obs: str


class Prediction(BaseModel):
    id: str
    latency_ms: float
    species: list[Candidate]
    genus: list[Candidate]
    family: list[Candidate]
    examples: dict[str, list[Example]]


class Health(BaseModel):
    status: str
    model_version: str
    classes: int
    uptime_s: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup so /health only answers once the model is ready.
    app.state.predictor = Predictor(os.environ.get("MODEL_DIR", "/opt/model"))
    app.state.started = time.monotonic()  # container age; low means a fresh cold start
    app.state.served = False  # flips on the first request, to flag cold starts in the log
    app.state.s3 = None
    yield


app = FastAPI(title="Slug Identifier", version="1.0.0", lifespan=lifespan)


def log_prediction(app: FastAPI, record: dict, bucket: str) -> None:
    if app.state.s3 is None:
        import boto3

        app.state.s3 = boto3.client("s3")
    key = f"predictions/dt={record['ts'][:10]}/{record['id']}.json"
    app.state.s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(record).encode())


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def page() -> str:
    return PAGE


@app.get("/health")
def health(request: Request) -> Health:
    n = len(request.app.state.predictor.labels["species"])
    uptime = round(time.monotonic() - request.app.state.started, 1)
    return Health(status="ok", model_version=MODEL_VERSION, classes=n, uptime_s=uptime)


# Plain `def` (not async): inference is CPU-bound and blocking, so FastAPI runs it in a
# threadpool instead of freezing the event loop.
@app.post("/v1/predict")
def predict(
    request: Request, file: Annotated[UploadFile, File(description="A photo of a slug")]
) -> Prediction:
    data = file.file.read()
    if not data:
        raise HTTPException(400, "empty file")
    cold = not request.app.state.served
    request.app.state.served = True

    start = time.perf_counter()
    try:
        pred = request.app.state.predictor(data)
    except Exception as e:  # unreadable image
        raise HTTPException(400, f"could not read image: {e}") from e
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    # The Lambda Web Adapter forwards Lambda's request id in this header.
    rid = request.headers.get("x-amzn-request-id") or str(uuid.uuid4())
    record = {
        "id": rid,
        "ts": datetime.now(UTC).isoformat(),
        "model_version": MODEL_VERSION,
        "species": pred["species"][0]["name"],
        "species_prob": pred["species"][0]["prob"],
        "genus": pred["genus"][0]["name"],
        "family": pred["family"][0]["name"],
        "top5": pred["species"],
        "latency_ms": latency_ms,
        "cold_start": cold,
        "image_bytes": len(data),
    }
    # Logged before returning: Lambda freezes the container once the response is sent, so a
    # background task could be cut off.
    if bucket := os.environ.get("LOG_BUCKET"):
        try:
            log_prediction(request.app, record, bucket)
        except Exception as e:  # logging must never break a prediction
            print(f"prediction log failed: {e}")
    return Prediction(id=rid, latency_ms=latency_ms, **pred)
