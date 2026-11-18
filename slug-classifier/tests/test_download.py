import io

import pandas as pd
import requests
from PIL import Image

from slugid.data.download import clean, download, image_path


def _jpeg(seed: int) -> bytes:
    """A distinct, deterministic test image (different seeds give different phashes)."""
    img = Image.new("RGB", (64, 64))
    img.putdata(
        [
            ((x * seed) % 256, (y * 7 * seed) % 256, (x * y) % 256)
            for y in range(64)
            for x in range(64)
        ]
    )
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content=b"", status=200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(str(self.status))


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, timeout):
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(status=404))


def _manifest(rows):
    return pd.DataFrame(rows, columns=["photo_id", "extension", "url", "split"])


def test_download_skips_missing_and_invalid_and_resumes(tmp_path):
    m = _manifest(
        [
            ("1", "jpg", "u/1", "train"),
            ("2", "jpg", "u/2", "train"),  # 404
            ("3", "jpg", "u/3", "val"),  # not an image
        ]
    )
    session = FakeSession({"u/1": FakeResponse(_jpeg(1)), "u/3": FakeResponse(b"<html>")})
    present = download(m, tmp_path, session, workers=2)
    assert present.tolist() == [True, False, False]
    assert image_path(tmp_path, next(m.itertuples())).exists()
    assert not list(tmp_path.glob("*.part"))

    # Re-running doesn't re-fetch files that already exist.
    session.calls.clear()
    download(m, tmp_path, session, workers=2)
    assert "u/1" not in session.calls


def test_repeated_photo_fetched_once(tmp_path):
    m = _manifest([("1", "jpg", "u/1", "train"), ("1", "jpg", "u/1", "test")])
    session = FakeSession({"u/1": FakeResponse(_jpeg(1))})
    present = download(m, tmp_path, session, workers=2)
    assert present.tolist() == [True, True]
    assert session.calls == ["u/1"]


def test_duplicates_kept_only_in_highest_priority_split(tmp_path):
    same, other = _jpeg(1), _jpeg(5)
    m = _manifest(
        [
            ("1", "jpg", "u/1", "test"),
            ("2", "jpg", "u/2", "train"),  # same image as 1 -> keep this one
            ("3", "jpg", "u/3", "val"),  # same image as 1 -> drop
            ("4", "jpg", "u/4", "val"),  # distinct -> keep
        ]
    )
    routes = {"u/1": same, "u/2": same, "u/3": same, "u/4": other}
    session = FakeSession({k: FakeResponse(v) for k, v in routes.items()})
    present = download(m, tmp_path, session, workers=2)
    cleaned = clean(m, tmp_path, present, workers=2)
    assert cleaned["photo_id"].tolist() == ["2", "4"]
