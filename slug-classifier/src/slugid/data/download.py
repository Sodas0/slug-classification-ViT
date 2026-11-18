"""Download manifest images, then drop missing files and cross-split duplicates.

Files are saved as <image-dir>/<photo_id>.<extension>. Existing valid files are skipped,
so the command can be re-run to resume. Writes a cleaned manifest for training.

Usage:
    python -m slugid.data.download --manifest data/manifest.parquet \
        --image-dir data/images --out data/manifest_clean.parquet
"""

import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

SPLIT_PRIORITY = ["train", "val", "test", "ood"]


def image_path(image_dir: Path, row) -> Path:
    return image_dir / f"{row.photo_id}.{row.extension}"


def make_session(workers: int) -> requests.Session:
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=workers)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers["User-Agent"] = "slugid-dataset-builder"
    return session


def is_valid_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def fetch(session, url: str, dest: Path) -> bool:
    """Download one image atomically. Returns True if a valid file exists at dest."""
    if dest.exists():
        return True
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return False
    if not is_valid_image(resp.content):
        return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(resp.content)
    os.replace(tmp, dest)
    return True


def download(manifest: pd.DataFrame, image_dir: Path, session, workers: int) -> pd.Series:
    """Download every row's image. Returns a boolean Series: file present."""
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = [image_path(image_dir, r) for r in manifest.itertuples()]
    with ThreadPoolExecutor(workers) as pool:
        jobs = zip(manifest["url"], paths, strict=True)
        results = pool.map(lambda a: fetch(session, *a), jobs)
        ok = list(tqdm(results, total=len(paths), desc="download", unit="img", mininterval=5))
    return pd.Series(ok, index=manifest.index)


def phash(path: Path) -> str:
    with Image.open(path) as img:
        return str(imagehash.phash(img))


def drop_duplicates(manifest: pd.DataFrame, hashes: pd.Series) -> pd.DataFrame:
    """Keep each perceptual hash only in its highest-priority split, once."""
    m = manifest.assign(
        _hash=hashes,
        _prio=manifest["split"].map({s: i for i, s in enumerate(SPLIT_PRIORITY)}),
    )
    m = m.sort_values(["_hash", "_prio", "photo_id"], kind="stable")
    m = m.drop_duplicates("_hash", keep="first")
    return m.drop(columns=["_hash", "_prio"]).sort_index()


def clean(manifest: pd.DataFrame, image_dir: Path, present: pd.Series, workers: int):
    kept = manifest[present]
    paths = [image_path(image_dir, r) for r in kept.itertuples()]
    with ThreadPoolExecutor(workers) as pool:
        results = tqdm(pool.map(phash, paths), total=len(paths), desc="phash", mininterval=5)
        hashes = pd.Series(list(results), index=kept.index)
    return drop_duplicates(kept, hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    present = download(manifest, args.image_dir, make_session(args.workers), args.workers)
    cleaned = clean(manifest, args.image_dir, present, args.workers)
    cleaned.to_parquet(args.out, index=False)

    failed = manifest.loc[~present, "url"]
    print(f"{present.sum()}/{len(manifest)} images present, {len(failed)} failed")
    if len(failed):
        print("First failures:\n  " + "\n  ".join(failed.head(5)))
    print(f"{present.sum() - len(cleaned)} duplicates dropped")
    print(cleaned["split"].value_counts().to_string())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
