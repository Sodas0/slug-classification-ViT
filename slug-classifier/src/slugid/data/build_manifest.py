"""Build the image manifest from an iNaturalist Open Data metadata snapshot.

Reads taxa/observations/photos (tab-separated, gzipped) and writes one parquet row per
photo with its label, taxonomy, observer and split. Splits are grouped by observer so no
person's photos appear in more than one split.

Usage:
    python -m slugid.data.build_manifest --config configs/data.yaml \
        --metadata-dir /path/to/inat-metadata --out data/manifest.parquet
"""

import argparse
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

PHOTO_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{size}.{ext}"
COLUMNS = {
    "taxa": ["taxon_id", "ancestry", "rank_level", "rank", "name", "active"],
    "observations": [
        "observation_uuid",
        "observer_id",
        "latitude",
        "longitude",
        "taxon_id",
        "quality_grade",
        "observed_on",
    ],
    "photos": ["photo_id", "observation_uuid", "extension", "license", "position"],
}
SPECIES_RANK_LEVEL = 10


def _source(con, metadata_dir: Path, table: str) -> str:
    """Return a DuckDB table expression for a metadata file, checking its columns."""
    path = metadata_dir / f"{table}.csv.gz"
    if not path.exists():
        path = metadata_dir / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{table}.csv(.gz) not found in {metadata_dir}")
    expr = f"read_csv('{path}', delim='\\t', header=true, quote='', escape='', all_varchar=true)"
    found = [d[0] for d in con.execute(f"select * from {expr} limit 0").description]
    missing = [c for c in COLUMNS[table] if c not in found]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; found {found}")
    return expr


def parse_ancestry(ancestry) -> list[str]:
    """Split an ancestry string of taxon ids (separator '/' or '\\')."""
    if not isinstance(ancestry, str) or not ancestry:
        return []
    return [t for t in re.split(r"[/\\]", ancestry) if t]


def normalize_quality(series: pd.Series) -> pd.Series:
    """Map 'Research Grade' / 'research' / 'needs_id' / 'Needs ID' to snake_case."""
    s = series.str.lower().str.strip().str.replace(r"\s+grade$", "", regex=True)
    return s.str.replace(r"\s+", "_", regex=True)


def resolve(entries, taxa: pd.DataFrame) -> dict[str, str]:
    """Map config entries (names or integer ids) to taxon ids. Fails on missing/ambiguous."""
    active = taxa[taxa["active"]]
    out, errors = {}, []
    for entry in entries:
        if isinstance(entry, int):
            matches = taxa[taxa["taxon_id"] == str(entry)]
        else:
            matches = active[active["name"] == entry]
        if len(matches) == 1:
            out[str(entry)] = matches["taxon_id"].iloc[0]
        elif matches.empty:
            errors.append(f"'{entry}': not found")
        else:
            options = ", ".join(f"{r.taxon_id} ({r.rank})" for r in matches.itertuples())
            errors.append(f"'{entry}': ambiguous, use one of the ids {options}")
    if errors:
        raise ValueError("Could not resolve taxa:\n  " + "\n  ".join(errors))
    return out


def label_taxa(taxa: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign group/label/family/genus to every taxon at species rank or below."""
    slug = set(resolve(cfg["slug_clades"], taxa).values())
    other = set(resolve(cfg["other_clades"], taxa).values())
    ood = set(resolve(cfg["unseen_negative_clades"], taxa).values())
    merged = {}
    for group in cfg.get("merge_groups", []):
        for tid in resolve(group["members"], taxa).values():
            merged[tid] = group["label"]

    rank = dict(zip(taxa["taxon_id"], taxa["rank"], strict=True))
    name = dict(zip(taxa["taxon_id"], taxa["name"], strict=True))
    clades = slug | other | ood
    rows = []
    for tid, ancestry, level in zip(
        taxa["taxon_id"], taxa["ancestry"], taxa["rank_level"], strict=True
    ):
        if not level <= SPECIES_RANK_LEVEL:
            continue
        path = parse_ancestry(ancestry) + [tid]
        hits = clades.intersection(path)
        if not hits:
            continue
        species = next((t for t in path if rank.get(t) == "species"), None)
        if species is None:
            continue
        family = next((name[t] for t in path if rank.get(t) == "family"), None)
        genus = next((name[t] for t in path if rank.get(t) == "genus"), None)
        if hits & slug:
            group, label = "slug", merged.get(species, name[species])
        elif hits & ood:
            group, label = "ood", "ood"
        else:
            group, label = "other", "other"
        rows.append((tid, group, label, name[species], genus, family))
    return pd.DataFrame(rows, columns=["taxon_id", "group", "label", "species", "genus", "family"])


def load_taxa(con, metadata_dir: Path) -> pd.DataFrame:
    taxa = con.execute(
        f"select taxon_id, ancestry, rank_level, rank, name, active "
        f"from {_source(con, metadata_dir, 'taxa')}"
    ).df()
    taxa["rank_level"] = pd.to_numeric(taxa["rank_level"], errors="coerce")
    taxa["active"] = taxa["active"].str.lower().isin(["true", "t", "1"])
    return taxa


def load_photos(con, metadata_dir: Path, labeled: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join labeled taxa -> observations -> photos; filter by quality and optional license."""
    con.register("labeled", labeled)
    obs = con.execute(
        f"select o.observation_uuid, o.observer_id, "
        f"try_cast(o.latitude as double) as latitude, "
        f"try_cast(o.longitude as double) as longitude, "
        f"o.observed_on, o.quality_grade, l.* "
        f"from {_source(con, metadata_dir, 'observations')} o join labeled l using (taxon_id)"
    ).df()
    obs = obs[normalize_quality(obs["quality_grade"]).isin(cfg["quality_grades"])]
    obs = obs.drop(columns="quality_grade")

    con.register("obs", obs)
    # No `licenses` key means keep every license; the license column is recorded either way.
    where = ""
    if cfg.get("licenses"):
        allowed = ", ".join(f"'{lic.upper()}'" for lic in cfg["licenses"])
        where = f"where upper(p.license) in ({allowed})"
    photos = con.execute(
        f"select p.photo_id, p.extension, upper(p.license) as license, "
        f"try_cast(p.position as integer) as position, obs.* "
        f"from {_source(con, metadata_dir, 'photos')} p join obs using (observation_uuid) "
        f"{where}"
    ).df()
    return photos


def select(photos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop slug labels with too few photos and cap images per label, favouring distinct obs."""
    n_photos = photos[photos["group"] == "slug"].groupby("label").size()
    rare = set(n_photos[n_photos < cfg["min_photos"]].index)
    photos = photos[~photos["label"].isin(rare)]

    # Deterministic shuffle, then prefer each observation's first photo before extra ones.
    photos = photos.sort_values("photo_id").reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    photos = photos.assign(_r=rng.random(len(photos)))
    photos = photos.sort_values(["label", "position", "_r"], na_position="last")
    caps = np.where(photos["group"] == "slug", cfg["max_images_per_class"], cfg["max_images_other"])
    keep = photos.groupby("label").cumcount().to_numpy() < caps
    return photos[keep].drop(columns="_r").reset_index(drop=True)


def assign_splits(photos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Observer-grouped, label-stratified split. Unseen negatives always go to 'ood'."""
    photos = photos.copy()
    photos["split"] = "ood"
    trainable = photos.index[photos["group"] != "ood"]
    sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    folds = sgkf.split(
        np.zeros(len(trainable)),
        photos.loc[trainable, "label"],
        groups=photos.loc[trainable, "observer_id"],
    )
    photos.loc[trainable, "split"] = "train"
    for fold, (_, held) in enumerate(folds):
        if fold > 1:
            break
        photos.loc[trainable[held], "split"] = "test" if fold == 0 else "val"
    return photos


def build(metadata_dir: Path, cfg: dict) -> pd.DataFrame:
    con = duckdb.connect()
    taxa = load_taxa(con, metadata_dir)
    labeled = label_taxa(taxa, cfg)
    photos = assign_splits(select(load_photos(con, metadata_dir, labeled, cfg), cfg), cfg)
    photos["url"] = [
        PHOTO_URL.format(photo_id=p, size=cfg["photo_size"], ext=e)
        for p, e in zip(photos["photo_id"], photos["extension"], strict=True)
    ]
    return photos.sort_values(["split", "label", "photo_id"]).reset_index(drop=True)


def summarize(manifest: pd.DataFrame) -> str:
    counts = manifest.pivot_table(
        index="label", columns="split", values="photo_id", aggfunc="count", fill_value=0
    ).reindex(columns=["train", "val", "test", "ood"], fill_value=0)
    lines = [
        f"{manifest['label'].nunique()} labels, {len(manifest)} photos, "
        f"{manifest['observer_id'].nunique()} observers",
        manifest["split"].value_counts().to_string(),
    ]
    trainable = counts.drop(index="ood", errors="ignore")
    empty = trainable[(trainable[["train", "val", "test"]] == 0).any(axis=1)]
    if not empty.empty:
        lines.append("WARNING: labels missing from a split:\n" + empty.to_string())
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    manifest = build(args.metadata_dir, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)
    print(summarize(manifest))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
