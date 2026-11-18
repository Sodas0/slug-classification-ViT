import gzip

import pytest

from slugid.data.build_manifest import build, normalize_quality, parse_ancestry

# taxon_id, parent_id, rank, rank_level, name
TREE = [
    ("1", None, "kingdom", 70, "Animalia"),
    ("2", "1", "family", 30, "Arionidae"),
    ("3", "2", "genus", 20, "Arion"),
    ("4", "3", "species", 10, "Arion ater"),
    ("5", "3", "species", 10, "Arion rufus"),
    ("6", "4", "subspecies", 5, "Arion ater ater"),
    ("7", "1", "family", 30, "Limacidae"),
    ("8", "7", "genus", 20, "Limax"),
    ("9", "8", "species", 10, "Limax maximus"),
    ("10", "8", "species", 10, "Limax cinereoniger"),
    ("11", "1", "family", 30, "Helicidae"),
    ("12", "11", "genus", 20, "Cornu"),
    ("13", "12", "species", 10, "Cornu aspersum"),
    ("14", "1", "family", 30, "Lumbricidae"),
    ("15", "14", "genus", 20, "Lumbricus"),
    ("16", "15", "species", 10, "Lumbricus terrestris"),
]
# taxon_id -> number of research-grade observations
OBS_PER_TAXON = {"4": 15, "5": 15, "6": 10, "9": 30, "10": 3, "13": 30, "16": 10}

CFG = {
    "slug_clades": ["Arionidae", "Limacidae"],
    "other_clades": ["Helicidae"],
    "unseen_negative_clades": ["Lumbricidae"],
    "merge_groups": [{"label": "Arion ater agg.", "members": ["Arion ater", "Arion rufus"]}],
    "quality_grades": ["research"],
    "licenses": ["CC0", "CC-BY"],
    "min_photos": 5,
    "max_images_per_class": 40,
    "max_images_other": 40,
    "photo_size": "medium",
    "n_folds": 5,
    "seed": 0,
}


def _write(path, header, rows):
    with gzip.open(path, "wt") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")


@pytest.fixture
def metadata_dir(tmp_path):
    parent = {t[0]: t[1] for t in TREE}

    def ancestry(tid):
        chain, p = [], parent[tid]
        while p:
            chain.insert(0, p)
            p = parent[p]
        return "/".join(chain)

    _write(
        tmp_path / "taxa.csv.gz",
        ["taxon_id", "ancestry", "rank_level", "rank", "name", "active"],
        [(t, ancestry(t), lvl, rank, name, "true") for t, _, rank, lvl, name in TREE],
    )

    obs, photos, pid = [], [], 0
    for tid, n in OBS_PER_TAXON.items():
        for i in range(n):
            uuid = f"o{tid}-{i}"
            observer = f"u{i % 12}"
            obs.append((uuid, observer, 1.0, 2.0, 50, tid, "research", "2024-01-01", 1.0))
            # 2 photos per observation; the second has a non-allowed license
            for pos, lic in [(0, "CC-BY"), (1, "CC-BY-ND")]:
                pid += 1
                photos.append((f"p{pid}", pid, uuid, observer, "jpg", lic, 500, 400, pos))
        # extra non-research observation that must be filtered out
        uuid = f"o{tid}-casual"
        obs.append((uuid, "u0", 1.0, 2.0, 50, tid, "casual", "2024-01-01", 1.0))
        pid += 1
        photos.append((f"p{pid}", pid, uuid, "u0", "jpg", "CC0", 500, 400, 0))

    _write(
        tmp_path / "observations.csv.gz",
        [
            "observation_uuid",
            "observer_id",
            "latitude",
            "longitude",
            "positional_accuracy",
            "taxon_id",
            "quality_grade",
            "observed_on",
            "anomaly_score",
        ],
        obs,
    )
    _write(
        tmp_path / "photos.csv.gz",
        [
            "photo_uuid",
            "photo_id",
            "observation_uuid",
            "observer_id",
            "extension",
            "license",
            "width",
            "height",
            "position",
        ],
        photos,
    )
    return tmp_path


def test_labels_merge_rollup_and_rare_class_drop(metadata_dir):
    m = build(metadata_dir, CFG)
    slugs = m[m["group"] == "slug"]
    # Arion ater, A. rufus and the A. ater subspecies merge; Limax cinereoniger (3 photos) drops.
    assert set(slugs["label"]) == {"Arion ater agg.", "Limax maximus"}
    assert (slugs["label"] == "Arion ater agg.").sum() == 40  # 15+15+10
    assert set(m.loc[m["group"] == "other", "label"]) == {"other"}
    assert set(m.loc[m["group"] == "ood", "split"]) == {"ood"}
    assert set(m.loc[m["group"] != "ood", "split"]) == {"train", "val", "test"}


def test_quality_and_license_filters(metadata_dir):
    m = build(metadata_dir, CFG)
    assert not m["observation_uuid"].str.endswith("casual").any()
    assert set(m["license"]) == {"CC-BY"}


def test_min_photos_counts_photos_not_observations(metadata_dir):
    # Without the license filter, L. cinereoniger has 6 photos from only 3 observations.
    cfg = {k: v for k, v in CFG.items() if k != "licenses"}
    m = build(metadata_dir, cfg)
    assert (m["label"] == "Limax cinereoniger").sum() == 6
    m = build(metadata_dir, {**cfg, "min_photos": 7})
    assert "Limax cinereoniger" not in set(m["label"])


def test_no_license_filter_keeps_all_licenses(metadata_dir):
    cfg = {k: v for k, v in CFG.items() if k != "licenses"}
    m = build(metadata_dir, cfg)
    assert {"CC-BY", "CC-BY-ND"} <= set(m["license"])


def test_no_observer_shared_between_splits(metadata_dir):
    m = build(metadata_dir, CFG)
    observers = {s: set(g["observer_id"]) for s, g in m.groupby("split")}
    assert not observers["train"] & observers["val"]
    assert not observers["train"] & observers["test"]
    assert not observers["val"] & observers["test"]


def test_cap_per_class(metadata_dir):
    cfg = {**CFG, "max_images_per_class": 12, "max_images_other": 7}
    counts = build(metadata_dir, cfg).groupby("label").size()
    assert counts["Arion ater agg."] == 12
    assert counts["Limax maximus"] == 12
    assert counts["other"] == 7


def test_deterministic(metadata_dir):
    assert build(metadata_dir, CFG).equals(build(metadata_dir, CFG))


def test_unresolved_clade_fails_loudly(metadata_dir):
    with pytest.raises(ValueError, match="Agriolimacidae"):
        build(metadata_dir, {**CFG, "slug_clades": ["Arionidae", "Agriolimacidae"]})


def test_missing_column_fails_loudly(metadata_dir):
    _write(metadata_dir / "taxa.csv.gz", ["taxon_id", "name"], [("1", "Animalia")])
    with pytest.raises(ValueError, match="missing columns"):
        build(metadata_dir, CFG)


def test_parse_ancestry_separators():
    assert parse_ancestry("1/2/3") == ["1", "2", "3"]
    assert parse_ancestry("1\\2\\3") == ["1", "2", "3"]
    assert parse_ancestry(None) == []


def test_normalize_quality():
    import pandas as pd

    s = pd.Series(["Research Grade", "research", "Needs ID", "needs_id", "Casual"])
    assert list(normalize_quality(s)) == ["research", "research", "needs_id", "needs_id", "casual"]
