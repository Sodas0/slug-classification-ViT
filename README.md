# slugid

Slug species classifier: BioCLIP ViT-B/16 fine-tuned on iNaturalist open data, with species/genus/family heads and an optional location prior. 223 classes. Validation top-1 is 94.2% (macro 94.1%), and top-5 is 99.1%.

| | |
|:-:|:-:|
| <img src="assets/arion_fuscus.jpg" width="360"> | <img src="assets/hexabranchus_lacer.jpg" width="360"> |
| **Arion fuscus** 0.79 · Arion subfuscus 0.10 | **Hexabranchus lacer** 0.90 |

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,train]"
pytest -q
```

## Pipeline

```bash
python -m slugid.data.build_manifest --config configs/data.yaml --metadata-dir <inat-metadata> --out data/manifest.parquet
python -m slugid.data.download --manifest data/manifest.parquet --image-dir data/images --out data/manifest_clean.parquet
python -m slugid.train --config configs/train_bioclip.yaml --manifest data/manifest_clean.parquet --image-dir data/images --out data/runs/bioclip
python -m slugid.geo --manifest data/manifest_clean.parquet --classes-from data/runs/bioclip/best.pt --val-logits data/runs/bioclip/val_logits.pt --out data/runs/geo
```
