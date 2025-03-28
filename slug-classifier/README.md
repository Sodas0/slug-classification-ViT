# Slug Classification System

A two-stage classification system for identifying and classifying slug species from images.

## Project Overview

This system implements a two-stage approach to slug classification:
1. Binary classifier - Identifies whether an image contains a slug (versus non-slug)
2. Species classifier - Classifies the specific species of slug using a Vision Transformer

## Repository Structure

```
slug-classifier/
├── api/                # API service for model deployment
│   ├── __init__.py
│   ├── app.py          # FastAPI/Flask application
│   └── routes.py       # API endpoint definitions
│
├── configs/            # Configuration files
│   ├── binary_classifier.yaml
│   └── species_classifier.yaml
│
├── data/               # Data processing and dataset creation
│   ├── __init__.py
│   ├── augmentation.py # Data augmentation utilities
│   ├── dataset.py      # PyTorch dataset classes
│   ├── preprocessing.py # Data preprocessing utilities
│   ├── raw/            # Raw downloaded images
│   ├── interim/        # Intermediate processed data
│   └── processed/      # Final processed datasets
│
├── docs/               # Documentation
│
├── models/             # Model definitions
│   ├── __init__.py
│   ├── base.py         # Base model classes
│   ├── binary/         # Binary classification models
│   │   ├── __init__.py
│   │   └── cnn.py      # CNN binary classifier 
│   └── species/        # Species classification models
│       ├── __init__.py
│       └── transformer.py # Vision Transformer for species classification
│
├── notebooks/          # Jupyter notebooks for experimentation
│
├── scripts/            # Utility scripts
│   ├── extract_data.py       # Extract metadata from CSV
│   ├── download_images.py    # Download images from URLs
│   ├── preprocess_images.py  # Preprocess and standardize images
│   ├── train_binary.py       # Train binary classifier
│   ├── train_species.py      # Train species classifier
│   └── predict.py            # Run inference on new images
│
├── tests/              # Test files
│
├── utils/              # Utility functions
│   ├── __init__.py
│   ├── logging.py      # Logging utilities
│   ├── metrics.py      # Evaluation metrics
│   └── visualization.py # Visualization utilities
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Pipeline Workflow

1. **Data Extraction and Preparation**
   - Parse CSV to extract image URLs, scientific names, etc.
   - Download images with error handling and retries
   - Validate images (corrupt files, dimensions)
   - Standardize images (resize to 224×224, format)
   - Apply preprocessing and augmentation

2. **Data Analysis**
   - Calculate class distribution
   - Taxonomic classification hierarchy
   - Generate visualizations
   - Examine image quality across classes

3. **Dataset Creation**
   - 80/10/10 train/validation/test split
   - Stratified sampling to maintain class distribution

## Two-Stage Classification Pipeline

1. **Binary Classifier (Slug vs. Non-Slug)**
   - Lightweight CNN (MobileNetV3, EfficientNet-B0)
   - Confidence thresholding to reduce false positives

2. **Vision Transformer (Slug Species Classification)**
   - Pre-trained ViT with custom classification head
   - Hierarchical classification (family → genus → species)
   - Advanced data augmentation and regularization techniques

## Getting Started

See individual script files for documentation and usage examples.

### Installation

```bash
pip install -r requirements.txt
```

## License

[MIT License](LICENSE)
