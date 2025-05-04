#!/usr/bin/env python3
"""
Simple prediction script for the slug classification system.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, vit_l_16
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Binary Classifier
class BinarySlugClassifier(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, dropout_rate=0.2):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 1)
        )
        self.backbone.classifier = self.classifier
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x, return_confidence=False):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            if return_confidence:
                return predictions.squeeze(), probabilities.squeeze()
            return predictions.squeeze()

# Species Classifier
class SlugSpeciesClassifier(nn.Module):
    def __init__(self, num_classes=1200, dropout_rate=0.2):
        super().__init__()
        self.backbone = vit_l_16(weights=None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, x, top_k=3):
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = F.softmax(logits, dim=1)
            top_k_confidences, top_k_indices = torch.topk(probabilities, k=top_k, dim=1)
            return top_k_indices, top_k_confidences

def load_model(model_path, model_class, device, **kwargs):
    """Simple function to load a model from a .pth file"""
    model = model_class(**kwargs)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict']) 
    else:
        # Try direct loading
        try:
            model.load_state_dict(checkpoint)
        except:
            print(f"Warning: Could not load checkpoint directly, trying to adapt keys")
            # If it's a full model saved with torch.save(model)
            if hasattr(checkpoint, 'state_dict'):
                model.load_state_dict(checkpoint.state_dict())
    
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Load and preprocess an image for model inference"""
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def predict_image(image_path, binary_model, species_model, device, class_mapping=None, threshold=0.5, top_k=3):
    """Run the prediction pipeline on a single image"""
    # Preprocess the image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Binary classification
    is_slug, slug_confidence = binary_model.predict(image_tensor, return_confidence=True)
    slug_confidence = float(slug_confidence.cpu().item())
    
    result = {
        "image_path": str(image_path),
        "is_slug": bool(slug_confidence >= threshold),
        "slug_confidence": slug_confidence,
        "species_predictions": []
    }
    
    # If it's a slug, predict species
    if result["is_slug"]:
        indices, confidences = species_model.predict(image_tensor, top_k=top_k)
        
        # Convert to Python types
        indices = indices.cpu().numpy().tolist()[0]
        confidences = confidences.cpu().numpy().tolist()[0]
        
        # Add species predictions
        for idx, conf in zip(indices, confidences):
            species_name = class_mapping.get(str(idx), f"species_{idx}") if class_mapping else f"species_{idx}"
            result["species_predictions"].append({
                "species_id": idx,
                "species_name": species_name,
                "confidence": conf
            })
    
    return result

def visualize_prediction(image_path, result, show=True, save_path=None):
    """Visualize prediction results"""
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    
    # Create caption text
    info = f"Slug: {result['is_slug']} ({result['slug_confidence']:.2f})"
    
    if result['is_slug'] and result['species_predictions']:
        for i, pred in enumerate(result['species_predictions']):
            info += f"\n{i+1}. {pred['species_name']} ({pred['confidence']:.2f})"
    
    plt.title(info, fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Slug classification prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--binary_model', type=str, required=True, help='Path to binary model .pth file')
    parser.add_argument('--species_model', type=str, required=True, help='Path to species model .pth file')
    parser.add_argument('--class_mapping', type=str, help='Path to class mapping JSON file')
    parser.add_argument('--num_classes', type=int, default=1200, help='Number of species classes')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--top_k', type=int, default=3, help='Number of species predictions')
    parser.add_argument('--visualize', action='store_true', help='Show prediction visualization')
    parser.add_argument('--output', type=str, help='Path to save results JSON')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    try:
        print(f"Loading binary model from {args.binary_model}")
        binary_model = load_model(args.binary_model, BinarySlugClassifier, device)
        
        print(f"Loading species model from {args.species_model}")
        species_model = load_model(args.species_model, SlugSpeciesClassifier, 
                                  device, num_classes=args.num_classes)
        
        # Load class mapping if provided
        class_mapping = None
        if args.class_mapping and os.path.exists(args.class_mapping):
            with open(args.class_mapping, 'r') as f:
                class_mapping = json.load(f)
                print(f"Loaded class mapping with {len(class_mapping)} classes")
    
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Run prediction
    result = predict_image(
        args.image, binary_model, species_model, device,
        class_mapping, args.threshold, args.top_k
    )
    
    # Print and save results
    print("\nPrediction Results:")
    print(f"Image: {result['image_path']}")
    print(f"Is Slug: {result['is_slug']} (Confidence: {result['slug_confidence']:.4f})")
    
    if result['is_slug'] and result['species_predictions']:
        print("Species Predictions:")
        for i, pred in enumerate(result['species_predictions'], 1):
            print(f"  {i}. {pred['species_name']} (Confidence: {pred['confidence']:.4f})")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Visualize if requested
    if args.visualize:
        visualize_prediction(args.image, result)

if __name__ == '__main__':
    main()