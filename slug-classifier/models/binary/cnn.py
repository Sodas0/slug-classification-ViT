"""
Binary classifier CNN model for distinguishing slugs from non-slugs.

This module should implement:
1. BinarySlugClassifier class - A CNN-based model for binary classification
2. Supporting functions for creating and configuring the classifier

Implementation guidelines:
- Use a lightweight CNN backbone (MobileNetV3, EfficientNet-B0, ResNet18)
- Support pre-trained weights and backbone freezing
- Add a custom classification head with proper regularization
- Include prediction functionality with confidence scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import os
import sys
from datetime import datetime
from torch.utils.data import DataLoader
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from SLUGdata.dataset import create_data_loaders

class BinarySlugClassifier(nn.Module):
    '''
    Binary classifier to identify slug vs. non-slug images.
    
    Uses MobileNetV3-Small pre-trained on ImageNet with a custom classification head.
    
    Args:
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone network
        dropout_rate: Dropout rate for classification head
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
    '''
    
    def __init__(self, pretrained=True, freeze_backbone=False, 
                 dropout_rate=0.2, learning_rate=1e-4, weight_decay=1e-5,
                 checkpoint_dir="models/binary/checkpoints",
                 log_dir="logs/binary"):
        super().__init__()
        
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights, progress=True)
    
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # replace classifier head with the binary classifier
        
        in_features = self.backbone.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 1)
        )
        
        self.backbone.classifier = self.classifier
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def forward(self, x):
        """Forward pass of the model."""
        return self.backbone(x)
    
    def predict(self, x, return_confidence=False):
        """
        Make predictions with optional confidence scores.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_confidence: Whether to return confidence scores
            
        Returns:
            If return_confidence is False:
                Tensor of shape (B,) with binary predictions (0 or 1)
            If return_confidence is True:
                Tuple of (predictions, confidences)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            
            if return_confidence:
                confidences = torch.max(probabilities, 1 - probabilities)
                return predictions.squeeze(), confidences.squeeze()
            return predictions.squeeze()
    
    def save_checkpoint(self, val_loss, is_best=False):
        """Save a checkpoint of the model."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.state_dict(),
            'val_loss': val_loss,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save model if it's the best so far
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.best_val_loss = val_loss
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint of the model."""
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        self.learning_rate = checkpoint['learning_rate']
        self.weight_decay = checkpoint['weight_decay']
    
   

def create_binary_classifier(config):
    '''
    Create a binary classifier from configuration.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        Configured BinarySlugClassifier
    '''
    return BinarySlugClassifier(
        pretrained=config.get('pretrained', True),
        freeze_backbone=config.get('freeze_backbone', False),
        dropout_rate=config.get('dropout_rate', 0.2),
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5),
        checkpoint_dir=config.get('checkpoint_dir', 'models/binary/checkpoints'),
        log_dir=config.get('log_dir', 'logs/binary')
    )



def main():
    # First, let's create the data loaders from the dataset module
    dataloaders_dict = create_data_loaders(
        root_dir='/Users/sohamdas/Desktop/myVit/slug-classifier/SLUGdata/interim/downloaded_images',
        metadata_file='/Users/sohamdas/Desktop/myVit/slug-classifier/SLUGdata/raw/slugs.csv',
        mode='binary',
        negative_dir='/Users/sohamdas/Desktop/myVit/slug-classifier/SLUGdata/interim/negative_examples',
        limit=100
        )
    
    #Replace config later with yaml
    config = {
        'pretrained': True,
        'freeze_backbone': False,
        'dropout_rate': .2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'checkpoint_dir': 'models/binary/checkpoints',
        'log_dir': 'logs/binary'
    }
    
    model = BinarySlugClassifier(config)
    model.train()  
    
    x = dataloaders_dict['train']
    print(isinstance(x, DataLoader))
    print(len(x))
    feature, label = next(iter(x))
    print(type(feature))
    print(type(label))
    print(f"Feature batch shape: {feature.size()}")
    print(f"Label batch shape: {len(label)}")
    
   # example pass from actual data
    
    output = model(feature)
    
    # print shapes
    print(f"Input shape: {feature.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (first few): {output[:2]}")
    
    # for binary classification, the output should be of shape [batch_size, 1]
    print("Forward pass test ok")
    
    
    

if __name__ == "__main__":
    main()