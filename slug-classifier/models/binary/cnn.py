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
import torch.backends.mps

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from SLUGdata.dataset import create_binary_data_loaders

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





def train_binary_classifier(model, train_loader, val_loader, criterion, optimizer, 
                           num_epochs=10, device='cuda' if torch.backends.mps.is_available() else 'cpu'):

    import torch.optim as optim
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    """
    Training loop for the Slug Classifier.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """

    model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch") as t:
            for inputs, targets in t:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets.float().unsqueeze(1))
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track loss
                train_loss += loss.item() * inputs.size(0)
                
                # Track predictions
                preds = (outputs > 0.5).float().cpu().detach().numpy()
                train_preds.extend(preds)
                train_targets.extend(targets.cpu().numpy())
                
                # Update tqdm description with current loss
                t.set_postfix(loss=loss.item())
        
        # Calculate epoch training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch") as t:
                for inputs, targets in t:
                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.float().unsqueeze(1))
                    
                    # Track loss
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Track predictions - convert to numpy for metrics calculation
                    preds = (outputs > 0.5).float().cpu().numpy()
                    
                    # Extend lists with numpy arrays
                    val_preds.extend(preds)
                    val_targets.extend(targets.cpu().numpy())
                    
                    # Update tqdm description with current loss
                    t.set_postfix(loss=loss.item())
        
        # Calculate epoch validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, zero_division=0)
        val_recall = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | '
              f'Val F1: {val_f1:.4f}')

    return model, history


def main():
    # First, let's create the data loaders from the dataset module
    dataloaders_dict = create_binary_data_loaders(
        root_dir='/Users/sohamdas/Desktop/Projects/myVit/slug-classifier/SLUGdata/interim/downloaded_images',
        metadata_file='/Users/sohamdas/Desktop/Projects/myVit/slug-classifier/SLUGdata/raw/slugs.csv',
        mode='binary',
        negative_dir='/Users/sohamdas/Desktop/Projects/myVit/slug-classifier/SLUGdata/interim/negative_examples')
    
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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'], weight_decay= config['weight_decay'])
    
    training_dataloader = dataloaders_dict['train']
    validation_dataloader = dataloaders_dict['val']
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    
    
    # for feature, label in training_dataloader:
    #     # Move data to device
    #     feature = feature.to(device)
    #     label = label.to(device)
        
        # Print statements for debugging
        # print(f"Feature batch shape: {feature.size()}")
        # print(f"Label batch shape: {label.size()}")
        # print(f"Feature data type: {feature.dtype}")
        # print(f"Label data type: {label.dtype}")
        # print(f"First few labels: {label[:5]}")
        
           
    model.train()  
    
    train_binary_classifier(
        model=model, 
        train_loader=training_dataloader, 
        val_loader=validation_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device)
    
        

#     x = dataloaders_dict['train']
#     print(isinstance(x, DataLoader))
#     print(len(x))
#     feature, label = next(iter(x))
#     print(type(feature))
#     print(type(label))
#     print(f"Feature batch shape: {feature.size()}")
#     print(f"Label batch shape: {len(label)}")
    
#    # example pass from actual data
    
#     output = model(feature)
    
#     # print shapes
#     print(f"Input shape: {feature.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Output values (first few): {output[:2]}")
    
#     # for binary classification, the output should be of shape [batch_size, 1]
#     print("Forward pass test ok")
    
    
    

if __name__ == "__main__":
    main()