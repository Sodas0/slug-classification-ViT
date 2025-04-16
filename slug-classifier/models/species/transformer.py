import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import TensorDataset

from speciesDATA.species_dataset import create_slug_species_data_loaders

# TODO: (order doesn't matter, just needs to get done)
    # create more organized and robust dataset and this time, include a zsh script for
        # dataset download []
    # dataset analysis to count number of classes and images/class for research []
    # script for dataloader should be able to configure #imgs/class []
    # import and freeze vit_b_16 or vit_b_32 [x]
    # example pass of fake data in expected shape [x]
    # clean up repo -> organize data []
    # etc. will write more as tasks get done based on needs.
    
        
class SlugSpeciesClassifier(nn.Module):
    '''
    Species classifier to identify species of slugs.
    
    Uses ViT-B/16.
    '''
    
    def __init__(self, pretrained=True, num_classes=1200, freeze_backbone=True, dropout_rate=0.2):
        super(SlugSpeciesClassifier, self).__init__()
        
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights, progress=True)
        
        in_features = self.backbone.heads.head.in_features
        
        # replace classifier with custom one
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # freeze backbone if flag
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # Unfreeze the head
            for param in self.backbone.heads.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
    
    
def create_species_classifier(config):
    '''
    Create a species classifier from configuration.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        Configured SlugSpeciesClassifier
    '''
    return SlugSpeciesClassifier(
        pretrained=config.get('pretrained', True),
        num_classes=config.get('num_classes', 10),  
        freeze_backbone=config.get('freeze_backbone', False),
        dropout_rate=config.get('dropout_rate', 0.2),
    )


#fake data for example pass
class SyntheticSlugDataset(Dataset):
    """
    Create a synthetic slug species dataset for testing.
    """
    def __init__(self, num_samples=1000, num_classes=100, img_size=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        
        
        self.images = torch.randn(num_samples, 3, img_size, img_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        
        
        for i in range(num_samples):
            class_idx = self.labels[i].item()
            row_pattern = (class_idx % 3) * (img_size // 3)
            col_pattern = (class_idx // 3) * (img_size // 3)
            # pattern for each class
            self.images[i, 0, row_pattern:row_pattern+8, col_pattern:col_pattern+8] += 2.0
            self.images[i, 1, row_pattern+4:row_pattern+12, col_pattern+4:col_pattern+12] += 2.0
            self.images[i, 2, row_pattern+2:row_pattern+10, col_pattern+2:col_pattern+10] += 2.0
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def create_data_loaders(dataset, batch_size=32, train_ratio=0.8):
    """
    Split dataset into training and validation sets and create data loaders.
    """
    # sizes
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_species_classifier(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5):
    """
    Train the species classifier.
    
    Args:
        model: The SlugSpeciesClassifier model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        num_epochs: Number of training epochs
    
    Returns:
        Trained model and training history
    """
    print(f"USING DEVICE: {device}")
    
    # tracking metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # training loop
    for epoch in range(num_epochs):
        # training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", 
                          unit="batch", position=0, leave=True)
        
        for inputs, targets in train_pbar:
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # zero the gradients
            optimizer.zero_grad()
            
            # forward 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # backward and optimize
            loss.backward()
            optimizer.step()
            

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            

            train_pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })
        
    
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        
        # store training metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", 
                        unit="batch", position=0, leave=True)
        
        with torch.no_grad():
            for inputs, targets in val_pbar:

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                

                val_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100.0 * correct / total
                })
        

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * correct / total
        

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return model, history


def main():
    # Configuration
    config = {
        'pretrained': True,
        'freeze_backbone': True,
        'dropout_rate': 0.2,
        'weight_decay': 1e-5,
        'learning_rate': 1e-4,
        'num_classes': 10,
        'batch_size': 32,
        'num_epochs': 3,
        'img_size': 224
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_species_classifier(config)
    model = model.to(device)
    
    print(model)
    
    
    # image_dir, csv_file, batch_size=32, num_workers=4, 
    #                               val_split=0.15, test_split=0.15, random_seed=42, 
    #                               negative_dir=None, max_samples=None
    data_dictionary = create_slug_species_data_loaders()
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train model
    trained_model, history = train_species_classifier(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=config['num_epochs']
    )
    
    print("Training donezo!")    

    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")


if __name__ == "__main__":
    main()