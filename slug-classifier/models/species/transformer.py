"""
Vision Transformer model for slug species classification.

This module should implement:
1. SlugSpeciesClassifier class - A Vision Transformer model for species classification
2. Supporting functions for creating and configuring the species classifier

Implementation:
- Use Vision Transformer architecture (ViT-B/16, ViT-B/32, (probably will use vit_b_16 cause training. but
    diff in params is like 2m, maybe will use 32 just for funsies)  
- Support pre-trained weights and backbone freezing - ViT needs to learn difference between slug species.
- Include confidence scoring and top-k predictions
- Main thing is to benchmark performance based on #imgs/class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import numpy as np
import sys
import os
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
import time
import matplotlib.pyplot as plt



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from speciesDATA.species_dataset import create_slug_species_data_loaders

# TODO: (order doesn't matter, just needs to get done)
    # create more organized and robust dataset and this time, include a sh script for
        # dataset download []
    # dataset analysis to count number of classes and images/class for research [x]
    # script for dataloader should be able to configure #imgs/class [x]
    # import and freeze vit_b_16 or vit_b_32 [x]
    # example pass of fake data in expected shape [x]
    # clean up repo -> organize data [x]
    # create script for entire build process []
    # save the index to class name mapp
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




def train_species_classifier(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, 
                           scheduler=None, checkpoint_dir='./checkpoints'):
    """
    Train the slug species classifier with a training loop.
    
    Args:
        model: The SlugSpeciesClassifier model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on ('cuda' or 'cpu')
        num_epochs: Number of training epochs
        scheduler: Learning rate scheduler (optional)
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Trained model and training history
    """
    
    print(f"Training on device: {device}")
    
    # creating checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # training stats for measurements
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'best_val_acc': 0.0, 'best_epoch': 0,
        'learning_rates': []
    }
    
    model = model.to(device)
    
    #### TRAINING LOOP ####
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Learning Rate: {current_lr:.6f}")
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # tqdm shenanigans
        train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
        
        # batch training
        for batch_idx, (images, targets) in enumerate(train_pbar):

            images, targets = images.to(device), targets.to(device)
            
            # hehe this is my favorite line to write
            optimizer.zero_grad()
            
            # forward, compute loss
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # BACKWARDS PROPAGATION OF ERRORS (its crazy how much math is happening in the background)
            loss.backward()
            
           
            
            optimizer.step()
            
            # update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(targets).sum().item()
            train_correct += batch_correct
            train_total += targets.size(0)
            
            # update progress bar
            batch_acc = 100. * batch_correct / targets.size(0)
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.2f}%',
                'correct': f'{batch_correct}/{targets.size(0)}'
            })
        
        # calculating training metrics for the epoch
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Class-wise accuracy tracking
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation", leave=False)
            
            for batch_idx, (images, targets) in enumerate(val_pbar):
                
                images, targets = images.to(device), targets.to(device)
                
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                val_correct += batch_correct
                val_total += targets.size(0)
                
                # tracking the class-wise accuracy
                for c in range(len(targets)):
                    label = targets[c].item()
                    class_correct[label] = class_correct.get(label, 0) + (predicted[c] == targets[c]).item()
                    class_total[label] = class_total.get(label, 0) + 1
                
                # update da progress bar
                batch_acc = 100. * batch_correct / targets.size(0)
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.2f}%'
                })
        
        # update val metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # display epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% ({train_correct}/{train_total})")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ({val_correct}/{val_total})")
        
        # display top 5 and bottom 5 classes by accuracy
        if len(class_total) >= 10:
            class_accuracies = {cls: 100.0 * class_correct.get(cls, 0) / total 
                               for cls, total in class_total.items() if total > 0}
            
            # get top and bottom classes 
            sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
            
            print("\n  Top 5 Classes:")
            for cls, acc in sorted_classes[:5]:
                print(f"    Class {cls}: {acc:.2f}% ({class_correct.get(cls, 0)}/{class_total.get(cls, 0)})")
                
            print("  Bottom 5 Classes:")
            for cls, acc in sorted_classes[-5:]:
                print(f"    Class {cls}: {acc:.2f}% ({class_correct.get(cls, 0)}/{class_total.get(cls, 0)})")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        
        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            history['best_epoch'] = epoch + 1
            
            # Save best model checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}_{timestamp}.pth')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history,
            }, checkpoint_path)
            
            print(f"  Best model saved: {checkpoint_path}")
        
        # save latest model
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
        }, latest_checkpoint_path)
    
    
    plot_training_history(history)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}% (Epoch {history['best_epoch']})")
    
    return model, history

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()
    

def main():
    # TODO:
        # implement argparse for ease of use and configurability
    image_dir = r"C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\positive_examples"
    csv_file = r"C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\raw\EVENMOREFILTEREDslug_data.csv"
    negative_dir = r"C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\negative_examples"
    
    train_loader, val_loader, stats = create_slug_species_data_loaders(image_dir=image_dir,
                                                                    csv_file=csv_file,
                                                                    negative_dir=negative_dir,
                                                                    batch_size=32,
                                                                    num_workers=1
                                                                    )

    num_classes = stats["dataset_info"]["num_classes"] if "num_classes" in stats["dataset_info"] else len(stats["dataset_info"]["classes"])
    print(f"Dataset contains {num_classes} classes")
    
        # configuration
    config = {
        'pretrained': True,
        'freeze_backbone': True,
        'dropout_rate': 0.2,
        'weight_decay': 1e-5,
        'learning_rate': 1e-4,
        'num_classes': num_classes,
        'batch_size': 32,
        'num_epochs': 10,
        'img_size': 224
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")
    model = create_species_classifier(config)
    model = model.to(device)
    
    ### BEFORE TRAINING STARTS, FOR MANUAL VALIDATION:
    # visualize a batch of training data,
    # measure dataloader performance,
    # compute batch statistics
   
    stats["visualize_batch"](train_loader, title="Training Samples")
    perf_stats = stats["measure_performance"](train_loader)
    batch_stats = stats["compute_batch_statistics"](train_loader)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # lr scheduler - ViTs probably do better with it
     
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # comment if wanna use cos annealing
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=config['num_epochs'], 
    #     eta_min=1e-6
    # )
    
    # create checkpoint directory
    checkpoint_dir = os.path.join('checkpoints', f'vit_slug_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    ### TRAINING MODEL ###
    trained_model, training_statistics = train_species_classifier(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=config['num_epochs'],
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir
    )
    
    print("training donezo")    
    
    # saving the final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'num_classes': config['num_classes'],
        'history': training_statistics
    }, final_model_path)
    
    print(f"Final model saved to: {final_model_path}")
    
    
if __name__ == "__main__":
    main()