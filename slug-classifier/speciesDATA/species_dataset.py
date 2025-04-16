"""
Dataset classes for the slug classification system.

This module should implement:
1. SlugDataset class - A PyTorch Dataset for slug images with binary or multi-class labels
2. Data loading utilities - Functions to create data loaders for training and evaluation

Implementation guidelines:
- Support both binary classification (slug vs. non-slug) and species classification
- Handle different data splits (train, validation, test)
- Apply appropriate transformations based on the split
- Generate class mappings for species classification
"""

# TODO: Implementation
    # maybe create class folders for each class. is definitely better than just raw images 🤣

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

### Unimportant imports ###
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
###                 ###

class SlugDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None, transform_type='train',
                 negative_dir=None, max_samples=None):
        """
        Dataset for multi-class slug species classification using common_name as labels.
        
        Args:
            image_dir (string): Directory with all the images
            csv_file (string): Path to the CSV file with metadata
            transform (callable, optional): Optional transform for the images
            transform_type (string): Type of transform to use (train, val, test, none)
            negative_dir (string, optional): Directory containing negative examples
            max_samples (int, optional): Maximum number of samples to include
        """
        self.image_dir = image_dir
        self.negative_dir = negative_dir
        self.metadata = pd.read_csv(csv_file)
        self.max_samples = max_samples

        if transform is not None:
            self.transform = transform
        else: 
            self.transform = self._get_transforms(transform_type)
        
        # Create a mapping from common_name to numerical indices
        unique_labels = sorted(self.metadata['common_name'].dropna().unique())
        self.classes = unique_labels
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(f"Found {len(self.classes)} unique slug species")
        
        # mapping image indices to the correct filenames
        self.image_files = {}
        self.negative_files = []
        
        # Load positive examples
        for filename in os.listdir(image_dir):
            if filename.startswith("image_") and (filename.endswith(".jpg") or filename.endswith(".jpeg")):
                try:
                    index = int(filename.split("_")[1].split(".")[0])
                    self.image_files[index] = filename
                except (ValueError, IndexError):
                    continue
        
        if negative_dir and os.path.exists(negative_dir):
            for filename in os.listdir(negative_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    self.negative_files.append(filename)
        
        # list of valid indices (both image exists and has valid metadata)
        self.valid_indices = []
        self.negative_indices = []
        
        # Process positive examples - require common_name to be present and in our class list
        for idx in range(len(self.metadata)):
            image_exists = idx in self.image_files
            
            # Check if the common_name is valid
            has_valid_label = (idx < len(self.metadata) and 
                              'common_name' in self.metadata.columns and 
                              pd.notna(self.metadata.loc[idx, 'common_name']) and
                              self.metadata.loc[idx, 'common_name'] in self.class_to_idx)
                
            if image_exists and has_valid_label:
                self.valid_indices.append(idx)
        
        # Process negative examples if any
        if self.negative_files:
            self.negative_indices = list(range(len(self.valid_indices), 
                                             len(self.valid_indices) + len(self.negative_files)))
            # Add negative class to class mapping
            if "non-slug" not in self.class_to_idx:
                self.classes.append("non-slug")
                self.class_to_idx["non-slug"] = len(self.classes) - 1
        
        # Apply max_samples limit if specified
        if self.max_samples is not None and len(self.valid_indices) + len(self.negative_indices) > self.max_samples:
            import random
            # If we have too many samples, randomly select up to max_samples
            combined_indices = self.valid_indices + self.negative_indices
            selected_indices = random.sample(combined_indices, self.max_samples)
            
            # Split back into positive and negative
            self.valid_indices = [idx for idx in selected_indices if idx < len(self.valid_indices)]
            self.negative_indices = [idx for idx in selected_indices if idx >= len(self.valid_indices)]
                
        print(f"Dataset created with {len(self.valid_indices)} slug samples across {len(self.classes)} species")
        if self.negative_indices:
            print(f"Added {len(self.negative_indices)} non-slug samples")
        
        # Print class distribution
        if len(self.valid_indices) > 0:
            class_counts = {}
            for idx in self.valid_indices:
                class_name = self.metadata.loc[idx, 'common_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Sort by count and print top 10
            print("Class distribution (top 10):")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cls}: {count} samples")
    
    def _get_transforms(self, transform_type):
        """
        Get standard transforms based on the specified type.
        """

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if transform_type == "none":
            return transforms.Compose([transforms.ToTensor()])
        
        elif transform_type == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize
            ])
            
        elif transform_type == 'val' or transform_type == 'test':
            return transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                normalize
            ])
            
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    
    def __len__(self):
        return len(self.valid_indices) + len(self.negative_indices)
    
    def __getitem__(self, idx):
        if idx < len(self.valid_indices):
            # Handle slug sample (positive)
            actual_idx = self.valid_indices[idx]
            img_name = os.path.join(self.image_dir, self.image_files[actual_idx])
            image = Image.open(img_name).convert('RGB')
            
            # Get common_name and convert to class index
            common_name = self.metadata.loc[actual_idx, 'common_name']
            label = self.class_to_idx[common_name]
        else:
            # Handle non-slug sample (negative)
            neg_idx = idx - len(self.valid_indices)
            img_name = os.path.join(self.negative_dir, self.negative_files[neg_idx])
            image = Image.open(img_name).convert('RGB')
            label = self.class_to_idx["non-slug"]

        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_slug_species_data_loaders(image_dir, csv_file, batch_size=32, num_workers=4, 
                                  val_split=0.15, test_split=0.15, random_seed=42, 
                                  negative_dir=None, max_samples=None):
    """
    Create data loaders for multi-class slug species classification.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create a dataset to get class info and indices
    temp_dataset = SlugDataset(
        image_dir=image_dir, 
        csv_file=csv_file, 
        transform_type='none',
        negative_dir=negative_dir,
        max_samples=max_samples
    )
    
    # Get all indices
    all_indices = list(range(len(temp_dataset)))
    
    # Get all labels for stratified sampling
    all_labels = []
    for i in all_indices:
        _, label = temp_dataset[i]
        all_labels.append(label)
    
    # Split train/validation/test with stratification
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=val_split + test_split,
        random_state=random_seed,
        stratify=all_labels  # Ensure class balance
    )
    
    # Get labels for the temporary indices
    temp_labels = [all_labels[i] for i in range(len(all_labels)) if i in temp_indices]
    
    if test_split > 0:
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=test_split / (val_split + test_split),
            random_state=random_seed,
            stratify=temp_labels  # Ensure class balance
        )
    else:
        val_indices = temp_indices
        test_indices = []
    
    # Create datasets with appropriate transforms
    train_dataset = SlugDataset(
        image_dir=image_dir,
        csv_file=csv_file,
        transform_type='train',
        negative_dir=negative_dir,
        max_samples=max_samples
    )
    
    val_dataset = SlugDataset(
        image_dir=image_dir,
        csv_file=csv_file,
        transform_type='val',
        negative_dir=negative_dir,
        max_samples=max_samples
    )
    
    # Create subsets using indices
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create a weighted sampler to handle class imbalance
    train_labels = []
    for idx in train_indices:
        _, label = train_dataset[idx]
        train_labels.append(label)
    
    # Count occurrences of each class
    class_counts = np.bincount(train_labels)
    
    # Compute weights - less frequent classes get higher weights
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create test dataset and loader if needed
    if test_split > 0:
        test_dataset = SlugDataset(
            image_dir=image_dir,
            csv_file=csv_file,
            transform_type='test',
            negative_dir=negative_dir,
            max_samples=max_samples
        )
        test_subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        test_loader = None
    
    # Print dataset information
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_subset)} samples")
    print(f"  Val: {len(val_subset)} samples")
    if test_loader:
        print(f"  Test: {len(test_subset)} samples")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': len(train_dataset.classes)
    }