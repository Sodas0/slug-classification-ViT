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

class SlugDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None, label_column='common_name', 
                 require_label=True, default_label=None):
        """
        Args:
            image_dir (string): Directory with all the images
            csv_file (string): Path to the CSV file with metadata
            transform (callable, optional): Optional transform for the images
            label_column (string): Name of the column containing the class labels
            require_label (bool): If True, only include samples with valid labels
            default_label (any): Default value to use when label is missing (if require_label=False)
        """
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        self.label_column = label_column
        
        # Create a mapping of image indices to their filenames
        self.image_files = {}
        for filename in os.listdir(image_dir):
            if filename.startswith("image_") and (filename.endswith(".jpg") or filename.endswith(".jpeg")):
                try:
                    index = int(filename.split("_")[1].split(".")[0]) # convoluted line to get the index number. but whatever works works
                    self.image_files[index] = filename
                except (ValueError, IndexError):
                    continue
        
        # Create a list of valid indices (both image exists and has valid metadata)
        self.valid_indices = []
        
        for idx in range(len(self.metadata)):
            image_exists = idx in self.image_files
            
            # Check the label (if required)
            if require_label:
                has_label = (idx < len(self.metadata) and 
                            self.label_column in self.metadata.columns and 
                            pd.notna(self.metadata.loc[idx, self.label_column]))
            else:
                has_label = True
                
            if image_exists and has_label:
                self.valid_indices.append(idx)
                
        print(f"Dataset created with {len(self.valid_indices)} valid samples out of {len(self.metadata)} total rows")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):

        actual_idx = self.valid_indices[idx]

        img_name = os.path.join(self.image_dir, self.image_files[actual_idx])
        image = Image.open(img_name).convert('RGB')
        image_metadata = self.metadata.iloc[actual_idx]
        label = image_metadata[self.label_column]

        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': label,
            'metadata': image_metadata.to_dict(),
            'idx': actual_idx  
        }


### UNIMPORTANT FUNCTION ###
def peek_dataset(dataset, num_samples=10, figsize=(15, 10)):
    """
    Visualize random samples from a dataset.
    
    Args:
        dataset: PyTorch dataset to visualize
        num_samples: Number of random samples to show
        figsize: Figure size (width, height) in inches
    """
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    fig, axs = plt.subplots(2, 5, figsize=figsize)
    axs = axs.flatten()
    
    # convert tensor to image for display
    def to_img(x):
        if isinstance(x, np.ndarray):
            return x
        # If a tensor, convert it
        if x.dim() == 3 and x.size(0) == 3:  # If normalized tensor (3,H,W)
            return x.permute(1, 2, 0).numpy()
        return x.numpy()
    
    # Display each sample
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img = sample['image']
        label = sample['label']
        metadata = sample['metadata']
        
       
        if hasattr(transforms, 'ToPILImage'):
            if isinstance(img, torch.Tensor):
                img_display = to_img(img)
            else:
                img_display = np.array(img)
        else:
            img_display = img
            

        axs[i].imshow(img_display)
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')
    
        print(f"Sample {i+1} (index {idx}, original idx {sample['idx']}):")
        print(f"  Label: {label}")
        print(f"  Image shape: {img.shape if hasattr(img, 'shape') else 'PIL Image'}")
        print(f"  Metadata highlights:")
        

        highlight_keys = ['common_name', 'scientific_name'] 
        highlights = {k: metadata[k] for k in highlight_keys if k in metadata}
        for key, value in highlights.items():
            print(f"    {key}: {value}")
        print()
    
    plt.tight_layout()
    plt.show()
    
    return indices

# test to ensure dataset exists
# dataset = SlugDataset(
#     image_dir="/Users/sohamdas/Desktop/myVit/slug-classifier/data/interim/downloaded_images",
#     csv_file="/Users/sohamdas/Desktop/myVit/slug-classifier/data/raw/observations-542622.csv",
#     label_column='common_name', 
#     require_label=True
# )

# random_indices = peek_dataset(dataset)

"""
def create_data_loaders(root_dir, metadata_file, batch_size=32, num_workers=4, 
                        train_transform=None, val_transform=None, mode='binary'):
    '''
    Create data loaders for training and validation.
    
    Args:
        root_dir: Root directory containing images
        metadata_file: Path to metadata file
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        train_transform: Transforms for training data
        val_transform: Transforms for validation data
        mode: 'binary' or 'species'
        
    Returns:
        Dictionary containing train and validation data loaders
    '''
    # TODO: Create datasets and data loaders
    pass
"""
