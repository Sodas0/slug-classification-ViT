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
                 require_label=True, default_label=None, transform_type='train',
                 negative_dir=None):
        """
        Args:
            image_dir (string): Directory with all the images
            csv_file (string): Path to the CSV file with metadata
            transform (callable, optional): Optional transform for the images
            label_column (string): Name of the column containing the class labels
            require_label (bool): If True, only include samples with valid labels
            default_label (any): Default value to use when label is missing (if require_label=False)
            transform_type (string): Type of transfrom to use (train, val, test, none)
            negative_dir (string, optional): Directory containing negative examples (non-slug images)
        """
        self.image_dir = image_dir
        self.negative_dir = negative_dir
        self.metadata = pd.read_csv(csv_file)
        self.label_column = label_column

        if transform is not None:
            self.transform = transform
        else: 
            self.transform = self._get_transforms(transform_type)
        
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
        
        # Load negative examples if directory provided
        if negative_dir and os.path.exists(negative_dir):
            for filename in os.listdir(negative_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    self.negative_files.append(filename)
        
        # list of valid indices (both image exists and has valid metadata)
        self.valid_indices = []
        self.negative_indices = []
        
        # Process positive examples
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
        
        # Process negative examples
        if self.negative_files:
            self.negative_indices = list(range(len(self.valid_indices), 
                                             len(self.valid_indices) + len(self.negative_files)))
                
        print(f"Dataset created with {len(self.valid_indices)} positive samples and {len(self.negative_indices)} negative samples")
    
    def _get_transforms(self, transform_type):
        """
        Get standard transforms based on the specified type.
        
        Args:
            transform_type (string): Type of transform ('train', 'val', 'test', or 'none')
            
        Returns:
            torchvision.transforms.Compose: Composition of transforms
        """
        
        # normalize = transforms.Normalize(
        #     mean=[0,0,0],
        #     std=[1,1,1]
        # )
        
        # imagenet norm vals
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if transform_type == "none":
            return transforms.Compose([transforms.ToTensor(),])
        
        elif transform_type == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),  # slightly larger than final size
                transforms.RandomCrop(224),     # Random crop to 224x224
                transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
                transforms.RandomRotation(15),  # Random rotation by up to 15 degrees
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Slight color jitter
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
            # Handle positive example
            actual_idx = self.valid_indices[idx]
            img_name = os.path.join(self.image_dir, self.image_files[actual_idx])
            image = Image.open(img_name).convert('RGB')
            image_metadata = self.metadata.iloc[actual_idx]
            label = image_metadata[self.label_column]
            metadata = image_metadata.to_dict()
        else:
            # Handle negative example
            neg_idx = idx - len(self.valid_indices)
            img_name = os.path.join(self.negative_dir, self.negative_files[neg_idx])
            image = Image.open(img_name).convert('RGB')
            label = 0  # Negative class
            metadata = {'source': 'negative_examples', 'filename': self.negative_files[neg_idx]}

        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': label,
            'metadata': metadata,
            'idx': idx
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


def create_data_loaders(root_dir, metadata_file, batch_size=32, num_workers=4, 
                        train_transform=None, val_transform=None, test_transform=None,
                        label_column='common_name', mode='binary', val_split=0.15, test_split=0.15,
                        random_seed=42, negative_dir=None):
    '''
    Create data loaders for training and validation.
    
    Args:
        root_dir: Root directory containing images
        metadata_file: Path to metadata CSV file
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        train_transform: Custom transforms for training data (optional)
        val_transform: Custom transforms for validation data (optional)
        test_transform: Custom transforms for test data (optional)
        label_column: Column name containing the label information
        mode: 'binary' (slug vs non-slug) or 'species' (multi-class)
        val_split: Proportion of data to use for validation
        test_split: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
        negative_dir: Directory containing negative examples (non-slug images)
        
    Returns:
        Dictionary containing train, validation, and test data loaders and class mappings
    '''
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    metadata = pd.read_csv(metadata_file)
    
    # Create full dataset to gain access to valid indices without additional processing
    full_dataset = SlugDataset(
        image_dir=root_dir, 
        csv_file=metadata_file, 
        transform=None,
        label_column=label_column,
        require_label=True,
        transform_type='none',
        negative_dir=negative_dir
    )
    
    valid_indices = full_dataset.valid_indices
    negative_indices = full_dataset.negative_indices
   
    class_to_idx = {}
    if mode == 'binary':
        # For binary classification, map all slug species to 1, everything else to 0
        slug_species = metadata[metadata['common_name'].str.contains('Slug|slug', na=False)][label_column].unique()
        class_to_idx = {species: 1 if species in slug_species else 0 for species in metadata[label_column].unique()}
        class_to_idx['negative_example'] = 0  # Add negative examples to class mapping
    else:  # species mode
        # For species classification, create a mapping from species to class index
        unique_species = sorted(metadata[metadata.index.isin(valid_indices)][label_column].unique())
        class_to_idx = {species: idx for idx, species in enumerate(unique_species)}
    
    # Split positive examples
    train_pos_indices, temp_pos_indices = train_test_split(
        valid_indices, 
        test_size=val_split + test_split,
        random_state=random_seed,
        stratify=metadata.loc[valid_indices, label_column].map(class_to_idx).values if len(class_to_idx) > 1 else None
    )

    if test_split > 0:
        val_pos_indices, test_pos_indices = train_test_split(
            temp_pos_indices,
            test_size=test_split / (val_split + test_split),
            random_state=random_seed,
            stratify=metadata.loc[temp_pos_indices, label_column].map(class_to_idx).values if len(class_to_idx) > 1 else None
        )
    else:
        val_pos_indices = temp_pos_indices
        test_pos_indices = []

    # Split negative examples similarly
    if negative_indices:
        train_neg_indices, temp_neg_indices = train_test_split(
            negative_indices,
            test_size=val_split + test_split,
            random_state=random_seed
        )

        if test_split > 0:
            val_neg_indices, test_neg_indices = train_test_split(
                temp_neg_indices,
                test_size=test_split / (val_split + test_split),
                random_state=random_seed
            )
        else:
            val_neg_indices = temp_neg_indices
            test_neg_indices = []
    else:
        train_neg_indices = val_neg_indices = test_neg_indices = []

    # Create datasets
    train_dataset = SlugDataset(
        image_dir=root_dir,
        csv_file=metadata_file,
        transform=train_transform,
        label_column=label_column,
        require_label=True,
        transform_type='train',
        negative_dir=negative_dir
    )

    val_dataset = SlugDataset(
        image_dir=root_dir,
        csv_file=metadata_file,
        transform=val_transform,
        label_column=label_column,
        require_label=True,
        transform_type='val',
        negative_dir=negative_dir
    )
    
    # Create subsets using the split indices
    train_subset = Subset(train_dataset, 
                         [train_dataset.valid_indices.index(idx) for idx in train_pos_indices if idx in train_dataset.valid_indices] +
                         [train_dataset.negative_indices.index(idx) for idx in train_neg_indices if idx in train_dataset.negative_indices])
    
    val_subset = Subset(val_dataset,
                       [val_dataset.valid_indices.index(idx) for idx in val_pos_indices if idx in val_dataset.valid_indices] +
                       [val_dataset.negative_indices.index(idx) for idx in val_neg_indices if idx in val_dataset.negative_indices])
    
    # Create weighted sampler for balanced training
    if mode == 'binary' and negative_indices:
        # Calculate weights for balanced sampling
        train_labels = [1 if idx < len(train_pos_indices) else 0 for idx in range(len(train_subset))]
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    else:
        sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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
    
    result = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'class_to_idx': class_to_idx,
        'idx_to_class': {v: k for k, v in class_to_idx.items()},
        'num_classes': len(class_to_idx)
    }
    
    # Print dataset stats
    print(f"Dataset splits:")
    print(f"  Train: {len(train_subset)} samples ({len(train_pos_indices)} positive, {len(train_neg_indices)} negative)")
    print(f"  Val: {len(val_subset)} samples ({len(val_pos_indices)} positive, {len(val_neg_indices)} negative)")
    if test_split > 0:
        test_subset = Subset(val_dataset,
                           [val_dataset.valid_indices.index(idx) for idx in test_pos_indices if idx in val_dataset.valid_indices] +
                           [val_dataset.negative_indices.index(idx) for idx in test_neg_indices if idx in val_dataset.negative_indices])
        print(f"  Test: {len(test_subset)} samples ({len(test_pos_indices)} positive, {len(test_neg_indices)} negative)")
    
    print(f"Number of classes: {len(class_to_idx)}")
    
    if mode == 'species':
        print(f"Class distribution in training set:")
        train_labels = [metadata.iloc[train_dataset.valid_indices[sample_idx]][label_column] for sample_idx in train_subset.indices]
        for class_name, count in pd.Series(train_labels).value_counts().items():
            print(f"  {class_name}: {count} samples")

    return result


mydict = create_data_loaders(
    root_dir='/Users/sohamdas/Desktop/myVit/slug-classifier/data/interim/downloaded_images',
    metadata_file='/Users/sohamdas/Desktop/myVit/slug-classifier/data/raw/observations-542622.csv',
    batch_size=32,
    num_workers=4,)

