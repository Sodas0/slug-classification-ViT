"""
Dataset classes for the slug classification system.

This module should implement:
1. SlugDataset class - A PyTorch Dataset for slug images with multi-class labels
2. Data loading utilities - Functions to create data loaders for training and evaluation

Implementation guidelines:
- Support both binary classification (slug vs. non-slug) and species classification
- Handle different data splits (train, validation, test)
- Apply appropriate transformations based on the split
- Generate class mappings for species classification
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np

class SlugDataset(Dataset):
    
    def __init__(self, image_dir, csv_file, negative_dir=None, transform=None, transform_type='train', max_samples=None):
        """
        Dataset for multi-class slug species classification using common_name as labels.
        
        Args:
            image_dir (string): Directory with all the images
            csv_file (string): Path to the CSV file with metadata
            negative_dir (string, optional): Directory containing negative examples
            transform (callable, optional): Optional transform for the images
            transform_type (string): Type of transform to use (train, val, none)
            max_samples (int, optional): Maximum number of samples PER CLASS to include
        """
        self.image_dir = image_dir
        self.negative_dir = negative_dir
        self.metadata = pd.read_csv(csv_file) # metadata file is used to associate labels with positive examples
        self.max_samples_per_class = max_samples  # Renamed to clarify purpose

        if transform is not None:
            self.transform = transform
        else: 
            self.transform = self._get_transforms(transform_type)
        
        # mapping from common_name to numerical indices for training
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
        
        # Add "non-slug" class if we have negative examples
        if self.negative_files:
            if "non-slug" not in self.class_to_idx:
                self.classes.append("non-slug")
                self.class_to_idx["non-slug"] = len(self.classes) - 1
        
        # Organize samples by class
        self.samples_by_class = {cls: [] for cls in self.class_to_idx.keys()}
        
        # Process positive examples - require common_name to be present and in our class list
        for idx in range(len(self.metadata)):
            image_exists = idx in self.image_files
            
            # Check if the common_name is valid
            has_valid_label = (idx < len(self.metadata) and 
                            'common_name' in self.metadata.columns and 
                            pd.notna(self.metadata.loc[idx, 'common_name']) and
                            self.metadata.loc[idx, 'common_name'] in self.class_to_idx)
                
            if image_exists and has_valid_label:
                class_name = self.metadata.loc[idx, 'common_name']
                self.samples_by_class[class_name].append(idx)
        
        # Process negative examples
        if self.negative_files:
            for neg_idx, _ in enumerate(self.negative_files):
                self.samples_by_class["non-slug"].append(f"neg_{neg_idx}")  # Use a prefix to distinguish negative examples
        
        # Balance the dataset to have max_samples_per_class from each class
        self.valid_indices = []
        self.negative_indices = []
        
        if self.max_samples_per_class is not None:
            for class_name, indices in self.samples_by_class.items():
                if len(indices) == 0:
                    print(f"Warning: Class '{class_name}' has 0 samples and will be skipped")
                    continue
                    
                # If we have more samples than max_samples_per_class, randomly select exactly that many
                if len(indices) > self.max_samples_per_class:
                    import random
                    selected_indices = random.sample(indices, self.max_samples_per_class)
                else:
                    # If we have fewer samples, use all of them with optional oversampling
                    if len(indices) < self.max_samples_per_class:
                        print(f"Warning: Class '{class_name}' has only {len(indices)} samples, " 
                            f"which is less than max_samples_per_class ({self.max_samples_per_class})")
                        
                        # for oversampling if I decide to do so.
                        # TODO:
                            # scrape web for images for the imbalanced classes and manually review them for optimal data.
                        """
                        import random
                        # Oversample by repeating examples
                        additional_needed = self.max_samples_per_class - len(indices)
                        oversampled_indices = random.choices(indices, k=additional_needed)
                        selected_indices = indices + oversampled_indices
                        """
                        
                        # If not oversampling, just use what we have
                        selected_indices = indices
                    else:
                        selected_indices = indices
                
                # Add the selected indices to our final indices lists
                if class_name == "non-slug":
                    for idx in selected_indices:
                        if isinstance(idx, str) and idx.startswith("neg_"):
                            neg_idx = int(idx.split("_")[1])
                            self.negative_indices.append(neg_idx)
                else:
                    self.valid_indices.extend(selected_indices)
        else:
            # If max_samples_per_class is None, use all samples
            for class_name, indices in self.samples_by_class.items():
                if class_name == "non-slug":
                    for idx in indices:
                        if isinstance(idx, str) and idx.startswith("neg_"):
                            neg_idx = int(idx.split("_")[1])
                            self.negative_indices.append(neg_idx)
                else:
                    self.valid_indices.extend(indices)
        
        # Shuffle indices for better training
        import random
        random.shuffle(self.valid_indices)
        random.shuffle(self.negative_indices)
        
        print(f"Dataset created with {len(self.valid_indices)} slug samples across {len(self.classes) - (1 if 'non-slug' in self.class_to_idx else 0)} species")
        if self.negative_indices:
            print(f"Added {len(self.negative_indices)} non-slug samples")        
       
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
    
    def get_dataset_information(self, peek=False):
        """
        Returns and prints comprehensive information about the dataset.
        
        Returns:
            dict: Dictionary containing dataset statistics and information
        """
        # Initialize information dictionary
        info = {
            'total_samples': len(self),
            'positive_samples': len(self.valid_indices),
            'negative_samples': len(self.negative_indices),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'transform_type': str(self.transform),
            'class_distribution': {}
        }
        
        # Calculate class distribution for positive examples
        if len(self.valid_indices) > 0:
            class_counts = {}
            for idx in self.valid_indices:
                class_name = self.metadata.loc[idx, 'common_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            info['class_distribution'] = class_counts
        
        # Add non-slug count if present
        if len(self.negative_indices) > 0:
            info['class_distribution']['non-slug'] = len(self.negative_indices)
        
        # Print summary information
        print(f"Dataset Summary:")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Slug samples: {info['positive_samples']}")
        print(f"  Non-slug samples: {info['negative_samples']}")
        print(f"  Number of classes: {info['num_classes']}")
        
        # Sort class distribution by count and print top N
        top_n = 10
        print(f"\nClass distribution (top {min(top_n, len(info['class_distribution']))} of {len(info['class_distribution'])}):")
        for cls, count in sorted(info['class_distribution'].items(), key=lambda x: x[1], reverse=True)[:top_n]:
            percentage = (count / info['total_samples']) * 100
            print(f"  {cls}: {count} samples ({percentage:.1f}%)")
        
        # Sample an image and get its dimensions
        if len(self) > 0:
            try:
                sample_img, _ = self[0]
                if isinstance(sample_img, torch.Tensor):
                    info['image_dimensions'] = tuple(sample_img.shape)
                    print(f"\nImage dimensions: {info['image_dimensions']} (C×H×W)")
            except Exception as e:
                print(f"Could not determine image dimensions: {e}")
        if peek and len(self) > 0:
            import matplotlib.pyplot as plt
            import random
            import numpy as np
            import torchvision.transforms as T
            
            # Save the current transform temporarily
            original_transform = self.transform
            
            # Set a simple transform just for visualization purposes
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])
            
            # Select random indices
            num_samples = min(10, len(self))
            random_indices = random.sample(range(len(self)), num_samples)
            
            # Create a grid of subplots
            fig = plt.figure(figsize=(15, 8))
            rows, cols = 2, 5
            
            for i, idx in enumerate(random_indices):
                # Get the image and label
                img, label = self[idx]
                
                # Get the actual index in our dataset structure
                if idx < len(self.valid_indices):
                    actual_idx = self.valid_indices[idx]
                    class_name = self.metadata.loc[actual_idx, 'common_name']
                    sample_type = "Slug"
                else:
                    neg_idx = idx - len(self.valid_indices)
                    actual_idx = neg_idx
                    class_name = "non-slug"
                    sample_type = "Non-slug"
                
                # Convert tensor to numpy for plotting
                if isinstance(img, torch.Tensor):
                    img = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
                    # If values are in [0,1] range, rescale to [0,255]
                    if img.max() <= 1.0:
                        img = img * 255
                    img = img.astype(np.uint8)
                
                # Create subplot
                ax = fig.add_subplot(rows, cols, i+1)
                ax.imshow(img)
                
                # Set title with class info
                title = f"{sample_type}\nClass: {class_name}\nLabel: {label}\nIndex: {actual_idx}"
                ax.set_title(title, fontsize=8)
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Restore the original transform
            self.transform = original_transform
            print("\nDisplayed 10 random samples from the dataset")
    
            
        return info

def create_slug_species_data_loaders(image_dir, csv_file, negative_dir=None, max_samples=200, batch_size=32, num_workers=1, 
                                  val_split=0.20, random_seed=42,):
    """
    Create data loaders for multi-class slug species classification.
    
    Args:
        image_dir (str): Directory with all the images
        csv_file (str): Path to the CSV file with metadata
        negative_dir (str, optional): Directory containing negative examples
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        val_split (float): Fraction of data to use for validation (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader, dataset_stats)
    """
    import numpy as np
    import time
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    import matplotlib.pyplot as plt
    import torch
    import math
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create full dataset with training transforms
    full_dataset = SlugDataset(
        image_dir=image_dir,
        csv_file=csv_file,
        negative_dir=negative_dir,
        transform_type='train',
        max_samples=max_samples
    )
    
    # Get dataset information
    dataset_info = full_dataset.get_dataset_information()
    
    # Create indices for train/val split
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    
    split_idx = int(np.floor(val_split * len(full_dataset)))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    
    # Create validation dataset with validation transforms
    val_dataset = SlugDataset(
        image_dir=image_dir,
        csv_file=csv_file,
        negative_dir=negative_dir,
        transform_type='val',
        max_samples=max_samples
    )
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    # Optional: Create class-balanced sampler for training
    # If dataset is imbalanced and you want equal representation of classes
    if hasattr(full_dataset, 'class_to_idx') and hasattr(full_dataset, 'metadata'):
        # Calculate class weights
        class_counts = dataset_info.get('class_distribution', {})
        if class_counts:
            # Create weights for each sample in the dataset
            weights = []
            for idx in train_indices:
                if idx < len(full_dataset.valid_indices):
                    actual_idx = full_dataset.valid_indices[idx]
                    class_name = full_dataset.metadata.loc[actual_idx, 'common_name']
                else:
                    class_name = "non-slug"
                
                # Weight is inverse of class frequency
                weight = 1.0 / class_counts.get(class_name, 1)
                weights.append(weight)
            
            # Create sampler
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_indices),
                replacement=True
            )
        else:
            sampler = None
    else:
        sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,  # Only shuffle if not using sampler
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Use pinned memory if CUDA is available
        drop_last=False  # Keep all samples
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # ---- Visualization Hook Function ----
    def visualize_batch(loader, num_samples=8, title="Sample Batch"):
        """Visualize a batch of data with properly handled image normalization"""
        data_iter = iter(loader)
        images, labels = next(data_iter)
        
        # Get class names if available
        class_names = {v: k for k, v in full_dataset.class_to_idx.items()} if hasattr(full_dataset, 'class_to_idx') else {}
        
        # Create a grid of images
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        # Define denormalization function if images were normalized
        def denormalize(tensor):
            # Standard ImageNet normalization values
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return tensor * std + mean
        
        for i in range(min(num_samples, len(images))):
            # Apply denormalization
            img_tensor = denormalize(images[i]) if images[i].max() <= 1.0 else images[i]
            
            # Transpose from (C,H,W) to (H,W,C)
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            
            # Clip values to valid range for display
            img = np.clip(img, 0, 1.0)
            
            # Display image
            axes[i].imshow(img)
            label_name = class_names.get(labels[i].item(), f"Class {labels[i].item()}")
            axes[i].set_title(f"{label_name}")
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
        return fig  # Return the figure for saving if needed
    
    # ---- Performance Metrics Function ----
    def measure_dataloader_performance(loader, num_batches=10):
        """Measure data loading performance"""
        total_time = 0
        batch_times = []
        batch_sizes = []
        
        # Warm-up
        for _ in range(2):
            _ = next(iter(loader))
        
        # Measure
        for i, (images, labels) in enumerate(loader):
            if i == 0:
                start_time = time.time()
            elif i <= num_batches:
                end_time = time.time()
                batch_time = end_time - start_time
                total_time += batch_time
                batch_times.append(batch_time)
                batch_sizes.append(len(images))
                start_time = end_time
            else:
                break
        
        # Calculate statistics
        avg_time = total_time / min(num_batches, len(batch_times))
        avg_throughput = sum(batch_sizes) / total_time if total_time > 0 else 0
        
        stats = {
            "avg_batch_time_ms": avg_time * 1000,
            "throughput_samples_per_sec": avg_throughput,
            "batch_times_ms": [t * 1000 for t in batch_times],
            "batch_sizes": batch_sizes
        }
        
        print(f"DataLoader Performance:")
        print(f"  Avg batch load time: {stats['avg_batch_time_ms']:.2f} ms")
        print(f"  Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec")
        
        return stats
    
    # ---- Batch Statistics Function ----
    def compute_batch_statistics(loader, num_batches=10):
        """Compute statistics on batch data"""
        means = []
        stds = []
        class_counts = {}
        
        for i, (images, labels) in enumerate(loader):
            if i >= num_batches:
                break
                
            # Compute per-batch channel-wise mean and std
            batch_mean = torch.mean(images, dim=[0, 2, 3]).cpu().numpy()
            batch_std = torch.std(images, dim=[0, 2, 3]).cpu().numpy()
            means.append(batch_mean)
            stds.append(batch_std)
            
            # Count class occurrences
            for label in labels.cpu().numpy():
                label_name = full_dataset.classes[label] if hasattr(full_dataset, 'classes') else f"Class {label}"
                class_counts[label_name] = class_counts.get(label_name, 0) + 1
        
        # Aggregate statistics
        if means:
            overall_mean = np.mean(means, axis=0)
            overall_std = np.mean(stds, axis=0)
        else:
            overall_mean = np.zeros(3)
            overall_std = np.zeros(3)
        
        stats = {
            "channel_means": overall_mean.tolist(),
            "channel_stds": overall_std.tolist(),
            "class_distribution": class_counts
        }
        
        print(f"Batch Statistics:")
        print(f"  Channel means: {stats['channel_means']}")
        print(f"  Channel stds: {stats['channel_stds']}")
        print(f"  Class distribution: {stats['class_distribution']}")
        
        return stats
    
    # Create dataset statistics dict with helper functions
    dataset_stats = {
        "dataset_info": dataset_info,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "num_classes": len(full_dataset.classes) if hasattr(full_dataset, 'classes') else 0,
        "batch_size": batch_size,
        "visualize_batch": visualize_batch,
        "measure_performance": measure_dataloader_performance,
        "compute_batch_statistics": compute_batch_statistics
    }
    
    print(f"Created data loaders:")
    print(f"  Training: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"  Validation: {len(val_loader)} batches, {len(val_dataset)} samples")
    
    return train_loader, val_loader, dataset_stats



# [for testing + debug]
# if __name__ == "__main__":
#     image_dir = r"C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\positive_examples"
#     csv_file = r"C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\raw\EVENMOREFILTEREDslug_data.csv"
#     negative_dir = r"C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\negative_examples"

#         # Create data loaders
#     train_loader, val_loader, stats = create_slug_species_data_loaders(
#         image_dir=image_dir,
#         csv_file=csv_file,
#         negative_dir=negative_dir,
#         batch_size=32,
#         num_workers=1
#     )
    
#     # Visualize a batch of training data
#     stats["visualize_batch"](train_loader, title="Training Samples")
#     perf_stats = stats["measure_performance"](train_loader)
#     batch_stats = stats["compute_batch_statistics"](train_loader)