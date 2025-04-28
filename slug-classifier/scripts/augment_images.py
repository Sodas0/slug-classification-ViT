#!/usr/bin/env python3
# augment_images_rgb_fixed.py - Generate augmented images with guaranteed RGB format
# Fixes color space issues that could cause incorrect color shifts

import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import concurrent.futures
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImageAugmenter:
    """Class to handle image augmentation operations."""
    
    def __init__(self, augmentations_per_image=20, seed=None, conservative_color=True):
        """
        Initialize the image augmenter.
        
        Args:
            augmentations_per_image: Number of augmented versions to create for each image
            seed: Random seed for reproducibility
            conservative_color: Whether to use conservative color adjustments
        """
        self.augmentations_per_image = augmentations_per_image
        self.conservative_color = conservative_color
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _ensure_rgb(self, image):
        """
        Ensure the image is in RGB format.
        
        Args:
            image: Input image array
            
        Returns:
            RGB image array
        """
        # Check if image is grayscale (2D array)
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            rgb_image = np.stack([image] * 3, axis=2)
            return rgb_image
        
        # Check if image has an alpha channel (RGBA)
        if image.shape[2] == 4:
            # Remove alpha channel
            rgb_image = image[:, :, :3]
            return rgb_image
        
        # Image should already be RGB (3 channels)
        if image.shape[2] == 3:
            return image
            
        # For any other case, try to convert
        print(f"Warning: Unusual image format with shape {image.shape}. Attempting to convert to RGB.")
        try:
            pil_image = Image.fromarray(image)
            rgb_image = pil_image.convert('RGB')
            return np.array(rgb_image)
        except Exception as e:
            print(f"Error converting image to RGB: {e}")
            # Return original as fallback
            return image
        
    def _rotate_image(self, image, angle_range=(-30, 30)):
        """Rotate the image by a random angle."""
        angle = random.uniform(*angle_range)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def _adjust_brightness_contrast(self, image, 
                                   brightness_range=(0.9, 1.1), 
                                   contrast_range=(0.9, 1.1)):
        """Adjust brightness and contrast of the image."""
      
        if self.conservative_color:
            brightness_range = (0.95, 1.05)
            contrast_range = (0.95, 1.05)
            
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 5)
        # Ensure it's still RGB
        return self._ensure_rgb(adjusted)
    
    def _flip(self, image, horizontal=True):
        """Flip the image horizontally or vertically."""
        if horizontal:
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
    
    def _add_noise(self, image, noise_level=0.05):
        """Add random noise to the image."""
    
        if self.conservative_color:
            noise_level = 0.02
            
        noise = np.random.randn(*image.shape) * 255 * random.uniform(0, noise_level)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image
    
    def _perspective_transform(self, image, scale=0.1):
        """Apply perspective transformation to the image."""
     
        if self.conservative_color:
            scale = 0.05
            
        height, width = image.shape[:2]
        
        src_points = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
        
       
        dst_points = np.float32([
            [random.uniform(0, width * scale), random.uniform(0, height * scale)],
            [random.uniform(width * (1 - scale), width), random.uniform(0, height * scale)],
            [random.uniform(0, width * scale), random.uniform(height * (1 - scale), height)],
            [random.uniform(width * (1 - scale), width), random.uniform(height * (1 - scale), height)]
        ])
        
       
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(image, perspective_matrix, (width, height), 
                                         borderMode=cv2.BORDER_REFLECT)
        return transformed
    
    def _color_jitter(self, image, saturation_range=(0.8, 0.9), value_range=(0.9, 1.0)):
        """Apply color jittering to the image without changing the hue."""
        
        if self.conservative_color:
           
            saturation_range = (0.98, 1.02)
            # Very tight brightness range
            value_range = (0.98, 1.02)
        
            if random.random() < 0.9:
                return image
    
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        if random.random() < 0.7:
            value_factor = random.uniform(*value_range)
            r = np.clip(r * value_factor, 0, 255).astype(np.uint8)
            g = np.clip(g * value_factor, 0, 255).astype(np.uint8)
            b = np.clip(b * value_factor, 0, 255).astype(np.uint8)
        

        if random.random() < 0.5:
            saturation_factor = random.uniform(*saturation_range)
            
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            
            
            r = np.clip(luminance + (r - luminance) * saturation_factor, 0, 255).astype(np.uint8)
            g = np.clip(luminance + (g - luminance) * saturation_factor, 0, 255).astype(np.uint8)
            b = np.clip(luminance + (b - luminance) * saturation_factor, 0, 255).astype(np.uint8)
        
        
        jittered = np.stack([r, g, b], axis=2)
        return jittered
    
    def _crop_and_resize(self, image, crop_factor_range=(0.8, 0.95)):
        """Randomly crop the image and resize it back to original dimensions."""

        if self.conservative_color:
            crop_factor_range = (0.85, 0.95)
            
        height, width = image.shape[:2]
        original_size = (width, height)
        
        crop_factor = random.uniform(*crop_factor_range)
        
        crop_width = int(width * crop_factor)
        crop_height = int(height * crop_factor)
        
        max_x = width - crop_width
        max_y = height - crop_height
        x = random.randint(0, max(0, max_x))
        y = random.randint(0, max(0, max_y))
        
        
        cropped = image[y:y+crop_height, x:x+crop_width]
        
        resized = cv2.resize(cropped, original_size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _gaussian_blur(self, image, kernel_range=(3, 5)):
        """Apply Gaussian blur to the image."""
        if self.conservative_color:
            kernel_size = 3
        else:
            kernel_size = random.randrange(*kernel_range)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def _validate_color_integrity(self, original, augmented, threshold=0.3):
        """
        Check that the augmented image doesn't have dramatic color shifts.
        
        Args:
            original: Original image in RGB
            augmented: Augmented image in RGB
            threshold: Maximum allowed color difference ratio
            
        Returns:
            True if colors are within acceptable range, False otherwise
        """
       
        orig_avg = np.mean(original, axis=(0, 1))
        
        aug_avg = np.mean(augmented, axis=(0, 1))
        

        rel_diff = np.abs(orig_avg - aug_avg) / (orig_avg + 1e-10)  
    
        max_diff = np.max(rel_diff)

        return max_diff < threshold
        
    def augment_image(self, image):
        """
        Apply a random combination of augmentations to the image.
        
        Args:
            image: Input image (numpy array in RGB format)
            
        Returns:
            Augmented image (in RGB format)
        """
        # Ensure input image is RGB
        image = self._ensure_rgb(image)
        
        original = image.copy()
        
        augmented = image.copy()
        
        if random.random() < 0.7:
            augmented = self._rotate_image(augmented)
            
        if random.random() < 0.7:
            augmented = self._adjust_brightness_contrast(augmented)
            
        if random.random() < 0.5:
            augmented = self._flip(augmented, horizontal=True)
            
        if random.random() < 0.3:
            augmented = self._flip(augmented, horizontal=False)
            
        if random.random() < 0.3:
            augmented = self._add_noise(augmented)
            
        if random.random() < 0.4:
            augmented = self._perspective_transform(augmented)
            
       
        prob = 0.3 if self.conservative_color else 0.6
        if random.random() < prob:
            augmented = self._color_jitter(augmented)
            
        if random.random() < 0.6:
            augmented = self._crop_and_resize(augmented)
            
        if random.random() < 0.3:
            augmented = self._gaussian_blur(augmented)
        
        # Final check to ensure output is RGB
        augmented = self._ensure_rgb(augmented)
        
        if self.conservative_color:
            attempts = 0
            max_attempts = 3
            
            while not self._validate_color_integrity(original, augmented) and attempts < max_attempts:
               
                augmented = original.copy()
    
                if random.random() < 0.7:
                    augmented = self._rotate_image(augmented)
                    
                if random.random() < 0.5:
                    augmented = self._flip(augmented, horizontal=True)
                    
                if random.random() < 0.4:
                    augmented = self._perspective_transform(augmented)
                    
                if random.random() < 0.6:
                    augmented = self._crop_and_resize(augmented)
                
                # Ensure it's still RGB
                augmented = self._ensure_rgb(augmented)
                
                attempts += 1
            
           
            if not self._validate_color_integrity(original, augmented):
                augmented = original.copy()
                
                if random.random() < 0.5:
                    augmented = self._flip(augmented, horizontal=True)
                else:
                    angle = random.uniform(-15, 15)  
                    height, width = augmented.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    augmented = cv2.warpAffine(augmented, rotation_matrix, (width, height), 
                                              borderMode=cv2.BORDER_REFLECT)
                
                # Final check to ensure output is RGB
                augmented = self._ensure_rgb(augmented)
        
        return augmented
    
    def generate_augmentations(self, image_path):
        """
        Generate multiple augmented versions of an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of augmented images (numpy arrays)
        """
        try:
            # Read image using PIL and explicitly convert to RGB
            pil_image = Image.open(str(image_path)).convert('RGB')
            image = np.array(pil_image)
            
            if image is None or image.size == 0:
                print(f"Warning: Could not read image {image_path}")
                return [], None
                
            # Verify the image is RGB
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = self._ensure_rgb(image)
                
            augmented_images = []
            for _ in range(self.augmentations_per_image):
                augmented = self.augment_image(image)
                augmented_images.append(augmented)
                
            return augmented_images, image 
        except Exception as e:
            print(f"Error augmenting image {image_path}: {e}")
            return [], None

def process_species_folder(species_folder, output_dir, label_data, augmenter, start_index):
    """
    Process all images in a species folder.
    
    Args:
        species_folder: Path to the species folder
        output_dir: Directory to save augmented images
        label_data: List to collect label information
        augmenter: ImageAugmenter instance
        start_index: Starting index for image numbering
        
    Returns:
        Next index to use
    """
    
    species_name = os.path.basename(species_folder)
    print(f"\nProcessing species: {species_name}")
  
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(species_folder).glob(f"*{ext}")))
        image_files.extend(list(Path(species_folder).glob(f"*{ext.upper()}")))
    
    
    image_files.sort()
    
    current_index = start_index
    
    
    for img_path in tqdm(image_files, desc=f"Augmenting {species_name}", leave=False):
       
        augmented_images, _ = augmenter.generate_augmentations(img_path)
        
        
        for aug_img in augmented_images:
            if aug_img is None or len(aug_img) == 0:
                continue
                
            
            pil_img = Image.fromarray(aug_img)
            
            # Ensure the PIL image is RGB before saving
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            output_filename = f"image_{current_index}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            
            pil_img.save(output_path, quality=95)
            
           
            label_data.append({
                'index': current_index,
                'common_name': species_name,
                'original_image': os.path.basename(str(img_path))
            })
            
        
            current_index += 1
    
    return current_index

def preview_augmentations(image_path, num_augmentations=9, conservative_color=True):
    """
    Generate and display augmentations for a single image.
    
    Args:
        image_path: Path to the image file
        num_augmentations: Number of augmentations to display
        conservative_color: Whether to use conservative color adjustments
    """
   
    augmenter = ImageAugmenter(augmentations_per_image=num_augmentations, conservative_color=conservative_color)
    
    
    augmented_images, original_image = augmenter.generate_augmentations(image_path)
    
    if not augmented_images or original_image is None:
        print(f"Failed to generate augmentations for {image_path}")
        return
    
    
    fig_size = min(4 * int(np.sqrt(num_augmentations + 1)), 16)
    fig = plt.figure(figsize=(fig_size, fig_size))
    rows = int(np.ceil(np.sqrt(num_augmentations + 1)))
    cols = int(np.ceil((num_augmentations + 1) / rows))
    
    
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(original_image)
    ax.set_title("Original")
    ax.axis('off')
    
    
    for i, aug_img in enumerate(augmented_images):
        if i >= num_augmentations:
            break
            
        ax = fig.add_subplot(rows, cols, i + 2)
        ax.imshow(aug_img)
        ax.set_title(f"Augmentation {i+1}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def augment_images(input_dir, output_dir, augmentations_per_image=20, num_workers=4, conservative_color=True):
    """
    Process all species folders and generate augmented images.
    
    Args:
        input_dir: Root directory containing species folders
        output_dir: Directory to save augmented images
        augmentations_per_image: Number of augmented versions per image
        num_workers: Number of worker processes for parallel processing
        conservative_color: Whether to use conservative color adjustments
        
    Returns:
        Path to the generated labels CSV file
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    augmenter = ImageAugmenter(augmentations_per_image=augmentations_per_image, conservative_color=conservative_color)
    
  
    species_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    species_folders.sort() 
    
    print(f"Found {len(species_folders)} species folders")
    
    
    label_data = []
    
    
    next_index = 0
    for folder in species_folders:
        next_index = process_species_folder(
            folder, output_dir, label_data, augmenter, next_index
        )
    
    
    csv_path = os.path.join(output_dir, "image_labels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'common_name', 'original_image'])
        writer.writeheader()
        writer.writerows(label_data)
    
    print(f"\nAugmentation complete!")
    print(f"Generated {len(label_data)} augmented images")
    print(f"Labels saved to {csv_path}")
    
    return csv_path

def main():
    parser = argparse.ArgumentParser(description='Augment slug images and create labels.')
    parser.add_argument('input_dir', help='Directory containing species folders')
    parser.add_argument('--output-dir', '-o', default='augmented_images_rgb_fixed',
                        help='Output folder for downloaded images')
    parser.add_argument('--augmentations', '-a', type=int, default=20,
                        help='Number of augmented versions per image')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--preview', '-p', type=str, default=None,
                        help='Preview augmentations for a single image without saving')
    parser.add_argument('--preview-count', '-pc', type=int, default=9,
                        help='Number of preview augmentations to display')
    parser.add_argument('--extreme-color', '-ec', action='store_true',
                        help='Use more extreme color augmentations (default is conservative)')
    
    args = parser.parse_args()
    
    # preview mode
    if args.preview:
        preview_augmentations(
            args.preview, 
            num_augmentations=args.preview_count,
            conservative_color=not args.extreme_color
        )
        return
    
    # normal mode
    start_time = time.time()
    augment_images(
        args.input_dir, 
        args.output_dir, 
        args.augmentations, 
        args.workers,
        conservative_color=not args.extreme_color
    )
    elapsed_time = time.time() - start_time
    
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()