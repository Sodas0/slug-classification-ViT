import os
import csv
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

class ImageAugmenter:
    """Class to handle image augmentation operations."""
    
    def __init__(self, seed=None):
        """Initialize the image augmenter."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _ensure_rgb(self, image):
        """Ensure the image is in RGB format."""
       
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=2)
        
        
        if image.shape[2] == 4:
            return image[:, :, :3]
        
        if image.shape[2] == 3:
            return image
        
        try:
            pil_image = Image.fromarray(image)
            rgb_image = pil_image.convert('RGB')
            return np.array(rgb_image)
        except Exception as e:
            print(f"Error converting image to RGB: {e}")
            return image
    
    def _rotate_image(self, image, angle_range=(-20, 20)):
        """Rotate the image by a random angle."""
        angle = random.uniform(*angle_range)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                 borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def _flip_horizontal(self, image):
        """Flip the image horizontally."""
        return cv2.flip(image, 1)
    
    def _crop_and_resize(self, image, crop_factor_range=(0.8, 0.95)):
        """Randomly crop the image and resize it back to original dimensions."""
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
    
    def augment_image(self, image):
        """Apply a basic set of augmentations to the image."""
        # Ensure input image is RGB
        image = self._ensure_rgb(image)
        
        augmented = image.copy()
        
        
        if random.random() < 0.7:
            augmented = self._rotate_image(augmented)
            
        
        if random.random() < 0.5:
            augmented = self._flip_horizontal(augmented)
            
        
        if random.random() < 0.6:
            augmented = self._crop_and_resize(augmented)
        
       
        return self._ensure_rgb(augmented)
    
    def generate_augmentations(self, image_path, num_augmentations=5):
        """Generate multiple augmented versions of an image."""
        try:
            
            pil_image = Image.open(str(image_path)).convert('RGB')
            image = np.array(pil_image)
            
            if image is None or image.size == 0:
                print(f"Warning: Could not read image {image_path}")
                return [], None
                
          
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = self._ensure_rgb(image)
                
            augmented_images = []
            for _ in range(num_augmentations):
                augmented = self.augment_image(image)
                augmented_images.append(augmented)
                
            return augmented_images, image 
        except Exception as e:
            print(f"Error augmenting image {image_path}: {e}")
            return [], None

def process_species_folder(species_folder, output_dir, label_data, augmenter, start_index, augmentations_per_image):
    """Process all images in a species folder."""
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
        augmented_images, _ = augmenter.generate_augmentations(img_path, augmentations_per_image)
        
        for aug_img in augmented_images:
            if aug_img is None or len(aug_img) == 0:
                continue
                
            pil_img = Image.fromarray(aug_img)
            
           
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

def augment_images(input_dir, output_dir, augmentations_per_image=5):
    """Process all species folders and generate augmented images."""
    os.makedirs(output_dir, exist_ok=True)
    
    augmenter = ImageAugmenter()
    
    species_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    species_folders.sort() 
    
    print(f"Found {len(species_folders)} species folders")
    
    label_data = []
    
    next_index = 0
    for folder in species_folders:
        next_index = process_species_folder(
            folder, output_dir, label_data, augmenter, next_index, augmentations_per_image
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
    parser = argparse.ArgumentParser(description='Augment images with simplified transformations.')
    parser.add_argument('input_dir', help='Directory containing species folders')
    parser.add_argument('--output-dir', '-o', default='augmented_images',
                        help='Output folder for augmented images')
    parser.add_argument('--augmentations', '-a', type=int, default=2,
                        help='Number of augmented versions per image')
    
    args = parser.parse_args()
    
    augment_images(
        args.input_dir, 
        args.output_dir, 
        args.augmentations
    )

if __name__ == "__main__":
    main()