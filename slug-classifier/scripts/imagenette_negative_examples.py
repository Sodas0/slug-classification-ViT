#!/usr/bin/env python3
"""
Script to copy and rename images from ImageNette dataset to negative examples folder.
"""

import os
import shutil
from pathlib import Path

def get_next_image_number(negative_dir):
    """Get the next available image number in the negative examples directory."""
    existing_files = os.listdir(negative_dir)
    max_number = 0
    
    for filename in existing_files:
        if filename.startswith('image_'):
            try:
                number = int(filename.split('_')[1].split('.')[0])
                max_number = max(max_number, number)
            except (ValueError, IndexError):
                continue
    
    return max_number + 1

def copy_and_rename_images(source_dir, target_dir, start_number):
    """Copy and rename images from source to target directory."""
    image_files = []
    total_images = 0
    
    # Walk through all subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                total_images += 1
    
    print(f'Found {total_images} images in {source_dir}')
    
    for i, source_path in enumerate(image_files):
        ext = os.path.splitext(source_path)[1].lower()
        if ext == '.png':
            ext = '.jpg'
        
        new_filename = f'image_{start_number + i}{ext}'
        target_path = os.path.join(target_dir, new_filename)
        
        if ext == '.jpg':
            shutil.copy2(source_path, target_path)
        else:
            from PIL import Image
            with Image.open(source_path) as img:
                img.convert('RGB').save(target_path, 'JPEG')
        
        if (i + 1) % 100 == 0:  # Print progress every 100 images
            print(f'Processed {i + 1}/{total_images} images')
    
    print(f'Completed copying {total_images} images from {source_dir}')

def main():
    negative_dir = 'slug-classifier/SLUGdata/interim/negative_examples'
    imagenette_dir = 'slug-classifier/SLUGdata/raw/imagenette2-320'
    
    next_number = get_next_image_number(negative_dir)
    print(f'Next available image number: {next_number}')
    
    for split in ['train', 'val']:
        source_dir = os.path.join(imagenette_dir, split)
        if os.path.exists(source_dir):
            print(f'\nProcessing {split} directory...')
            copy_and_rename_images(source_dir, negative_dir, next_number)
            # Count total images in the split directory
            total_images = sum(1 for root, _, files in os.walk(source_dir) 
                             for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            next_number += total_images

if __name__ == '__main__':
    main() 