#!/usr/bin/env python3
"""
Script to delete negative example images starting from a specific number.
"""

import os
from pathlib import Path

def delete_images_from_number(negative_dir, start_number):
    """Delete all images with numbers greater than or equal to start_number."""
    deleted_count = 0
    
    for filename in os.listdir(negative_dir):
        if filename.startswith('image_'):
            try:
                number = int(filename.split('_')[1].split('.')[0])
                if number >= start_number:
                    file_path = os.path.join(negative_dir, filename)
                    os.remove(file_path)
                    deleted_count += 1
                    if deleted_count % 100 == 0:
                        print(f'Deleted {deleted_count} images...')
            except (ValueError, IndexError):
                continue
    
    print(f'Total images deleted: {deleted_count}')

def main():
    negative_dir = 'slug-classifier/binaryDatainterim/negative_examples'
    start_number = 59000
    
    print(f'Deleting images from image_{start_number} onwards...')
    delete_images_from_number(negative_dir, start_number)

if __name__ == '__main__':
    main() 