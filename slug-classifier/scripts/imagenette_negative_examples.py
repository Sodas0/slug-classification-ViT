import os
import shutil
from pathlib import Path

def get_next_image_number(negative_dir):
    """Get the next available image number in the negative examples directory."""
    try:
        # Check if directory exists
        if not os.path.exists(negative_dir):
            print(f"WARNING: Negative directory does not exist: {negative_dir}")
            return 1
            
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
    except Exception as e:
        print(f"Error in get_next_image_number: {e}")
        return 1

def copy_and_rename_images(source_dir, target_dir, start_number):
    """Copy and rename images from source to target directory."""
    # Check if directories exist
    if not os.path.exists(source_dir):
        print(f"ERROR: Source directory does not exist: {source_dir}")
        return
    
    if not os.path.exists(target_dir):
        print(f"ERROR: Target directory does not exist: {target_dir}")
        try:
            os.makedirs(target_dir)
            print(f"Created target directory: {target_dir}")
        except Exception as e:
            print(f"Failed to create target directory: {e}")
            return
    
    image_files = []
    total_images = 0
    
    # Walk through all subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
                total_images += 1
    
    print(f'Found {total_images} images in {source_dir}')
    
    if total_images == 0:
        print(f"WARNING: No images found in {source_dir}")
        return
    
    processed_count = 0
    for i, source_path in enumerate(image_files):
        try:
            ext = os.path.splitext(source_path)[1].lower()
            new_ext = '.jpg' if ext == '.png' else ext
            
            new_filename = f'image_{start_number + i}{new_ext}'
            target_path = os.path.join(target_dir, new_filename)
            
            print(f"Copying: {source_path} -> {target_path}")
            
            if ext == '.jpg' or ext == '.jpeg':
                shutil.copy2(source_path, target_path)
            else:
                try:
                    from PIL import Image
                    with Image.open(source_path) as img:
                        img.convert('RGB').save(target_path, 'JPEG')
                except ImportError:
                    print("WARNING: PIL not installed, trying direct copy for PNG")
                    shutil.copy2(source_path, target_path)
                except Exception as e:
                    print(f"Error converting image {source_path}: {e}")
                    continue
            
            processed_count += 1
            if (i + 1) % 100 == 0:  # Print progress every 100 images
                print(f'Processed {i + 1}/{total_images} images')
                
        except Exception as e:
            print(f"Error processing file {source_path}: {e}")
    
    print(f'Completed copying {processed_count}/{total_images} images from {source_dir}')

def main():
    negative_dir = r'C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\negative_examples'
    #'slug-classifier/binaryDatainterim/negative_examples'  #MACBOOK
    
    imagenette_dir = r'C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\speciesDATA\raw\imagenette2-320'
    #'slug-classifier/binaryDataraw/imagenette2-320'  #MACBOOK
    
    print(f"Checking if directories exist:")
    print(f"Negative dir exists: {os.path.exists(negative_dir)}")
    print(f"Imagenette dir exists: {os.path.exists(imagenette_dir)}")
    
    try:
        next_number = get_next_image_number(negative_dir)
        print(f'Next available image number: {next_number}')
        
        for split in ['train', 'val']:
            source_dir = os.path.join(imagenette_dir, split)
            if os.path.exists(source_dir):
                print(f'\nProcessing {split} directory: {source_dir}')
                copy_and_rename_images(source_dir, negative_dir, next_number)
                # Count total images in the split directory
                total_images = sum(1 for root, _, files in os.walk(source_dir) 
                                for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                next_number += total_images
            else:
                print(f"WARNING: Split directory does not exist: {source_dir}")
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == '__main__':
    main()