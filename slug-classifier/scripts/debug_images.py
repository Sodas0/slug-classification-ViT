"""
Debug script to identify and handle images with channel issues.
"""
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

#Paths - update these to match your setup
ROOT_DIR = r'C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\data\interim\downloaded_images'
NEGATIVE_DIR = r'C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\data\interim\negative_examples'
PROBLEM_LOG = 'problem_images.log'

def check_image_channels(image_path):
    """Check if an image has exactly 3 RGB channels after conversion."""
    try:
        # Open and explicitly convert to RGB
        img = Image.open(image_path).convert('RGB')

#Convert to tensor
        tensor = transforms.ToTensor()(img)

#Check the number of channels
        if tensor.shape[0] != 3:
            return False, f"Image {image_path} has {tensor.shape[0]} channels after conversion"

        return True, None
    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"

def fix_image(image_path):
    """Fix an image by explicitly saving it as RGB."""
    try:
        img = Image.open(image_path).convert('RGB')
        img.save(image_path, format='JPEG')
        return True
    except Exception as e:
        return False

def scan_directory(directory):
    """Scan a directory for problematic images."""
    problem_images = []

    print(f"Scanning directory: {directory}")
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_path in tqdm(image_files, desc="Checking images"):
        ok, message = check_image_channels(image_path)
        if not ok:
            problem_images.append((image_path, message))

    return problem_images

#Main scanning and fixing logic
with open(PROBLEM_LOG, 'w') as log_file:
    # First check positive examples
    log_file.write("Checking positive examples directory...\n")
    pos_problems = scan_directory(ROOT_DIR)

#Then check negative examples if the directory exists
    if os.path.exists(NEGATIVE_DIR):
        log_file.write("\nChecking negative examples directory...\n")
        neg_problems = scan_directory(NEGATIVE_DIR)


#Summary
    log_file.write("\nSummary:\n")
    log_file.write(f"Found {len(pos_problems)} problematic images in positive examples\n")
    if os.path.exists(NEGATIVE_DIR):
        log_file.write(f"Found {len(neg_problems)} problematic images in negative examples\n")

print(f"Scan complete. Check {PROBLEM_LOG} for details.")
print(f"Total problematic images: {len(pos_problems) + len(neg_problems if os.path.exists(NEGATIVE_DIR) else [])}")