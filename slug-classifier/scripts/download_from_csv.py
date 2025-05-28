# download_images_filtered.py: downloads images only for species in the filter list

import csv
import os
import requests
import concurrent.futures
from urllib.parse import urlparse
from pathlib import Path
import time
import re

def load_filter_species(filter_csv):
    """
    Load the list of species to include from the filter CSV file.
    
    Args:
        filter_csv: Path to the CSV file with filtered species
        
    Returns:
        Dictionary mapping taxon_ids to common names
    """
    species_info = {}
    try:
        with open(filter_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'taxon_id' in row and row['taxon_id']:
                    # Convert to string to ensure matching works correctly
                    taxon_id = str(row['taxon_id']).strip()
                    common_name = row.get('common_name', '')
                    scientific_name = row.get('scientific_name', '')
                    
                    # Use common name if available, otherwise use scientific name
                    display_name = common_name if common_name else scientific_name
                    
                    # Store both the display name and scientific name
                    species_info[taxon_id] = {
                        'display_name': display_name,
                        'scientific_name': scientific_name
                    }
    except Exception as e:
        print(f"Error loading filter species: {e}")
    
    print(f"Loaded {len(species_info)} species to include from filter file")
    return species_info

def create_safe_folder_name(name):
    """
    Create a safe folder name from a species name.
    
    Args:
        name: Original name
        
    Returns:
        Safe folder name
    """
    if not name:
        return "unnamed_species"
        
    # Replace spaces with underscores and remove special characters
    safe_name = re.sub(r'[^\w\s-]', '', name)
    safe_name = re.sub(r'[\s]+', '_', safe_name)
    return safe_name.lower()

def download_image(url, base_folder, folder_name, image_index):
    """
    Download an image from URL and save it to a species-specific folder.
    
    Args:
        url: The URL of the image
        base_folder: Base output folder
        folder_name: Name of the subfolder for this species
        image_index: Index to use in filename
    
    Returns:
        Path to saved image or None if download failed
    """
    try:
        # Create species folder
        species_folder = os.path.join(base_folder, folder_name)
        os.makedirs(species_folder, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create filename with image index
        parsed_url = urlparse(url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = '.jpg'
        filename = f"image_{image_index}{ext}"
        
        output_path = os.path.join(species_folder, filename)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {folder_name}/{filename}")
        return output_path
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def download_filtered_images(data_csv, filter_csv, output_folder, url_column='image_url', 
                            max_per_species=None, max_workers=10):
    """
    Download images from URLs in a CSV file, filtering by species in the filter CSV.
    
    Args:
        data_csv: Path to the CSV file with image URLs
        filter_csv: Path to the CSV file with species to include
        output_folder: Folder to save the images
        url_column: Name of the column containing image URLs
        max_per_species: Maximum number of images to download per species
        max_workers: Maximum number of concurrent download workers
    
    Returns:
        Dictionary with species names and counts of downloaded images
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load species filter
    species_info = load_filter_species(filter_csv)
    
    if not species_info:
        print("No species to filter by. Exiting.")
        return {}
    
    # Read URLs from CSV and filter by species
    download_tasks = []
    species_counts = {}  # Track counts per species
    folder_names = {}    # Map taxon_ids to folder names
    
    try:
        with open(data_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames and url_column not in reader.fieldnames:
                print(f"Error: Column '{url_column}' not found in CSV file")
                print(f"Available columns: {', '.join(reader.fieldnames)}")
                return {}
            
            for row in reader:
                taxon_id = str(row.get('taxon_id', '')).strip()
                
                # Skip if not in our allowed list
                if taxon_id not in species_info:
                    continue
                
                url = row.get(url_column)
                if not url:
                    continue
                
                # Get species info
                species_data = species_info[taxon_id]
                display_name = species_data['display_name']
                
                # Create folder name if not already done
                if taxon_id not in folder_names:
                    folder_name = create_safe_folder_name(display_name)
                    
                    # Ensure folder name is unique
                    if folder_name in folder_names.values():
                        # If duplicate, append scientific name
                        scientific_suffix = create_safe_folder_name(species_data['scientific_name'])
                        folder_name = f"{folder_name}_{scientific_suffix}"
                    
                    folder_names[taxon_id] = folder_name
                
                # Get folder name
                folder_name = folder_names[taxon_id]
                
                # Initialize count for this species if not already done
                if folder_name not in species_counts:
                    species_counts[folder_name] = 0
                
                # Skip if we've reached the max for this species
                if max_per_species and species_counts[folder_name] >= max_per_species:
                    continue
                
                # Add to download tasks
                download_tasks.append((
                    url, 
                    output_folder,
                    folder_name,
                    species_counts[folder_name]
                ))
                
                # Increment count
                species_counts[folder_name] += 1
    
    except Exception as e:
        print(f"Error reading data CSV file: {e}")
        return {}
    
    print(f"Found {len(download_tasks)} images to download from {len(species_counts)} species")
    
    # Reset counts for tracking successful downloads
    for species in species_counts:
        species_counts[species] = 0
    
    # Download images in parallel
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary to store futures and their corresponding info
        future_to_info = {
            executor.submit(download_image, url, output_folder, folder_name, idx): 
            (folder_name, url)
            for url, output_folder, folder_name, idx in download_tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_info):
            folder_name, url = future_to_info[future]
            try:
                image_path = future.result()
                if image_path:
                    species_counts[folder_name] += 1
            except Exception as e:
                print(f"Worker error processing {url}: {e}")
    
    elapsed_time = time.time() - start_time
    
    # Calculate total downloads
    total_downloads = sum(species_counts.values())
    print(f"Downloaded {total_downloads} images in {elapsed_time:.2f} seconds")
    
    # Save download statistics
    stats_file = os.path.join(output_folder, "download_stats.csv")
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['folder_name', 'images_downloaded'])
        for folder_name, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow([folder_name, count])
    
    print(f"Download statistics saved to {stats_file}")
    
    # Save folder mapping for reference
    mapping_file = os.path.join(output_folder, "folder_mapping.csv")
    with open(mapping_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['taxon_id', 'folder_name', 'display_name', 'scientific_name'])
        for taxon_id, folder_name in folder_names.items():
            writer.writerow([
                taxon_id, 
                folder_name, 
                species_info[taxon_id]['display_name'],
                species_info[taxon_id]['scientific_name']
            ])
    
    print(f"Folder mapping saved to {mapping_file}")
    return species_counts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download images only for species in filter list')
    parser.add_argument('data_csv', help='Path to the CSV file with image URLs')
    parser.add_argument('filter_csv', help='Path to the CSV file with species to include')
    parser.add_argument('--output', '-o', default='positive_examples', 
                        help='Output folder for downloaded images')
    parser.add_argument('--column', '-c', default='image_url',
                        help='Name of the column containing image URLs')
    parser.add_argument('--max-per-species', '-m', type=int, default=None,
                        help='Maximum number of images per species')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='Maximum number of concurrent workers (default: 10)')
    
    args = parser.parse_args()
    
    download_filtered_images(
        args.data_csv, 
        args.filter_csv, 
        args.output, 
        args.column, 
        args.max_per_species,
        args.workers
    )