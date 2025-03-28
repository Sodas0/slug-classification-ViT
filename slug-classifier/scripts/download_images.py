import csv
import os
import requests
import concurrent.futures
from urllib.parse import urlparse
from pathlib import Path
import time

def download_image(url, output_folder, row_index=None):
    """
    Download an image from URL and save it to the output folder.
    
    Args:
        url: The URL of the image
        output_folder: Folder to save the image
        row_index: Optional index to use in filename
    
    Returns:
        Path to saved image or None if download failed
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()  
        
        if row_index is not None:
            parsed_url = urlparse(url)
            path = parsed_url.path
            ext = os.path.splitext(path)[1]
            if not ext:
                ext = '.jpg'
            filename = f"image_{row_index}{ext}"
        else:
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                return None
        
        output_path = os.path.join(output_folder, filename)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return output_path
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def download_images_parallel(csv_file, output_folder, url_column='image_url', limit=None, max_workers=10):
    """
    Download images from URLs in a CSV file using parallel processing.
    
    Args:
        csv_file: Path to the CSV file
        output_folder: Folder to save the images
        url_column: Name of the column containing image URLs
        limit: Maximum number of images to download
        max_workers: Maximum number of concurrent download workers
    
    Returns:
        List of paths to downloaded images
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read URLs from CSV
    urls_to_download = []
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames and url_column not in reader.fieldnames:
                print(f"Error: Column '{url_column}' not found in CSV file")
                print(f"Available columns: {', '.join(reader.fieldnames)}")
                return []
            
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                    
                url = row.get(url_column)
                if url:
                    urls_to_download.append((url, i))
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    # Download images in parallel
    start_time = time.time()
    downloaded_images = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary to store futures and their corresponding indices
        future_to_url = {
            executor.submit(download_image, url, output_folder, idx): (url, idx) 
            for url, idx in urls_to_download
        }
        
        for future in concurrent.futures.as_completed(future_to_url):
            url, idx = future_to_url[future]
            try:
                image_path = future.result()
                if image_path:
                    downloaded_images.append(image_path)
            except Exception as e:
                print(f"Worker error processing {url} (index {idx}): {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Downloaded {len(downloaded_images)} images in {elapsed_time:.2f} seconds")
    return downloaded_images

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download images from URLs in a CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--output', '-o', default='downloaded_images', 
                        help='Output folder for downloaded images')
    parser.add_argument('--column', '-c', default='image_url',
                        help='Name of the column containing image URLs')
    parser.add_argument('--limit', '-l', type=int, default=10,
                        help='Maximum number of images to download (default: 10)')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='Maximum number of concurrent workers (default: 10)')
    
    args = parser.parse_args()
    
    download_images_parallel(args.csv_file, args.output, args.column, args.limit, args.workers)