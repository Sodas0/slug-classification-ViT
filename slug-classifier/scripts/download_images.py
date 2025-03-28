import csv
import os
import requests
from urllib.parse import urlparse
from pathlib import Path

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
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get the image data
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Determine filename - either from URL or use index
        if row_index is not None:
            # Extract file extension from URL
            parsed_url = urlparse(url)
            path = parsed_url.path
            ext = os.path.splitext(path)[1]
            if not ext:
                # Default to .jpg if no extension found
                ext = '.jpg'
            filename = f"image_{row_index}{ext}"
        else:
            # Use the filename from the URL
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                return None
        
        # Full path for the output file
        output_path = os.path.join(output_folder, filename)
        
        # Save the image
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return output_path
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def download_images_from_csv(csv_file, output_folder, url_column='image_url', limit=None):
    """
    Download images from URLs in a CSV file.
    
    Args:
        csv_file: Path to the CSV file
        output_folder: Folder to save the images
        url_column: Name of the column containing image URLs
        limit: Maximum number of images to download
    
    Returns:
        List of paths to downloaded images
    """
    downloaded_images = []
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if the URL column exists
        if reader.fieldnames and url_column not in reader.fieldnames:
            print(f"Error: Column '{url_column}' not found in CSV file")
            print(f"Available columns: {', '.join(reader.fieldnames)}")
            return downloaded_images
        
        # Download each image
        for i, row in enumerate(reader):
            # Stop if we've reached the limit
            if limit is not None and i >= limit:
                break
                
            url = row.get(url_column)
            if url:
                image_path = download_image(url, output_folder, i)
                if image_path:
                    downloaded_images.append(image_path)
    
    print(f"Downloaded {len(downloaded_images)} images")
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
    
    args = parser.parse_args()
    
    download_images_from_csv(args.csv_file, args.output, args.column, args.limit) 