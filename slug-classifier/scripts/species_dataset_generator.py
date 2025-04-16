import requests
import csv
import time
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("slug_downloader.log"),
        logging.StreamHandler()
    ]
)

class SlugCSVGenerator:
    def __init__(self, output_file="slug_data.csv", min_images=250, max_species=None):
        """
        Initialize the Slug CSV Generator
        
        Args:
            output_file (str): CSV file to save downloaded data
            min_images (int): Minimum number of images required for a species to be included
            max_species (int): Maximum number of species to process (None for all)
        """
        self.base_url = "https://api.inaturalist.org/v1"
        self.output_file = output_file
        self.min_images = min_images
        self.max_species = max_species
        self.slug_taxa = []
        self.observation_count = 0
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create a metadata directory
        self.metadata_dir = "slug_metadata"
        if not os.path.exists(self.metadata_dir):
            os.makedirs(self.metadata_dir)
            
        # Initialize CSV file with headers
        self.initialize_csv()
    
    def initialize_csv(self):
        """Initialize the CSV file with headers"""
        headers = [
            'id', 'uuid', 'quality_grade', 'url', 'image_url', 'sound_url', 
            'description', 'species_guess', 'scientific_name', 'common_name', 
            'iconic_taxon_name', 'taxon_id'
        ]
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()

    def is_slug(self, taxon):
        """Check if a taxon is a slug based on its name or ancestors"""
        if not taxon:
            return False
        
        # Check scientific name
        scientific_name = taxon.get('name', '').lower()
        if 'slug' in scientific_name:
            return True
        
        # Check common name
        common_name = taxon.get('preferred_common_name', '').lower()
        if common_name and 'slug' in common_name:
            return True
        
        # Check if it belongs to a slug family or order
        if 'ancestry' in taxon:
            ancestry = taxon.get('ancestry', '')
            return '/Limacidae' in ancestry or '/Arionidae' in ancestry or '/Agriolimacidae' in ancestry
            
        return False
    
    def find_slug_taxa(self, per_page=200, max_pages=10):
        """
        Find all slug taxa using multiple search approaches
        
        Args:
            per_page: Number of results per page
            max_pages: Maximum number of pages to search through
        """
        logging.info("Searching for slug taxa...")
        
        # Search strategies for finding slug taxa
        search_queries = [
            {"q": "slug", "per_page": per_page},  # Basic slug search
            {"q": "banana slug", "per_page": per_page},  # Specific type
            {"q": "sea slug", "per_page": per_page},  # Sea slugs
            {"taxon_name": "Limacidae", "per_page": per_page},  # Slug family
            {"taxon_name": "Arionidae", "per_page": per_page},  # Another slug family
            {"taxon_name": "Agriolimacidae", "per_page": per_page},  # Another slug family
            {"q": "nudibranch", "per_page": per_page},  # Sea slugs (nudibranchs)
            {"q": "Gastropoda slug", "per_page": per_page}  # General gastropod search to find more related taxa
        ]
        
        for search_params in search_queries:
            logging.info(f"Searching with params: {search_params}")
            self._search_with_params(search_params, max_pages)
            # Respect rate limits
            time.sleep(1)
            
        # Remove duplicates
        unique_ids = set()
        unique_taxa = []
        
        for taxon in self.slug_taxa:
            if taxon['id'] not in unique_ids:
                unique_ids.add(taxon['id'])
                unique_taxa.append(taxon)
                
        self.slug_taxa = unique_taxa
        
        # Filter only for species-level taxa
        species_taxa = [taxon for taxon in self.slug_taxa if taxon.get('rank') == 'species']
        logging.info(f"Found {len(self.slug_taxa)} unique slug taxa, of which {len(species_taxa)} are species-level")
        
        # Set to species-only
        self.slug_taxa = species_taxa
        
        # Limit to max_species if specified
        if self.max_species:
            self.slug_taxa = self.slug_taxa[:self.max_species]
            
        logging.info(f"Will process {len(self.slug_taxa)} slug species")
        
        # Save taxa to file for reference
        self._save_taxa_to_file()
    
    def _search_with_params(self, params, max_pages):
        """Helper method to search with specific parameters"""
        search_url = f"{self.base_url}/taxa"
        
        for page in range(1, max_pages + 1):
            # Add page parameter
            page_params = params.copy()
            page_params["page"] = page
            
            try:
                response = requests.get(search_url, params=page_params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # If no results, we've reached the end
                    if not results:
                        break
                    
                    # Filter for slugs
                    slug_results = [taxon for taxon in results if self.is_slug(taxon)]
                    self.slug_taxa.extend(slug_results)
                    
                    logging.info(f"Page {page}: Found {len(slug_results)} slug taxa")
                    
                    # If we didn't get a full page, no need to continue
                    if len(results) < params["per_page"]:
                        break
                else:
                    logging.error(f"Failed to get taxa: {response.status_code}")
                    break
            except Exception as e:
                logging.error(f"Error searching for taxa: {e}")
                break
            
            # Respect rate limits
            time.sleep(0.5)
    
    def _save_taxa_to_file(self):
        """Save the list of slug taxa to a JSON file for reference"""
        import json
        taxa_file = os.path.join(self.metadata_dir, "slug_taxa.json")
        
        # Create a simplified representation of each taxon
        simplified_taxa = []
        for taxon in self.slug_taxa:
            simplified = {
                'id': taxon.get('id'),
                'name': taxon.get('name'),
                'common_name': taxon.get('preferred_common_name'),
                'rank': taxon.get('rank'),
                'observations_count': taxon.get('observations_count', 0),
                'photos_count': taxon.get('photos_count', 0)
            }
            simplified_taxa.append(simplified)
            
        # Save to file
        with open(taxa_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_taxa, f, indent=2)
            
        logging.info(f"Saved taxa information to {taxa_file}")
    
    def check_observation_count(self, taxon_id):
        """
        Check how many observations with photos exist for a taxon
        
        Args:
            taxon_id: The taxon ID to check
            
        Returns:
            The number of observations with photos
        """
        observations_url = f"{self.base_url}/observations"
        params = {
            "taxon_id": taxon_id,
            "photos": "true",  # Only count observations with photos
            "per_page": 1,     # We just need the total count, not actual results
            "page": 1
        }
        
        try:
            response = requests.get(observations_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                total_results = data.get("total_results", 0)
                return total_results
            else:
                logging.error(f"Failed to get observation count: {response.status_code}")
                return 0
        except Exception as e:
            logging.error(f"Error checking observation count: {e}")
            return 0
    
    def get_observations_for_taxon(self, taxon_id, per_page=200, max_pages=10):
        """
        Get observations with images for a specific taxon
        
        Args:
            taxon_id: The taxon ID to get observations for
            per_page: Number of observations per page
            max_pages: Maximum number of pages to fetch
            
        Returns:
            List of observations
        """
        observations = []
        observations_url = f"{self.base_url}/observations"
        
        for page in range(1, max_pages + 1):
            params = {
                "taxon_id": taxon_id,
                "photos": "true",  # Only get observations with photos
                "per_page": per_page,
                "page": page,
                "quality_grade": "any",  # Include all quality grades to maximize results
                "order_by": "observed_on",
                "order": "desc"
            }
            
            try:
                response = requests.get(observations_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # If no results, we've reached the end
                    if not results:
                        break
                    
                    # Add to observations list
                    observations.extend(results)
                    
                    logging.info(f"Page {page}: Got {len(results)} observations")
                    
                    # If we didn't get a full page, we've reached the end
                    if len(results) < per_page:
                        break
                else:
                    logging.error(f"Failed to get observations: {response.status_code}")
                    break
            except Exception as e:
                logging.error(f"Error getting observations: {e}")
                break
            
            # Respect rate limits
            time.sleep(0.5)
        
        logging.info(f"Got total of {len(observations)} observations for taxon {taxon_id}")
        return observations
    
    def save_observations_to_csv(self, observations):
        """Save a batch of observations to CSV file"""
        if not observations:
            return 0
        
        count = 0
        with open(self.output_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'id', 'uuid', 'quality_grade', 'url', 'image_url', 'sound_url', 
                'description', 'species_guess', 'scientific_name', 'common_name', 
                'iconic_taxon_name', 'taxon_id'
            ])
            
            for obs in observations:
                # Extract the needed fields
                image_url = ''
                if 'photos' in obs and obs['photos'] and 'url' in obs['photos'][0]:
                    image_url = obs['photos'][0]['url'].replace('square', 'medium')
                
                # Skip if no valid image URL
                if not image_url:
                    continue
                
                row = {
                    'id': obs.get('id', ''),
                    'uuid': obs.get('uuid', ''),
                    'quality_grade': obs.get('quality_grade', ''),
                    'url': f"https://www.inaturalist.org/observations/{obs.get('id', '')}" if 'id' in obs else '',
                    'image_url': image_url,
                    'sound_url': obs['sounds'][0]['file_url'] if 'sounds' in obs and obs['sounds'] else '',
                    'description': obs.get('description', ''),
                    'species_guess': obs.get('species_guess', ''),
                    'scientific_name': obs['taxon']['name'] if 'taxon' in obs and 'name' in obs['taxon'] else '',
                    'common_name': obs['taxon']['preferred_common_name'] if 'taxon' in obs and 'preferred_common_name' in obs['taxon'] else '',
                    'iconic_taxon_name': obs['taxon']['iconic_taxon_name'] if 'taxon' in obs and 'iconic_taxon_name' in obs['taxon'] else '',
                    'taxon_id': obs['taxon']['id'] if 'taxon' in obs and 'id' in obs['taxon'] else ''
                }
                writer.writerow(row)
                count += 1
                self.observation_count += 1
        
        return count
    
    def process_taxon(self, taxon):
        """
        Process a single taxon: get observations and add to CSV
        
        Args:
            taxon: The taxon to process
            
        Returns:
            Dict with processing results
        """
        taxon_id = taxon.get('id')
        scientific_name = taxon.get('name', 'unknown')
        common_name = taxon.get('preferred_common_name', '')
        rank = taxon.get('rank', 'unknown')
        
        logging.info(f"Processing species {scientific_name} ({common_name}) - ID: {taxon_id}")
        
        # Check observation count first
        observation_count = self.check_observation_count(taxon_id)
        logging.info(f"Species {scientific_name} has {observation_count} observations with photos")
        
        # Skip if less than minimum required images
        if observation_count < self.min_images:
            logging.info(f"Skipping species {scientific_name} - insufficient images ({observation_count} < {self.min_images})")
            return {
                'taxon_id': taxon_id,
                'scientific_name': scientific_name,
                'common_name': common_name,
                'rank': rank,
                'observations_found': observation_count,
                'observations_added': 0,
                'skipped_reason': 'insufficient_images'
            }
        
        # Get observations
        observations = self.get_observations_for_taxon(taxon_id)
        
        if not observations:
            logging.warning(f"No observations found for species {taxon_id}")
            return {
                'taxon_id': taxon_id,
                'scientific_name': scientific_name,
                'common_name': common_name,
                'rank': rank,
                'observations_found': 0,
                'observations_added': 0
            }
        
        # Save observations to CSV
        observations_added = self.save_observations_to_csv(observations)
        
        logging.info(f"Added {observations_added} observations for species {scientific_name} to CSV")
        
        return {
            'taxon_id': taxon_id,
            'scientific_name': scientific_name,
            'common_name': common_name,
            'rank': rank,
            'observations_found': len(observations),
            'observations_added': observations_added
        }
    
    def generate_csv(self):
        """Main method to generate the CSV"""
        # Find all slug taxa
        self.find_slug_taxa()
        
        if not self.slug_taxa:
            logging.error("No slug species found to process")
            return
        
        # Process each taxon
        results = []
        species_with_enough_images = 0
        
        for i, taxon in enumerate(self.slug_taxa):
            logging.info(f"Processing taxon {i+1}/{len(self.slug_taxa)}")
            result = self.process_taxon(taxon)
            results.append(result)
            
            # Count species with enough images
            if 'skipped_reason' not in result:
                species_with_enough_images += 1
            
            # Save progress
            self._save_progress(results)
        
        # Log final summary
        total_observations = self.observation_count
        
        logging.info(f"CSV generation complete! Added {total_observations} observations to CSV")
        logging.info(f"Found {species_with_enough_images} species with at least {self.min_images} images")
        
        # Generate CSV summary
        self._generate_summary_csv(results)
        
        return results
    
    def _save_progress(self, results):
        """Save progress information"""
        import json
        progress_file = os.path.join(self.metadata_dir, "progress.json")
        
        summary = {
            'total_taxa': len(results),
            'species_with_enough_images': sum(1 for r in results if 'skipped_reason' not in r),
            'species_insufficient_images': sum(1 for r in results if r.get('skipped_reason') == 'insufficient_images'),
            'min_images_threshold': self.min_images,
            'total_observations': self.observation_count,
            'taxa_processed': [
                {
                    'taxon_id': r.get('taxon_id'),
                    'scientific_name': r.get('scientific_name'),
                    'common_name': r.get('common_name'),
                    'rank': r.get('rank'),
                    'observations_found': r.get('observations_found', 0),
                    'observations_added': r.get('observations_added', 0),
                    'skipped': 'skipped_reason' in r,
                    'skipped_reason': r.get('skipped_reason', '') if 'skipped_reason' in r else ''
                }
                for r in results
            ]
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        logging.info(f"Updated progress: {self.observation_count} observations added to CSV")
    
    def _generate_summary_csv(self, results):
        """Generate a CSV summary of the processed taxa"""
        summary_file = os.path.join(self.metadata_dir, "species_summary.csv")
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Taxon ID', 'Scientific Name', 'Common Name', 'Observations Found', 
                'Observations Added', 'Skipped', 'Reason'
            ])
            
            for r in results:
                writer.writerow([
                    r.get('taxon_id', ''),
                    r.get('scientific_name', ''),
                    r.get('common_name', ''),
                    r.get('observations_found', 0),
                    r.get('observations_added', 0),
                    'Yes' if 'skipped_reason' in r else 'No',
                    r.get('skipped_reason', '') if 'skipped_reason' in r else ''
                ])
                
        logging.info(f"Generated CSV summary: {summary_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate CSV file with slug observations from iNaturalist')
    parser.add_argument('--output', '-o', default='/Users/sohamdas/Desktop/Projects/myVit/slug-classifier/speciesDATA/raw/slug_data.csv', 
                        help='Output CSV file (default: slug_data.csv)')
    parser.add_argument('--min-images', '-m', type=int, default=250,
                        help='Minimum number of images required for a species (default: 250)')
    parser.add_argument('--max-species', '-s', type=int, default=None,
                        help='Maximum number of species to process (default: all)')
    
    args = parser.parse_args()
    
    # Create and run the generator
    generator = SlugCSVGenerator(
        output_file=args.output,
        min_images=args.min_images,
        max_species=args.max_species
    )
    generator.generate_csv()