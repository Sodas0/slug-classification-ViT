import requests
import csv
import time
import os
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inaturalist_download.log"),
        logging.StreamHandler()
    ]
)

class INaturalistCSVGenerator:
    def __init__(self, output_file="inaturalist_data.csv", total_images=30000):
        """
        Initialize the iNaturalist CSV generator.
        
        Args:
            output_file (str): CSV file to save downloaded data
            total_images (int): Total number of images/observations to download
        """
        self.base_url = "https://api.inaturalist.org/v1"
        self.output_file = output_file
        self.total_images = total_images
        self.slugs_taxon_ids = []
        self.similar_to_slugs_taxon_ids = []
        self.observation_count = 0
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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
        """Check if a taxon is a slug based on its name"""
        if not taxon:
            return False
        
        # Check both scientific name and common name
        scientific_name = taxon.get('name', '').lower()
        common_name = taxon.get('preferred_common_name', '').lower()
        
        # Exclude if either name contains 'slug'
        return 'slug' in scientific_name or 'slug' in common_name
    
    def get_slug_taxon_ids(self):
        """Find taxon IDs for actual slugs to exclude them"""
        search_url = f"{self.base_url}/taxa"
        params = {
            "q": "slug",
            "per_page": 100
        }
        
        logging.info("Finding slug taxon IDs to exclude...")
        response = requests.get(search_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            for taxon in data.get("results", []):
                if self.is_slug(taxon):
                    self.slugs_taxon_ids.append(taxon["id"])
                    
                    # Also add parent taxon IDs to make sure we exclude the whole slug family
                    if "ancestor_ids" in taxon:
                        self.slugs_taxon_ids.extend(taxon["ancestor_ids"])
            
            self.slugs_taxon_ids = list(set(self.slugs_taxon_ids))  # Remove duplicates
            logging.info(f"Found {len(self.slugs_taxon_ids)} slug-related taxon IDs to exclude")
        else:
            logging.error(f"Failed to get slug taxon IDs: {response.status_code}")
    
    def find_similar_to_slugs(self):
        """Find taxon IDs similar to slugs but not actual slugs"""
        similar_terms = [
            "snail", "mollusk", "gastropod", "worm", "leech", 
            "planarian", "flatworm", "nudibranch", "caterpillar", 
            "larva", "earthworm", "annelid"
        ]
        
        for term in similar_terms:
            search_url = f"{self.base_url}/taxa"
            params = {
                "q": term,
                "per_page": 100
            }
            
            logging.info(f"Searching for '{term}' taxa...")
            response = requests.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                for taxon in data.get("results", []):
                    # Don't add if it's a slug or in the slug exclusion list
                    if not self.is_slug(taxon) and taxon["id"] not in self.slugs_taxon_ids:
                        self.similar_to_slugs_taxon_ids.append(taxon["id"])
            else:
                logging.error(f"Failed to search for '{term}': {response.status_code}")
            
            # Respect rate limits
            time.sleep(1)
        
        self.similar_to_slugs_taxon_ids = list(set(self.similar_to_slugs_taxon_ids))
        logging.info(f"Found {len(self.similar_to_slugs_taxon_ids)} taxa similar to slugs")
    
    def get_observations(self, taxon_id=None, per_page=200):
        """Get observations with images"""
        observations_url = f"{self.base_url}/observations"
        params = {
            "photos": "true",  # Only get observations with photos
            "per_page": per_page,
            "quality_grade": "research",  # High quality observations
            "order": "desc",
            "order_by": "created_at"
        }
        
        if taxon_id:
            params["taxon_id"] = taxon_id
        
        # For similar-to-slugs taxa we want to ensure we get those
        if taxon_id in self.similar_to_slugs_taxon_ids:
            # Get more from the slug-like taxa
            params["per_page"] = min(per_page * 2, 200)
        
        remaining = self.total_images - self.observation_count
        if remaining <= 0:
            return []
        
        if remaining < params["per_page"]:
            params["per_page"] = remaining
        
        try:
            response = requests.get(observations_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                observations = []
                
                for obs in data.get("results", []):
                    # Only include observations with photos and valid taxon
                    if 'photos' in obs and obs['photos'] and 'taxon' in obs:
                        # Skip if it's a slug or has 'slug' in the name
                        if self.is_slug(obs['taxon']) or obs['taxon']['id'] in self.slugs_taxon_ids:
                            continue
                        
                        # Make sure the image URL is available and points to a medium-sized image
                        image_url = ''
                        if obs['photos'] and 'url' in obs['photos'][0]:
                            image_url = obs['photos'][0]['url'].replace('square', 'medium')
                        
                        # Only include if there's a valid image URL
                        if image_url:
                            observations.append(obs)
                
                logging.info(f"Retrieved {len(observations)} valid observations" + 
                            (f" for taxon {taxon_id}" if taxon_id else ""))
                return observations
            else:
                logging.error(f"Failed to get observations: {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Error getting observations: {e}")
            return []
    
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
        
        return count
    
    def get_diverse_taxa(self, num_taxa=300):
        """Get a diverse set of taxa from various taxonomic groups"""
        diverse_taxa = []
        
        # First, prioritize similar-to-slugs taxa
        diverse_taxa.extend(self.similar_to_slugs_taxon_ids)
        
        # Then get taxa from major taxonomic groups to ensure diversity
        taxonomic_groups = [
            "animalia",  # Animals
            "plantae",   # Plants
            "fungi",     # Fungi
            "chromista", # Marine algae and related organisms
            "protozoa",  # Single-celled eukaryotes
        ]
        
        for group in taxonomic_groups:
            search_url = f"{self.base_url}/taxa"
            params = {
                "q": group,
                "rank": "family",  # Get family-level taxa for diversity
                "per_page": 100
            }
            
            logging.info(f"Getting diverse taxa from group '{group}'...")
            response = requests.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                for taxon in data.get("results", []):
                    # Skip if it's in the slug exclusion list
                    if taxon["id"] not in self.slugs_taxon_ids:
                        diverse_taxa.append(taxon["id"])
            else:
                logging.error(f"Failed to get taxa for group '{group}': {response.status_code}")
            
            # Respect rate limits
            time.sleep(1)
        
        # If we still need more taxa, get popular ones
        if len(diverse_taxa) < num_taxa:
            popular_url = f"{self.base_url}/taxa"
            params = {
                "per_page": 200,
                "order_by": "observations_count",
                "order": "desc"
            }
            
            logging.info("Getting additional popular taxa...")
            response = requests.get(popular_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                for taxon in data.get("results", []):
                    # Skip if it's already in our list or in the slug exclusion list
                    if taxon["id"] not in diverse_taxa and taxon["id"] not in self.slugs_taxon_ids:
                        diverse_taxa.append(taxon["id"])
            else:
                logging.error(f"Failed to get popular taxa: {response.status_code}")
        
        # Remove duplicates
        diverse_taxa = list(set(diverse_taxa))
        
        # Shuffle the list to ensure randomness
        random.shuffle(diverse_taxa)
        
        # Trim the list to the requested number
        return diverse_taxa[:num_taxa]
    
    def generate_csv(self):
        """Main method to generate the CSV"""
        # First, identify slugs to exclude
        self.get_slug_taxon_ids()
        
        # Find taxa similar to slugs
        self.find_similar_to_slugs()
        
        # First, add observations from taxa similar to slugs
        logging.info("Getting observations similar to slugs...")
        for taxon_id in self.similar_to_slugs_taxon_ids[:30]:  # Limit to top 30 to avoid too many API calls
            observations = self.get_observations(taxon_id, per_page=100)
            count = self.save_observations_to_csv(observations)
            self.observation_count += count
            logging.info(f"Added {count} observations (Total: {self.observation_count}/{self.total_images})")
            
            # Check if we've reached our target
            if self.observation_count >= self.total_images:
                logging.info(f"Reached target of {self.total_images} observations. Stopping.")
                return
            
            # Respect rate limits
            time.sleep(1)
        
        # Get a diverse set of taxa
        diverse_taxa = self.get_diverse_taxa(num_taxa=200)
        
        # Get observations for each taxon
        for taxon_id in diverse_taxa:
            # Check if we've reached our target
            if self.observation_count >= self.total_images:
                logging.info(f"Reached target of {self.total_images} observations. Stopping.")
                break
            
            # Get observations for this taxon
            observations = self.get_observations(taxon_id, per_page=50)
            count = self.save_observations_to_csv(observations)
            self.observation_count += count
            logging.info(f"Added {count} observations for taxon {taxon_id} (Total: {self.observation_count}/{self.total_images})")
            
            # Respect rate limits
            time.sleep(0.5)
        
        # If we still need more observations, get general observations
        if self.observation_count < self.total_images:
            logging.info(f"Getting additional general observations to reach target...")
            batch_size = 200
            while self.observation_count < self.total_images:
                observations = self.get_observations(per_page=batch_size)
                if not observations:
                    logging.warning("No more observations available. Stopping.")
                    break
                
                count = self.save_observations_to_csv(observations)
                self.observation_count += count
                logging.info(f"Added {count} general observations (Total: {self.observation_count}/{self.total_images})")
                
                # Respect rate limits
                time.sleep(1)
        
        logging.info(f"CSV generation complete. Generated {self.observation_count} records.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate iNaturalist CSV file with image URLs')
    parser.add_argument('--output', '-o', default='inaturalist_data.csv', 
                        help='Output CSV file (default: inaturalist_data.csv)')
    parser.add_argument('--count', '-c', type=int, default=30000,
                        help='Number of observations to include (default: 30000)')
    
    args = parser.parse_args()
    
    # Create and run the generator
    generator = INaturalistCSVGenerator(output_file=args.output, total_images=args.count)
    generator.generate_csv()