import requests
import csv
import time
import os
import random

class SlugDatasetGenerator:
    def __init__(self, slugs_output="slugs.csv", non_slugs_output="non_slugs.csv", min_images_per_species=250):
        """
        Initialize the combined slug dataset generator.
        
        Args:
            slugs_output (str): CSV file for slug observations
            non_slugs_output (str): CSV file for non-slug observations  
            min_images_per_species (int): Minimum images required per slug species
        """
        self.base_url = "https://api.inaturalist.org/v1"
        self.slugs_output = slugs_output
        self.non_slugs_output = non_slugs_output
        self.min_images_per_species = min_images_per_species
        self.slug_taxa = []
        self.slug_taxon_ids = []
        
        # CSV headers
        self.csv_headers = [
            'id', 'quality_grade', 'url', 'image_url', 'species_guess', 
            'scientific_name', 'common_name', 'iconic_taxon_name', 'taxon_id'
        ]
        
        # Initialize CSV files
        self.initialize_csv_files()
    
    def initialize_csv_files(self):
        """Initialize both CSV files with headers"""
        for output_file in [self.slugs_output, self.non_slugs_output]:
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.csv_headers)
                writer.writeheader()
    
    def is_slug(self, taxon):
        """Comprehensive slug detection using taxonomic hierarchy and name filtering"""
        if not taxon:
            return False
        
        scientific_name = taxon.get('name', '').lower()
        common_name = taxon.get('preferred_common_name', '').lower()
        all_names = f"{scientific_name} {common_name}"
        
        # Check for slug families in taxonomic hierarchy
        slug_families = [
            'limacidae', 'arionidae', 'philomycidae', 'veronicellidae', 
            'agriolimacidae', 'onchidiidae', 'pleurobranchidae'
        ]
        
        # Check ancestors for slug families
        if 'ancestors' in taxon:
            ancestor_names = [ancestor.get('name', '').lower() for ancestor in taxon['ancestors']]
            ancestor_text = ' '.join(ancestor_names)
            if any(slug_family in ancestor_text for slug_family in slug_families):
                return True
        
        # Check ancestry string if available
        if 'ancestry' in taxon:
            ancestry = taxon.get('ancestry', '').lower()
            if any(f'/{family}' in ancestry for family in slug_families):
                return True
        
        # Name-based detection with exclusions
        if 'slug' in all_names:
            # Exclude things that have 'slug' but aren't actual slugs
            false_positives = [
                'moth', 'butterfly', 'caterpillar', 'beetle', 'fly', 'wasp', 'bee', 
                'spider', 'snake', 'lizard', 'fish', 'bird', 'frog', 'toad', 
                'salamander', 'newt', 'plant', 'flower', 'leaf', 'tree'
            ]
            
            if any(fp in all_names for fp in false_positives):
                return False
            
            # If mollusk with 'slug' in name, likely real slug
            iconic_taxon = taxon.get('iconic_taxon_name', '').lower()
            if iconic_taxon == 'mollusca':
                return True
        
        # Check for slug genera
        slug_genera = [
            'limax', 'arion', 'deroceras', 'milax', 'lehmannia',
            'tandonia', 'limacus', 'malacolimax'
        ]
        
        if any(genus in scientific_name for genus in slug_genera):
            return True
        
        return False
    
    def find_slug_taxa(self):
        """Find all qualifying slug species"""
        print("🔍 Searching for slug species...")
        
        search_queries = [
            {"q": "slug", "per_page": 200},
            {"q": "banana slug", "per_page": 200},
            {"q": "sea slug", "per_page": 200},
            {"taxon_name": "Limacidae", "per_page": 200},
            {"taxon_name": "Arionidae", "per_page": 200},
            {"taxon_name": "Agriolimacidae", "per_page": 200},
            {"q": "limax", "per_page": 200},
            {"q": "arion", "per_page": 200},
            {"q": "deroceras", "per_page": 200}
        ]
        
        all_slug_taxa = []
        
        for search_params in search_queries:
            search_url = f"{self.base_url}/taxa"
            
            try:
                response = requests.get(search_url, params=search_params)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    slug_results = [taxon for taxon in results if self.is_slug(taxon)]
                    all_slug_taxa.extend(slug_results)
                    print(f"   Found {len(slug_results)} slug taxa with query: {search_params}")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"   Error with query {search_params}: {e}")
        
        # Remove duplicates
        unique_taxa = {}
        for taxon in all_slug_taxa:
            unique_taxa[taxon['id']] = taxon
        
        # Filter for species-level only
        species_taxa = [taxon for taxon in unique_taxa.values() if taxon.get('rank') == 'species']
        
        print(f"   Found {len(species_taxa)} unique slug species")
        
        # Check observation counts and filter
        qualifying_taxa = []
        print("📊 Checking observation counts for each species...")
        
        for i, taxon in enumerate(species_taxa):
            observation_count = self.check_observation_count(taxon['id'])
            
            if observation_count >= self.min_images_per_species:
                qualifying_taxa.append(taxon)
                print(f"   ✅ {taxon.get('name', 'unknown')} ({taxon.get('preferred_common_name', '')}) - {observation_count} observations")
            else:
                print(f"   ❌ {taxon.get('name', 'unknown')} - only {observation_count} observations (need {self.min_images_per_species})")
            
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{len(species_taxa)} species checked")
            
            time.sleep(0.3)  # Rate limiting
        
        self.slug_taxa = qualifying_taxa
        self.slug_taxon_ids = [taxon['id'] for taxon in qualifying_taxa]
        
        print(f"🎯 Final result: {len(qualifying_taxa)} slug species qualify for dataset")
        return qualifying_taxa
    
    def check_observation_count(self, taxon_id):
        """Check total observations with photos for a taxon"""
        observations_url = f"{self.base_url}/observations"
        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "per_page": 1,
            "page": 1
        }
        
        try:
            response = requests.get(observations_url, params=params)
            if response.status_code == 200:
                data = response.json()
                return data.get("total_results", 0)
        except Exception:
            pass
        return 0
    
    def get_observations_for_taxon(self, taxon_id, max_observations=250):
        """Get observations for a specific taxon"""
        observations = []
        observations_url = f"{self.base_url}/observations"
        page = 1
        
        while len(observations) < max_observations:
            params = {
                "taxon_id": taxon_id,
                "photos": "true",
                "per_page": min(200, max_observations - len(observations)),
                "page": page,
                "quality_grade": "any",
                "order_by": "observed_on",
                "order": "desc"
            }
            
            try:
                response = requests.get(observations_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if not results:
                        break
                    
                    observations.extend(results)
                    page += 1
                    time.sleep(0.3)
                else:
                    break
            except Exception:
                break
        
        return observations[:max_observations]
    
    def get_non_slug_observations(self, total_needed):
        """Get diverse non-slug observations"""
        print(f"🔍 Collecting {total_needed} non-slug observations...")
        
        # Search terms for diverse non-slug creatures
        search_terms = [
            "snail", "mollusk", "gastropod", "worm", "leech", "caterpillar",
            "larva", "earthworm", "planarian", "flatworm", "beetle", "spider",
            "insect", "arthropod", "fish", "bird", "mammal", "reptile",
            "amphibian", "plant", "fungi", "lichen", "moss"
        ]
        
        all_observations = []
        
        # Get observations from various search terms
        for term in search_terms:
            if len(all_observations) >= total_needed:
                break
                
            print(f"   Searching for: {term}")
            
            search_url = f"{self.base_url}/observations"
            params = {
                "q": term,
                "photos": "true",
                "per_page": min(200, total_needed - len(all_observations)),
                "quality_grade": "research",
                "order": "desc",
                "order_by": "created_at"
            }
            
            try:
                response = requests.get(search_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    # Filter out slugs
                    non_slug_results = []
                    for obs in results:
                        if 'taxon' in obs and obs['taxon']:
                            if not self.is_slug(obs['taxon']) and obs['taxon']['id'] not in self.slug_taxon_ids:
                                if 'photos' in obs and obs['photos']:
                                    non_slug_results.append(obs)
                    
                    all_observations.extend(non_slug_results)
                    print(f"      Added {len(non_slug_results)} non-slug observations")
                
                time.sleep(0.5)
            except Exception as e:
                print(f"      Error searching for {term}: {e}")
        
        # If we need more, get general observations
        if len(all_observations) < total_needed:
            print("   Getting additional general observations...")
            
            observations_url = f"{self.base_url}/observations"
            page = 1
            
            while len(all_observations) < total_needed:
                params = {
                    "photos": "true",
                    "per_page": min(200, total_needed - len(all_observations)),
                    "page": page,
                    "quality_grade": "research",
                    "order": "desc",
                    "order_by": "created_at"
                }
                
                try:
                    response = requests.get(observations_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        
                        if not results:
                            break
                        
                        # Filter out slugs
                        non_slug_results = []
                        for obs in results:
                            if 'taxon' in obs and obs['taxon']:
                                if not self.is_slug(obs['taxon']) and obs['taxon']['id'] not in self.slug_taxon_ids:
                                    if 'photos' in obs and obs['photos']:
                                        non_slug_results.append(obs)
                        
                        all_observations.extend(non_slug_results)
                        page += 1
                        time.sleep(0.5)
                    else:
                        break
                except Exception:
                    break
        
        # Shuffle and return the needed amount
        random.shuffle(all_observations)
        return all_observations[:total_needed]
    
    def save_observations_to_csv(self, observations, output_file):
        """Save observations to specified CSV file"""
        count = 0
        
        with open(output_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_headers)
            
            for obs in observations:
                # Get image URL
                image_url = ''
                if 'photos' in obs and obs['photos'] and 'url' in obs['photos'][0]:
                    image_url = obs['photos'][0]['url'].replace('square', 'medium')
                
                if not image_url:
                    continue
                
                row = {
                    'id': obs.get('id', ''),
                    'quality_grade': obs.get('quality_grade', ''),
                    'url': f"https://www.inaturalist.org/observations/{obs.get('id', '')}" if 'id' in obs else '',
                    'image_url': image_url,
                    'species_guess': obs.get('species_guess', ''),
                    'scientific_name': obs['taxon']['name'] if 'taxon' in obs and 'name' in obs['taxon'] else '',
                    'common_name': obs['taxon']['preferred_common_name'] if 'taxon' in obs and 'preferred_common_name' in obs['taxon'] else '',
                    'iconic_taxon_name': obs['taxon']['iconic_taxon_name'] if 'taxon' in obs and 'iconic_taxon_name' in obs['taxon'] else '',
                    'taxon_id': obs['taxon']['id'] if 'taxon' in obs and 'id' in obs['taxon'] else ''
                }
                writer.writerow(row)
                count += 1
        
        return count
    
    def generate_dataset(self):
        """Main method to generate the complete dataset"""
        print("🚀 Starting slug dataset generation...")
        
        # Step 1: Find qualifying slug species
        qualifying_slugs = self.find_slug_taxa()
        
        if not qualifying_slugs:
            print("❌ No qualifying slug species found!")
            return
        
        # Step 2: Collect slug observations
        print(f"\n📸 Collecting {self.min_images_per_species} images for each of {len(qualifying_slugs)} slug species...")
        total_slug_observations = 0
        
        for i, taxon in enumerate(qualifying_slugs):
            species_name = taxon.get('name', 'unknown')
            common_name = taxon.get('preferred_common_name', '')
            
            print(f"   [{i+1}/{len(qualifying_slugs)}] Getting images for {species_name} ({common_name})")
            
            observations = self.get_observations_for_taxon(taxon['id'], self.min_images_per_species)
            count = self.save_observations_to_csv(observations, self.slugs_output)
            total_slug_observations += count
            
            print(f"      ✅ Added {count} observations")
        
        print(f"🎯 Slug collection complete: {total_slug_observations} total observations")
        
        # Step 3: Collect non-slug observations
        non_slug_needed = len(qualifying_slugs) * self.min_images_per_species
        print(f"\n🌍 Collecting {non_slug_needed} non-slug observations for balance...")
        
        non_slug_observations = self.get_non_slug_observations(non_slug_needed)
        non_slug_count = self.save_observations_to_csv(non_slug_observations, self.non_slugs_output)
        
        print(f"🎯 Non-slug collection complete: {non_slug_count} observations")
        
        # Final summary
        print(f"\n✅ Dataset generation complete!")
        print(f"   📁 Slugs: {total_slug_observations} observations in {self.slugs_output}")
        print(f"   📁 Non-slugs: {non_slug_count} observations in {self.non_slugs_output}")
        print(f"   🎯 Total: {total_slug_observations + non_slug_count} observations")
        print(f"   ⚖️  Balance: {total_slug_observations/(total_slug_observations + non_slug_count)*100:.1f}% slugs, {non_slug_count/(total_slug_observations + non_slug_count)*100:.1f}% non-slugs")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate balanced slug vs non-slug dataset')
    parser.add_argument('--slugs-output', '-s', default='slugs.csv', 
                        help='Output CSV file for slug observations (default: slugs.csv)')
    parser.add_argument('--non-slugs-output', '-n', default='non_slugs.csv',
                        help='Output CSV file for non-slug observations (default: non_slugs.csv)')
    parser.add_argument('--min-images', '-m', type=int, default=250,
                        help='Minimum images per slug species (default: 250)')
    
    args = parser.parse_args()
    
    # Create and run the generator
    generator = SlugDatasetGenerator(
        slugs_output=args.slugs_output,
        non_slugs_output=args.non_slugs_output,
        min_images_per_species=args.min_images
    )
    generator.generate_dataset()