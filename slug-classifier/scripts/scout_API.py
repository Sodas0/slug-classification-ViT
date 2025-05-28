import requests
import pandas as pd
import time
from collections import Counter
import re

def is_actual_slug(taxon_name, description):
    """
    Check if the taxon is actually a slug and not something else with 'slug' in the name.
    
    Args:
        taxon_name (str): Name of the taxon
        description (str): Description or notes about the taxon
        
    Returns:
        bool: True if it's a slug, False otherwise
    """
    
    non_slug_terms = [
        'moth', 'snake', 'fish', 'caterpillar', 'insect', 'beetle', 
        'fly', 'bird', 'lizard', 'reptile', 'amphibian'
    ]
    
    for term in non_slug_terms:
        if term.lower() in taxon_name.lower() or (description and term.lower() in description.lower()):
            return False
    slug_indicators = [
        'slug', 'gastropod', 'mollusc', 'mollusk', 'arionidae', 'limacidae', 
        'agriolimacidae', 'philomycidae', 'veronicellidae'
    ]
    
    for indicator in slug_indicators:
        if indicator.lower() in taxon_name.lower() or (description and indicator.lower() in description.lower()):
            return True
    return False

def get_slug_species_data(min_image_count=100):
    """
    Get statistics on slug species from iNaturalist with a minimum number of images.
    
    Args:
        min_image_count (int): Minimum number of images required for a species
    
    Returns:
        dict: Statistics about slug species and their image counts
    """
    print("Searching for slug species on iNaturalist...")
    search_params = {
        'q': 'slug',
        'per_page': 100,  # Maximum allowed by API
        'is_active': 'true',
        'rank': 'species'
    }
    
    all_slug_taxa = []
    page = 1
    
    while True:
        search_params['page'] = page
        response = requests.get('https://api.inaturalist.org/v1/taxa', params=search_params)
        
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break
        
        data = response.json()
        results = data.get('results', [])
        
        if not results:
            break
            
        all_slug_taxa.extend(results)
        page += 1
        time.sleep(1)
        
        print(f"Fetched page {page-1}, found {len(results)} potential taxa")
        if len(results) < search_params['per_page']:
            break
    
    print(f"Found a total of {len(all_slug_taxa)} potential slug taxa")
    slug_species = []
    for taxon in all_slug_taxa:
        name = taxon.get('name', '')
        preferred_name = taxon.get('preferred_common_name', '')
        description = taxon.get('wikipedia_summary', '')
        
        display_name = preferred_name if preferred_name else name
        
        if is_actual_slug(display_name, description):
            slug_species.append({
                'id': taxon.get('id'),
                'name': name,
                'preferred_name': display_name,
                'rank': taxon.get('rank'),
                'observations_count': taxon.get('observations_count', 0),
                'photos_count': taxon.get('taxon_photos_count', 0)
            })
    
    print(f"Filtered to {len(slug_species)} actual slug species")
    slug_species_with_images = []
    
    for species in slug_species:
        species_id = species['id']
        obs_params = {
            'taxon_id': species_id,
            'per_page': 1,
            'photos': 'true',
            'quality_grade': 'research',
            'order': 'desc',
            'order_by': 'created_at'
        }
        
        obs_response = requests.get('https://api.inaturalist.org/v1/observations', params=obs_params)
        
        if obs_response.status_code == 200:
            obs_data = obs_response.json()
            total_count = obs_data.get('total_results', 0)
            
            photo_params = {
                'taxon_id': species_id,
                'per_page': 1,
                'photos': 'true',
                'quality_grade': 'research'
            }
            
            photo_response = requests.get('https://api.inaturalist.org/v1/observations', params=photo_params)
            
            if photo_response.status_code == 200:
                photo_data = photo_response.json()
                photo_count = photo_data.get('total_results', 0)
                
                species['observations_with_photos'] = photo_count
                
                if photo_count >= min_image_count:
                    slug_species_with_images.append(species)
                    print(f"Found species {species['preferred_name']} with {photo_count} images")
        
        # API rate limiting
        time.sleep(1)
    
    # calculating statistics
    stats = {
        'total_potential_taxa': len(all_slug_taxa),
        'actual_slug_species': len(slug_species),
        'slug_species_with_min_images': len(slug_species_with_images),
        'min_image_count': min_image_count,
        'species_data': slug_species_with_images
    }
    
    return stats

def run_slug_analysis():
    """Run the full analysis and save results"""
    min_images = 200
    stats = get_slug_species_data(min_images)
    
    print("\n=== SLUG SPECIES ANALYSIS ===")
    print(f"Total potential taxa found: {stats['total_potential_taxa']}")
    print(f"Actual slug species found: {stats['actual_slug_species']}")
    print(f"Slug species with at least {min_images} images: {stats['slug_species_with_min_images']}")
    
    if stats['species_data']:
        df = pd.DataFrame(stats['species_data'])
        df = df.sort_values(by='observations_with_photos', ascending=False)
        
        print("\nTop slug species by image count:")
        for i, row in df.head(10).iterrows():
            print(f"{row['preferred_name']}: {row['observations_with_photos']} images")
            
        df.to_csv('slug_species_with_min_images.csv', index=False)
        print("\nFull data saved to 'slug_species_with_min_images.csv'")
    else:
        print("\nNo slug species found with the minimum image count.")

if __name__ == "__main__":
    run_slug_analysis()