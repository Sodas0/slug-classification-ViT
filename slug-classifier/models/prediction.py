import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# 1. Paste your class_names list here
class_names = [
    'african_banana_slug', 'ash-black_slug', 'black-spotted_semi-slug', 'black_slug', 'blue-margin_headshield_slug', 
    'brown_leatherback_slug', 'brown_slug', 'budapest_slug', 'buttons_banana_slug', 'california_banana_slug', 
    'caribbean_leatherleaf_slug', 'carpathian_blue_slug', 'celtic_sea_slug', 'chestnut_slug', 'chinese_slug', 
    'common_slug-eater', 'conemenos_slug', 'cratena_slug', 'crimson_foot_semi-slug', 'cuban_leaf_slug', 
    'earshell_slug', 'faithful_sea_slug', 'florida_leatherleaf_slug', 'green_cellar_slug', 'greenhouse_slug', 
    'grey_side-gilled_sea_slug', 'harold_dundees_leatherback_slug', 'headband_headshield_slug', 'hedgehog_slug',
     'hills_side-gill_slug', 'humped_ancula_sea_slug', 'iridescent_semi-slug', 'kerry_slug', 'kirks_tailed_slug', 
     'leathery_sea_slug', 'lemon_slug', 'leopard_sea_slug', 'leopard_slug', 'lettuce_sea_slug', 'lovely_headshield_slug',
      'meadow_slug', 'milky_slug', 'northern_dusky_slug', 'orange-clubbed_sea_slug', 'orange-edged_sapsucking_slug', 
      'pacific_banana_slug', 'pilsbrys_head_shield_slug', 'puerto_rican_semi-slug', 'pustulose_wart_slug', 'pyjama_slug', 
      'red_triangle_slug', 'ringed_sap-sucking_slug', 'sap-sucking_slug', 'slender_banana_slug', 'slender_sap_sucking_slug',
       'southern_pacific_banana_slug', 'spanish_slug', 'spotted_white_sea_slug', 'striped_garden_slug', 
       'striped_greenhouse_slug', 'swallowtail_headshield_slug', 'tree_slug', 'tropical_leatherleaf_slug', 
       'umbrella_slug', 'varicose_wart_slug', 'vine_slug', 'warty_slug', 'western_dusky_slug', 
       'white-speckled_headshield_slug', 'yellow-shelled_semi-slug', 'yellow_cellar_slug', 'yellow_umbrella_slug',
        'yellow_umbrella_slug_tylodina_perversa', 'unsure_lol'
]

# 2. Load your model checkpoint
checkpoint_path = r'C:\Users\soham\Desktop\slug-classification-ViT\slug-classifier\models\species\checkpoints\vit_slug_classifier_20250429_011357\final_model.pth'  # Update if needed
checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint['config']
num_classes = config['num_classes']

# 3. Define your model (copy the SlugSpeciesClassifier from transformer.py)
from torchvision.models import vit_l_16

import torch.nn as nn
class SlugSpeciesClassifier(nn.Module):
    def __init__(self, num_classes=74, dropout_rate=0.2):
        super().__init__()
        self.backbone = vit_l_16(weights=None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# 4. Instantiate and load weights
model = SlugSpeciesClassifier(num_classes=num_classes, dropout_rate=config.get('dropout_rate', 0.2))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 5. Preprocess your image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = r'C:\Users\soham\Desktop\slug-classification-ViT\mysterySlug.png'  # <-- Set your image path here
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# 6. Run prediction
with torch.no_grad():
    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1)
    topk_probs, topk_indices = torch.topk(probabilities, k=3, dim=1)

# 7. Print results
threshold = 0.6  # 60% threshold for top-1 prediction

for i in range(topk_indices.size(1)):
    idx = topk_indices[0, i].item()
    prob = topk_probs[0, i].item()
    # For the top-1 prediction, apply the threshold
    if i == 0 and prob < threshold:
        print(f"Prediction {i+1}: unsure_lol (confidence: {prob:.2%} < threshold)")
    else:
        print(f"Prediction {i+1}: {class_names[idx]} (confidence: {prob:.2%})")
