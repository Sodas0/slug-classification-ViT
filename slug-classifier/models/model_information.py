import torch
from torchvision import models
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_classes = [
    models.mobilenet_v3_small(pretrained=True),
    models.vit_b_16(pretrained=True),
    models.vit_b_32(pretrained=True),
]


input_size = (3, 224, 224) 

for i, model in enumerate(model_classes):
    model_name = type(model).__name__
    print("="*40)
    print(f"MODEL SUMMARY FOR: {model_name}")
    print("="*40)
    
    model = model.to(device)
    summary(model, input_size)