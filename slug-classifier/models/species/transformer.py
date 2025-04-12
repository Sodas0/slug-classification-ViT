"""
Vision Transformer model for slug species classification.

This module should implement:
1. SlugSpeciesClassifier class - A Vision Transformer model for species classification
2. Supporting functions for creating and configuring the species classifier

Implementation:
- Use Vision Transformer architecture (ViT-B/16, ViT-B/32, (probably will use vit_b_16 cause training. but
    diff in params is like 2m, maybe will use 32 just for funsies)  
- Support pre-trained weights and backbone freezing - ViT needs to learn difference between slug species.
- Add support for hierarchical classification (family → genus → species)(?) maybe
- Include confidence scoring and top-k predictions
- Main thing is to benchmark performance based on #imgs/class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import numpy as np
import os
import sys
from datetime import datetime
from torch.utils.data import DataLoader
from torchsummary import summary


# TODO: (order doesn't matter, just needs to get done)
    # create more organized and robust dataset and this time, include a sh script for
        # dataset download []
    # dataset analysis to count number of classes and images/class for research []
    # script for dataloader should be able to configure #imgs/class []
    # import and freeze vit_b_16 or vit_b_32 []
    # example pass of fake data in expected shape []
    # etc. will write more as tasks get done based on needs.

class SlugSpeciesClassifier(nn.Module):
    '''
    Species classifier to identify species of slugs.
    
    Uses ViT-B/16.
    
    Args:

    '''
    
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = vit_b_16(weights=weights, progress=True)
        

    def forward(self, x):
        return self.backbone(x)
    
    
    
model = SlugSpeciesClassifier()
summary(model)