"""
Binary classifier CNN model for distinguishing slugs from non-slugs.

This module should implement:
1. BinarySlugClassifier class - A CNN-based model for binary classification
2. Supporting functions for creating and configuring the classifier

Implementation guidelines:
- Use a lightweight CNN backbone (MobileNetV3, EfficientNet-B0, ResNet18)
- Support pre-trained weights and backbone freezing
- Add a custom classification head with proper regularization
- Include prediction functionality with confidence scoring
"""

# TODO: Implementation

# Example structure:
"""
class BinarySlugClassifier(ClassificationModel):
    '''
    Binary classifier to identify slug vs. non-slug images.
    
    Uses a lightweight CNN architecture (MobileNetV3 or EfficientNet)
    pre-trained on ImageNet with a custom classification head.
    
    Args:
        model_name: Base model architecture ('mobilenet_v3_small', 'efficientnet_b0', etc.)
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone network
        dropout_rate: Dropout rate for classification head
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
    '''
    
    def __init__(self, model_name='mobilenet_v3_small', pretrained=True, 
                 freeze_backbone=False, dropout_rate=0.2, 
                 learning_rate=1e-4, weight_decay=1e-5):
        # TODO: Initialize model
        pass
    
    def forward(self, x):
        # TODO: Forward pass
        pass
    
    def predict(self, x, return_confidence=False):
        # TODO: Make predictions with optional confidence scores
        pass


def create_binary_classifier(config):
    '''
    Create a binary classifier from configuration.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        Configured BinarySlugClassifier
    '''
    # TODO: Parse config and create model
    pass
"""
