import torch
import torch.nn as nn
import torchvision.models as models
import os
from ..config import MODELS_DIR

class CustomCNN(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()
        # Calculate final feature map size based on input size
        feature_size = (((input_size - 4) // 2 - 4) // 2 - 4) // 2
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

def create_custom_cnn():
    """Create a custom CNN model for micro-expression recognition."""
    return CustomCNN(input_size=224)

def create_mobilenet():
    """Create MobileNetV2 model with transfer learning."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model

def create_efficientnet():
    """Create EfficientNetB0 model with transfer learning."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(1280, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    return model

def save_model(model, model_name):
    """Save model to the models directory."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}.pt"))

def load_model(model_input, model_class=None):
    """Load model from a .pt file path or from the models directory by name."""
    # Determine whether input is a full path to a .pt file
    if model_input.endswith('.pt') and os.path.isfile(model_input):
        model_path = model_input
        model_name = os.path.splitext(os.path.basename(model_input))[0]
    else:
        model_name = model_input
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    
    if model_class is None:
        key = model_name.lower()
        if 'cnn' in key:
            model = create_custom_cnn()
        elif 'mobile' in key:
            model = create_mobilenet()
        elif 'efficient' in key:
            model = create_efficientnet()
        else:
            raise ValueError(f"Cannot infer model type from name: {model_name}")
    else:
        model = model_class()
    
    model.load_state_dict(torch.load(model_path))
            
    
    return model 