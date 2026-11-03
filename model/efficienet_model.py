import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B3_Weights


def create_efficientnet_b3(num_classes=2, pretrained=True):

    # Load pretrained EfficientNet-B3
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None

    model = models.efficientnet_b3(weights=weights)

    # Get the number of input features
    in_features = model.classifier[1].in_features

    # Replace the original classifier
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model