"""ResNet builders for binary smile classification."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


def build_resnet_smile_classifier(name: str = "resnet18", pretrained: bool = True, dropout: float = 0.0):
    model_name = name.lower()

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif model_name == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet model name: {name}")

    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    else:
        model.fc = nn.Linear(in_features, 1)
    return model