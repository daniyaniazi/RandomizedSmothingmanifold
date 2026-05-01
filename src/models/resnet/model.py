"""ResNet classifiers for CelebA / CelebA-HQ image classification.

Supports resnet18, resnet34, resnet50. The final FC layer is replaced with a
single-logit binary head (Smiling, or any other binary attribute).
"""

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

_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights),
    "resnet34": (resnet34, ResNet34_Weights),
    "resnet50": (resnet50, ResNet50_Weights),
}


def build_resnet_classifier(
    name: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.0,
    num_classes: int = 1,
) -> nn.Module:
    """Return a ResNet-based classifier.

    Args:
        name: ``resnet18``, ``resnet34``, or ``resnet50``.
        pretrained: Load ImageNet weights when ``True``.
        dropout: If > 0, insert a Dropout layer before the final Linear.
        num_classes: Output dimensionality. 1 = binary sigmoid head.

    Returns:
        A ``torchvision.models.ResNet`` with a custom FC head.
    """
    name = name.lower()
    if name not in _BUILDERS:
        raise ValueError(f"Unsupported ResNet: {name}. Choose from {list(_BUILDERS)}")

    build_fn, weights_cls = _BUILDERS[name]
    weights = weights_cls.DEFAULT if pretrained else None
    model = build_fn(weights=weights)

    in_features = model.fc.in_features
    if dropout > 0.0:
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


# Convenience alias kept for compatibility
build_resnet_smile_classifier = build_resnet_classifier
