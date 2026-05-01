"""Backward-compatibility shim. New code should import from src.models.resnet."""

from src.models.resnet import build_resnet_classifier as build_resnet_smile_classifier  # noqa: F401
