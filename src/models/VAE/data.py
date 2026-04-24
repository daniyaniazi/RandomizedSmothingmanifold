"""
Dataset construction and image utility helpers.
Supports: imagefolder, cifar10, mnist.
All datasets are normalised to the same mean/std so ConvVAE is dataset-agnostic.
"""

from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_MEAN: List[float] = [0.5, 0.5, 0.5]
DEFAULT_STD: List[float] = [0.5, 0.5, 0.5]


def build_transform(
    image_size: int,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD,
) -> transforms.Compose:
    """Return a standard Compose that resizes, crops, converts to RGB, and normalises."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_dataset(
    dataset_name: str,
    data_root: str,
    image_size: int = 64,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD,
) -> Tuple[Dataset, int]:
    """
    Build a torchvision dataset from a short name.

    Returns (dataset, in_channels).
    in_channels is always 3 (images are converted to RGB).
    """
    transform = build_transform(image_size=image_size, mean=mean, std=std)
    name = dataset_name.lower()

    if name == "imagefolder":
        ds = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
    elif name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
    elif name == "mnist":
        ds = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )
    else:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            "Choose from: imagefolder, cifar10, mnist"
        )

    return ds, 3  # always 3 channels after RGB conversion


def denormalize(
    x: torch.Tensor,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD,
) -> torch.Tensor:
    """
    Reverse normalisation and clamp to [0, 1].
    Works for both single images (C,H,W) and batches (N,C,H,W).
    """
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype)

    # Broadcast to (..., C, 1, 1)
    while mean_t.dim() < x.dim():
        mean_t = mean_t.unsqueeze(-1)
        std_t = std_t.unsqueeze(-1)

    return torch.clamp(x * std_t + mean_t, 0.0, 1.0)


def tensor_to_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert a (C,H,W) float tensor in [0,1] to a (H,W,C) uint8 numpy array."""
    return np.transpose(t.numpy(), (1, 2, 0))
