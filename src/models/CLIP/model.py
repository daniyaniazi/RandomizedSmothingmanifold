"""CLIP wrapper for RoCOCO image-text retrieval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def _load_clip(model_name: str = "ViT-B/32", device: str = "cuda"):
    try:
        import clip as _clip
    except ImportError:
        raise ImportError("pip install git+https://github.com/openai/CLIP.git")
    model, preprocess = _clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, _clip


class CLIPWrapper:
    """Thin wrapper around OpenAI CLIP.

    encode_images / encode_texts return L2-normalized float32 CPU tensors.
    CLIP outputs are already approximately unit-norm; we normalize explicitly
    to guarantee exact unit vectors for dot-product cosine similarity.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess, self._clip = _load_clip(model_name, str(self.device))
        self.model_name = model_name
        self.dim: int = self.model.visual.output_dim

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: List[str],
        batch_size: int = 256,
        num_workers: int = 4,
        desc: str = "Encoding images",
    ) -> torch.Tensor:
        """Encode images from disk paths using the RoCoCoDataset image-only loader.

        Returns: (N, dim) float32 CPU tensor, L2-normalized.
        """
        from src.dataloaders.rococo import RoCoCoImageDataset
        from torch.utils.data import DataLoader

        ds     = RoCoCoImageDataset(image_paths, self.preprocess)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

        embeddings = torch.zeros(len(image_paths), self.dim)
        for batch_imgs, indices in tqdm(loader, desc=desc, leave=False):
            feats = self.model.encode_image(batch_imgs.to(self.device)).float()
            feats = F.normalize(feats, dim=-1).cpu()
            embeddings[indices] = feats
        return embeddings

    @torch.no_grad()
    def encode_texts(
        self,
        captions: List[str],
        batch_size: int = 512,
        desc: str = "Encoding texts",
    ) -> torch.Tensor:
        """Encode text captions.

        Returns: (M, dim) float32 CPU tensor, L2-normalized.
        """
        embeddings = torch.zeros(len(captions), self.dim)
        for start in tqdm(range(0, len(captions), batch_size), desc=desc, leave=False):
            batch  = captions[start:start + batch_size]
            tokens = self._clip.tokenize(batch, truncate=True).to(self.device)
            feats  = self.model.encode_text(tokens).float()
            feats  = F.normalize(feats, dim=-1).cpu()
            embeddings[start:start + len(batch)] = feats
        return embeddings
