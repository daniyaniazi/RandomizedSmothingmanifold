"""Generate and cache CLIP embeddings for all COCO images and RoCOCO captions.

Run ONCE before any evaluation. Outputs:
  {embedding_cache_dir}/image_embeddings.pt
    → {"embeddings": Tensor(N,512), "image_ids": List[str]}

  {embedding_cache_dir}/{ann_stem}_captions.pt
    → {"embeddings": Tensor(M,512), "captions": List[str], "image_indices": List[int]}

Usage:
    python -m src.experiments.clip_embedding.generate_embeddings \\
        --config src/configs/experiments/rococo_clip_baseline.yaml

    # Force re-encode even if cache exists
    python -m src.experiments.clip_embedding.generate_embeddings \\
        --config src/configs/experiments/rococo_clip_baseline.yaml --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.configs.rococo_clip_schema import load_rococo_config
from src.dataloaders.rococo import RoCoCoDataset
from src.models.CLIP.model import CLIPWrapper


def generate_image_embeddings(
    clip: CLIPWrapper,
    dataset: RoCoCoDataset,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    force: bool = False,
) -> Path:
    out_path = cache_dir / "image_embeddings.pt"
    if out_path.exists() and not force:
        print(f"  SKIP image embeddings — already exists: {out_path}")
        return out_path

    print(f"  Encoding {len(dataset):,} images …")
    embs = clip.encode_images(
        dataset.image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        desc="Image embeddings",
    )
    torch.save({"embeddings": embs, "image_ids": dataset.image_ids}, out_path)
    print(f"  Saved image embeddings: {out_path}  shape={tuple(embs.shape)}")
    return out_path


def generate_caption_embeddings(
    clip: CLIPWrapper,
    dataset: RoCoCoDataset,
    ann_stem: str,
    cache_dir: Path,
    batch_size: int,
    force: bool = False,
) -> Path:
    out_path = cache_dir / f"{ann_stem}_captions.pt"
    if out_path.exists() and not force:
        print(f"  SKIP {ann_stem} captions — already exists: {out_path}")
        return out_path

    captions, img2txt, wrongtext = dataset.get_retrieval_pool(ann_stem)
    if not captions:
        print(f"  SKIP {ann_stem} — no captions found")
        return out_path

    print(f"  Encoding {len(captions):,} captions for {ann_stem} "
          f"(GT pool + {len(wrongtext)} adversarial) …")
    embs = clip.encode_texts(captions, batch_size=batch_size,
                             desc=f"Caption embs [{ann_stem}]")
    torch.save({
        "embeddings":  embs,
        "captions":    captions,
        "img2txt":     img2txt,      # image_idx → list of GT caption indices
        "wrongtext":   wrongtext,    # adversarial caption indices (RSMS)
    }, out_path)
    print(f"  Saved {ann_stem} captions: {out_path}  "
          f"shape={tuple(embs.shape)}  wrongtext={len(wrongtext)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for RoCOCO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true",
                        help="Re-encode even if cache files exist")
    args = parser.parse_args()

    cfg = load_rococo_config(args.config)
    cache_dir = Path(cfg.embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"CLIP model:   {cfg.clip_model}")
    print(f"Cache dir:    {cache_dir}")
    print(f"Device:       {cfg.device}")

    clip = CLIPWrapper(cfg.clip_model, cfg.device)
    print(f"CLIP dim:     {clip.dim}")

    dataset = RoCoCoDataset(
        image_dir=cfg.image_dir,
        annotation_dir=cfg.annotation_dir,
        annotation_files=cfg.annotation_files,
    )

    # 1. Image embeddings
    generate_image_embeddings(clip, dataset, cache_dir,
                              cfg.batch_size, cfg.num_workers, args.force)

    # 2. Caption embeddings for each annotation file
    for ann_file in cfg.annotation_files:
        from pathlib import Path as _P
        ann_stem = _P(ann_file).stem
        generate_caption_embeddings(clip, dataset, ann_stem,
                                    cache_dir, cfg.batch_size, args.force)

    print("\nDone. All embeddings saved to:", cache_dir)


if __name__ == "__main__":
    main()
