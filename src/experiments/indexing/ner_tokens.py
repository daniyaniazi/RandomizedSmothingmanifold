"""Build NER token embedding index.

Extracts token embeddings from a trained NER model and saves them
for use in manifold smoothing certification.

Usage:
    python -m src.experiments.indexing.ner_tokens \\
        --config CONFIG --checkpoint CHECKPOINT --out-dir OUTPUT_DIR

Example:
    python -m src.experiments.indexing.ner_tokens \\
        --config src/configs/experiments/ner_conll2003_bert_certify.yaml \\
        --checkpoint output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt \\
        --out-dir output/ner_conll2003_bert/token_embeddings/train/last \\
        --split train

Output files:
    - token_vectors.npz: Embedding vectors (N x D)
    - token_metadata.json: Token texts and labels
    - label_map.json: Label ID to name mapping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.configs.schema import DataloaderConfig, DatasetConfig
from src.dataloaders.ner_conll import build_conll_dataloaders
from src.indexing.ner_token_index import (
    extract_token_vectors,
    deduplicate_token_vectors,
    save_token_index_artifacts,
    build_token_neighbor_index,
)
from src.indexing.annoy_indexing import save_annoy_index
from src.indexing.faiss_indexing.backend import save_faiss_index
from src.models.transformer.ner.model import TransformerNER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labeled NER token-vector artifacts.")
    parser.add_argument("--config", type=str, required=True, help="Path to resolved experiment config YAML.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained NER checkpoint.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory for saved token vectors and metadata.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split to index.")
    parser.add_argument("--layer-index", type=int, default=None, help="Optional transformer layer index to extract.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for faster debugging.")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication of same (token, label) embeddings.")
    parser.add_argument("--backend", type=str, default="annoy", choices=["annoy", "faiss", "torch"], help="Index backend.")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Distance metric.")
    parser.add_argument("--n-trees", type=int, default=50, help="Number of Annoy trees.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    raw_dataset_cfg = dict(config.get("dataset", {}))
    raw_dataset_cfg["num_workers"] = 0
    raw_loader_cfg = dict(config.get("dataloader", {}))
    raw_loader_cfg["shuffle_train"] = False

    dataset_cfg = DatasetConfig(**raw_dataset_cfg)
    loader_cfg = DataloaderConfig(**raw_loader_cfg)
    encoder_name = config["model"]["encoder_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_conll_dataloaders(
        dataset_cfg=dataset_cfg,
        loader_cfg=loader_cfg,
        encoder_name=encoder_name,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    num_labels = int(state_dict["classifier.weight"].shape[0])

    model = TransformerNER(
        encoder_name=encoder_name,
        num_labels=num_labels,
        id2label=data.id2label,
        label2id=data.label2id,
        dropout=config.get("model", {}).get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    split_to_loader = {
        "train": data.train_loader,
        "validation": data.val_loader,
        "test": data.test_loader,
    }
    loader = split_to_loader[args.split]

    vectors, token_texts, label_ids = extract_token_vectors(
        model=model,
        loader=loader,
        device=device,
        tokenizer_name=data.tokenizer_name,
        layer_index=args.layer_index,
        max_batches=args.max_batches,
    )

    if not args.no_dedup:
        vectors, token_texts, label_ids = deduplicate_token_vectors(
            vectors, token_texts, label_ids,
        )

    save_token_index_artifacts(out_dir, vectors, token_texts, label_ids)
    (out_dir / "label_map.json").write_text(json.dumps({"id2label": data.id2label}, indent=2))

    # ── Build and save the search index ──
    print(f"Building {args.backend} index (metric={args.metric}, n_trees={args.n_trees})...")
    index = build_token_neighbor_index(
        vectors=vectors,
        backend=args.backend,
        metric=args.metric,
        n_trees=args.n_trees,
    )

    # Save index to the path certification expects
    index_dir = out_dir.parent.parent / "indexes" / args.split / _layer_tag(args.layer_index) / args.metric / args.backend
    index_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "annoy":
        index_path = index_dir / "index.ann"
        save_annoy_index(index, str(index_path))
    elif args.backend == "faiss":
        index_path = index_dir / "index.faiss"
        save_faiss_index(index, str(index_path))
    else:
        index_path = index_dir / "(in-memory)"

    print(f"saved_vectors={out_dir / 'token_vectors.npz'}")
    print(f"saved_metadata={out_dir / 'token_metadata.json'}")
    print(f"saved_label_map={out_dir / 'label_map.json'}")
    print(f"saved_index={index_path}")
    print(f"num_vectors={vectors.shape[0]}")
    print(f"embedding_dim={vectors.shape[1]}")


def _layer_tag(layer_index: int | None) -> str:
    return f"layer_{layer_index}" if layer_index is not None else "last"


if __name__ == "__main__":
    main()