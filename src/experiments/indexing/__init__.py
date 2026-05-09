"""Indexing scripts for building kNN indexes.

Scripts:
    - ner_tokens: Extract NER token embeddings and build index
    - celeba_images: Extract image embeddings (pixel/latent) and build index

Usage:
    python -m src.experiments.indexing.ner_tokens --config ... --checkpoint ... --out-dir ...
    python -m src.experiments.indexing.celeba_images --config ... --space pixel|latent
"""
