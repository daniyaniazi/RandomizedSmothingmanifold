"""Experiment runner scripts.

This module contains executable scripts for running experiments.

Structure:
    experiments/
    ├── training/           # Train models
    │   ├── resnet_smile/   # CelebA smile classifier
    │   └── vae/            # VAE for latent space
    │
    ├── indexing/           # Build kNN indexes
    │   ├── ner_tokens.py   # NER token embeddings
    │   └── celeba_images.py # Image embeddings (pixel/latent)
    │
    └── certify/            # Run certification
        ├── ner.py          # NER certification
        └── celeba.py       # CelebA certification

Usage Examples:

    # 1. TRAINING
    python -m src.experiments.training.resnet_smile.main \\
        --config src/configs/training/smile_resnet_celeba.yaml
    
    python -m src.experiments.training.vae.main \\
        --config src/configs/training/vae_celeba.yaml

    # 2. INDEXING (for manifold smoothing)
    python -m src.experiments.indexing.ner_tokens \\
        --config src/configs/experiments/ner_conll2003_bert_certify.yaml \\
        --checkpoint output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt \\
        --out-dir output/ner_conll2003_bert/token_embeddings/train/last

    python -m src.experiments.indexing.celeba_images \\
        --config src/configs/experiments/certify_celeba_latent_128.yaml \\
        --space latent

    # 3. CERTIFICATION
    python -m src.experiments.certify.ner \\
        --config src/configs/experiments/ner_conll2003_bert_certify.yaml \\
        --checkpoint output/ner_conll2003_bert/ner_bert_conll2003_finetune/model.pt

    python -m src.experiments.certify.celeba \\
        --config src/configs/experiments/certify_celeba_latent_128.yaml

These scripts use modular components from:
    - src/smoothing/    # Smoother classes (Isotropic, Manifold)
    - src/certify/      # Certification logic (voting, radii)
    - src/indexing/     # kNN index building
    - src/tasks/        # Task-specific utilities & visualization
    - src/models/       # Model architectures
"""
