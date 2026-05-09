"""Train NER token classification model.

Usage:
    python -m src.experiments.training.ner.main \
        --config src/configs/experiments/ner_conll2003_bert.yaml

Example:
    # Train BERT-based NER model on CoNLL-2003
    python -m src.experiments.training.ner.main \
        --config src/configs/experiments/ner_conll2003_bert.yaml

    # Train DistilBERT-based NER model
    python -m src.experiments.training.ner.main \
        --config src/configs/experiments/ner_conll2003_distilbert.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import the actual training function
from src.models.transformer.ner.train import main as train_main


def main():
    """Train NER model."""
    parser = argparse.ArgumentParser(
        description="Train NER token classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train BERT-based NER model
  python -m src.experiments.training.ner.main \\
      --config src/configs/experiments/ner_conll2003_bert.yaml

  # Train DistilBERT-based NER model  
  python -m src.experiments.training.ner.main \\
      --config src/configs/experiments/ner_conll2003_distilbert.yaml
        """
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML")
    
    args = parser.parse_args()
    
    # Validate config path
    if not Path(args.config).exists():
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)
    
    # Override sys.argv for the training script
    sys.argv = ["train", "--config", args.config]
    train_main()


if __name__ == "__main__":
    main()
