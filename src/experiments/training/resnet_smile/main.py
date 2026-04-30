"""Main entrypoint for training ResNet smile classifiers."""

from __future__ import annotations

import argparse

from src.configs.train_smile_io import load_smile_training_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet on smile classification")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/training/smile_resnet_celeba.yaml",
        help="Path to smile training YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_smile_training_config(args.config)
    from src.train.train_smile_resnet import run_smile_training

    run_smile_training(cfg)


if __name__ == "__main__":
    main()