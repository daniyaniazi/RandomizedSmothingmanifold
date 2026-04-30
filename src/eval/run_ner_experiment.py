"""Compatibility shim for old NER-specific eval entrypoint."""

from .run_experiment import parse_args, run


if __name__ == "__main__":
    args = parse_args()
    run(args.config, checkpoint_path=args.checkpoint, metrics_out_dir=args.metrics_out_dir)
