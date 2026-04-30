"""Compatibility shim for old NER-specific sweep entrypoint."""

from .cert_n_sweep import main


if __name__ == "__main__":
    main()
