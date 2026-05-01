"""Backward-compatibility shim for token indexing.

New code should import from src.indexing.ner_token_index.
"""

from src.indexing.ner_token_index import *  # noqa: F401,F403
