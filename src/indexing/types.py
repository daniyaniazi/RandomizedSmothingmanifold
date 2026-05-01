from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class NeighborIndex:
    backend: str
    dim: int
    metric: str
    vectors: np.ndarray | None = None
    index: Any = None
