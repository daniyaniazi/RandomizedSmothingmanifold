from __future__ import annotations

import numpy as np
import torch


def to_numpy_array(x: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr
