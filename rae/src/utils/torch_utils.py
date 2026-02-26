from __future__ import annotations

import pickle
from typing import Any

import torch


def safe_torch_load(path: str, map_location: Any = "cpu") -> Any:
    """
    Load checkpoints compatibly across PyTorch versions.

    PyTorch 2.6 defaults torch.load(..., weights_only=True), which can fail for
    legacy checkpoints. We first try the safe path and only fall back to
    weights_only=False when needed.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch without the weights_only kwarg.
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        print(
            f"[torch.load] weights_only=True failed for {path}; "
            "retrying with weights_only=False. Ensure this file is trusted."
        )
        return torch.load(path, map_location=map_location, weights_only=False)
