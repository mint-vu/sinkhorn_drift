from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EMAState:
    decay: float
    shadow: dict[str, torch.Tensor]


class EMA:
    """
    Simple EMA of parameters (float32 shadow).
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().float().clone()
                continue
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(dtype=p.dtype, device=p.device))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict, device: torch.device | str | None = None) -> None:
        self.decay = float(state["decay"])
        if device is not None:
            self.shadow = {k: v.clone().to(device) for k, v in state["shadow"].items()}
        else:
            self.shadow = {k: v.clone() for k, v in state["shadow"].items()}

