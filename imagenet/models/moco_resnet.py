from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def _extract_checkpoint_state_dict(ckpt_obj) -> dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise ValueError("Unsupported MoCo checkpoint format: expected dict with 'state_dict' or a raw tensor state dict.")


def _map_moco_keys_to_resnet(
    state_dict: dict[str, torch.Tensor],
    *,
    encoder_prefix: str = "module.encoder_q.",
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith(encoder_prefix):
            continue
        new_key = key[len(encoder_prefix) :]
        if new_key.startswith("fc.") or new_key.startswith("head."):
            continue
        out[new_key] = value
    return out


def _build_resnet_backbone(arch: str) -> nn.Module:
    if arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported MoCo arch: {arch} (supported: resnet50)")
    model.fc = nn.Identity()
    return model


@dataclass(frozen=True)
class MoCoMeta:
    arch: str
    ckpt_path: str
    loaded_keys: int


class MoCoResNetFeatureEncoder(nn.Module):
    """
    ResNet feature wrapper for MoCo checkpoints.

    Exposes 'forward_feature_maps(...)' with the same interface used by MAE encoder.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_feature_maps(self, x: torch.Tensor, *, every_n_blocks: int = 2) -> list[torch.Tensor]:
        if every_n_blocks <= 0:
            raise ValueError(f"every_n_blocks must be > 0, got {every_n_blocks}")

        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected RGB input [N,3,H,W], got {tuple(x.shape)}")

        def run_stage(inp: torch.Tensor, stage: nn.Sequential) -> tuple[torch.Tensor, list[torch.Tensor]]:
            outs: list[torch.Tensor] = []
            h = inp
            n_blocks = len(stage)
            for i, block in enumerate(stage, start=1):
                h = block(h)
                if i % every_n_blocks == 0:
                    outs.append(h)
            if n_blocks % every_n_blocks != 0:
                outs.append(h)
            return h, outs

        b = self.backbone
        h = b.conv1(x)
        h = b.bn1(h)
        h = b.relu(h)
        h = b.maxpool(h)

        maps: list[torch.Tensor] = []
        h, out1 = run_stage(h, b.layer1)
        maps.extend(out1)
        h, out2 = run_stage(h, b.layer2)
        maps.extend(out2)
        h, out3 = run_stage(h, b.layer3)
        maps.extend(out3)
        h, out4 = run_stage(h, b.layer4)
        maps.extend(out4)
        return maps


def load_moco_resnet_encoder(
    ckpt_path: str,
    *,
    arch: str = "resnet50",
    device: torch.device,
) -> tuple[MoCoResNetFeatureEncoder, MoCoMeta]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = _extract_checkpoint_state_dict(ckpt)

    mapped = _map_moco_keys_to_resnet(state_dict, encoder_prefix="module.encoder_q.")
    if not mapped:
        mapped = _map_moco_keys_to_resnet(state_dict, encoder_prefix="encoder_q.")
    if not mapped:
        raise ValueError(
            "No encoder_q keys found in MoCo checkpoint. Expected prefixes 'module.encoder_q.' or 'encoder_q.'."
        )

    backbone = _build_resnet_backbone(str(arch))
    model_state = backbone.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    for key, value in mapped.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value

    if not compatible:
        raise ValueError("No compatible backbone keys found after mapping MoCo checkpoint.")

    missing, unexpected = backbone.load_state_dict(compatible, strict=False)
    unexpected_set = set(unexpected)
    if unexpected_set:
        raise RuntimeError(f"Unexpected keys while loading MoCo backbone: {sorted(unexpected_set)}")

    non_fc_missing = [k for k in missing if not k.startswith("fc.")]
    if non_fc_missing:
        raise RuntimeError(f"Missing non-fc keys while loading MoCo backbone: {non_fc_missing[:20]}")

    encoder = MoCoResNetFeatureEncoder(backbone=backbone)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    meta = MoCoMeta(
        arch=str(arch),
        ckpt_path=str(Path(ckpt_path).resolve()),
        loaded_keys=len(compatible),
    )
    return encoder, meta
