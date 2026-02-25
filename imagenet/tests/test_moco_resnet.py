from __future__ import annotations

import torch
from torchvision import models

from imagenet.models.moco_resnet import _map_moco_keys_to_resnet, load_moco_resnet_encoder


def test_map_moco_keys_to_resnet_filters_prefix_and_heads() -> None:
    state_dict = {
        "module.encoder_q.layer1.0.conv1.weight": torch.randn(64, 64, 3, 3),
        "module.encoder_q.fc.weight": torch.randn(1000, 2048),
        "module.encoder_q.head.weight": torch.randn(128, 2048),
        "module.encoder_k.layer1.0.conv1.weight": torch.randn(64, 64, 3, 3),
    }
    mapped = _map_moco_keys_to_resnet(state_dict, encoder_prefix="module.encoder_q.")
    assert "layer1.0.conv1.weight" in mapped
    assert "fc.weight" not in mapped
    assert "head.weight" not in mapped
    assert all(not k.startswith("module.") for k in mapped.keys())


def test_load_moco_resnet_encoder_from_synthetic_checkpoint(tmp_path) -> None:
    backbone = models.resnet50(weights=None)
    source = backbone.state_dict()

    ckpt_state = {}
    for key, value in source.items():
        ckpt_state[f"module.encoder_q.{key}"] = value.clone()
    ckpt_state["module.encoder_q.fc.0.weight"] = torch.randn(2048, 2048)
    ckpt_state["module.encoder_q.fc.2.weight"] = torch.randn(128, 2048)

    ckpt_path = tmp_path / "moco_synth.pt"
    torch.save({"state_dict": ckpt_state}, ckpt_path)

    encoder, meta = load_moco_resnet_encoder(str(ckpt_path), arch="resnet50", device=torch.device("cpu"))
    assert meta.arch == "resnet50"
    assert meta.loaded_keys > 0

    x = torch.randn(2, 3, 224, 224, dtype=torch.float32)
    maps = encoder.forward_feature_maps(x, every_n_blocks=2)
    assert len(maps) > 0
    for fmap in maps:
        assert torch.isfinite(fmap).all()
