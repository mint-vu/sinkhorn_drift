"""MNIST models: ConvAE (autoencoder) and MLPGenerator (drifting generator)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """Convolutional autoencoder: MNIST 1x28x28 <-> latent_dim.

    Encoder: Conv(1->32) -> Conv(32->64) -> FC(3136->128) -> FC(128->latent_dim)
    Decoder: FC(latent_dim->128) -> FC(128->3136) -> ConvT(64->32) -> ConvT(32->1)
    """

    def __init__(self, latent_dim: int = 6):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7 -> 3136 -> 128 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

        # Decoder: latent_dim -> 128 -> 3136 -> 64x7x7 -> 32x14x14 -> 1x28x28
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 28, 28] -> [B, latent_dim]"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim] -> [B, 1, 28, 28]"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        return self.decode(z), z


class MLPGenerator(nn.Module):
    """MLP generator for drifting in latent space.

    Interface matches DiTLatentB2: forward(noise, class_labels, omega) -> latent.
    """

    def __init__(
        self,
        num_classes: int = 10,
        noise_dim: int = 6,
        latent_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 3,
        class_emb_dim: int = 32,
        omega_emb_dim: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        # Class conditioning: +1 for unconditional token (index = num_classes)
        self.class_emb = nn.Embedding(num_classes + 1, class_emb_dim)

        # Omega conditioning
        self.omega_mlp = nn.Sequential(
            nn.Linear(1, omega_emb_dim),
            nn.SiLU(),
            nn.Linear(omega_emb_dim, omega_emb_dim),
        )

        # MLP body: [noise + class_emb + omega_emb] -> hidden -> ... -> latent_dim
        in_dim = noise_dim + class_emb_dim + omega_emb_dim
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        noise: torch.Tensor,
        class_labels: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """
        noise:        [B, noise_dim]
        class_labels: [B] int64, 0..num_classes-1 or -1 for unconditional
        omega:        [B] float, CFG scale
        returns:      [B, latent_dim]
        """
        b = noise.shape[0]
        cls = class_labels.clone()
        cls = torch.where(cls < 0, torch.full_like(cls, self.num_classes), cls)
        c_emb = self.class_emb(cls)
        o_emb = self.omega_mlp(omega.view(b, 1))
        inp = torch.cat([noise, c_emb, o_emb], dim=-1)
        return self.mlp(inp)
