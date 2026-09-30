from __future__ import annotations

import math

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, nonlinear: bool = True):
        super().__init__()
        hidden = max(16, min(128, input_dim // 2))
        if nonlinear:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, input_dim))
        else:
            # Sans biais : cas pédagogique proche de l'ACP sur données centrées.
            self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
            self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encode(x))


class ConvAutoencoder1D(nn.Module):
    """CNN 1D qui garantit une reconstruction à la longueur exacte."""
    def __init__(self, input_length: int = 336, latent_dim: int = 8):
        super().__init__()
        self.input_length = input_length
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(8, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        encoded_length = math.ceil(math.ceil(math.ceil(input_length / 2) / 2) / 2)
        self.encoded_shape = (32, encoded_length)
        flat = 32 * encoded_length
        self.to_latent = nn.Linear(flat, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 4, stride=2, padding=1),
        )

    def encode(self, x):
        h = self.encoder_conv(x.unsqueeze(1))
        return self.to_latent(h.flatten(1))

    def forward(self, x):
        z = self.encode(x)
        h = self.from_latent(z).view(-1, *self.encoded_shape)
        reconstruction = self.decoder_conv(h).squeeze(1)
        return reconstruction[:, : self.input_length]


class ConditionalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, context_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 64), nn.ReLU(), nn.Linear(64, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, context):
        return self.decoder(torch.cat([self.encode(x), context], dim=1))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def encode(self, x):
        h = self.hidden(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
