"""
Autoencoder determinístico (baseline del VAE).

Idéntico al VAE salvo por un punto: el encoder termina en UNA capa lineal que devuelve
z directamente, sin mu, sin logvar y sin reparametrización. El decoder es exactamente el
mismo módulo que usa el VAE (se importa de model.py).

Sirve para mostrar empíricamente por qué hace falta el término KL: entrenado solo con
reconstrucción, el AE ubica cada cara en un punto aislado del latente y deja huecos, así
que decodificar un z al azar no produce una cara.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

import config
from model import Decoder, count_params


class EncoderAE(nn.Module):
    """(3, 64, 64) -> z en R^latent_dim. Mismo stack convolucional que el VAE."""

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()  # 256 * 4 * 4 = 4096
        # Única diferencia con el VAE: una sola cabeza, sin distribución.
        self.fc_z = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        return self.fc_z(self.flatten(self.conv(x)))


class AE(nn.Module):
    """Encoder determinístico + el mismo decoder del VAE."""

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = EncoderAE(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Sin muestreo: la reconstrucción es una función determinística de x.
        return self.decode(self.encode(x))


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    model = AE()
    model.eval()

    x = torch.randn(8, 3, 64, 64)
    with torch.no_grad():
        z = model.encode(x)
        x_recon = model(x)

    print(f"entrada:  {tuple(x.shape)}")
    print(f"z:        {tuple(z.shape)}")
    print(f"x_recon:  {tuple(x_recon.shape)}")
    assert z.shape == (8, config.LATENT_DIM)
    assert x_recon.shape == (8, 3, 64, 64)
    assert 0.0 <= x_recon.min() and x_recon.max() <= 1.0, "la salida debe estar en [0, 1]"

    n_enc, n_dec = count_params(model.encoder), count_params(model.decoder)
    print(f"\nParámetros encoder: {n_enc:,}")
    print(f"Parámetros decoder: {n_dec:,}")
    print(f"Parámetros totales: {n_enc + n_dec:,}")
    # El AE tiene exactamente una cabeza Linear(4096, 128) menos que el VAE (524.416
    # parámetros): 2.958.659 - 524.416 = 2.434.243.
    assert n_enc + n_dec == 2_434_243, f"se esperaban 2.434.243 parámetros, hay {n_enc + n_dec}"
    print("\nOK")
