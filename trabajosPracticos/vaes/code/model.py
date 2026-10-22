"""
Autoencoder Variacional convolucional para rostros de 64x64x3.

Encoder: 4 convoluciones stride 2 (64 -> 32 -> 16 -> 8 -> 4) hasta (256, 4, 4), y dos
cabezas lineales independientes que parametrizan la posterior q(z|x) = N(mu, sigma^2 I).
Decoder: espejo con convoluciones transpuestas, salida Sigmoid en [0, 1].
"""
import torch
import torch.nn as nn

import config


class Encoder(nn.Module):
    """(3, 64, 64) -> (mu, logvar), ambos de dimensión latent_dim."""

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
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.flatten(self.conv(x))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """z de dimensión latent_dim -> (3, 64, 64) con píxeles en [0, 1]."""

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # (32, 32, 32) -> (3, 64, 64). Sin BatchNorm: Sigmoid da directo los píxeles.
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """Encoder + truco de reparametrización + Decoder."""

    def __init__(self, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, con eps ~ N(0, I).

        Mueve la estocasticidad a eps (que no lleva gradiente) para que el muestreo
        quede dentro del grafo y mu/logvar sean entrenables por retropropagación.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device=None):
        """Genera n caras nuevas decodificando z ~ N(0, I)."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    model = VAE()
    model.eval()

    x = torch.randn(8, 3, 64, 64)
    with torch.no_grad():
        x_recon, mu, logvar = model(x)

    print(f"entrada:      {tuple(x.shape)}")
    print(f"x_recon:      {tuple(x_recon.shape)}")
    print(f"mu:           {tuple(mu.shape)}")
    print(f"logvar:       {tuple(logvar.shape)}")
    assert x_recon.shape == (8, 3, 64, 64)
    assert mu.shape == (8, config.LATENT_DIM)
    assert logvar.shape == (8, config.LATENT_DIM)
    assert 0.0 <= x_recon.min() and x_recon.max() <= 1.0, "la salida debe estar en [0, 1]"

    with torch.no_grad():
        muestras = model.sample(4)
    assert muestras.shape == (4, 3, 64, 64)
    print(f"sample(4):    {tuple(muestras.shape)}")

    print(f"\nParámetros encoder: {count_params(model.encoder):,}")
    print(f"Parámetros decoder: {count_params(model.decoder):,}")
    print(f"Parámetros totales: {count_params(model):,}")
    print("\nOK")
