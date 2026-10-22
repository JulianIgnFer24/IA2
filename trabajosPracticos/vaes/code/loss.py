"""
Pérdida del VAE: ELBO negativo = reconstrucción (BCE) + divergencia KL.

Ambos términos se devuelven sumados sobre el batch (reduction="sum"); el promedio
por imagen lo hace train.py al cerrar cada época.

Nota: todo corre en float32. No se usa autocast/AMP porque binary_cross_entropy no es
numéricamente segura en precisión mixta.
"""
import torch
import torch.nn.functional as F


def vae_loss(x_recon, x, mu, logvar):
    """Devuelve (total, reconstrucción, KL), los tres como escalares sumados."""
    # Verosimilitud: Bernoulli por píxel, con x en [0, 1] como probabilidad objetivo.
    recon = F.binary_cross_entropy(x_recon, x, reduction="sum")
    # KL[ N(mu, sigma^2) || N(0, I) ], forma cerrada.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl
