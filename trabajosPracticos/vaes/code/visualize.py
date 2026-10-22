"""
Figuras del VAE: reconstrucciones, muestreo, interpolaciones y curvas de pérdida.

Todas las funciones ponen el modelo en eval() y desactivan el grafo. Es imprescindible:
el modelo tiene BatchNorm, y en modo train() usaría las estadísticas del batch (con
batches chicos o un z arbitrario las imágenes salen distorsionadas).

Uso: python code/visualize.py --checkpoint outputs/checkpoints/vae_epoch_200.pt
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from model import VAE


def _to_img(t: torch.Tensor):
    """(3, H, W) en [0, 1] -> (H, W, 3) en CPU/NumPy, listo para imshow."""
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def _guardar(fig, nombre: str) -> Path:
    ruta = config.FIGURE_DIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {ruta}")
    return ruta


def _sufijo(epoch) -> str:
    return f"epoca_{epoch}" if epoch is not None else "final"


@torch.no_grad()
def plot_reconstructions(model, x_fixed, epoch=None):
    """Grilla de 2 filas: originales arriba, reconstrucciones abajo."""
    model.eval()
    device = next(model.parameters()).device
    x = x_fixed.to(device)
    x_recon, _, _ = model(x)

    n = x.size(0)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(_to_img(x[i]))
        axes[1, i].imshow(_to_img(x_recon[i]))
        for fila in (0, 1):
            axes[fila, i].axis("off")
    axes[0, 0].set_title("Original", loc="left", fontsize=10)
    axes[1, 0].set_title("Reconstrucción", loc="left", fontsize=10)
    titulo = "Reconstrucciones sobre validación"
    if epoch is not None:
        titulo += f" — época {epoch}"
    fig.suptitle(titulo, fontsize=12)
    return _guardar(fig, f"reconstrucciones_{_sufijo(epoch)}.png")


@torch.no_grad()
def plot_samples(model, z_fixed, epoch=None):
    """Grilla 8x8 de caras nuevas decodificadas desde z ~ N(0, I)."""
    model.eval()
    device = next(model.parameters()).device
    muestras = model.decode(z_fixed.to(device))

    n = muestras.size(0)
    lado = int(n ** 0.5)
    fig, axes = plt.subplots(lado, lado, figsize=(lado, lado))
    for i, ax in enumerate(axes.flat):
        ax.imshow(_to_img(muestras[i]))
        ax.axis("off")
    titulo = "Muestras desde z ~ N(0, I)"
    if epoch is not None:
        titulo += f" — época {epoch}"
    fig.suptitle(titulo, fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.94)
    return _guardar(fig, f"muestras_{_sufijo(epoch)}.png")


@torch.no_grad()
def interpolate(model, img_a, img_b, steps: int = 10):
    """Interpolación lineal en el latente entre dos imágenes.

    Usa mu (no z muestreado) para que la transición sea determinística.
    Devuelve el tensor (steps, 3, 64, 64) de imágenes decodificadas.
    """
    model.eval()
    device = next(model.parameters()).device
    par = torch.stack([img_a, img_b]).to(device)

    mu, _ = model.encode(par)
    z_a, z_b = mu[0], mu[1]

    ts = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    z_t = (1 - ts) * z_a + ts * z_b
    return model.decode(z_t)


@torch.no_grad()
def plot_interpolations(model, val_dataset, n_pairs: int = 3, steps: int = 10, epoch=None):
    """n_pairs filas: imagen A | transición | imagen B, con pares fijos de validación."""
    model.eval()
    # Índices fijos para poder comparar la misma figura entre épocas.
    pares = [(0, 1), (2, 3), (4, 5)][:n_pairs]

    fig, axes = plt.subplots(n_pairs, steps + 2, figsize=(1.1 * (steps + 2), 1.3 * n_pairs))
    axes = axes.reshape(n_pairs, steps + 2)

    for fila, (ia, ib) in enumerate(pares):
        img_a, img_b = val_dataset[ia], val_dataset[ib]
        transicion = interpolate(model, img_a, img_b, steps=steps)

        axes[fila, 0].imshow(_to_img(img_a))
        for j in range(steps):
            axes[fila, j + 1].imshow(_to_img(transicion[j]))
        axes[fila, steps + 1].imshow(_to_img(img_b))
        for ax in axes[fila]:
            ax.axis("off")

    axes[0, 0].set_title("A", fontsize=9)
    axes[0, steps + 1].set_title("B", fontsize=9)
    titulo = "Interpolación en el espacio latente"
    if epoch is not None:
        titulo += f" — época {epoch}"
    fig.suptitle(titulo, fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.88)
    return _guardar(fig, f"interpolaciones_{_sufijo(epoch)}.png")


def plot_loss_curves(history, nombre: str = "curvas_perdida.png"):
    """Tres paneles (total, reconstrucción, KL) con train vs validación por época."""
    if not history:
        print("  (historial vacío, no se grafican curvas)")
        return None

    epocas = [h["epoch"] for h in history]
    paneles = [
        ("total", "Pérdida total (ELBO negativo)"),
        ("recon", "Reconstrucción (BCE)"),
        ("kl", "Divergencia KL"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (clave, titulo) in zip(axes, paneles):
        ax.plot(epocas, [h[f"train_{clave}"] for h in history], label="Entrenamiento")
        ax.plot(epocas, [h[f"val_{clave}"] for h in history], label="Validación")
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.set_ylabel("nats por imagen")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Curvas de pérdida del VAE", fontsize=13)
    fig.tight_layout()
    return _guardar(fig, nombre)


def load_history_csv(ruta=config.LOSS_CSV):
    """Lee outputs/loss_history.csv como lista de dicts con valores numéricos."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return []
    with ruta.open(newline="") as f:
        filas = list(csv.DictReader(f))
    return [
        {k: (int(v) if k == "epoch" else float(v)) for k, v in fila.items()}
        for fila in filas
    ]


def load_checkpoint(ruta, device=config.DEVICE):
    """Reconstruye el VAE desde un checkpoint y devuelve (modelo, checkpoint)."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {ruta.resolve()}")

    ckpt = torch.load(ruta, map_location=device, weights_only=False)
    latent_dim = ckpt.get("config", {}).get("latent_dim", config.LATENT_DIM)
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint cargado: {ruta}  (época {ckpt.get('epoch', '?')})")
    return model, ckpt


def main():
    parser = argparse.ArgumentParser(description="Figuras del VAE desde un checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="ruta al .pt (p. ej. outputs/checkpoints/vae_epoch_200.pt)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="carpeta de imágenes (default: archive/cropped)")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    import data as data_mod  # import diferido: solo hace falta en la CLI

    config.set_seeds()
    config.print_device()

    model, ckpt = load_checkpoint(args.checkpoint)
    epoch = ckpt.get("epoch")

    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.DATA_DIR
    _, _, val_ds = data_mod.build_dataloaders(
        data_dir=data_dir, num_workers=args.num_workers
    )

    # Los tensores fijos vienen del checkpoint para que las figuras sean comparables
    # con las generadas durante el entrenamiento.
    x_fixed = ckpt.get("x_fixed")
    if x_fixed is None:
        x_fixed = torch.stack([val_ds[i] for i in range(config.N_RECON)])
    z_fixed = ckpt.get("z_fixed")
    if z_fixed is None:
        g = torch.Generator().manual_seed(config.SEED)
        z_fixed = torch.randn(config.N_SAMPLES, model.latent_dim, generator=g)

    print("\nGenerando figuras:")
    plot_reconstructions(model, x_fixed, epoch)
    plot_samples(model, z_fixed, epoch)
    plot_interpolations(model, val_ds, n_pairs=3, epoch=epoch)
    plot_loss_curves(ckpt.get("history") or load_history_csv())


if __name__ == "__main__":
    main()
