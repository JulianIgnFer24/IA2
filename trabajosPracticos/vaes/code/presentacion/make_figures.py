"""
Figuras y resultados numéricos para la presentación.

Compara el autoencoder determinístico baseline contra el VAE de las épocas 30 y 200.
Toda la inferencia va con model.eval() + torch.no_grad() (hay BatchNorm) y todas las
figuras usan las mismas imágenes de validación y el mismo z aleatorio, para que se
puedan leer una al lado de la otra.

Salidas: outputs/presentacion/figuras/*.png y outputs/presentacion/resultados.json
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config
from data import build_dataloaders
from model import count_params
from model_ae import AE
from visualize import _to_img, load_checkpoint

PRESENTACION_DIR = config.OUTPUT_DIR / "presentacion"
FIGURAS_DIR = PRESENTACION_DIR / "figuras"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
RESULTADOS_JSON = PRESENTACION_DIR / "resultados.json"

BASELINE_DIR = config.OUTPUT_DIR / "baseline"
AE_CHECKPOINT = BASELINE_DIR / "ae_final.pt"
AE_CSV = BASELINE_DIR / "ae_loss_history.csv"

AMARILLO = "#FFD90F"
GRIS = "#333333"
AZUL = "#1f77b4"
NARANJA = "#ff7f0e"

# Imágenes de validación usadas en todas las figuras (índices fijos).
IDX_CARAS = list(range(8))
IDX_PIPELINE = 4
IDX_VARIACIONES = 4
PARES_INTERPOLACION = [(0, 1), (2, 3), (4, 5)]
N_TSNE = 3000  # imágenes que se codifican para el t-SNE


def guardar(fig, nombre: str) -> Path:
    ruta = FIGURAS_DIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {nombre}")
    return ruta


def grilla(ax_grid, imgs, titulo=None):
    """Dibuja imgs (tensor Nx3xHxW) sobre un array de ejes ya creado."""
    for ax, img in zip(ax_grid.flat, imgs):
        ax.imshow(_to_img(img))
        ax.axis("off")
    if titulo:
        ax_grid.flat[0].figure.suptitle(titulo)


def leer_csv(ruta):
    with Path(ruta).open(newline="") as f:
        return [
            {k: (int(v) if k == "epoch" else float(v)) for k, v in fila.items()}
            for fila in csv.DictReader(f)
        ]


# ---------------------------------------------------------------------------
# 1. Pipeline: cara -> latente -> reconstrucción
# ---------------------------------------------------------------------------
@torch.no_grad()
def fig_pipeline(vae, x):
    mu, _ = vae.encode(x.unsqueeze(0).to(config.DEVICE))
    recon = vae.decode(mu)[0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4),
                             gridspec_kw={"width_ratios": [1, 1.6, 1]})

    axes[0].imshow(_to_img(x))
    axes[0].set_title("Imagen original\n64 × 64 × 3 = 12.288 valores", fontsize=11)
    axes[0].axis("off")

    mapa = mu[0].cpu().numpy().reshape(8, 16)
    im = axes[1].imshow(mapa, cmap="RdBu_r", vmin=-np.abs(mapa).max(), vmax=np.abs(mapa).max())
    axes[1].set_title("Vector latente\n128 valores (μ)", fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)

    axes[2].imshow(_to_img(recon))
    axes[2].set_title("Reconstrucción\n64 × 64 × 3", fontsize=11)
    axes[2].axis("off")

    # Las flechas se ubican en los huecos reales entre ejes (la barra de color corre
    # el panel del medio, así que posiciones fijas se superponen).
    fig.canvas.draw()
    cajas = [ax.get_position() for ax in axes]
    huecos = [(cajas[0].x1 + cajas[1].x0) / 2, (cajas[1].x1 + 0.055 + cajas[2].x0) / 2]
    for x_pos, etiqueta in zip(huecos, ["encoder", "decoder"]):
        fig.text(x_pos, 0.52, "→", fontsize=26, ha="center", va="center", color=GRIS)
        fig.text(x_pos, 0.40, etiqueta, fontsize=9, ha="center", color=GRIS,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    return guardar(fig, "pipeline_ejemplo.png")


# ---------------------------------------------------------------------------
# 2-3. Reconstrucción y generación del AE
# ---------------------------------------------------------------------------
@torch.no_grad()
def fig_ae_reconstruccion(ae, caras):
    x = caras.to(config.DEVICE)
    recon = ae(x)

    n = x.size(0)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.4))
    for i in range(n):
        axes[0, i].imshow(_to_img(x[i]))
        axes[1, i].imshow(_to_img(recon[i]))
        axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left", fontsize=10)
    axes[1, 0].set_title("Reconstrucción del AE", loc="left", fontsize=10)
    fig.suptitle("El autoencoder reconstruye bien lo que ya vio", fontsize=13)
    return guardar(fig, "ae_reconstruccion.png")


@torch.no_grad()
def fig_ae_generacion(ae, z_random):
    imgs = ae.decode(z_random.to(config.DEVICE))
    fig, axes = plt.subplots(4, 4, figsize=(7, 7.4))
    grilla(axes, imgs)
    fig.suptitle("AE: decodificar z ~ N(0, I) al azar", fontsize=13)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.93)
    return guardar(fig, "ae_generacion_aleatoria.png")


# ---------------------------------------------------------------------------
# 4. Espacio latente: PCA y t-SNE
# ---------------------------------------------------------------------------
@torch.no_grad()
def latentes_validacion(ae, vae, val_loader):
    """Devuelve (z del AE, mu del VAE) de todo el conjunto de validación."""
    zs_ae, mus_vae = [], []
    for x in val_loader:
        x = x.to(config.DEVICE)
        zs_ae.append(ae.encode(x).cpu())
        mus_vae.append(vae.encode(x)[0].cpu())
    return torch.cat(zs_ae).numpy(), torch.cat(mus_vae).numpy()


def fig_espacio_latente(z_ae, mu_vae, seed=config.SEED):
    """PCA 2D de cada latente, con 500 puntos de N(0, I) superpuestos."""
    rng = np.random.default_rng(seed)
    prior = rng.standard_normal((500, config.LATENT_DIM))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (datos, titulo) in zip(
        axes,
        [(z_ae, "Autoencoder determinístico\n(z, sin regularizar)"),
         (mu_vae, "VAE\n(μ, regularizado por KL)")],
    ):
        pca = PCA(n_components=2).fit(datos)
        p_datos = pca.transform(datos)
        p_prior = pca.transform(prior)

        ax.scatter(p_datos[:, 0], p_datos[:, 1], s=14, alpha=0.6, c=AZUL,
                   label=f"caras de validación (n={len(datos)})")
        ax.scatter(p_prior[:, 0], p_prior[:, 1], s=14, alpha=0.6, c=NARANJA,
                   marker="^", label="muestras de N(0, I) (n=500)")
        ax.set_title(titulo, fontsize=12)
        ax.set_xlabel("componente principal 1")
        ax.set_ylabel("componente principal 2")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        # El rango de los ejes es la información clave: si las caras viven en un rango
        # mucho mayor que el del prior, muestrear de N(0, I) cae fuera de los datos.
        ax.text(0.02, 0.02,
                f"rango PC1: [{p_datos[:, 0].min():.1f}, {p_datos[:, 0].max():.1f}]",
                transform=ax.transAxes, fontsize=9, color=GRIS)

    fig.suptitle("¿Dónde caen las muestras aleatorias respecto de los datos? (PCA)",
                 fontsize=13)
    fig.tight_layout()
    return guardar(fig, "espacio_latente.png")


@torch.no_grad()
def fig_tsne(ae, vae, dataset, n=N_TSNE, seed=config.SEED):
    """t-SNE 2D de los latentes del AE y del VAE sobre una muestra del dataset.

    A diferencia del ejemplo clásico con MNIST, este dataset no tiene etiquetas de
    clase, así que cada punto se colorea con el color promedio de su propia imagen:
    si el latente está organizado, los colores quedan agrupados.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    imgs, zs_ae, mus_vae = [], [], []
    for i in range(0, len(idx), 256):
        lote = torch.stack([dataset[int(j)] for j in idx[i:i + 256]]).to(config.DEVICE)
        imgs.append(lote.mean(dim=(2, 3)).cpu())  # color promedio RGB
        zs_ae.append(ae.encode(lote).cpu())
        mus_vae.append(vae.encode(lote)[0].cpu())

    colores = torch.cat(imgs).numpy().clip(0, 1)
    # Los rostros promedian casi todos a un oliva parecido, así que se estira cada canal
    # a su rango completo: no cambia qué punto es cuál, solo hace visible la diferencia.
    lo, hi = colores.min(axis=0), colores.max(axis=0)
    colores = ((colores - lo) / np.maximum(hi - lo, 1e-8)).clip(0, 1)
    z_ae = torch.cat(zs_ae).numpy()
    mu_vae = torch.cat(mus_vae).numpy()
    print(f"  (t-SNE sobre {len(colores)} imágenes, 128 dims -> 2)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (datos, titulo) in zip(
        axes,
        [(z_ae, "Autoencoder determinístico (z)"), (mu_vae, "VAE (μ)")],
    ):
        emb = TSNE(n_components=2, init="pca", random_state=seed,
                   perplexity=30).fit_transform(datos)
        # Normalización a [0, 1] para que ambos paneles compartan escala.
        emb = (emb - emb.min()) / (emb.max() - emb.min())
        ax.scatter(emb[:, 0], emb[:, 1], c=colores, s=9)
        ax.set_title(titulo, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("t-SNE del espacio latente — cada punto es una cara, coloreada con su "
                 "color promedio (contraste realzado)", fontsize=13)
    fig.tight_layout()
    return guardar(fig, "tsne_latente.png")


# ---------------------------------------------------------------------------
# 5-6. La nube gaussiana del VAE
# ---------------------------------------------------------------------------
@torch.no_grad()
def fig_variaciones_mu(vae, x, seed=config.SEED, amplif=6):
    """Original, decode(mu) sin ruido, 7 muestras, y la diferencia amplificada.

    La nube que aprende el VAE es angosta (sigma ~ 0,4), así que a simple vista las
    muestras parecen idénticas. La fila de abajo muestra |muestra - decode(mu)|
    multiplicada por `amplif` para que se vea DÓNDE varía cada una.
    """
    mu, logvar = vae.encode(x.unsqueeze(0).to(config.DEVICE))
    std = torch.exp(0.5 * logvar)

    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(7, config.LATENT_DIM, generator=g).to(config.DEVICE)
    base = vae.decode(mu)
    muestras = vae.decode(mu + std * eps)
    dif = (muestras - base).abs() * amplif

    fig, axes = plt.subplots(2, 9, figsize=(14, 3.9))
    axes[0, 0].imshow(_to_img(x)); axes[0, 0].set_title("Original", fontsize=9)
    axes[0, 1].imshow(_to_img(base[0]))
    axes[0, 1].set_title("decode(μ)\nsin ruido", fontsize=9)
    for i in range(7):
        axes[0, i + 2].imshow(_to_img(muestras[i]))
        axes[0, i + 2].set_title(f"muestra {i + 1}", fontsize=9)
        axes[1, i + 2].imshow(_to_img(dif[i]))
    for col in (0, 1):
        axes[1, col].axis("off")
    axes[1, 2].text(-0.12, 0.5, f"diferencia\ncontra μ (×{amplif})",
                    transform=axes[1, 2].transAxes, fontsize=9,
                    ha="right", va="center", color=GRIS)
    for ax in axes.flat:
        ax.axis("off")

    d_media = (muestras - base).abs().mean().item()
    fig.suptitle(f"El VAE codifica una nube: z = μ + σ · ε    "
                 f"(σ medio {std.mean():.2f} → las muestras difieren {d_media:.3f} "
                 f"por píxel)", fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.12, top=0.78)
    return guardar(fig, "vae_variaciones_mu.png")


@torch.no_grad()
def fig_nube_agrandada(vae, x, seed=config.SEED, ks=(0, 1, 2, 3, 6)):
    """Filas con k creciente, columnas con el mismo eps: z = mu + k*sigma*eps.

    Se incluye k = 6, bastante más allá de lo que aprendió el modelo, porque con el
    sigma real (~0,4) hasta k = 3 la cara casi no se mueve. Cada fila lleva la
    diferencia medida por píxel contra decode(mu), para que la progresión sea un número
    y no una impresión.
    """
    mu, logvar = vae.encode(x.unsqueeze(0).to(config.DEVICE))
    std = torch.exp(0.5 * logvar)

    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(8, config.LATENT_DIM, generator=g).to(config.DEVICE)
    base = vae.decode(mu)

    fig, axes = plt.subplots(len(ks), 8, figsize=(11, 1.0 + 0.95 * len(ks)))
    for fila, k in enumerate(ks):
        imgs = vae.decode(mu + k * std * eps)
        dif = (imgs - base).abs().mean().item()
        for col in range(8):
            axes[fila, col].imshow(_to_img(imgs[col]))
            axes[fila, col].axis("off")
        axes[fila, 0].text(-0.08, 0.5, f"k = {k}\n({dif:.3f}/píxel)",
                           transform=axes[fila, 0].transAxes, fontsize=10,
                           va="center", ha="right", color=GRIS)
    fig.suptitle(f"z = μ + k · σ · ε   —   con σ medio {std.mean():.2f}, hay que pasarse "
                 "bastante de k = 1 para que la cara cambie", fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.90, left=0.12)
    return guardar(fig, "vae_nube_agrandada.png")


# ---------------------------------------------------------------------------
# 7. Diagramas de arquitectura
# ---------------------------------------------------------------------------
TEAL_TRAP = "#1F7A6E"
TEAL_RELLENO = "#E3EFED"
NARANJA_TRAP = "#CE6A28"
NARANJA_RELLENO = "#FBE8DA"


def _lienzo(ancho_u, alto_u, escala=0.1):
    """Ejes con aspecto igual: 1 unidad de datos = `escala` pulgadas.

    Hace falta para que las caras salgan cuadradas y no estiradas.
    """
    fig, ax = plt.subplots(figsize=(ancho_u * escala, alto_u * escala))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, ancho_u); ax.set_ylim(0, alto_u)
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def _trapecio(ax, x, w, yc, h_izq, h_der, texto, borde, relleno, fontsize=13):
    """Trapecio del encoder (angosta hacia la derecha) o del decoder (ensancha)."""
    ax.add_patch(mpatches.Polygon(
        [(x, yc - h_izq / 2), (x + w, yc - h_der / 2),
         (x + w, yc + h_der / 2), (x, yc + h_izq / 2)],
        closed=True, facecolor=relleno, edgecolor=borde, linewidth=2.6, zorder=3))
    ax.text(x + w / 2, yc, texto, ha="center", va="center", fontsize=fontsize,
            weight="bold", color="#222222", zorder=4)


def _caja(ax, x, yc, w, h, texto, fontsize=12, borde="#333333", relleno="white",
          lw=2.0, negrita=True):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, yc - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor=relleno, edgecolor=borde, linewidth=lw, zorder=3))
    ax.text(x + w / 2, yc, texto, ha="center", va="center", fontsize=fontsize,
            weight="bold" if negrita else "normal", color="#222222", zorder=4)


def _flecha(ax, x1, y1, x2, y2, color="#444444", lw=1.9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                                mutation_scale=15))


def _cara(ax, img, x, yc, lado, etiquetas):
    """Imagen cuadrada con su epígrafe debajo."""
    ax.imshow(img, extent=[x, x + lado, yc - lado / 2, yc + lado / 2], zorder=3)
    y = yc - lado / 2 - 3.2
    for linea in etiquetas:
        ax.text(x + lado / 2, y, linea, ha="center", va="top", fontsize=12,
                color="#222222")
        y -= 5.0


@torch.no_grad()
def fig_arquitectura(modelo, x, variacional: bool):
    """Diagrama conceptual encoder / latente / decoder, con caras reales a los lados.

    Mismo lenguaje visual que la clase teórica: trapecios que angostan y ensanchan,
    la imagen de entrada a la izquierda y la reconstrucción a la derecha. El del VAE
    agrega las dos cabezas lineales y el sorteo reparametrizado en el medio.
    """
    modelo.eval()
    xb = x.unsqueeze(0).to(config.DEVICE)
    recon = (modelo(xb)[0] if variacional else modelo(xb))[0]
    entrada, salida = _to_img(x), _to_img(recon)

    if not variacional:
        fig, ax = _lienzo(140, 46)
        yc, lado = 30, 24
        _cara(ax, entrada, 3, yc, lado, ["imagen x", "64 × 64 × 3"])
        _flecha(ax, 29, yc, 35, yc)
        _trapecio(ax, 37, 15, yc, 26, 15, "encoder\ne(x)", TEAL_TRAP, TEAL_RELLENO)
        _flecha(ax, 54, yc, 60, yc)
        _caja(ax, 62, yc, 16, 15, "z\nlatente")
        _flecha(ax, 80, yc, 86, yc)
        _trapecio(ax, 88, 15, yc, 15, 26, "decoder\nd(z)", NARANJA_TRAP,
                  NARANJA_RELLENO)
        _flecha(ax, 105, yc, 111, yc)
        _cara(ax, salida, 113, yc, lado, ["reconstrucción x̂", "64 × 64 × 3"])
        ax.text(70, 45, f"12.288 números   →   {config.LATENT_DIM} números   →   "
                          f"12.288 números", ha="center", va="top", fontsize=13,
                color="#222222")
        ax.text(70, 9, "lo que no pasa por el cuello\nno se recupera", ha="center",
                va="top", fontsize=12, style="italic", color="#555555")
    else:
        fig, ax = _lienzo(160, 52)
        yc, lado = 30, 22
        _cara(ax, entrada, 2, yc, lado, ["3 × 64 × 64"])
        _flecha(ax, 26, yc, 31, yc)
        _trapecio(ax, 33, 14, yc, 26, 14, "encoder", TEAL_TRAP, TEAL_RELLENO,
                  fontsize=12)
        _caja(ax, 53, 42, 18, 11, f"μ  ({config.LATENT_DIM})", fontsize=12,
              borde=TEAL_TRAP)
        _caja(ax, 53, 18, 18, 11, f"σ  ({config.LATENT_DIM})", fontsize=12,
              borde=TEAL_TRAP)
        _flecha(ax, 47.5, yc + 2, 52, 40)
        _flecha(ax, 47.5, yc - 2, 52, 22)
        _caja(ax, 78, yc, 23, 16, f"z = μ + σ·ε\n({config.LATENT_DIM})", fontsize=12)
        _flecha(ax, 71.5, 41, 77, 34)
        _flecha(ax, 71.5, 19, 77, 26)
        _caja(ax, 80, 47, 19, 9, "ε ~ N(0, I)", fontsize=11, borde="#8A8A8A",
              relleno="#F2F2F2", lw=1.6, negrita=False)
        _flecha(ax, 89.5, 42, 89.5, 38.5, color="#8A8A8A")
        _flecha(ax, 102, yc, 107, yc)
        _trapecio(ax, 109, 14, yc, 14, 26, "decoder", NARANJA_TRAP, NARANJA_RELLENO,
                  fontsize=12)
        _flecha(ax, 125, yc, 130, yc)
        _cara(ax, salida, 132, yc, lado, ["3 × 64 × 64"])
        ax.text(66, 9, "dos cabezas lineales: media y desvío por dimensión",
                ha="center", va="top", fontsize=12.5, color=TEAL_TRAP)

    nombre = "vae_arquitectura.png" if variacional else "ae_arquitectura.png"
    return guardar(fig, nombre)


# ---------------------------------------------------------------------------
# 8. Curvas de pérdida
# ---------------------------------------------------------------------------
def fig_curvas(history_vae, history_ae):
    mejor = min(history_vae, key=lambda r: r["val_total"])
    ep = [r["epoch"] for r in history_vae]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    paneles = [
        ("recon", "Reconstrucción (BCE)"),
        ("kl", "Divergencia KL"),
        ("total", "Pérdida total (BCE + KL)"),
    ]
    for ax, (clave, titulo) in zip(axes, paneles):
        ax.plot(ep, [r[f"train_{clave}"] for r in history_vae], label="VAE train", color=AZUL)
        ax.plot(ep, [r[f"val_{clave}"] for r in history_vae], label="VAE val", color=NARANJA)
        if clave == "recon" and history_ae:
            ax.plot([r["epoch"] for r in history_ae],
                    [r["train_recon"] for r in history_ae],
                    label="AE train (sin KL)", color="#2ca02c", linestyle="--")
        ax.axvline(mejor["epoch"], color=GRIS, linestyle=":", linewidth=1.3)
        ax.annotate(f"mín. val\népoca {mejor['epoch']}",
                    xy=(mejor["epoch"], ax.get_ylim()[1]),
                    xytext=(mejor["epoch"] + 12, 0.80), textcoords=("data", "axes fraction"),
                    fontsize=9, color=GRIS)
        ax.set_title(titulo); ax.set_xlabel("Época"); ax.set_ylabel("nats por imagen")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle("Curvas de pérdida por imagen", fontsize=13)
    fig.tight_layout()
    return guardar(fig, "curvas_perdida.png")


# ---------------------------------------------------------------------------
# 9-11. Comparaciones entre modelos y épocas
# ---------------------------------------------------------------------------
@torch.no_grad()
def fig_recon_30_vs_200(vae30, vae200, caras):
    x = caras.to(config.DEVICE)
    r30, r200 = vae30(x)[0], vae200(x)[0]

    n = x.size(0)
    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 5.0))
    for i in range(n):
        for fila, img in enumerate([x[i], r30[i], r200[i]]):
            axes[fila, i].imshow(_to_img(img))
            axes[fila, i].axis("off")
    for fila, etiqueta in enumerate(["Original", "VAE época 30", "VAE época 200"]):
        axes[fila, 0].set_title(etiqueta, loc="left", fontsize=10)
    fig.suptitle("Reconstrucción: época 30 vs época 200", fontsize=13)
    return guardar(fig, "reconstrucciones_30_vs_200.png")


@torch.no_grad()
def _comparar_dos(izq_imgs, der_imgs, izq_tit, der_tit, titulo, nombre):
    fig, subfigs = plt.subplots(4, 8, figsize=(14, 7.4))
    for col_off, imgs in [(0, izq_imgs), (4, der_imgs)]:
        for i in range(16):
            ax = subfigs[i // 4, col_off + i % 4]
            ax.imshow(_to_img(imgs[i]))
            ax.axis("off")
    fig.text(0.27, 0.935, izq_tit, ha="center", fontsize=12, color=GRIS)
    fig.text(0.74, 0.935, der_tit, ha="center", fontsize=12, color=GRIS)
    fig.suptitle(titulo, fontsize=14, y=0.99)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.90)
    return guardar(fig, nombre)


@torch.no_grad()
def fig_ae_vs_vae(ae, vae200, z_random):
    z = z_random.to(config.DEVICE)
    return _comparar_dos(ae.decode(z), vae200.decode(z),
                         "Autoencoder (sin KL)", "VAE época 200",
                         "Los mismos z ~ N(0, I) decodificados por cada modelo",
                         "ae_vs_vae_generacion.png")


@torch.no_grad()
def fig_vae_30_vs_200(vae30, vae200, z_random):
    z = z_random.to(config.DEVICE)
    return _comparar_dos(vae30.decode(z), vae200.decode(z),
                         "VAE época 30", "VAE época 200",
                         "Caras generadas con los mismos z ~ N(0, I)",
                         "vae_muestras_epoca30_vs_200.png")


# ---------------------------------------------------------------------------
# 12. Interpolaciones
# ---------------------------------------------------------------------------
@torch.no_grad()
def fig_interpolacion(vae, val_ds, ia, ib, n_fig, steps=10):
    img_a, img_b = val_ds[ia], val_ds[ib]
    par = torch.stack([img_a, img_b]).to(config.DEVICE)
    mu, _ = vae.encode(par)
    ts = torch.linspace(0, 1, steps, device=config.DEVICE).view(-1, 1)
    transicion = vae.decode((1 - ts) * mu[0] + ts * mu[1])

    fig, axes = plt.subplots(1, steps + 2, figsize=(1.15 * (steps + 2), 1.9))
    axes[0].imshow(_to_img(img_a)); axes[0].set_title("A", fontsize=10)
    for j in range(steps):
        axes[j + 1].imshow(_to_img(transicion[j]))
    axes[-1].imshow(_to_img(img_b)); axes[-1].set_title("B", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04)
    return guardar(fig, f"interpolacion_{n_fig}.png")


# ---------------------------------------------------------------------------
# resultados.json
# ---------------------------------------------------------------------------
def minutos_de_log(ruta):
    """Tiempo real de pared que imprimió el script de entrenamiento, si está el log."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return None
    m = re.search(r"Entrenamiento terminado en ([\d.]+) min", ruta.read_text(errors="ignore"))
    return float(m.group(1)) if m else None


@torch.no_grad()
def uso_del_latente(vae, val_loader):
    """Cuántas de las 128 dimensiones usa realmente el VAE.

    Una dimensión con sigma ~ 1 quedó igual al prior: el encoder no guardó nada ahí.
    Una con sigma chico lleva información.
    """
    stds = []
    for x in val_loader:
        _, logvar = vae.encode(x.to(config.DEVICE))
        stds.append(torch.exp(0.5 * logvar).cpu())
    std = torch.cat(stds).mean(dim=0)
    return {
        "sigma_medio": round(float(std.mean()), 4),
        "dims_activas": int((std < 0.5).sum()),
        "dims_colapsadas": int((std > 0.9).sum()),
        "dims_totales": config.LATENT_DIM,
    }


def escribir_resultados(history_vae, history_ae, vae200, ae, ckpt_ae, minutos_vae,
                        latente=None):
    mejor = min(history_vae, key=lambda r: r["val_total"])

    def fila(h, e):
        r = h[e - 1]
        return {k: round(v, 2) for k, v in r.items() if k != "epoch"}

    resultados = {
        "vae": {
            "parametros": count_params(vae200),
            "epocas": len(history_vae),
            # Dos medidas: el tiempo real de pared y la suma de las épocas (que no
            # incluye el arranque, los checkpoints ni las figuras intermedias).
            "minutos_entrenamiento": minutos_de_log(config.OUTPUT_DIR / "train_200.log")
                                     or round(minutos_vae, 1),
            "minutos_solo_epocas": round(minutos_vae, 1),
            "uso_del_latente": latente,
            "epoca_30": fila(history_vae, 30),
            "epoca_200": fila(history_vae, len(history_vae)),
            "mejor_validacion": {
                "epoca": mejor["epoch"],
                "val_total": round(mejor["val_total"], 2),
                "val_recon": round(mejor["val_recon"], 2),
                "val_kl": round(mejor["val_kl"], 2),
            },
        },
        "ae": {
            "parametros": count_params(ae),
            "epocas": len(history_ae),
            "minutos_entrenamiento": round(ckpt_ae.get("minutos", 0.0), 1),
            "minutos_solo_epocas": round(sum(r["seconds"] for r in history_ae) / 60, 1),
            "final": {k: round(v, 2) for k, v in history_ae[-1].items() if k != "epoch"},
            "mejor_validacion": {
                "epoca": min(history_ae, key=lambda r: r["val_recon"])["epoch"],
                "val_recon": round(min(r["val_recon"] for r in history_ae), 2),
            },
        },
        "comparacion_recon_final": {
            "vae_train_recon": round(history_vae[-1]["train_recon"], 2),
            "ae_train_recon": round(history_ae[-1]["train_recon"], 2),
            "diferencia": round(history_vae[-1]["train_recon"] - history_ae[-1]["train_recon"], 2),
        },
        "entrenamiento": {
            "optimizador": "Adam",
            "lr": config.LR,
            "batch_size": config.BATCH_SIZE,
            "latent_dim": config.LATENT_DIM,
            "img_size": config.IMG_SIZE,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            "torch": torch.__version__,
        },
    }
    RESULTADOS_JSON.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"\nresultados: {RESULTADOS_JSON}")
    return resultados


def main():
    config.set_seeds()
    config.print_device()

    if not AE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Falta el AE baseline: {AE_CHECKPOINT}. "
            "Correr primero: python code/baseline/train_ae.py --epochs 200"
        )

    # --- modelos ---
    ckpt_ae = torch.load(AE_CHECKPOINT, map_location=config.DEVICE, weights_only=False)
    ae = AE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    ae.load_state_dict(ckpt_ae["model_state_dict"])
    ae.eval()

    vae30, _ = load_checkpoint(config.CHECKPOINT_DIR / "vae_epoch_30.pt")
    vae200, ckpt200 = load_checkpoint(config.CHECKPOINT_DIR / "vae_epoch_200.pt")

    # --- datos ---
    _, val_loader, val_ds = build_dataloaders(num_workers=0)
    dataset_completo = val_ds.dataset
    caras = torch.stack([val_ds[i] for i in IDX_CARAS])

    g = torch.Generator().manual_seed(config.SEED)
    z_random = torch.randn(16, config.LATENT_DIM, generator=g)

    history_vae = leer_csv(config.LOSS_CSV)
    history_ae = leer_csv(AE_CSV)
    minutos_vae = sum(r["seconds"] for r in history_vae) / 60

    print(f"\nAE:  {count_params(ae):,} parámetros, {len(history_ae)} épocas")
    print(f"VAE: {count_params(vae200):,} parámetros, {len(history_vae)} épocas")

    print("\nFiguras:")
    fig_pipeline(vae200, caras[IDX_PIPELINE])
    fig_ae_reconstruccion(ae, caras)
    fig_ae_generacion(ae, z_random)

    z_ae, mu_vae = latentes_validacion(ae, vae200, val_loader)
    fig_espacio_latente(z_ae, mu_vae)
    fig_tsne(ae, vae200, dataset_completo)

    fig_variaciones_mu(vae200, val_ds[IDX_VARIACIONES])
    fig_nube_agrandada(vae200, val_ds[IDX_VARIACIONES])
    fig_arquitectura(ae, val_ds[IDX_PIPELINE], variacional=False)
    fig_arquitectura(vae200, val_ds[IDX_PIPELINE], variacional=True)
    fig_curvas(history_vae, history_ae)
    fig_recon_30_vs_200(vae30, vae200, caras)
    fig_ae_vs_vae(ae, vae200, z_random)
    fig_vae_30_vs_200(vae30, vae200, z_random)
    for n, (ia, ib) in enumerate(PARES_INTERPOLACION, start=1):
        fig_interpolacion(vae200, val_ds, ia, ib, n)

    latente = uso_del_latente(vae200, val_loader)
    print(f"\nUso del latente: {latente['dims_activas']}/{latente['dims_totales']} "
          f"dimensiones activas (σ medio {latente['sigma_medio']})")
    res = escribir_resultados(history_vae, history_ae, vae200, ae, ckpt_ae, minutos_vae,
                              latente)

    # Chequeo del supuesto central: sin KL, el AE debería reconstruir mejor.
    c = res["comparacion_recon_final"]
    print(f"\nReconstrucción final en train — VAE {c['vae_train_recon']} vs "
          f"AE {c['ae_train_recon']} (diferencia {c['diferencia']:+.2f})")
    if c["diferencia"] <= 0:
        print("  ATENCIÓN: el VAE reconstruye mejor que el AE, lo contrario de lo esperado.")


if __name__ == "__main__":
    main()
