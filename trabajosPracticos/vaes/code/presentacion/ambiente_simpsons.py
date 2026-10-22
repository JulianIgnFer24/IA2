"""
Ambientación de la presentación: los motivos de la apertura de Los Simpson.

El cielo con nubes, el pizarrón de Bart y el sofá se DIBUJAN acá en vez de bajar
capturas de la apertura: las capturas son material con copyright, y además dibujarlos
permite que entren en la paleta del resto de la presentación y den a cualquier
resolución.

Lo único descargado es `donut.svg` de Wikimedia Commons, que está en dominio público
sin restricciones de uso.

Las caras son las del propio dataset del trabajo.

Salidas: outputs/presentacion/ambiente/*.png
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from data import build_transform

AMBIENTE = config.OUTPUT_DIR / "presentacion" / "ambiente"
AMBIENTE.mkdir(parents=True, exist_ok=True)

# Paleta de la serie
CIELO = "#6FA8DC"
CIELO_HONDO = "#4E88BE"
NUBE = "#FFFFFF"
AMARILLO = "#FFD90F"
PIZARRON = "#2C4A3E"
MARCO = "#8A5A3B"
TIZA = "#F2F0E6"
SOFA = "#C96A4A"
PARED = "#E8C66B"


def guardar(fig, nombre, transparente=False):
    ruta = AMBIENTE / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", pad_inches=0,
                transparent=transparente,
                facecolor="none" if transparente else fig.get_facecolor())
    plt.close(fig)
    print(f"  {nombre}")
    return ruta


def _nube(ax, x, y, escala, color=NUBE, alpha=1.0):
    """Nube de la apertura: círculos superpuestos de tamaños distintos."""
    bultos = [(-1.5, -0.1, 0.75), (-0.7, 0.28, 1.0), (0.25, 0.38, 1.15),
              (1.15, 0.12, 0.85), (1.85, -0.18, 0.6), (0.3, -0.3, 1.0)]
    for dx, dy, r in bultos:
        ax.add_patch(plt.Circle((x + dx * escala, y + dy * escala), r * escala,
                                color=color, alpha=alpha, zorder=2, linewidth=0))


def fig_cielo(ancho=13.333, alto=7.5, n_nubes=9, seed=7):
    """Cielo con nubes: fondo de la portada."""
    rng = random.Random(seed)
    fig, ax = plt.subplots(figsize=(ancho, alto))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, ancho); ax.set_ylim(0, alto); ax.axis("off")

    # Degradado vertical del cielo
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, ancho, 0, alto], aspect="auto", origin="lower",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  "cielo", [CIELO_HONDO, CIELO]), zorder=0)

    # Banda superior: nubes grandes a lo ancho.
    for _ in range(n_nubes):
        _nube(ax, rng.uniform(-0.5, ancho + 0.5), rng.uniform(alto * 0.74, alto * 1.04),
              rng.uniform(0.5, 0.95), alpha=rng.uniform(0.92, 1.0))
    # Columna derecha: dos nubes a media altura, lejos del título.
    for _ in range(2):
        _nube(ax, rng.uniform(ancho * 0.78, ancho + 0.4),
              rng.uniform(alto * 0.34, alto * 0.62), rng.uniform(0.5, 0.75), alpha=0.95)
    # Nubes bajas tenues, solo para dar profundidad al pie.
    for _ in range(3):
        _nube(ax, rng.uniform(ancho * 0.25, ancho), rng.uniform(0, alto * 0.16),
              rng.uniform(0.5, 0.85), alpha=0.22)
    return guardar(fig, "cielo.png")


def fig_pizarron(lineas, nombre, ancho=9.0, alto=5.0, seed=3):
    """El gag del pizarrón: la misma frase escrita varias veces, con temblor.

    `lineas` es la frase (o lista de frases) que escribe Bart.
    """
    if isinstance(lineas, str):
        lineas = [lineas]
    rng = random.Random(seed)

    fig, ax = plt.subplots(figsize=(ancho, alto))
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor(MARCO)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # Marco de madera y paño verde
    ax.add_patch(mp.FancyBboxPatch((0.18, 0.18), 9.64, 9.64,
                                   boxstyle="round,pad=0.02,rounding_size=0.1",
                                   facecolor=PIZARRON, edgecolor="#6E4428",
                                   linewidth=6, zorder=1))
    # Restos de tiza: manchas claras muy suaves
    for _ in range(40):
        ax.add_patch(plt.Circle((rng.uniform(0.5, 9.5), rng.uniform(0.6, 9.4)),
                                rng.uniform(0.05, 0.35), color=TIZA,
                                alpha=rng.uniform(0.012, 0.05), zorder=2, linewidth=0))

    n = len(lineas)
    paso = min(1.3, 7.6 / max(n, 1))
    y0 = 5.0 + (n - 1) * paso / 2   # bloque centrado en el paño
    for i, texto in enumerate(lineas):
        ax.text(0.75 + rng.uniform(-0.1, 0.1), y0 - i * paso + rng.uniform(-0.07, 0.07),
                texto, fontsize=19 if n <= 5 else 16, color=TIZA, family="DejaVu Sans",
                rotation=rng.uniform(-0.9, 0.9), va="center", zorder=3, alpha=0.93)

    # Canaleta inferior con la tiza
    ax.add_patch(mp.Rectangle((0.18, 0.05), 9.64, 0.42, facecolor="#6E4428",
                              edgecolor="none", zorder=4))
    ax.add_patch(mp.FancyBboxPatch((8.6, 0.16), 0.75, 0.16,
                                   boxstyle="round,pad=0.01,rounding_size=0.08",
                                   facecolor=TIZA, edgecolor="none", zorder=5))
    return guardar(fig, nombre)


def fig_sofa(ancho=9.0, alto=5.0):
    """El gag del sofá: sala vacía, para el cierre."""
    fig, ax = plt.subplots(figsize=(ancho, alto))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")

    ax.add_patch(mp.Rectangle((0, 1.5), 10, 4.1, facecolor=PARED, edgecolor="none"))
    ax.add_patch(mp.Rectangle((0, 0), 10, 1.5, facecolor="#9C5B3C", edgecolor="none"))

    # Sofá
    ax.add_patch(mp.FancyBboxPatch((1.6, 1.0), 6.8, 1.9,
                                   boxstyle="round,pad=0.02,rounding_size=0.25",
                                   facecolor=SOFA, edgecolor="#9B4A30", linewidth=3))
    ax.add_patch(mp.FancyBboxPatch((1.6, 2.3), 6.8, 1.5,
                                   boxstyle="round,pad=0.02,rounding_size=0.3",
                                   facecolor=SOFA, edgecolor="#9B4A30", linewidth=3))
    for x in (1.35, 8.15):
        ax.add_patch(mp.FancyBboxPatch((x, 1.1), 0.75, 2.3,
                                       boxstyle="round,pad=0.02,rounding_size=0.3",
                                       facecolor="#B85A3E", edgecolor="#9B4A30",
                                       linewidth=3))
    for x in (3.3, 5.0, 6.7):
        ax.plot([x, x], [1.15, 2.85], color="#9B4A30", linewidth=2.5)

    # Cuadro y lámpara, como en la sala
    ax.add_patch(mp.Rectangle((4.1, 4.1), 1.9, 1.15, facecolor="#7FB3D5",
                              edgecolor="#6E4428", linewidth=4))
    ax.add_patch(mp.Polygon([[8.9, 3.3], [9.9, 3.3], [9.65, 4.1], [9.15, 4.1]],
                            closed=True, facecolor="#E8A33D", edgecolor="#9B4A30",
                            linewidth=2))
    ax.plot([9.4, 9.4], [1.5, 3.3], color="#9B4A30", linewidth=3)
    return guardar(fig, "sofa.png")


def fig_tira_caras(n=12, alto_px=140, seed=11):
    """Tira horizontal de caras del dataset, para franjas decorativas."""
    rng = np.random.default_rng(seed)
    carpeta = config.PROJECT_DIR / "archive" / "cropped"
    rutas = sorted(p for p in carpeta.iterdir() if p.suffix == ".png")
    elegidas = [rutas[i] for i in rng.choice(len(rutas), n, replace=False)]

    tr = build_transform()
    tira = Image.new("RGB", (n * alto_px, alto_px))
    for i, r in enumerate(elegidas):
        with Image.open(r) as im:
            arr = (tr(im.convert("RGB")).permute(1, 2, 0).numpy() * 255).astype("uint8")
        tira.paste(Image.fromarray(arr).resize((alto_px, alto_px), Image.LANCZOS),
                   (i * alto_px, 0))
    ruta = AMBIENTE / "tira_caras.png"
    tira.save(ruta)
    print("  tira_caras.png")
    return ruta


def main():
    print("Ambiente Simpsons:")
    fig_cielo()
    fig_sofa()
    fig_tira_caras()
    fig_pizarron(["Un autoencoder no puede inventar caras"] * 5,
                 "pizarron_acto1.png")
    fig_pizarron(["Voy a codificar una nube, no un punto"] * 5,
                 "pizarron_acto2.png", seed=5)
    fig_pizarron(["Voy a entrenar con todos los datos juntos"] * 5,
                 "pizarron_acto3.png", seed=9)
    fig_pizarron(["El dato es su propia etiqueta",
                  "Sin KL el latente tiene huecos",
                  "La KL ata las nubes a N(0, I)",
                  "z = mu + sigma * epsilon",
                  "Simplificar es una direccion"],
                 "pizarron_cierre.png", seed=13)


if __name__ == "__main__":
    main()
