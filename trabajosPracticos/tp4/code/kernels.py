"""
Visualización de los kernels aprendidos por las capas convolucionales (TP4).

Carga los modelos entrenados de results/*.keras y grafica, para cada uno, los
pesos de su primera capa convolucional: una grilla con un kernel por filtro.

Uso:
    python code/kernels.py            # primera conv de todos los experimentos
    python code/kernels.py 1 3        # sólo los experimentos 1 y 3
    python code/kernels.py --todas    # todas las capas conv de cada modelo

Escala de color: común a toda la capa y simétrica alrededor de cero (rojo =
peso positivo, azul = negativo, blanco = ~0), de modo que los kernels sean
comparables entre sí y se vea qué filtros aprendieron respuestas fuertes y
cuáles quedaron cerca de cero.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common import RESULTS_DIR, save_fig
from models import EXPERIMENTS

MAX_FILTROS = 64  # tope de kernels dibujados por capa (las capas profundas tienen cientos)
NCOLS = 8


def capas_conv(model: tf.keras.Model) -> list:
    """Capas Conv2D del modelo, en orden, incluyendo las anidadas en submodelos."""
    encontradas = []

    def recorrer(capas):
        for c in capas:
            if isinstance(c, tf.keras.layers.Conv2D):
                encontradas.append(c)
            elif hasattr(c, "layers"):
                recorrer(c.layers)

    recorrer(model.layers)
    return encontradas


def kernels_2d(layer: tf.keras.layers.Conv2D) -> tuple[np.ndarray, int]:
    """
    (kernels, canales_entrada) con kernels de forma (n_filtros, k, k).

    Los pesos vienen como (k, k, C_in, F). En la primera capa C_in = 1 y el
    kernel es directamente una imagen; en las capas profundas se promedian los
    canales de entrada para poder mostrarlo como una sola imagen.
    """
    w = layer.get_weights()[0]
    canales = w.shape[2]
    plano = w.mean(axis=2) if canales > 1 else w[:, :, 0, :]
    return np.moveaxis(plano, -1, 0), canales


def plot_kernels(layer: tf.keras.layers.Conv2D, exp: dict) -> None:
    kernels, canales = kernels_2d(layer)
    n_total, k = len(kernels), kernels.shape[1]
    kernels = kernels[:MAX_FILTROS]
    n = len(kernels)

    ncols = min(NCOLS, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.15 * ncols + 1.2, 1.15 * nrows + 1.4))
    axes = np.atleast_1d(axes).ravel()

    lim = float(np.abs(kernels).max()) or 1.0  # escala simétrica común a la capa
    for ax, kernel, i in zip(axes, kernels, range(n)):
        ax.imshow(kernel, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(f"{i + 1}", fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes[n:]:
        ax.axis("off")

    detalle = f"{n_total} kernels de {k}×{k}"
    if canales > 1:
        detalle += f" (promedio de {canales} canales de entrada)"
    if n < n_total:
        detalle += f" — se muestran los primeros {n}"
    detalle += f"  ·  escala ±{lim:.2f}"
    fig.suptitle(
        f"Exp {exp['id']} — {exp['nombre']} · capa `{layer.name}`\n{detalle}",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, f"kernels_{exp['key']}_{layer.name}.png")


def main() -> None:
    args = sys.argv[1:]
    todas = "--todas" in args
    ids = {int(a) for a in args if a != "--todas"}
    seleccion = [e for e in EXPERIMENTS if not ids or e["id"] in ids]
    if not seleccion:
        raise SystemExit(f"Ids válidos: {[e['id'] for e in EXPERIMENTS]}")

    for exp in seleccion:
        path = RESULTS_DIR / f"{exp['key']}.keras"
        if not path.exists():
            print(f"Exp {exp['id']}: falta {path.name}, se omite (correr train.py primero)")
            continue

        model = tf.keras.models.load_model(path, compile=False)
        convs = capas_conv(model)
        if not convs:
            print(f"Exp {exp['id']}: el modelo no tiene capas convolucionales")
            continue

        print(f"\nExp {exp['id']} — {exp['nombre']}: {len(convs)} capas conv")
        for layer in convs if todas else convs[:1]:
            plot_kernels(layer, exp)


if __name__ == "__main__":
    main()
