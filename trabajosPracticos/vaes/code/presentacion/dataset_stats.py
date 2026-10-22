"""
Estadísticas reales del dataset y figuras de dataset / preprocesamiento.

Recorre archive/cropped leyendo solo las cabeceras de los PNG (PIL no decodifica la
imagen hasta que se la usa), así que medir 9.877 archivos cuesta menos de un segundo.

Salidas: outputs/presentacion/dataset_stats.json,
         outputs/presentacion/figuras/{dataset_ejemplos,preprocesamiento}.png
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config
from data import EXTENSIONES, build_transform

PRESENTACION_DIR = config.OUTPUT_DIR / "presentacion"
FIGURAS_DIR = PRESENTACION_DIR / "figuras"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
STATS_JSON = PRESENTACION_DIR / "dataset_stats.json"


def guardar(fig, nombre: str) -> Path:
    ruta = FIGURAS_DIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  figura: {ruta.name}")
    return ruta


def recolectar_stats(data_dir: Path) -> dict:
    """Cantidad, peso en disco, dimensiones, modos de color y extensiones."""
    paths = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES
    )
    if not paths:
        raise FileNotFoundError(f"No hay imágenes en {data_dir}")

    bytes_total = sum(p.stat().st_size for p in paths)
    dims, modos, exts = Counter(), Counter(), Counter()
    for p in paths:
        with Image.open(p) as im:  # solo lee la cabecera
            dims[f"{im.size[0]}x{im.size[1]}"] += 1
            modos[im.mode] += 1
        exts[p.suffix.lower()] += 1

    return {
        "directorio": str(data_dir),
        "n_imagenes": len(paths),
        # Tres medidas distintas del "tamaño": la suma de bytes de los archivos en MB
        # decimales y en MiB, y lo que ocupan los bloques en disco (`du -sh`, que suma
        # ~4 KB por archivo de overhead y da los 402 MB que reporta el sistema).
        "tamano_mb": round(bytes_total / 1000**2, 1),
        "tamano_mib": round(bytes_total / 1024**2, 1),
        "tamano_bytes": bytes_total,
        "dimensiones": dict(dims.most_common()),
        "modos_color": dict(modos.most_common()),
        "extensiones": dict(exts.most_common()),
        "img_size_entrenamiento": config.IMG_SIZE,
        "valores_por_imagen_original": 200 * 200 * 3,
        "valores_por_imagen_64": config.IMG_SIZE**2 * 3,
        "latent_dim": config.LATENT_DIM,
        "factor_compresion": round(config.IMG_SIZE**2 * 3 / config.LATENT_DIM, 1),
        "val_split": config.VAL_SPLIT,
        "n_train": len(paths) - round(len(paths) * config.VAL_SPLIT),
        "n_val": round(len(paths) * config.VAL_SPLIT),
    }


def fig_ejemplos(paths, stats):
    """Grilla 4x4 de imágenes originales elegidas al azar con semilla fija."""
    rng = random.Random(config.SEED)
    muestra = rng.sample(paths, 16)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, p in zip(axes.flat, muestra):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.axis("off")
    fig.suptitle(
        f"Dataset: {stats['n_imagenes']:,} rostros de 200×200".replace(",", "."),
        fontsize=14,
    )
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.94)
    return guardar(fig, "dataset_ejemplos.png")


def fig_preprocesamiento(paths):
    """3 ejemplos: original 200x200 junto a su versión 64x64, al mismo tamaño en pantalla."""
    rng = random.Random(config.SEED + 1)
    muestra = rng.sample(paths, 3)
    transform = build_transform()

    fig, axes = plt.subplots(3, 2, figsize=(6, 9))
    for fila, p in enumerate(muestra):
        original = Image.open(p).convert("RGB")
        # El tensor se vuelve a imagen para mostrar exactamente lo que ve la red.
        reducida = transform(original).permute(1, 2, 0).numpy()

        axes[fila, 0].imshow(original)
        axes[fila, 1].imshow(reducida)
        for col in (0, 1):
            axes[fila, col].axis("off")

    axes[0, 0].set_title("Original 200×200", fontsize=12)
    axes[0, 1].set_title("Redimensionada 64×64", fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return guardar(fig, "preprocesamiento.png")


def main():
    print(f"Leyendo {config.DATA_DIR}")
    stats = recolectar_stats(config.DATA_DIR)
    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"  imágenes:    {stats['n_imagenes']:,}".replace(",", "."))
    print(f"  en disco:    {stats['tamano_mb']} MB ({stats['tamano_mib']} MiB)")
    print(f"  dimensiones: {stats['dimensiones']}")
    print(f"  modos:       {stats['modos_color']}")
    print(f"  extensiones: {stats['extensiones']}")
    print(f"  compresión:  {stats['valores_por_imagen_64']} -> {stats['latent_dim']} "
          f"({stats['factor_compresion']}x)")
    print(f"  json: {STATS_JSON}")

    paths = sorted(
        p for p in config.DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES
    )
    print("\nFiguras:")
    fig_ejemplos(paths, stats)
    fig_preprocesamiento(paths)


if __name__ == "__main__":
    main()
