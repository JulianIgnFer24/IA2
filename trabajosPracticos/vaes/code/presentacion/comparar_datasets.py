"""
Compara los dos experimentos: archive/cropped contra archive/simplified.

Advertencia metodológica que atraviesa todo este script: las pérdidas de los dos
datasets NO son comparables en valor absoluto. `simplified` tiene imágenes de colores
planos (3 colores únicos por imagen contra ~9.500 en `cropped`), así que su BCE mínima
alcanzable es mucho más baja. Que un número sea menor no dice que el modelo «ande
mejor»: dice que la tarea es más fácil. Lo comparable es el comportamiento: calidad
visual de lo generado, uso del latente y distancia entre train y validación.

Salidas: outputs/presentacion/figuras/comp_*.png y outputs/presentacion/comparacion.json
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from data import SimpsonsFacesDataset, build_dataloaders, build_transform
from model import VAE
from model_ae import AE

RAIZ = config.PROJECT_DIR
FIGURAS = RAIZ / "outputs" / "presentacion" / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)
JSON_SALIDA = RAIZ / "outputs" / "presentacion" / "comparacion.json"

AZUL, NARANJA, GRIS = "#1f77b4", "#ff7f0e", "#333333"

DATASETS = {
    "cropped": {
        "datos": RAIZ / "archive" / "cropped",
        "out": RAIZ / "outputs",
        "etiqueta": "cropped (fotogramas reales)",
    },
    "simplified": {
        "datos": RAIZ / "archive" / "simplified",
        "out": RAIZ / "outputs" / "simplified",
        "etiqueta": "simplified (colores planos)",
    },
}


def guardar(fig, nombre):
    ruta = FIGURAS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {nombre}")
    return ruta


def img(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def leer_csv(ruta):
    with Path(ruta).open(newline="") as f:
        return [{k: (int(v) if k == "epoch" else float(v)) for k, v in fila.items()}
                for fila in csv.DictReader(f)]


def cargar(nombre):
    """Devuelve (vae, ae, val_dataset, history_vae, history_ae) de un experimento."""
    cfg = DATASETS[nombre]
    ck = torch.load(cfg["out"] / "checkpoints" / "vae_epoch_200.pt",
                    map_location=config.DEVICE, weights_only=False)
    vae = VAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    vae.load_state_dict(ck["model_state_dict"]); vae.eval()

    ck_ae = torch.load(cfg["out"] / "baseline" / "ae_final.pt",
                       map_location=config.DEVICE, weights_only=False)
    ae = AE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    ae.load_state_dict(ck_ae["model_state_dict"]); ae.eval()

    _, val_loader, val_ds = build_dataloaders(data_dir=cfg["datos"], num_workers=0,
                                              verbose=False)
    return {
        "vae": vae, "ae": ae, "val_ds": val_ds, "val_loader": val_loader,
        "hist_vae": leer_csv(cfg["out"] / "loss_history.csv"),
        "hist_ae": leer_csv(cfg["out"] / "baseline" / "ae_loss_history.csv"),
        "ck_ae": ck_ae,
    }


# ---------------------------------------------------------------------------
def fig_datasets(n=4, seed=config.SEED):
    """La misma imagen en sus dos versiones, para que se vea la diferencia."""
    rng = np.random.default_rng(seed)
    nombres = sorted(p.name for p in DATASETS["cropped"]["datos"].iterdir()
                     if p.suffix == ".png")
    elegidos = [nombres[i] for i in rng.choice(len(nombres), n, replace=False)]
    tr = build_transform()

    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8))
    for col, nombre in enumerate(elegidos):
        for fila, clave in enumerate(["cropped", "simplified"]):
            ruta = DATASETS[clave]["datos"] / nombre
            with Image.open(ruta) as im:
                arr = tr(im.convert("RGB")).permute(1, 2, 0).numpy()
            axes[fila, col].imshow(arr)
            axes[fila, col].axis("off")
    axes[0, 0].set_ylabel("cropped")
    for fila, etiqueta in enumerate(["cropped", "simplified"]):
        axes[fila, 0].text(-0.08, 0.5, etiqueta, transform=axes[fila, 0].transAxes,
                           rotation=90, va="center", ha="right", fontsize=11, color=GRIS)
    fig.suptitle("La misma cara en las dos versiones del dataset", fontsize=13)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, top=0.91)
    return guardar(fig, "comp_dataset.png")


@torch.no_grad()
def fig_reconstruccion(exp, n=6):
    """Original y reconstrucción del VAE, una fila por dataset."""
    fig, axes = plt.subplots(4, n, figsize=(1.6 * n, 7.0))
    filas = [("cropped", "Original", "Reconstrucción"),
             ("simplified", "Original", "Reconstrucción")]
    for b, (clave, t1, t2) in enumerate(filas):
        ds = exp[clave]["val_ds"]
        x = torch.stack([ds[i] for i in range(n)]).to(config.DEVICE)
        recon, _, _ = exp[clave]["vae"](x)
        for i in range(n):
            axes[2 * b, i].imshow(img(x[i]))
            axes[2 * b + 1, i].imshow(img(recon[i]))
            axes[2 * b, i].axis("off"); axes[2 * b + 1, i].axis("off")
        axes[2 * b, 0].set_title(f"{clave} — {t1}", loc="left", fontsize=10)
        axes[2 * b + 1, 0].set_title(t2, loc="left", fontsize=10)
    fig.suptitle("Reconstrucción del VAE en cada dataset", fontsize=13)
    fig.subplots_adjust(hspace=0.35, wspace=0.05, top=0.93)
    return guardar(fig, "comp_reconstruccion.png")


@torch.no_grad()
def fig_generacion(exp, n=8, seed=config.SEED):
    """Caras nuevas desde el MISMO z en los dos datasets."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, config.LATENT_DIM, generator=g).to(config.DEVICE)

    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.9))
    for fila, clave in enumerate(["cropped", "simplified"]):
        imgs = exp[clave]["vae"].decode(z)
        for i in range(n):
            axes[fila, i].imshow(img(imgs[i]))
            axes[fila, i].axis("off")
        axes[fila, 0].text(-0.08, 0.5, clave, transform=axes[fila, 0].transAxes,
                           rotation=90, va="center", ha="right", fontsize=11, color=GRIS)
    fig.suptitle("Caras nuevas desde los mismos z ~ N(0, I)", fontsize=13)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, top=0.88)
    return guardar(fig, "comp_generacion.png")


def fig_curvas(exp):
    """Pérdida total y KL por dataset. Ejes separados: las escalas no son comparables."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for clave, color in [("cropped", AZUL), ("simplified", NARANJA)]:
        h = exp[clave]["hist_vae"]
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_total"] for r in h], color=color, label=f"{clave} train")
        axes[0].plot(ep, [r["val_total"] for r in h], color=color, linestyle="--",
                     alpha=0.75, label=f"{clave} val")
        axes[1].plot(ep, [r["train_kl"] for r in h], color=color, label=clave)
        gap = [r["val_total"] - r["train_total"] for r in h]
        axes[2].plot(ep, gap, color=color, label=clave)

    axes[0].set_title("Pérdida total del VAE (nats/imagen)")
    axes[1].set_title("Divergencia KL (nats/imagen)")
    axes[2].set_title("Brecha validación − entrenamiento")
    axes[2].axhline(0, color=GRIS, linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.set_xlabel("Época"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Los dos entrenamientos, lado a lado "
                 "(la pérdida absoluta no es comparable entre datasets)", fontsize=12)
    fig.tight_layout()
    return guardar(fig, "comp_curvas.png")


@torch.no_grad()
def uso_latente(vae, val_loader):
    stds = []
    for x in val_loader:
        _, logvar = vae.encode(x.to(config.DEVICE))
        stds.append(torch.exp(0.5 * logvar).cpu())
    std = torch.cat(stds).mean(dim=0)
    return {"sigma_medio": round(float(std.mean()), 4),
            "dims_activas": int((std < 0.5).sum()),
            "dims_colapsadas": int((std > 0.9).sum())}


@torch.no_grad()
def nitidez(vae, val_ds, n=128, seed=config.SEED):
    """Nitidez media (varianza del laplaciano) de reconstrucciones contra originales.

    Es una medida comparable entre datasets porque se reporta como COCIENTE: cuánta
    de la nitidez del original conserva la reconstrucción.
    """
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=config.DEVICE).view(1, 1, 3, 3)

    def lap(t):
        gris = t.mean(dim=1, keepdim=True)
        return torch.nn.functional.conv2d(gris, k).var(dim=(1, 2, 3))

    x = torch.stack([val_ds[i] for i in range(min(n, len(val_ds)))]).to(config.DEVICE)
    recon, _, _ = vae(x)
    o, r = lap(x).mean().item(), lap(recon).mean().item()
    return {"nitidez_original": round(o, 5), "nitidez_reconstruccion": round(r, 5),
            "conservada_pct": round(100 * r / o, 1)}


@torch.no_grad()
def realismo(vae, val_ds, n_gen=256, seed=config.SEED):
    """¿Qué tan lejos del conjunto de datos caen las muestras generadas?

    Para cada cara generada se mide la distancia al vecino más cercano entre las
    imágenes reales de validación. Ese número solo no es comparable entre datasets
    (simplified tiene menos variedad de píxeles), así que se normaliza por la distancia
    típica entre dos imágenes reales del mismo dataset.

    Cociente ~1: las muestras caen tan cerca de los datos como los datos entre sí.
    Cociente >> 1: las muestras caen fuera de la variedad de caras.
    """
    reales = torch.stack([val_ds[i] for i in range(len(val_ds))]).to(config.DEVICE)
    reales = reales.flatten(1)

    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_gen, config.LATENT_DIM, generator=g).to(config.DEVICE)
    gen = vae.decode(z).flatten(1)

    d_gen = torch.cdist(gen, reales).min(dim=1).values.mean().item()

    # Distancia típica entre dos imágenes reales distintas (se ignora la diagonal).
    d_rr = torch.cdist(reales, reales)
    d_rr.fill_diagonal_(float("inf"))
    d_real = d_rr.min(dim=1).values.mean().item()

    return {"dist_generada": round(d_gen, 3), "dist_entre_reales": round(d_real, 3),
            "cociente": round(d_gen / d_real, 2)}


def main():
    config.set_seeds()
    config.print_device()

    faltan = [n for n, c in DATASETS.items()
              if not (c["out"] / "checkpoints" / "vae_epoch_200.pt").is_file()]
    if faltan:
        raise FileNotFoundError(f"Faltan los checkpoints de: {faltan}")

    exp = {n: cargar(n) for n in DATASETS}

    print("\nFiguras comparativas:")
    fig_datasets()
    fig_reconstruccion(exp)
    fig_generacion(exp)
    fig_curvas(exp)

    resumen = {}
    for clave in DATASETS:
        e = exp[clave]
        hv, ha = e["hist_vae"], e["hist_ae"]
        mejor = min(hv, key=lambda r: r["val_total"])
        resumen[clave] = {
            "etiqueta": DATASETS[clave]["etiqueta"],
            "vae": {
                "train_total": round(hv[-1]["train_total"], 2),
                "val_total": round(hv[-1]["val_total"], 2),
                "train_recon": round(hv[-1]["train_recon"], 2),
                "train_kl": round(hv[-1]["train_kl"], 2),
                "mejor_val_epoca": mejor["epoch"],
                "mejor_val_total": round(mejor["val_total"], 2),
                "brecha_final": round(hv[-1]["val_total"] - hv[-1]["train_total"], 2),
                "minutos": round(sum(r["seconds"] for r in hv) / 60, 1),
            },
            "ae": {
                "train_recon": round(ha[-1]["train_recon"], 2),
                "val_recon": round(ha[-1]["val_recon"], 2),
            },
            "ventaja_ae_recon": round(hv[-1]["train_recon"] - ha[-1]["train_recon"], 2),
            "uso_latente": uso_latente(e["vae"], e["val_loader"]),
            "nitidez": nitidez(e["vae"], e["val_ds"]),
            "realismo": realismo(e["vae"], e["val_ds"]),
        }

    resumen["_nota"] = (
        "Las pérdidas de cropped y simplified no son comparables en valor absoluto: "
        "simplified tiene colores planos, así que su BCE mínima alcanzable es mucho "
        "menor. Lo comparable es el comportamiento (uso del latente, brecha train/val, "
        "nitidez conservada) y la calidad visual."
    )
    JSON_SALIDA.write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    print(f"\ncomparacion.json -> {JSON_SALIDA}\n")
    for clave in DATASETS:
        r = resumen[clave]
        print(f"{clave:11s} VAE total {r['vae']['train_total']:8.1f} | "
              f"KL {r['vae']['train_kl']:6.1f} | "
              f"dims activas {r['uso_latente']['dims_activas']:3d}/128 | "
              f"brecha {r['vae']['brecha_final']:+7.1f} | "
              f"nitidez {r['nitidez']['conservada_pct']:5.1f} % | "
              f"muestras a {r['realismo']['cociente']:.2f}× de los datos")


if __name__ == "__main__":
    main()
