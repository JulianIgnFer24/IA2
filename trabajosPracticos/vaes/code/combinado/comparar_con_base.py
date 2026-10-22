"""
¿Agregar simplified como datos extra mejoró al modelo en la tarea original?

La pregunta hay que responderla sobre imágenes CROPPED, que es la tarea original, y con
los dos modelos evaluados exactamente sobre las mismas imágenes.

El problema metodológico: los dos modelos tienen particiones distintas. De las 494
imágenes de validación del modelo base, el combinado vio 468 durante su entrenamiento.
Compararlos sobre esas 494 inflaría al combinado. La única comparación limpia es sobre
las imágenes que quedaron fuera del entrenamiento de LOS DOS: son 26.

Con n = 26 se usa comparación pareada (la misma imagen en los dos modelos) y un intervalo
de confianza por bootstrap, que es mucho más potente que comparar dos promedios sueltos.
También se reporta el número sesgado, etiquetado como tal.

Un confundidor que conviene tener presente: el combinado hizo el doble de pasos de
gradiente (146 batches por época contra 73), aunque cada imagen cropped la vio las mismas
200 veces. Así que «más datos» viene acompañado de «más optimización».

Salidas: outputs/combinado/analisis/comparativa_base.png y comparativa_base.json
"""
import csv
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, random_split

import config
from data import SimpsonsFacesDataset
from model import VAE

OUT = config.PROJECT_DIR / "outputs"
ANALISIS = OUT / "combinado" / "analisis"
ANALISIS.mkdir(parents=True, exist_ok=True)

AZUL, NARANJA, GRIS = "#1F7A6E", "#CE6A28", "#333333"
N_BOOT = 10000


def cargar_vae(ruta):
    ck = torch.load(ruta, map_location=config.DEVICE, weights_only=False)
    m = VAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def particiones():
    """Reproduce las dos particiones y devuelve los índices cropped limpios.

    Devuelve (indices_limpios, indices_val_base, n_filtrados_por_fuga).
    """
    cr = SimpsonsFacesDataset(config.PROJECT_DIR / "archive" / "cropped")
    si = SimpsonsFacesDataset(config.PROJECT_DIR / "archive" / "simplified")
    n = len(cr)

    g = torch.Generator().manual_seed(config.SEED)
    n_val = round(n * config.VAL_SPLIT)
    _, val_base = random_split(cr, [n - n_val, n_val], generator=g)
    idx_base = set(val_base.indices)

    comb = ConcatDataset([cr, si]); m = len(comb)
    g = torch.Generator().manual_seed(config.SEED)
    m_val = round(m * config.VAL_SPLIT)
    tr_c, val_c = random_split(comb, [m - m_val, m_val], generator=g)
    val_comb = {i for i in val_c.indices if i < n}

    limpios = sorted(idx_base & val_comb)
    return cr, limpios, sorted(idx_base), len(idx_base) - len(limpios)


@torch.no_grad()
def metricas_por_imagen(vae, x):
    """BCE, KL y total por imagen (sin promediar entre imágenes)."""
    recon, mu, logvar = vae(x)
    bce = F.binary_cross_entropy(recon, x, reduction="none").flatten(1).sum(1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    return bce.cpu().numpy(), kl.cpu().numpy(), (bce + kl).cpu().numpy(), recon


@torch.no_grad()
def nitidez(x, recon):
    """Proporción de la nitidez del original que conserva la reconstrucción."""
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                     device=x.device).view(1, 1, 3, 3)

    def lap(t):
        return F.conv2d(t.mean(dim=1, keepdim=True), k).var(dim=(1, 2, 3))

    return (lap(recon) / lap(x)).cpu().numpy() * 100


def boot_ci(d, n=N_BOOT, seed=config.SEED):
    """IC del 95 % de la media por bootstrap sobre las diferencias pareadas."""
    rng = np.random.default_rng(seed)
    medias = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


@torch.no_grad()
def distancia_a_los_datos(vae, reales, n_gen=256, seed=config.SEED):
    """Distancia de las caras generadas al vecino real más cercano, normalizada."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_gen, config.LATENT_DIM, generator=g).to(config.DEVICE)
    gen = vae.decode(z).flatten(1)
    d_gen = torch.cdist(gen, reales).min(dim=1).values.mean().item()
    d_rr = torch.cdist(reales, reales); d_rr.fill_diagonal_(float("inf"))
    return d_gen / d_rr.min(dim=1).values.mean().item()


def fig_comparativa(par, nit, dif, ic, res):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # 1. Pérdida de reconstrucción pareada, imagen por imagen
    ax = axes[0]
    x = np.arange(len(par["base_bce"]))
    orden = np.argsort(par["base_bce"])
    ax.plot(x, par["base_bce"][orden], "o-", ms=4, color=AZUL,
            label="solo cropped")
    ax.plot(x, par["comb_bce"][orden], "s-", ms=4, color=NARANJA,
            label="cropped + simplified")
    ax.set_title("Reconstrucción por imagen\n(26 caras fuera de ambos entrenamientos)",
                 fontsize=11)
    ax.set_xlabel("imagen (ordenada por dificultad)")
    ax.set_ylabel("BCE (nats)"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 2. Diferencias pareadas
    ax = axes[1]
    colores = [NARANJA if v < 0 else AZUL for v in dif]
    ax.bar(x, np.sort(dif), color=[colores[i] for i in np.argsort(dif)])
    ax.axhline(0, color=GRIS, linewidth=1)
    ax.axhline(dif.mean(), color="black", linestyle="--", linewidth=1.4,
               label=f"media {dif.mean():+.1f}")
    ax.fill_between([-1, len(dif)], ic[0], ic[1], color="black", alpha=0.1,
                    label=f"IC 95 % [{ic[0]:+.0f}, {ic[1]:+.0f}]")
    ax.set_xlim(-1, len(dif))
    ax.set_title("Diferencia pareada por imagen\n(negativo = mejora el combinado)",
                 fontsize=11)
    ax.set_xlabel("imagen"); ax.set_ylabel("BCE combinado − BCE base")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # 3. Nitidez conservada
    ax = axes[2]
    ax.boxplot([nit["base"], nit["comb"]], tick_labels=["solo\ncropped",
                                                        "cropped +\nsimplified"],
               patch_artist=True,
               boxprops=dict(facecolor="#DCE9E7"), medianprops=dict(color="black"))
    ax.set_title("Nitidez conservada en la reconstrucción", fontsize=11)
    ax.set_ylabel("% de la nitidez del original"); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("¿Mejoró el modelo al agregar simplified? — evaluado sobre las mismas "
                 "caras cropped", fontsize=13)
    fig.tight_layout()
    ruta = ANALISIS / "comparativa_base.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {ruta.name}")


def main():
    config.set_seeds()
    config.print_device()

    base = cargar_vae(OUT / "checkpoints" / "vae_epoch_200.pt")
    comb = cargar_vae(OUT / "combinado" / "checkpoints" / "vae_epoch_200.pt")

    cr, limpios, val_base, n_fuga = particiones()
    print(f"\nValidación del modelo base: {len(val_base)} imágenes cropped")
    print(f"  de esas, el combinado vio {n_fuga} durante su entrenamiento")
    print(f"  comparación limpia sobre: {len(limpios)} imágenes\n")

    x = torch.stack([cr[i] for i in limpios]).to(config.DEVICE)
    b_bce, b_kl, b_tot, b_rec = metricas_por_imagen(base, x)
    c_bce, c_kl, c_tot, c_rec = metricas_por_imagen(comb, x)
    dif = c_bce - b_bce
    ic = boot_ci(dif)
    nit = {"base": nitidez(x, b_rec), "comb": nitidez(x, c_rec)}

    # Mismo cálculo sobre las 494, que está sesgado a favor del combinado.
    x494 = torch.stack([cr[i] for i in val_base]).to(config.DEVICE)
    b494, _, _, _ = metricas_por_imagen(base, x494)
    c494, _, _, _ = metricas_por_imagen(comb, x494)

    # Generación: no usa particiones, así que es limpia por construcción.
    reales = torch.stack([cr[i] for i in val_base]).to(config.DEVICE).flatten(1)
    gen_base = distancia_a_los_datos(base, reales)
    gen_comb = distancia_a_los_datos(comb, reales)

    fig_comparativa({"base_bce": b_bce, "comb_bce": c_bce}, nit, dif, ic, None)

    gana = int((dif < 0).sum())
    res = {
        "n_limpias": len(limpios),
        "n_val_base": len(val_base),
        "n_vistas_por_combinado": n_fuga,
        "comparacion_limpia": {
            "base_bce": round(float(b_bce.mean()), 2),
            "combinado_bce": round(float(c_bce.mean()), 2),
            "diferencia_media": round(float(dif.mean()), 2),
            "ic95": [round(ic[0], 2), round(ic[1], 2)],
            "imagenes_en_que_gana_el_combinado": f"{gana}/{len(dif)}",
            "significativo": bool(ic[1] < 0 or ic[0] > 0),
            "base_kl": round(float(b_kl.mean()), 2),
            "combinado_kl": round(float(c_kl.mean()), 2),
            "base_total": round(float(b_tot.mean()), 2),
            "combinado_total": round(float(c_tot.mean()), 2),
            "nitidez_base_pct": round(float(nit["base"].mean()), 2),
            "nitidez_combinado_pct": round(float(nit["comb"].mean()), 2),
        },
        "comparacion_sesgada_494": {
            "_nota": ("el combinado entrenó con 468 de estas 494 imágenes; el número "
                      "le favorece y NO sirve para concluir"),
            "base_bce": round(float(b494.mean()), 2),
            "combinado_bce": round(float(c494.mean()), 2),
        },
        "generacion": {
            "_nota": "no depende de las particiones: muestrear del prior es limpio",
            "base_dist_a_datos": round(gen_base, 3),
            "combinado_dist_a_datos": round(gen_comb, 3),
        },
        "confundidor": ("el combinado hizo el doble de pasos de gradiente (146 batches "
                        "por época contra 73), aunque cada imagen cropped la vio las "
                        "mismas 200 veces"),
    }
    (ANALISIS / "comparativa_base.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))

    c = res["comparacion_limpia"]
    print(f"Sobre las {len(limpios)} imágenes limpias:")
    print(f"  reconstrucción  base {c['base_bce']:8.2f}  |  combinado "
          f"{c['combinado_bce']:8.2f}")
    print(f"  diferencia pareada {c['diferencia_media']:+8.2f}  "
          f"IC95 [{ic[0]:+.1f}, {ic[1]:+.1f}]  "
          f"{'SIGNIFICATIVA' if c['significativo'] else 'no significativa'}")
    print(f"  gana el combinado en {c['imagenes_en_que_gana_el_combinado']} imágenes")
    print(f"  nitidez conservada  base {c['nitidez_base_pct']:.1f} %  |  "
          f"combinado {c['nitidez_combinado_pct']:.1f} %")
    print(f"\nGeneración (distancia a los datos, menor es mejor):")
    print(f"  base {gen_base:.3f}  |  combinado {gen_comb:.3f}")
    print(f"\nPara contraste, el número SESGADO sobre las 494: "
          f"base {b494.mean():.2f} | combinado {c494.mean():.2f}")


if __name__ == "__main__":
    main()
