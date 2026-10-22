"""
Análisis del VAE entrenado con los dos estilos juntos.

Un solo modelo que vio cropped y simplified abre preguntas que los modelos por separado
no tenían:

  1. ¿El latente separa los dos estilos en dos nubes, o los mezcla?
  2. ¿Qué estilo genera cuando se muestrea del prior: uno, el otro, o híbridos?
  3. ¿Reconstruye bien los dos?
  4. ¿«Estilo» quedó codificado como una DIRECCIÓN del latente? Si la resta
     μ(simplified) − μ(cropped) apunta siempre para el mismo lado, se le puede sumar a
     una cara nueva para simplificarla sin haberla visto nunca simplificada.

Salidas: outputs/combinado/analisis/*.png y outputs/combinado/analisis/analisis.json
"""
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
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import config
from data import build_transform
from model import VAE, count_params

RAIZ = config.PROJECT_DIR
OUT = RAIZ / "outputs" / "combinado"
ANALISIS = OUT / "analisis"
ANALISIS.mkdir(parents=True, exist_ok=True)

CARPETAS = {"cropped": RAIZ / "archive" / "cropped",
            "simplified": RAIZ / "archive" / "simplified"}

TEAL, NARANJA, GRIS = "#1F7A6E", "#CE6A28", "#333333"
N_PARES = 1500          # pares (cropped, simplified) de la misma cara para el análisis
N_MUESTRAS = 64         # caras generadas desde el prior


def guardar(fig, nombre):
    ruta = ANALISIS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {nombre}")
    return ruta


def im(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def nombres_comunes(n=None, seed=config.SEED):
    """Archivos que existen en las dos carpetas (son las mismas caras)."""
    a = {p.name for p in CARPETAS["cropped"].iterdir() if p.suffix == ".png"}
    b = {p.name for p in CARPETAS["simplified"].iterdir() if p.suffix == ".png"}
    comunes = sorted(a & b)
    if n and n < len(comunes):
        idx = np.random.default_rng(seed).choice(len(comunes), n, replace=False)
        comunes = [comunes[i] for i in sorted(idx)]
    return comunes


def cargar_lote(nombres, estilo, tr):
    return torch.stack([tr(Image.open(CARPETAS[estilo] / n).convert("RGB"))
                        for n in nombres])


@torch.no_grad()
def codificar(vae, nombres, estilo, tr, lote=256):
    """μ de cada imagen, en bloques para no llenar la memoria."""
    mus = []
    for i in range(0, len(nombres), lote):
        x = cargar_lote(nombres[i:i + lote], estilo, tr).to(config.DEVICE)
        mus.append(vae.encode(x)[0].cpu())
    return torch.cat(mus)


def variacion_total(x):
    """Textura media: diferencia absoluta entre píxeles vecinos.

    Una imagen de colores planos da un valor muy bajo; un fotograma real, alto. Sirve
    para ubicar objetivamente en qué estilo cae una imagen generada.
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    dv = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    return ((dh + dv) / 2)


# ---------------------------------------------------------------------------
@torch.no_grad()
def fig_reconstruccion(vae, nombres, tr, n=6):
    """¿Reconstruye bien los dos estilos? Las mismas caras en sus dos versiones."""
    sel = nombres[:n]
    fig, axes = plt.subplots(4, n, figsize=(1.6 * n, 7.0))
    for b, estilo in enumerate(["cropped", "simplified"]):
        x = cargar_lote(sel, estilo, tr).to(config.DEVICE)
        recon, _, _ = vae(x)
        for i in range(n):
            axes[2 * b, i].imshow(im(x[i]))
            axes[2 * b + 1, i].imshow(im(recon[i]))
            axes[2 * b, i].axis("off"); axes[2 * b + 1, i].axis("off")
        axes[2 * b, 0].set_title(f"{estilo} — original", loc="left", fontsize=10)
        axes[2 * b + 1, 0].set_title("reconstrucción", loc="left", fontsize=10)
    fig.suptitle("Un solo modelo reconstruyendo los dos estilos", fontsize=13)
    fig.subplots_adjust(hspace=0.35, wspace=0.05, top=0.93)
    return guardar(fig, "comb_reconstruccion.png")


def fig_latente(mu_c, mu_s, seed=config.SEED):
    """¿El latente separa los estilos? PCA y t-SNE coloreados por estilo."""
    X = np.concatenate([mu_c, mu_s])
    y = np.array([0] * len(mu_c) + [1] * len(mu_s))
    rng = np.random.default_rng(seed)
    prior = rng.standard_normal((500, config.LATENT_DIM))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    pca = PCA(n_components=2).fit(X)
    P, Pp = pca.transform(X), pca.transform(prior)
    axes[0].scatter(*P[y == 0].T, s=7, alpha=0.5, c=TEAL, label="cropped")
    axes[0].scatter(*P[y == 1].T, s=7, alpha=0.5, c=NARANJA, label="simplified")
    axes[0].scatter(*Pp.T, s=7, alpha=0.4, c="0.35", marker="^", label="N(0, I)")
    axes[0].set_title("PCA de μ, coloreado por estilo", fontsize=12)
    axes[0].set_xlabel("componente principal 1"); axes[0].set_ylabel("componente 2")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=9)

    sub = rng.choice(len(X), min(3000, len(X)), replace=False)
    emb = TSNE(n_components=2, init="pca", random_state=seed,
               perplexity=30).fit_transform(X[sub])
    ys = y[sub]
    axes[1].scatter(*emb[ys == 0].T, s=7, alpha=0.6, c=TEAL, label="cropped")
    axes[1].scatter(*emb[ys == 1].T, s=7, alpha=0.6, c=NARANJA, label="simplified")
    axes[1].set_title("t-SNE de μ", fontsize=12)
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].legend(fontsize=9)

    fig.suptitle("¿El modelo separa los dos estilos en el espacio latente?", fontsize=13)
    fig.tight_layout()
    return guardar(fig, "comb_latente.png")


@torch.no_grad()
def estilo_de_las_muestras(vae, mu_c, mu_s, seed=config.SEED, n=512):
    """¿En qué estilo genera el modelo? Medido en el latente, no en píxeles.

    La textura por píxel no sirve: el desenfoque del VAE también la baja, así que una
    cara generada en estilo cropped pero borrosa se confunde con una simplified. Acá se
    proyectan los z sorteados sobre la dirección de estilo v = μ(simp) − μ(crop) y se
    los compara con las proyecciones de las caras reales, que es donde el modelo
    efectivamente codificó el estilo.
    """
    v = (mu_s - mu_c).mean(dim=0)
    v = v / v.norm()
    pc = (mu_c @ v).numpy()
    ps = (mu_s @ v).numpy()

    g = torch.Generator().manual_seed(seed + 1)
    z = torch.randn(n, config.LATENT_DIM, generator=g)
    pg = (z @ v).numpy()

    # Umbral: punto medio entre las medias de las dos nubes reales.
    umbral = (pc.mean() + ps.mean()) / 2
    hacia_simp = ps.mean() > pc.mean()
    pct_simp = float(((pg > umbral) == hacia_simp).mean()) * 100

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(pc, bins=40, alpha=0.6, color=TEAL, density=True, label="cropped reales")
    ax.hist(ps, bins=40, alpha=0.6, color=NARANJA, density=True,
            label="simplified reales")
    ax.hist(pg, bins=40, histtype="step", linewidth=2.2, color="black", density=True,
            label="z ~ N(0, I)")
    ax.axvline(umbral, color=GRIS, linestyle=":", linewidth=1.3)
    ax.set_title("Proyección sobre la dirección de estilo", fontsize=12)
    ax.set_xlabel("← cropped          simplified →"); ax.set_ylabel("densidad")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()
    guardar(fig, "comb_estilo_proyeccion.png")

    return {"proy_cropped": round(float(pc.mean()), 3),
            "proy_simplified": round(float(ps.mean()), 3),
            "proy_generadas": round(float(pg.mean()), 3),
            "pct_generadas_lado_simplified": round(pct_simp, 1),
            "solapamiento_real": round(float(
                min(pc.max(), ps.max()) - max(pc.min(), ps.min())) /
                float(max(pc.max(), ps.max()) - min(pc.min(), ps.min())), 3)}


@torch.no_grad()
def fig_generacion(vae, tv_reales, seed=config.SEED):
    """¿Qué estilo sale al muestrear del prior? Grilla + histograma de textura."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(N_MUESTRAS, config.LATENT_DIM, generator=g).to(config.DEVICE)
    muestras = vae.decode(z)
    tv_gen = variacion_total(muestras).cpu().numpy()

    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])

    sub = gs[0].subgridspec(6, 6, wspace=0.05, hspace=0.05)
    orden = np.argsort(tv_gen)   # de más plano a más texturado
    for i in range(36):
        ax = fig.add_subplot(sub[i // 6, i % 6])
        ax.imshow(im(muestras[orden[i]])); ax.axis("off")
    fig.text(0.07, 0.93, "36 caras generadas desde z ~ N(0, I), ordenadas por textura",
             fontsize=12, color=GRIS)

    ax = fig.add_subplot(gs[1])
    ax.hist(tv_reales["cropped"], bins=40, alpha=0.6, color=TEAL,
            label="cropped reales", density=True)
    ax.hist(tv_reales["simplified"], bins=40, alpha=0.6, color=NARANJA,
            label="simplified reales", density=True)
    ax.hist(tv_gen, bins=30, histtype="step", linewidth=2.2, color="black",
            label="generadas", density=True)
    ax.set_title("Textura (variación total por píxel)", fontsize=12)
    ax.set_xlabel("más plano  →  más texturado"); ax.set_ylabel("densidad")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)

    fig.suptitle("¿Qué estilo genera el modelo combinado?", fontsize=13)
    fig.tight_layout()
    guardar(fig, "comb_generacion.png")
    return tv_gen


@torch.no_grad()
def fig_direccion_estilo(vae, mu_c, mu_s, nombres, tr, n_demo=5, seed=config.SEED):
    """¿«Simplificar» es una dirección fija del latente?

    v = media de (μ_simplified − μ_cropped). Si la dirección es consistente, sumarle v
    a una cara cropped que el modelo nunca vio simplificada debería simplificarla.
    """
    dif = mu_s - mu_c
    v = dif.mean(dim=0)
    cos = F.cosine_similarity(dif, v.unsqueeze(0), dim=1)

    # Caras de prueba que NO entraron en el cálculo de v.
    reservadas = nombres[-n_demo:]
    x = cargar_lote(reservadas, "cropped", tr).to(config.DEVICE)
    x_ref = cargar_lote(reservadas, "simplified", tr).to(config.DEVICE)
    mu, _ = vae.encode(x)
    vd = v.to(config.DEVICE)

    alfas = [0.0, 0.5, 1.0, 1.5]
    fig, axes = plt.subplots(n_demo, len(alfas) + 2, figsize=(1.55 * (len(alfas) + 2),
                                                              1.6 * n_demo))
    for f in range(n_demo):
        axes[f, 0].imshow(im(x[f]))
        for j, a in enumerate(alfas):
            axes[f, j + 1].imshow(im(vae.decode((mu[f] + a * vd).unsqueeze(0))[0]))
        axes[f, -1].imshow(im(x_ref[f]))
        for ax in axes[f]:
            ax.axis("off")
    cabeceras = ["original\n(cropped)"] + [f"μ + {a}·v" for a in alfas] + \
                ["simplified\nreal"]
    for j, c in enumerate(cabeceras):
        axes[0, j].set_title(c, fontsize=9)
    fig.suptitle(f"Aritmética latente: v = media de μ(simplified) − μ(cropped)   "
                 f"(coseno medio {cos.mean():.2f})", fontsize=12)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, top=0.86)
    guardar(fig, "comb_direccion_estilo.png")

    return {"coseno_medio": round(float(cos.mean()), 3),
            "coseno_mediano": round(float(cos.median()), 3),
            "fraccion_positivos": round(float((cos > 0).float().mean()), 3),
            "norma_v": round(float(v.norm()), 3),
            "norma_media_dif": round(float(dif.norm(dim=1).mean()), 3)}


@torch.no_grad()
def fig_interpolacion_estilos(vae, nombres, tr, n=3, pasos=9):
    """Interpolar entre una cara y SU PROPIA versión simplificada."""
    sel = nombres[:n]
    xc = cargar_lote(sel, "cropped", tr).to(config.DEVICE)
    xs = cargar_lote(sel, "simplified", tr).to(config.DEVICE)
    mc, _ = vae.encode(xc)
    ms, _ = vae.encode(xs)

    ts = torch.linspace(0, 1, pasos, device=config.DEVICE).view(-1, 1)
    fig, axes = plt.subplots(n, pasos + 2, figsize=(1.2 * (pasos + 2), 1.45 * n))
    for f in range(n):
        tramo = vae.decode((1 - ts) * mc[f] + ts * ms[f])
        axes[f, 0].imshow(im(xc[f]))
        for j in range(pasos):
            axes[f, j + 1].imshow(im(tramo[j]))
        axes[f, -1].imshow(im(xs[f]))
        for ax in axes[f]:
            ax.axis("off")
    axes[0, 0].set_title("cropped", fontsize=9)
    axes[0, -1].set_title("simplified", fontsize=9)
    fig.suptitle("Interpolar entre una cara y su propia versión simplificada",
                 fontsize=12)
    fig.subplots_adjust(wspace=0.04, hspace=0.05, top=0.84)
    return guardar(fig, "comb_interpolacion_estilos.png")


def separacion(mu_c, mu_s):
    """Qué tan separadas están las dos nubes, en unidades de su propia dispersión."""
    cc, cs = mu_c.mean(0), mu_s.mean(0)
    d = float(np.linalg.norm(cc - cs))
    disp = float((np.linalg.norm(mu_c - cc, axis=1).mean() +
                  np.linalg.norm(mu_s - cs, axis=1).mean()) / 2)
    return {"distancia_centroides": round(d, 3),
            "dispersion_media": round(disp, 3),
            "cociente": round(d / disp, 3)}


def main():
    config.set_seeds()
    config.print_device()

    ck = OUT / "checkpoints" / "vae_epoch_200.pt"
    if not ck.is_file():
        raise FileNotFoundError(f"Falta {ck}. Correr primero code/combinado/run_combinado.py")
    ckpt = torch.load(ck, map_location=config.DEVICE, weights_only=False)
    vae = VAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    vae.load_state_dict(ckpt["model_state_dict"]); vae.eval()
    print(f"Modelo combinado: {count_params(vae):,} parámetros, "
          f"época {ckpt['epoch']}")

    tr = build_transform()
    nombres = nombres_comunes(N_PARES)
    print(f"Pares (cropped, simplified) analizados: {len(nombres)}")

    mu_c = codificar(vae, nombres, "cropped", tr)
    mu_s = codificar(vae, nombres, "simplified", tr)

    # Textura de las imágenes reales, para ubicar a las generadas.
    tv_reales = {}
    for estilo in CARPETAS:
        x = cargar_lote(nombres[:512], estilo, tr).to(config.DEVICE)
        tv_reales[estilo] = variacion_total(x).cpu().numpy()

    print("\nFiguras:")
    fig_reconstruccion(vae, nombres, tr)
    fig_latente(mu_c.numpy(), mu_s.numpy())
    tv_gen = fig_generacion(vae, tv_reales)
    estilo_info = fig_direccion_estilo(vae, mu_c, mu_s, nombres, tr)
    proy = estilo_de_las_muestras(vae, mu_c, mu_s)
    fig_interpolacion_estilos(vae, nombres, tr)

    # ¿A qué estilo se parecen las generadas, por textura?
    umbral = (tv_reales["simplified"].mean() + tv_reales["cropped"].mean()) / 2
    res = {
        "parametros": count_params(vae),
        "epocas": ckpt["epoch"],
        "n_pares_analizados": len(nombres),
        "separacion_latente": separacion(mu_c.numpy(), mu_s.numpy()),
        "direccion_estilo": estilo_info,
        "estilo_generado": proy,
        "textura": {
            "cropped_real": round(float(tv_reales["cropped"].mean()), 4),
            "simplified_real": round(float(tv_reales["simplified"].mean()), 4),
            "generadas": round(float(tv_gen.mean()), 4),
            "umbral": round(float(umbral), 4),
            "_nota": ("la textura por píxel NO distingue el estilo de una imagen "
                      "generada: el desenfoque del VAE la baja igual que la "
                      "simplificación. Para eso está estilo_generado."),
        },
    }
    (ANALISIS / "analisis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\nanalisis.json -> {ANALISIS / 'analisis.json'}")
    sep = res["separacion_latente"]
    tx = res["textura"]
    dr = res["direccion_estilo"]
    print(f"\n  Separación de estilos en el latente: {sep['cociente']:.2f} "
          f"(distancia entre centroides / dispersión)")
    print(f"  Dirección de estilo: coseno medio {dr['coseno_medio']:.2f}, "
          f"{dr['fraccion_positivos'] * 100:.0f} % de los pares apunta para el mismo lado")
    print(f"  Textura real — cropped {tx['cropped_real']:.4f} | "
          f"simplified {tx['simplified_real']:.4f}")
    print(f"  Estilo de lo generado (proyección latente): "
          f"{proy['pct_generadas_lado_simplified']:.0f} % del lado simplified")
    print(f"    cropped {proy['proy_cropped']:.2f} | "
          f"simplified {proy['proy_simplified']:.2f} | "
          f"generadas {proy['proy_generadas']:.2f}")


if __name__ == "__main__":
    main()
