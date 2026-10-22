"""
PCA + clustering — TP2.

Para cada escenario (X_comp sin winpercent, X_full con winpercent):

  - PCA completo: scree plot de varianza explicada y análisis de loadings
    (qué features definen PC1/PC2).
  - Biplot 2D: golosinas proyectadas + vectores de features.
  - Elección de n_componentes para retener ~85% de varianza.
  - Clustering en el espacio PCA: K-means (con el K elegido en kmeans.py)
    y, como extra, jerárquico con linkage average (mejor cofenética).
  - Comparación ARI entre las etiquetas obtenidas en el espacio PCA y en
    el espacio original.
  - Figura 2D final coloreada por cluster con nombres anotados.

Guarda resultados en results/pca_clustering.pkl
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from common import (
    TP2_DIR,
    build_features,
    load_data,
    save_fig,
    scenario_tag,
)

VAR_THRESHOLD = 0.85  # varianza acumulada objetivo
RESULTS_PATH = TP2_DIR / "results" / "pca_clustering.pkl"
KMEANS_PATH = TP2_DIR / "results" / "kmeans.pkl"


def plot_scree(pca: PCA, scen: str) -> int:
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    n_comp = int(np.searchsorted(cum, VAR_THRESHOLD) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(var) + 1)
    ax.bar(x, var, color="steelblue", alpha=0.8, label="individual")
    ax.plot(x, cum, marker="o", color="darkorange", label="acumulada")
    ax.axhline(VAR_THRESHOLD, color="gray", ls="--", lw=1)
    ax.axvline(n_comp, color="seagreen", ls="--", lw=1,
               label=f"{n_comp} comp. ≈ {cum[n_comp - 1]:.0%}")
    ax.set_xticks(x)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Proporción de varianza explicada")
    ax.set_title(f"Scree plot PCA ({scenario_tag(scen)})")
    ax.legend()
    save_fig(fig, f"pca_scree_{scen}.png")
    return n_comp


def plot_loadings(pca: PCA, cols: list, scen: str) -> pd.DataFrame:
    load = pd.DataFrame(
        pca.components_[:3].T,
        index=cols,
        columns=["PC1", "PC2", "PC3"],
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    load.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Loadings de PC1–PC3 ({scenario_tag(scen)})")
    ax.set_ylabel("Peso del feature en el componente")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, f"pca_loadings_{scen}.png")
    return load


def plot_biplot(Z: np.ndarray, pca: PCA, labels: np.ndarray, cols: list,
                names: list, k: int, scen: str, metodo: str) -> None:
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(11, 8))
    palette = sns.color_palette("tab10", k)
    for c in range(k):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=45, color=palette[c],
                   label=f"cluster {c}")
    for (x, y), n in zip(Z, names):
        ax.annotate(n, (x, y), fontsize=5.5, alpha=0.7,
                    xytext=(2, 2), textcoords="offset points")
    # Vectores de features (escalados para visibilidad)
    scale = np.abs(Z[:, :2]).max() * 0.9
    for i, col in enumerate(cols):
        vx, vy = pca.components_[0, i] * scale, pca.components_[1, i] * scale
        ax.arrow(0, 0, vx, vy, color="red", alpha=0.6,
                 head_width=0.06 * scale, length_includes_head=True)
        ax.text(vx * 1.1, vy * 1.1, col, color="red", fontsize=8,
                ha="center", va="center")
    ax.set_title(f"Biplot PCA + {metodo} k={k} ({scenario_tag(scen)}) — "
                 f"PC1 {var[0]:.0%}, PC2 {var[1]:.0%}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    save_fig(fig, f"pca_biplot_{metodo.lower().replace('-', '')}_{scen}.png")


def run_scenario(scen: str, X: np.ndarray, cols: list, names: list,
                 best_k: int, labels_orig: np.ndarray) -> dict:
    print(f"\n{'=' * 70}\nEscenario: {scen} ({scenario_tag(scen)})\n{'=' * 70}")

    pca_full = PCA(random_state=42).fit(X)
    n_comp = plot_scree(pca_full, scen)
    load = plot_loadings(pca_full, cols, scen)
    print(f"\nVarianza acumulada con 2 comp.: "
          f"{pca_full.explained_variance_ratio_[:2].sum():.1%}")
    print(f"Componentes para >= {VAR_THRESHOLD:.0%} de varianza: {n_comp}")
    print("\nLoadings PC1–PC3:")
    print(load.round(3).to_string())

    # Clustering en espacio PCA reducido
    Zr = pca_full.transform(X)[:, :n_comp]
    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    lab_km = km.fit_predict(Zr)
    sil_km = silhouette_score(Zr, lab_km)
    ari_km = adjusted_rand_score(labels_orig, lab_km)
    print(f"\nK-means en PCA({n_comp}): silhouette={sil_km:.3f}, "
          f"ARI vs. espacio original={ari_km:.3f}")

    # Extra: jerárquico average en el espacio PCA
    Zh = linkage(Zr, method="average", metric="euclidean")
    lab_hier = fcluster(Zh, t=best_k, criterion="maxclust") - 1
    sil_hier = silhouette_score(Zr, lab_hier)
    ari_hier = adjusted_rand_score(labels_orig, lab_hier)
    ari_km_hier = adjusted_rand_score(lab_km, lab_hier)
    print(f"Jerárquico average en PCA({n_comp}): silhouette={sil_hier:.3f}, "
          f"ARI vs. espacio original={ari_hier:.3f}")
    print(f"ARI K-means(PCA) vs jerárquico(PCA): {ari_km_hier:.3f}")

    # Biplots con las etiquetas de cada método (PC1/PC2)
    Z2 = pca_full.transform(X)[:, :2]
    plot_biplot(Z2, pca_full, lab_km, cols, names, best_k, scen, "KMeans")
    plot_biplot(Z2, pca_full, lab_hier, cols, names, best_k, scen,
                "Jerarquico")

    return {
        "n_components": n_comp,
        "var_2comp": float(pca_full.explained_variance_ratio_[:2].sum()),
        "loadings": load,
        "kmeans_pca": {"labels": lab_km, "silhouette": sil_km,
                       "ari_vs_orig": ari_km},
        "hier_pca": {"labels": lab_hier, "silhouette": sil_hier,
                     "ari_vs_orig": ari_hier},
        "ari_km_hier_pca": ari_km_hier,
    }


def main() -> None:
    df = load_data()
    names = df["competitorname"].tolist()
    feats = build_features(df)

    with open(KMEANS_PATH, "rb") as f:
        km_results = pickle.load(f)

    results = {}
    for scen, (X, cols) in feats.items():
        best_k = km_results[scen]["best_k"]
        labels_orig = km_results[scen]["labels"]
        results[scen] = run_scenario(scen, X, cols, names, best_k, labels_orig)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados guardados en {RESULTS_PATH.relative_to(TP2_DIR)}")


if __name__ == "__main__":
    main()
