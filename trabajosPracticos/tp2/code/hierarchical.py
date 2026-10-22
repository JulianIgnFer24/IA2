"""
Clustering jerárquico aglomerativo — TP2.

Compara los linkage single, complete y average para los dos escenarios
(X_comp sin winpercent, X_full con winpercent):

  - Dendrogramas etiquetados con el nombre de cada golosina.
  - Coeficiente de correlación cofenética por linkage (fidelidad del
    dendrograma a las distancias originales).
  - Silhouette de los cortes k = 2..8 para cada linkage.
  - Corte a un mismo k y comparación de asignaciones entre linkages
    con Adjusted Rand Index (ARI) + tabla cruzada.
  - Proyección PCA 2D coloreada por el clustering del mejor linkage.

Guarda resultados en results/hierarchical.pkl
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from common import (
    FIGS_DIR,
    TP2_DIR,
    build_features,
    internal_metrics,
    load_data,
    save_fig,
    scenario_tag,
)

LINKAGES = ["single", "complete", "average"]
K_RANGE = range(2, 9)
RESULTS_PATH = TP2_DIR / "results" / "hierarchical.pkl"


def plot_dendrograms(Zs: dict, names: list, scen: str) -> None:
    """Tres dendrogramas apilados (uno por linkage)."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    for ax, link in zip(axes, LINKAGES):
        dendrogram(
            Zs[link],
            labels=names,
            leaf_font_size=5.5,
            leaf_rotation=90,
            color_threshold=None,
            ax=ax,
        )
        ax.set_title(f"Linkage {link}", fontsize=12)
        ax.set_ylabel("Distancia")
        ax.tick_params(axis="x", labelsize=5.5)
    fig.suptitle(
        f"Dendrogramas — clustering jerárquico ({scenario_tag(scen)})", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_fig(fig, f"hier_dendros_{scen}.png")


def plot_silhouette_cuts(sil: pd.DataFrame, scen: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for link in LINKAGES:
        ax.plot(sil.index, sil[link], marker="o", label=link)
    ax.set_xlabel("k (número de clusters del corte)")
    ax.set_ylabel("Silhouette")
    ax.set_title(f"Silhouette por corte del dendrograma ({scenario_tag(scen)})")
    ax.set_xticks(list(sil.index))
    ax.legend(title="linkage")
    save_fig(fig, f"hier_silhouette_{scen}.png")


def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, names: list, link: str,
                      k: int, scen: str) -> None:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum()
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", k)
    for c in range(k):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=45, color=palette[c], label=f"cluster {c}")
    for (x, y), n in zip(Z, names):
        ax.annotate(n, (x, y), fontsize=5.5, alpha=0.75,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_title(
        f"Corte k={k} del linkage {link} ({scenario_tag(scen)}) — "
        f"proyección PCA ({var:.0%} var. explicada)"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    save_fig(fig, f"hier_pca_{scen}.png")


def run_scenario(scen: str, X: np.ndarray, names: list) -> dict:
    print(f"\n{'=' * 70}\nEscenario: {scen} ({scenario_tag(scen)})\n{'=' * 70}")

    # Linkages + correlación cofenética
    dist = pdist(X, metric="euclidean")
    Zs, coph = {}, {}
    for link in LINKAGES:
        Zs[link] = linkage(X, method=link, metric="euclidean")
        coph[link], _ = cophenet(Zs[link], dist)
    print("\nCorrelación cofenética (mayor = más fiel a las distancias):")
    for link in LINKAGES:
        print(f"  {link:8s}: {coph[link]:.4f}")

    plot_dendrograms(Zs, names, scen)

    # Silhouette de los cortes k = 2..8
    sil = pd.DataFrame(index=list(K_RANGE), columns=LINKAGES, dtype=float)
    for link in LINKAGES:
        for k in K_RANGE:
            lab = fcluster(Zs[link], t=k, criterion="maxclust")
            sil.loc[k, link] = silhouette_score(X, lab)
    print("\nSilhouette por corte:")
    print(sil.round(3).to_string())
    plot_silhouette_cuts(sil, scen)

    # Mejor linkage = mayor cofenética; k = mejor silhouette de ese linkage
    best_link = max(coph, key=coph.get)
    best_k = int(sil[best_link].idxmax())
    print(f"\nMejor linkage por cofenética: {best_link} (coph = {coph[best_link]:.4f})")
    print(f"k elegido (máx. silhouette con {best_link}): k = {best_k}")

    # Cortes de los 3 linkages al mismo k y comparación con ARI
    cuts = {link: fcluster(Zs[link], t=best_k, criterion="maxclust")
            for link in LINKAGES}
    ari = pd.DataFrame(np.eye(3), index=LINKAGES, columns=LINKAGES)
    for i, a in enumerate(LINKAGES):
        for b in LINKAGES[i + 1:]:
            ari.loc[a, b] = ari.loc[b, a] = adjusted_rand_score(cuts[a], cuts[b])
    print(f"\nARI entre cortes (k = {best_k}):")
    print(ari.round(3).to_string())

    for link in LINKAGES:
        ct = pd.crosstab(cuts[best_link], cuts[link],
                         rownames=[best_link], colnames=[link])
        print(f"\nTabla cruzada {best_link} vs {link}:")
        print(ct.to_string())

    best_labels = cuts[best_link] - 1  # 0..k-1
    mets = internal_metrics(X, best_labels)
    print(f"\nMétricas del corte elegido ({best_link}, k={best_k}): "
          f"silhouette={mets['silhouette']:.3f}, "
          f"CH={mets['calinski_harabasz']:.1f}, "
          f"DB={mets['davies_bouldin']:.3f}")

    plot_pca_clusters(X, best_labels, names, best_link, best_k, scen)

    return {
        "coph": coph,
        "silhouette_cuts": sil,
        "best_link": best_link,
        "best_k": best_k,
        "ari": ari,
        "labels": best_labels,
        "metrics": mets,
    }


def main() -> None:
    df = load_data()
    names = df["competitorname"].tolist()
    feats = build_features(df)

    results = {}
    for scen, (X, _cols) in feats.items():
        results[scen] = run_scenario(scen, X, names)

    # ¿Cambian las asignaciones al agregar winpercent?
    ari_scen = adjusted_rand_score(results["comp"]["labels"],
                                   results["full"]["labels"])
    print(f"\nARI entre mejores cortes de comp vs full: {ari_scen:.3f}")
    results["ari_comp_vs_full"] = ari_scen

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados guardados en {RESULTS_PATH.relative_to(TP2_DIR)}")


if __name__ == "__main__":
    main()
