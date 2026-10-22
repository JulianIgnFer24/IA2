"""
K-means — TP2.

Para cada escenario (X_comp sin winpercent, X_full con winpercent):

  - K = 2..10: inercia (método del codo), silhouette y Calinski-Harabasz.
  - Elección de K con criterio combinado.
  - Caracterización de clusters: heatmap de medias de features (escala
    original) y listado de golosinas por cluster.
  - winpercent medio por cluster (análisis de popularidad).
  - Proyección PCA 2D coloreada por cluster.
  - Comparación de asignaciones entre K distintos con ARI.

Guarda resultados en results/kmeans.pkl
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)

from common import (
    TP2_DIR,
    WIN_COL,
    build_features,
    load_data,
    save_fig,
    scenario_tag,
)

K_RANGE = range(2, 11)
RESULTS_PATH = TP2_DIR / "results" / "kmeans.pkl"


def plot_select_k(metrics: pd.DataFrame, scen: str) -> None:
    """Codo (inercia), silhouette y CH en tres paneles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(metrics.index, metrics["inertia"], marker="o", color="steelblue")
    axes[0].set_title("Método del codo")
    axes[0].set_ylabel("Inercia (SSE intra-cluster)")
    axes[1].plot(metrics.index, metrics["silhouette"], marker="o", color="darkorange")
    axes[1].set_title("Silhouette (mayor = mejor)")
    axes[2].plot(metrics.index, metrics["calinski_harabasz"], marker="o",
                 color="seagreen")
    axes[2].set_title("Calinski-Harabasz (mayor = mejor)")
    for ax in axes:
        ax.set_xlabel("K")
        ax.set_xticks(list(metrics.index))
    fig.suptitle(f"Selección de K — K-means ({scenario_tag(scen)})", y=1.03)
    fig.tight_layout()
    save_fig(fig, f"kmeans_select_k_{scen}.png")


def plot_heatmap(df: pd.DataFrame, labels: np.ndarray, cols: list,
                 k: int, scen: str) -> pd.DataFrame:
    """Medias de features por cluster (escala original)."""
    means = df[cols].groupby(labels).mean()
    means.index = [f"cluster {i} (n={int((labels == i).sum())})"
                   for i in means.index]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    sns.heatmap(means, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5,
                cbar_kws={"label": "media del feature"}, ax=ax)
    ax.set_title(f"Perfil de clusters K-means k={k} ({scenario_tag(scen)})")
    ax.set_ylabel("")
    save_fig(fig, f"kmeans_heatmap_{scen}.png")
    return means


def plot_pca(X: np.ndarray, labels: np.ndarray, names: list,
             k: int, scen: str) -> None:
    Z = PCA(n_components=2, random_state=42).fit_transform(X)
    var = PCA(n_components=2, random_state=42).fit(X).explained_variance_ratio_.sum()
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", k)
    for c in range(k):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=45, color=palette[c], label=f"cluster {c}")
    for (x, y), n in zip(Z, names):
        ax.annotate(n, (x, y), fontsize=5.5, alpha=0.75,
                    xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"K-means k={k} ({scenario_tag(scen)}) — "
                 f"proyección PCA ({var:.0%} var. explicada)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    save_fig(fig, f"kmeans_pca_{scen}.png")


def run_scenario(scen: str, X: np.ndarray, cols: list, df: pd.DataFrame,
                 names: list) -> dict:
    print(f"\n{'=' * 70}\nEscenario: {scen} ({scenario_tag(scen)})\n{'=' * 70}")

    # Barrido de K
    metrics = pd.DataFrame(
        index=list(K_RANGE),
        columns=["inertia", "silhouette", "calinski_harabasz"],
        dtype=float,
    )
    fits = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        lab = km.fit_predict(X)
        fits[k] = lab
        metrics.loc[k] = [
            km.inertia_,
            silhouette_score(X, lab),
            calinski_harabasz_score(X, lab),
        ]
    print("\nMétricas por K:")
    print(metrics.round(3).to_string())
    plot_select_k(metrics, scen)

    best_k = int(metrics["silhouette"].idxmax())
    print(f"\nK elegido por máximo silhouette: K = {best_k}")
    labels = fits[best_k]

    # Caracterización
    means = plot_heatmap(df, labels, cols, best_k, scen)
    print("\nMedias de features por cluster (escala original):")
    print(means.round(3).to_string())

    print(f"\nwinpercent medio por cluster (no usado como feature en este "
          f"escenario si scen=comp):")
    win_by_cluster = df[WIN_COL].groupby(labels).agg(["mean", "std", "count"])
    print(win_by_cluster.round(2).to_string())

    print("\nGolosinas por cluster:")
    members = {}
    for c in range(best_k):
        miembros = [n for n, lab in zip(names, labels) if lab == c]
        members[c] = miembros
        print(f"  cluster {c} ({len(miembros)}): {', '.join(miembros)}")

    plot_pca(X, labels, names, best_k, scen)

    # ARI entre K consecutivos (estabilidad de la partición al mover K)
    ari_ks = {k: adjusted_rand_score(fits[k], fits[k + 1])
              for k in list(K_RANGE)[:-1]}
    print("\nARI entre asignaciones de K y K+1:")
    for k, v in ari_ks.items():
        print(f"  K={k} vs K={k + 1}: {v:.3f}")

    return {
        "metrics": metrics,
        "best_k": best_k,
        "labels": labels,
        "cluster_means": means,
        "win_by_cluster": win_by_cluster,
        "members": members,
        "ari_consecutive": ari_ks,
    }


def main() -> None:
    df = load_data()
    names = df["competitorname"].tolist()
    feats = build_features(df)

    results = {}
    for scen, (X, cols) in feats.items():
        results[scen] = run_scenario(scen, X, cols, df, names)

    ari_scen = adjusted_rand_score(results["comp"]["labels"],
                                   results["full"]["labels"])
    print(f"\nARI entre mejores K-means de comp vs full: {ari_scen:.3f}")
    results["ari_comp_vs_full"] = ari_scen

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados guardados en {RESULTS_PATH.relative_to(TP2_DIR)}")


if __name__ == "__main__":
    main()
