"""
DBSCAN — TP2.

Para cada escenario (X_comp sin winpercent, X_full con winpercent):

  - k-distance plot (k = min_samples) para estimar eps por la "rodilla".
  - Grid de hiperparámetros: eps x min_samples, reportando n° de clusters,
    n° de puntos de ruido y silhouette (excluyendo ruido).
  - Heatmaps del grid para visualizar la sensibilidad a los hiperparámetros.
  - Caracterización del mejor modelo (winpercent medio, golosinas de ruido).
  - Proyección PCA 2D coloreada por cluster (ruido en gris).

Guarda resultados en results/dbscan.pkl
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from common import (
    TP2_DIR,
    WIN_COL,
    build_features,
    internal_metrics,
    load_data,
    save_fig,
    scenario_tag,
)

EPS_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
MIN_SAMPLES_GRID = [3, 4, 5, 8]
KDIST_K = 4  # vecinos para el k-distance plot
RESULTS_PATH = TP2_DIR / "results" / "dbscan.pkl"


def plot_kdistance(X: np.ndarray, scen: str) -> None:
    nn = NearestNeighbors(n_neighbors=KDIST_K).fit(X)
    dist, _ = nn.kneighbors(X)
    dk = np.sort(dist[:, -1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(dk)), dk, color="steelblue")
    for eps in EPS_GRID:
        ax.axhline(eps, color="gray", ls=":", lw=0.8)
        ax.text(len(dk) - 1, eps, f" eps={eps}", va="bottom", fontsize=8,
                color="dimgray")
    ax.set_xlabel(f"Puntos ordenados por distancia al {KDIST_K}° vecino")
    ax.set_ylabel(f"Distancia al {KDIST_K}° vecino más cercano")
    ax.set_title(f"k-distance plot ({scenario_tag(scen)})")
    save_fig(fig, f"dbscan_kdist_{scen}.png")


def plot_grid_heatmaps(grid: pd.DataFrame, scen: str) -> None:
    pivots = {
        "n_clusters": grid.pivot(index="min_samples", columns="eps",
                                 values="n_clusters"),
        "n_noise": grid.pivot(index="min_samples", columns="eps",
                              values="n_noise"),
        "silhouette": grid.pivot(index="min_samples", columns="eps",
                                 values="silhouette"),
    }
    titles = {
        "n_clusters": "N° de clusters",
        "n_noise": "N° de puntos de ruido",
        "silhouette": "Silhouette (sin ruido)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (key, piv) in zip(axes, pivots.items()):
        sns.heatmap(piv, annot=True, fmt=".2f" if key == "silhouette" else "d",
                    cmap="viridis", ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(titles[key])
        ax.set_xlabel("eps")
        ax.set_ylabel("min_samples" if ax is axes[0] else "")
    fig.suptitle(f"DBSCAN — sensibilidad a hiperparámetros "
                 f"({scenario_tag(scen)})", y=1.04)
    fig.tight_layout()
    save_fig(fig, f"dbscan_grid_{scen}.png")


def plot_pca(X: np.ndarray, labels: np.ndarray, names: list, scen: str,
             eps: float, min_samples: int) -> None:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum()
    clusters = sorted(set(labels) - {-1})
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("tab10", len(clusters))
    for i, c in enumerate(clusters):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=45, color=palette[i],
                   label=f"cluster {c}")
    if (labels == -1).any():
        m = labels == -1
        ax.scatter(Z[m, 0], Z[m, 1], s=45, color="lightgray", marker="x",
                   label="ruido")
        for (x, y), n in zip(Z[m], np.array(names)[m]):
            ax.annotate(n, (x, y), fontsize=6, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"DBSCAN eps={eps}, min_samples={min_samples} "
                 f"({scenario_tag(scen)}) — PCA ({var:.0%} var.)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    save_fig(fig, f"dbscan_pca_{scen}.png")


def run_scenario(scen: str, X: np.ndarray, df: pd.DataFrame,
                 names: list) -> dict:
    print(f"\n{'=' * 70}\nEscenario: {scen} ({scenario_tag(scen)})\n{'=' * 70}")

    plot_kdistance(X, scen)

    # Grid de hiperparámetros
    rows = []
    models = {}
    for ms in MIN_SAMPLES_GRID:
        for eps in EPS_GRID:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            mets = internal_metrics(X, lab)
            rows.append({"min_samples": ms, "eps": eps, **mets})
            models[(ms, eps)] = lab
    grid = pd.DataFrame(rows)
    print("\nGrid de hiperparámetros:")
    print(grid.round(3).to_string(index=False))
    plot_grid_heatmaps(grid, scen)

    # Máximo silhouette crudo (suele ser degenerado: mucho ruido, clusters
    # diminutos) vs. mejor configuración "usable" (ruido <= 25% de los puntos)
    n = len(X)
    valid = grid.dropna(subset=["silhouette"])
    raw = valid.loc[valid["silhouette"].idxmax()]
    print(f"\nMáximo silhouette crudo (degenerado): min_samples="
          f"{int(raw['min_samples'])}, eps={raw['eps']} -> clusters="
          f"{int(raw['n_clusters'])}, ruido={int(raw['n_noise'])}/{n}, "
          f"silhouette={raw['silhouette']:.3f}")

    usable = valid[valid["n_noise"] <= 0.25 * n]
    best = usable.loc[usable["silhouette"].idxmax()]
    best_ms, best_eps = int(best["min_samples"]), float(best["eps"])
    labels = models[(best_ms, best_eps)]
    print(f"Mejor configuración usable (ruido <= 25%): min_samples={best_ms}, "
          f"eps={best_eps} -> clusters={int(best['n_clusters'])}, "
          f"ruido={int(best['n_noise'])}, silhouette={best['silhouette']:.3f}")

    # Caracterización del mejor modelo
    print("\nwinpercent medio por cluster (ruido aparte):")
    tmp = pd.DataFrame({"label": labels, WIN_COL: df[WIN_COL]})
    print(tmp.groupby("label")[WIN_COL].agg(["mean", "count"]).round(2)
          .to_string())

    noise = [n for n, lab in zip(names, labels) if lab == -1]
    if noise:
        print(f"\nGolosinas marcadas como ruido ({len(noise)}):")
        print("  " + ", ".join(noise))

    plot_pca(X, labels, names, scen, best_eps, best_ms)

    return {
        "grid": grid,
        "best_min_samples": best_ms,
        "best_eps": best_eps,
        "raw_max_silhouette": raw.to_dict(),
        "labels": labels,
        "noise_names": noise,
    }


def main() -> None:
    df = load_data()
    names = df["competitorname"].tolist()
    feats = build_features(df)

    results = {}
    for scen, (X, _cols) in feats.items():
        results[scen] = run_scenario(scen, X, df, names)

    ari_scen = adjusted_rand_score(results["comp"]["labels"],
                                   results["full"]["labels"])
    print(f"\nARI entre mejores DBSCAN de comp vs full: {ari_scen:.3f}")
    results["ari_comp_vs_full"] = ari_scen

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResultados guardados en {RESULTS_PATH.relative_to(TP2_DIR)}")


if __name__ == "__main__":
    main()
