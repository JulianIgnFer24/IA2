"""
Predictor de cluster para vectores nuevos — TP2 (candy-data).

Reconstruye la regla de decisión de cada método a partir de los
resultados guardados en results/*.pkl (sin re-entrenar):

  - K-means: centroide más cercano en el espacio escalado. Los
    centroides son la media de los puntos de cada cluster según las
    etiquetas guardadas; en convergencia, "centroide más cercano" es
    exactamente la regla de asignación de k-means.
  - Jerárquico (linkage average, el elegido en el informe): se asigna
    el cluster con menor distancia promedio a sus miembros.
  - DBSCAN: cluster del core point más cercano a distancia <= eps;
    si no hay core points alcanzables, el punto es ruido (-1).
    Los core points se recomputan de los datos escalados con el eps
    y min_samples guardados (un punto etiquetado es core si tiene
    >= min_samples vecinos a distancia <= eps, contándose a sí mismo).

El StandardScaler se re-ajusta sobre candy-data.csv (determinístico,
idéntico al pipeline de entrenamiento de common.build_features).

Uso CLI: ``python predict.py`` corre un chequeo de consistencia que
re-clasifica las 85 golosinas del dataset y las compara con las
etiquetas guardadas (k-means y DBSCAN deben dar 100 %).
"""
import base64
import io
import pickle
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from common import (
    FEATURES_COMP,
    FEATURES_FULL,
    NAME_COL,
    SCENARIOS,
    TP2_DIR,
    WIN_COL,
    load_data,
    scenario_tag,
)

FEATURES = {"comp": FEATURES_COMP, "full": FEATURES_FULL}

RESULTS_DIR = TP2_DIR / "results"
METHODS = ("kmeans", "hierarchical", "dbscan")
METHOD_TITLES = {
    "kmeans": "K-means",
    "hierarchical": "Jerárquico (average)",
    "dbscan": "DBSCAN",
}


def _load_results() -> dict:
    """Lee los tres pkl de results/."""
    out = {}
    for m in METHODS:
        with open(RESULTS_DIR / f"{m}.pkl", "rb") as f:
            out[m] = pickle.load(f)
    return out


@lru_cache(maxsize=1)
def _context() -> dict:
    """
    Datos + estructuras de asignación por escenario (se calcula una vez).

    Por escenario:
      scaler, X, cols, etiquetas de cada método, centroides de k-means,
      clusters de cada método, core points de DBSCAN, PCA 2D ajustado.
    """
    df = load_data()
    res = _load_results()
    ctx = {"df": df, "names": df[NAME_COL].tolist(), "scenarios": {}}
    for scen in SCENARIOS:
        cols = list(FEATURES[scen])
        scaler = StandardScaler().fit(df[cols].to_numpy(dtype=float))
        X = scaler.transform(df[cols].to_numpy(dtype=float))

        km_labels = np.asarray(res["kmeans"][scen]["labels"], dtype=int)
        hi_labels = np.asarray(res["hierarchical"][scen]["labels"], dtype=int)
        db_labels = np.asarray(res["dbscan"][scen]["labels"], dtype=int)

        # K-means: centroides = media de puntos escalados por cluster
        km_clusters = sorted(set(km_labels))
        centroids = np.vstack([X[km_labels == c].mean(axis=0) for c in km_clusters])

        # DBSCAN: core points recomputados con eps/min_samples guardados
        eps = float(res["dbscan"][scen]["best_eps"])
        min_samples = int(res["dbscan"][scen]["best_min_samples"])
        D = pairwise_distances(X)
        n_neighbors = (D <= eps).sum(axis=1)  # incluye al propio punto
        core_mask = (db_labels != -1) & (n_neighbors >= min_samples)

        # PCA 2D para la visualización (mismo criterio que las figs del TP)
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X)

        ctx["scenarios"][scen] = {
            "cols": cols,
            "scaler": scaler,
            "X": X,
            "km_labels": km_labels,
            "hi_labels": hi_labels,
            "db_labels": db_labels,
            "km_clusters": km_clusters,
            "centroids": centroids,
            "hier_clusters": sorted(set(hi_labels)),
            "db_clusters": sorted(set(db_labels) - {-1}),
            "eps": eps,
            "min_samples": min_samples,
            "core_mask": core_mask,
            "pca": pca,
            "Z": Z,
        }
    return ctx


def feature_cols(scen: str) -> list:
    """Columnas (en orden) que espera el vector para el escenario dado."""
    return list(_context()["scenarios"][scen]["cols"])


def _profile_table(df: pd.DataFrame, cols: list, labels: np.ndarray,
                   assigned: int) -> list:
    """Filas de medias por cluster en escala original (+ winpercent medio)."""
    means = df[cols].groupby(labels).mean()
    win = df[WIN_COL].groupby(labels).mean()
    counts = pd.Series(labels).value_counts().sort_index()
    rows = []
    for c in means.index:
        rows.append({
            "cluster": int(c),
            "n": int(counts[c]),
            "assigned": bool(c == assigned),
            "vals": [round(float(v), 3) for v in means.loc[c]],
            "win_mean": round(float(win[c]), 1),
        })
    return rows


def predict(vec, scen: str) -> dict:
    """
    Asigna un vector nuevo (escala original, orden de feature_cols(scen))
    a un cluster según los tres métodos.

    Devuelve un dict con las predicciones, distancias, datos para el
    scatter PCA, perfil de clusters k-means y miembros por método.
    """
    ctx = _context()
    if scen not in ctx["scenarios"]:
        raise ValueError(f"Escenario inválido: {scen!r} (usar {SCENARIOS})")
    s = ctx["scenarios"][scen]
    df, names = ctx["df"], ctx["names"]

    x = np.asarray(vec, dtype=float).ravel()
    if x.size != len(s["cols"]):
        raise ValueError(
            f"El vector debe tener {len(s['cols'])} valores "
            f"({', '.join(s['cols'])}); recibió {x.size}"
        )
    xs = s["scaler"].transform(x.reshape(1, -1))[0]

    # --- K-means: centroide más cercano -------------------------------
    d_cent = np.linalg.norm(s["centroids"] - xs, axis=1)
    km_cluster = s["km_clusters"][int(np.argmin(d_cent))]
    km_distances = [
        {"cluster": int(c), "dist": round(float(d), 3), "assigned": bool(c == km_cluster)}
        for c, d in sorted(zip(s["km_clusters"], d_cent), key=lambda t: t[1])
    ]

    # --- Jerárquico (average): menor distancia promedio a miembros ----
    d_train = np.linalg.norm(s["X"] - xs, axis=1)
    hier_avg = {c: float(d_train[s["hi_labels"] == c].mean()) for c in s["hier_clusters"]}
    hier_cluster = min(hier_avg, key=hier_avg.get)
    hier_distances = [
        {"cluster": int(c), "dist": round(d, 3), "assigned": bool(c == hier_cluster)}
        for c, d in sorted(hier_avg.items(), key=lambda t: t[1])
    ]

    # --- DBSCAN: core point más cercano a distancia <= eps ------------
    core_idx = np.flatnonzero(s["core_mask"])
    d_core = d_train[core_idx]
    min_core = float(d_core.min())
    if min_core <= s["eps"]:
        db_cluster = int(s["db_labels"][core_idx[int(np.argmin(d_core))]])
        db_noise = False
    else:
        db_cluster, db_noise = -1, True

    # --- PCA 2D del punto nuevo ---------------------------------------
    z_new = s["pca"].transform(xs.reshape(1, -1))[0]

    # --- Miembros y perfiles ------------------------------------------
    assigned = {"kmeans": km_cluster, "hierarchical": hier_cluster, "dbscan": db_cluster}
    members = {}
    profiles = {}
    for method, labels in (("kmeans", s["km_labels"]),
                           ("hierarchical", s["hi_labels"]),
                           ("dbscan", s["db_labels"])):
        c = assigned[method]
        members[method] = [n for n, lab in zip(names, labels) if lab == c]
        profiles[method] = _profile_table(df, s["cols"], labels, c)

    cards = []
    for method in METHODS:
        c = assigned[method]
        if method == "dbscan" and db_noise:
            detail = (f"Sin core points a distancia ≤ eps={s['eps']:.2f} "
                      f"(mínima: {min_core:.2f})")
        elif method == "kmeans":
            detail = f"Distancia al centroide: {d_cent.min():.2f}"
        elif method == "hierarchical":
            detail = f"Distancia promedio a miembros: {hier_avg[hier_cluster]:.2f}"
        else:
            detail = f"Core point más cercano a {min_core:.2f} (eps={s['eps']:.2f})"
        cards.append({
            "key": method,
            "title": METHOD_TITLES[method],
            "cluster": c,
            "noise": bool(method == "dbscan" and db_noise),
            "n_members": len(members[method]),
            "detail": detail,
        })

    return {
        "scenario": scen,
        "scenario_tag": scenario_tag(scen),
        "cols": s["cols"],
        "vector": {c: float(v) for c, v in zip(s["cols"], x)},
        "cards": cards,
        "km_distances": km_distances,
        "hier_distances": hier_distances,
        "dbscan_info": {
            "eps": s["eps"],
            "min_samples": s["min_samples"],
            "min_core_dist": round(min_core, 3),
            "noise": db_noise,
        },
        "pca": {
            "Z": s["Z"],
            "z_new": z_new,
            "var": float(s["pca"].explained_variance_ratio_.sum()),
            "color_labels": s["km_labels"],
            "km_clusters": s["km_clusters"],
            "names": names,
        },
        "profile_kmeans": profiles["kmeans"],
        "members": members,
        "assigned": assigned,
    }


def scatter_png_b64(pred: dict) -> str:
    """Scatter PCA 2D (coloreado por k-means) con el punto nuevo destacado."""
    p = pred["pca"]
    Z, labels, names = p["Z"], p["color_labels"], p["names"]
    z_new = p["z_new"]
    palette = sns.color_palette("tab10", len(p["km_clusters"]))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, c in enumerate(p["km_clusters"]):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=45, color=palette[i],
                   label=f"cluster {c}")
    for (x, y), n in zip(Z, names):
        ax.annotate(n, (x, y), fontsize=5.5, alpha=0.7,
                    xytext=(2, 2), textcoords="offset points")
    ax.scatter([z_new[0]], [z_new[1]], marker="*", s=500, c="black",
               label="vector nuevo", zorder=5)
    ax.annotate("NUEVO", z_new, fontsize=9, weight="bold",
                xytext=(8, 8), textcoords="offset points")
    ax.set_title(f"Proyección PCA ({pred['scenario_tag']}) — "
                 f"{p['var']:.0%} var. explicada — color = k-means")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, dpi=130, bbox_inches="tight", format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def check_consistency() -> None:
    """Re-clasifica las 85 golosinas y compara con las etiquetas guardadas."""
    ctx = _context()
    df = ctx["df"]
    for scen, s in ctx["scenarios"].items():
        X_raw = df[s["cols"]].to_numpy(dtype=float)
        ok = {"kmeans": 0, "hierarchical": 0, "dbscan": 0}
        for i in range(len(df)):
            r = predict(X_raw[i], scen)
            ok["kmeans"] += r["assigned"]["kmeans"] == int(s["km_labels"][i])
            ok["hierarchical"] += r["assigned"]["hierarchical"] == int(s["hi_labels"][i])
            ok["dbscan"] += r["assigned"]["dbscan"] == int(s["db_labels"][i])
        n = len(df)
        print(f"[{scen}] consistencia sobre {n} puntos: "
              f"k-means {ok['kmeans']}/{n}, "
              f"jerárquico {ok['hierarchical']}/{n}, "
              f"DBSCAN {ok['dbscan']}/{n}")


if __name__ == "__main__":
    check_consistency()
