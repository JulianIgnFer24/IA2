"""
Utilidades compartidas del TP2 — Clustering (candy-data.csv).

Carga del dataset, construcción de las dos matrices de features
(X_comp: sin winpercent, X_full: con winpercent), escalado,
métricas internas y helpers de graficado.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
TP2_DIR = CODE_DIR.parent
DATA_PATH = TP2_DIR / "candy-data.csv"
FIGS_DIR = TP2_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Columnas
# ---------------------------------------------------------------------------
NAME_COL = "competitorname"
BINARY_COLS = [
    "chocolate",
    "fruity",
    "caramel",
    "peanutyalmondy",
    "nougat",
    "crispedricewafer",
    "hard",
    "bar",
    "pluribus",
]
CONT_COLS = ["sugarpercent", "pricepercent"]  # percentiles en [0, 1]
WIN_COL = "winpercent"  # popularidad (0-100), variable de análisis

FEATURES_COMP = BINARY_COLS + CONT_COLS            # 11 features, sin winpercent
FEATURES_FULL = FEATURES_COMP + [WIN_COL]          # 12 features, con winpercent

# Etiquetas de los dos escenarios de clustering
SCENARIOS = ("comp", "full")

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Carga y escalado
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Lee candy-data.csv y hace limpieza mínima de nombres."""
    df = pd.read_csv(DATA_PATH)
    # El CSV tiene caracteres mal codificados (Õ en vez de ')
    df[NAME_COL] = df[NAME_COL].str.replace("Õ", "'", regex=False)
    return df


def build_features(df: pd.DataFrame) -> dict:
    """
    Devuelve dict con las dos matrices escaladas:
      {"comp": (X, cols), "full": (X, cols)}
    StandardScaler en todas las columnas para que winpercent (0-100)
    no domine las distancias euclídeas.
    """
    out = {}
    for scen, cols in (("comp", FEATURES_COMP), ("full", FEATURES_FULL)):
        X = StandardScaler().fit_transform(df[cols].to_numpy(dtype=float))
        out[scen] = (X, cols)
    return out


# ---------------------------------------------------------------------------
# Métricas internas (no hay ground truth)
# ---------------------------------------------------------------------------
def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Silhouette, Calinski-Harabasz y Davies-Bouldin.
    Si hay ruido (label -1, DBSCAN) se calculan excluyéndolo.
    Devuelve NaN si no es posible calcularlas (< 2 clusters).
    """
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())
    res = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    if n_clusters >= 2 and mask.sum() > n_clusters:
        res["silhouette"] = silhouette_score(X[mask], labels[mask])
        res["calinski_harabasz"] = calinski_harabasz_score(X[mask], labels[mask])
        res["davies_bouldin"] = davies_bouldin_score(X[mask], labels[mask])
    return res


# ---------------------------------------------------------------------------
# Helpers de graficado
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, name: str) -> Path:
    """Guarda la figura en figs/ con nombre dado y la cierra."""
    path = FIGS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.relative_to(TP2_DIR)}")
    return path


def scenario_tag(scen: str) -> str:
    """Sufijo legible para títulos: 'sin winpercent' / 'con winpercent'."""
    return "sin winpercent" if scen == "comp" else "con winpercent"
