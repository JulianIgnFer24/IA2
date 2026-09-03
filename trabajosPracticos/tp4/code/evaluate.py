"""
Evaluación y reporte del TP4.

Toma results/experimentos.pkl y los modelos entrenados y genera:
  - comparación de curvas de aprendizaje entre los 4 experimentos,
  - barras de accuracy train/val/test y del gap de sobreajuste,
  - parámetros entrenables vs. accuracy de test,
  - matriz de confusión del mejor modelo global (conteos y normalizada),
  - accuracy por clase, ejemplos mal clasificados,
  - muestras del dataset y efecto del aumento de datos,
  - tablas Markdown (results/tabla_resumen.md, results/metricas_por_clase.md).
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from common import (
    CLASS_NAMES,
    ERASING_AREA,
    ERASING_PROB,
    RESULTS_DIR,
    ROTATION_FACTOR,
    augmentation_layers,
    load_data,
    random_erasing_layers,
    save_fig,
    set_seeds,
)

RESULTS_PATH = RESULTS_DIR / "experimentos.pkl"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def load_resultados():
    with open(RESULTS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["resultados"], data["mejor_key"]


# ---------------------------------------------------------------------------
# Comparación entre experimentos
# ---------------------------------------------------------------------------
def plot_comparacion_curvas(resultados: list) -> None:
    """Curvas de los 4 experimentos superpuestas (train punteado, val sólido)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, r in zip(PALETTE, resultados):
        h = r["history"]
        ep = range(1, len(h["loss"]) + 1)
        lab = f"Exp {r['id']} — {r['nombre']}"
        axes[0].plot(ep, h["val_loss"], color=color, label=lab)
        axes[0].plot(ep, h["loss"], color=color, ls=":", alpha=0.6)
        axes[1].plot(ep, h["val_accuracy"], color=color, label=lab)
        axes[1].plot(ep, h["accuracy"], color=color, ls=":", alpha=0.6)

    axes[0].set_title("Pérdida — validación (sólido) vs entrenamiento (punteado)")
    axes[0].set_ylabel("Pérdida")
    axes[1].set_title("Exactitud — validación (sólido) vs entrenamiento (punteado)")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.set_xlabel("Época")
    axes[0].legend(fontsize=8)
    fig.suptitle("Comparación de curvas de aprendizaje", fontsize=13, fontweight="bold")
    save_fig(fig, "comparacion_curvas.png")


def plot_comparacion_accuracy(resultados: list) -> None:
    """Barras agrupadas de accuracy en train / validación / test."""
    etiquetas = [f"Exp {r['id']}\n{r['nombre']}" for r in resultados]
    series = {
        "Train": [r["train_acc"] for r in resultados],
        "Validación": [r["val_acc"] for r in resultados],
        "Test": [r["test_acc"] for r in resultados],
    }
    x = np.arange(len(resultados))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (nombre, vals) in enumerate(series.items()):
        pos = x + (i - 1) * w
        ax.bar(pos, vals, w, label=nombre, color=PALETTE[i])
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x, etiquetas)
    ax.set_ylim(min(min(v) for v in series.values()) - 0.03, 1.005)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exactitud por conjunto y experimento", fontweight="bold")
    ax.legend()
    save_fig(fig, "comparacion_accuracy.png")


def plot_sobreajuste(resultados: list) -> None:
    """Gap train-validación (accuracy y pérdida) como indicador de sobreajuste."""
    etiquetas = [f"Exp {r['id']}" for r in resultados]
    gap_acc = [r["train_acc"] - r["val_acc"] for r in resultados]
    gap_loss = [r["val_loss"] - r["train_loss"] for r in resultados]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, titulo, ylab in (
        (axes[0], gap_acc, "Gap de exactitud (train − val)", "Δ accuracy"),
        (axes[1], gap_loss, "Gap de pérdida (val − train)", "Δ pérdida"),
    ):
        ax.bar(etiquetas, vals, color=PALETTE)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(titulo)
        ax.set_ylabel(ylab)
        ax.margins(y=0.18)
    fig.suptitle("Indicadores de sobreajuste (valores altos = más sobreajuste)", fontsize=12, fontweight="bold")
    save_fig(fig, "gap_sobreajuste.png")


def plot_parametros(resultados: list) -> None:
    """Parámetros entrenables por experimento y su relación con el test accuracy."""
    etiquetas = [f"Exp {r['id']}\n{r['nombre']}" for r in resultados]
    params = [r["params_entrenables"] for r in resultados]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(etiquetas, params, color=PALETTE)
    for i, p in enumerate(params):
        axes[0].text(i, p, f"{p:,}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Parámetros entrenables")
    axes[0].set_title("Complejidad del modelo")
    axes[0].margins(y=0.15)

    for color, r in zip(PALETTE, resultados):
        axes[1].scatter(r["params_entrenables"], r["test_acc"], s=110, color=color, zorder=3)
        axes[1].annotate(
            f"Exp {r['id']}",
            (r["params_entrenables"], r["test_acc"]),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=9,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Parámetros entrenables (escala log)")
    axes[1].set_ylabel("Accuracy en test")
    axes[1].set_title("Parámetros vs. desempeño en test")
    save_fig(fig, "parametros_vs_accuracy.png")


# ---------------------------------------------------------------------------
# Mejor modelo global
# ---------------------------------------------------------------------------
def plot_matriz_confusion(y_true, y_pred, mejor: dict) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 8},
    )
    axes[0].set_title("Conteos absolutos")
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1], vmin=0, vmax=1,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 8},
    )
    axes[1].set_title("Normalizada por clase real (recall en la diagonal)")
    for ax in axes:
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(
        f"Matriz de confusión en test — mejor modelo: Exp {mejor['id']} — {mejor['nombre']} "
        f"(test acc = {mejor['test_acc']:.4f})",
        fontsize=13,
        fontweight="bold",
    )
    save_fig(fig, "matriz_confusion_mejor.png")
    return cm


def plot_accuracy_por_clase(cm: np.ndarray, mejor: dict) -> np.ndarray:
    recall = np.diag(cm) / cm.sum(axis=1)
    orden = np.argsort(recall)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([CLASS_NAMES[i] for i in orden], recall[orden], color="#4C72B0")
    for i, v in enumerate(recall[orden]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Recall (accuracy por clase)")
    ax.set_title(f"Desempeño por clase — Exp {mejor['id']} ({mejor['nombre']})", fontweight="bold")
    save_fig(fig, "accuracy_por_clase.png")
    return recall


def plot_errores(X_test, y_true, y_pred, probs, mejor: dict, n: int = 15) -> None:
    """Ejemplos mal clasificados con mayor confianza (los errores 'más graves')."""
    errores = np.flatnonzero(y_true != y_pred)
    conf = probs[errores, y_pred[errores]]
    peores = errores[np.argsort(-conf)][:n]

    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    for ax, idx in zip(axes.ravel(), peores):
        ax.imshow(X_test[idx].squeeze(), cmap="gray")
        ax.set_title(
            f"real: {CLASS_NAMES[y_true[idx]]}\npred: {CLASS_NAMES[y_pred[idx]]} ({probs[idx, y_pred[idx]]:.2f})",
            fontsize=8,
        )
        ax.axis("off")
    fig.suptitle(
        f"Errores más confiados en test — Exp {mejor['id']} ({mejor['nombre']})",
        fontsize=13,
        fontweight="bold",
    )
    save_fig(fig, "errores_mejor_modelo.png")


# ---------------------------------------------------------------------------
# Dataset y aumento de datos
# ---------------------------------------------------------------------------
def plot_dataset(X_train, y_train) -> None:
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for c in range(10):
        idx = np.flatnonzero(y_train == c)[:3]
        for fila, i in enumerate(idx):
            axes[fila, c].imshow(X_train[i].squeeze(), cmap="gray")
            axes[fila, c].axis("off")
        axes[0, c].set_title(f"{c}: {CLASS_NAMES[c]}", fontsize=8)
    fig.suptitle("Fashion-MNIST — 3 muestras por clase (train)", fontsize=13, fontweight="bold")
    save_fig(fig, "dataset_muestras.png")


def plot_augmentacion(X_train) -> None:
    aug = augmentation_layers()
    muestras = X_train[:8]
    rotadas = aug(muestras, training=True).numpy()
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for j in range(8):
        axes[0, j].imshow(muestras[j].squeeze(), cmap="gray")
        axes[1, j].imshow(rotadas[j].squeeze(), cmap="gray")
        for fila in (0, 1):
            axes[fila, j].axis("off")
    fig.suptitle(
        f"Aumento de datos: original (arriba) vs rotación aleatoria factor={ROTATION_FACTOR} (abajo)",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "augmentacion_ejemplos.png")


def plot_random_erasing(X_train) -> None:
    erasing = random_erasing_layers()
    muestras = X_train[:8]
    ocluidas = erasing(muestras, training=True).numpy()
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for j in range(8):
        axes[0, j].imshow(muestras[j].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(ocluidas[j].squeeze(), cmap="gray", vmin=0, vmax=1)
        for fila in (0, 1):
            axes[fila, j].axis("off")
    fig.suptitle(
        f"Random Erasing: original (arriba) vs ocluida (abajo) — p={ERASING_PROB}, "
        f"área {ERASING_AREA[0]:.0%}-{ERASING_AREA[1]:.0%}",
        fontsize=12,
        fontweight="bold",
    )
    save_fig(fig, "random_erasing_ejemplos.png")


# ---------------------------------------------------------------------------
# Tablas Markdown
# ---------------------------------------------------------------------------
def tabla_resumen(resultados: list, mejor_key: str) -> str:
    filas = [
        "| Exp ID | Arquitectura | Pooling | Tamaño Kernel | Dropout | Data Aug. | Train Acc | Val Acc | Test Acc | Param. Totales |",
        "|:---:|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    for r in resultados:
        marca = " ⭐" if r["key"] == mejor_key else ""
        filas.append(
            f"| {r['id']} | {r['nombre']}{marca} | {r['pooling']} | {r['kernel']} | {r['dropout']} | "
            f"{r['data_aug']} | {r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | "
            f"{r['params_entrenables']:,} |"
        )
    return "\n".join(filas)


def tabla_por_clase(y_true, y_pred) -> str:
    rep = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, digits=4, zero_division=0
    )
    filas = [
        "| Clase | Precision | Recall | F1-score | Soporte |",
        "|---|---:|---:|---:|---:|",
    ]
    for nombre in CLASS_NAMES:
        m = rep[nombre]
        filas.append(
            f"| {nombre} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |"
        )
    m = rep["macro avg"]
    filas.append(
        f"| **Promedio macro** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | "
        f"{int(m['support'])} |"
    )
    return "\n".join(filas)


def main() -> None:
    set_seeds()
    resultados, mejor_key = load_resultados()
    mejor = next(r for r in resultados if r["key"] == mejor_key)

    (X_train, y_train), _, (X_test, y_test) = load_data()

    print("Gráficos comparativos...")
    plot_comparacion_curvas(resultados)
    plot_comparacion_accuracy(resultados)
    plot_sobreajuste(resultados)
    plot_parametros(resultados)
    plot_dataset(X_train, y_train)
    plot_augmentacion(X_train)
    plot_random_erasing(X_train)

    print(f"Mejor modelo: Exp {mejor['id']} — {mejor['nombre']} (test acc {mejor['test_acc']:.4f})")
    modelo = tf.keras.models.load_model(mejor["modelo_path"])
    probs = modelo.predict(X_test, batch_size=512, verbose=0)
    y_pred = probs.argmax(axis=1)

    cm = plot_matriz_confusion(y_test, y_pred, mejor)
    recall = plot_accuracy_por_clase(cm, mejor)
    plot_errores(X_test, y_test, y_pred, probs, mejor)

    np.save(RESULTS_DIR / "matriz_confusion_mejor.npy", cm)
    (RESULTS_DIR / "tabla_resumen.md").write_text(tabla_resumen(resultados, mejor_key) + "\n")
    (RESULTS_DIR / "metricas_por_clase.md").write_text(tabla_por_clase(y_test, y_pred) + "\n")

    print("\n" + tabla_resumen(resultados, mejor_key))
    print("\n" + tabla_por_clase(y_test, y_pred))
    peor = int(np.argmin(recall))
    confusiones = cm[peor].copy()
    confusiones[peor] = 0
    print(
        f"\nClase más difícil: {CLASS_NAMES[peor]} (recall {recall[peor]:.4f}), "
        f"confundida sobre todo con {CLASS_NAMES[int(confusiones.argmax())]} ({confusiones.max()} casos)."
    )
    for r in resultados:
        print(
            f"Exp {r['id']}: épocas={r['epocas_corridas']} (mejor {r['mejor_epoca']}), "
            f"gap train-val={r['train_acc'] - r['val_acc']:+.4f}, tiempo={r['tiempo_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
