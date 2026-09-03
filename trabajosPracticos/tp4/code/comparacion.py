"""
Comparación global de todos los experimentos del TP4.

Toma results/experimentos.pkl y los modelos entrenados y genera la vista de conjunto
de las 8 configuraciones probadas, en orden cronológico:

  1. LeNet-5 clásica          -> 2. LeNet-5 optimizada
  7. AlexNet angosta+Dropout  -> 8. + rotación
  3. AlexNet ancha sin Dropout-> 4. + rotación
  5. WRN-28-10                -> 6. + Random Erasing

Salidas: results/cmp_*.png y results/cmp_tabla.md
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

from common import CLASS_NAMES, RESULTS_DIR, load_data, save_fig, set_seeds

RESULTS_PATH = RESULTS_DIR / "experimentos.pkl"

# Orden en que se corrieron los experimentos (los ids 7-8 preceden a 3-4)
ORDEN = [1, 2, 7, 8, 3, 4, 5, 6]

# Qué introduce cada paso respecto del anterior
CAMBIO = {
    1: "punto de partida",
    2: "kernels 3x3, max pool, padding same",
    7: "profundidad (5 conv) + Dropout 0.5",
    8: "+ rotación leve",
    3: "conv4/5 más anchas (512/256), sin Dropout",
    4: "+ rotación leve",
    5: "arquitectura residual ancha",
    6: "+ Random Erasing",
}

# Color = familia de arquitectura (identidad); el estilo de línea separa las variantes.
# Los tres tonos pasan las seis comprobaciones del validador con --pairs all.
FAMILIA = {"LeNet-5": "#2a78d6", "AlexNet": "#eb6834", "WRN-28-10": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # train / validación / test

# Rampa secuencial de un solo tono para las magnitudes (recall, confusión)
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul_seq",
    ["#eef5fe", "#cde2fb", "#b7d3f6", "#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95"],
)

GRIS_EJE = "#9a9a94"
TINTA = "#2b2b28"


def estilo_base(ax, titulo: str = "", ylab: str = "", xlab: str = "") -> None:
    """Grilla y ejes recesivos, sin marcos innecesarios."""
    ax.set_axisbelow(True)
    ax.grid(True, color="#e4e4e0", lw=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS_EJE)
    ax.tick_params(colors=GRIS_EJE, labelsize=9)
    for etiqueta in ax.get_xticklabels() + ax.get_yticklabels():
        etiqueta.set_color(TINTA)
    if titulo:
        ax.set_title(titulo, fontsize=12, fontweight="bold", color=TINTA, pad=10)
    ax.set_ylabel(ylab, fontsize=10, color=TINTA)
    ax.set_xlabel(xlab, fontsize=10, color=TINTA)


def cargar():
    with open(RESULTS_PATH, "rb") as f:
        datos = pickle.load(f)
    por_id = {r["id"]: r for r in datos["resultados"]}
    faltan = [i for i in ORDEN if i not in por_id]
    if faltan:
        raise SystemExit(f"Faltan experimentos {faltan}: corré primero train.py {' '.join(map(str, faltan))}")
    return [por_id[i] for i in ORDEN]


def familia(r: dict) -> str:
    return r["arquitectura"]


def etiqueta(r: dict) -> str:
    return f"Exp {r['id']} · {r['nombre']}"


# ---------------------------------------------------------------------------
# 1. Evolución cronológica
# ---------------------------------------------------------------------------
def plot_evolucion(resultados: list) -> None:
    """Serie única: cómo fue moviéndose el test accuracy a lo largo del trabajo."""
    acc = [r["test_acc"] for r in resultados]
    y = np.arange(len(resultados))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(acc, y, color="#c9c9c4", lw=2, zorder=1)
    for yi, r in zip(y, resultados):
        ax.scatter(r["test_acc"], yi, s=140, color=FAMILIA[familia(r)], zorder=3,
                   edgecolor="white", linewidth=2)
        ax.annotate(
            f"  {r['test_acc']:.4f}   {CAMBIO[r['id']]}",
            (r["test_acc"], yi), textcoords="offset points", xytext=(10, -4),
            fontsize=9, color=TINTA, va="center",
        )

    estilo_base(ax, "Evolución del accuracy en test a lo largo del TP", "", "Accuracy en test")
    ax.set_yticks(y, [etiqueta(r) for r in resultados], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(min(acc) - 0.004, max(acc) + 0.035)
    manijas = [plt.Line2D([], [], marker="o", ls="", color=c, ms=9, label=f) for f, c in FAMILIA.items()]
    ax.legend(handles=manijas, frameon=False, fontsize=9, loc="lower right")
    save_fig(fig, "cmp_evolucion.png")


# ---------------------------------------------------------------------------
# 2. Curvas de aprendizaje de las 8 corridas
# ---------------------------------------------------------------------------
def plot_curvas(resultados: list) -> None:
    """Color por familia, línea llena/punteada según la variante dentro de la familia."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    vistas = {}
    for r in resultados:
        fam = familia(r)
        estilo = "-" if vistas.get(fam, 0) % 2 == 0 else "--"
        ancho = 2.0 if vistas.get(fam, 0) % 2 == 0 else 1.6
        vistas[fam] = vistas.get(fam, 0) + 1
        h = r["history"]
        ep = range(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["val_loss"], color=FAMILIA[fam], ls=estilo, lw=ancho, label=etiqueta(r))
        axes[1].plot(ep, h["val_accuracy"], color=FAMILIA[fam], ls=estilo, lw=ancho, label=etiqueta(r))

    # marca del mínimo de cada curva de pérdida (etiqueta directa selectiva)
    mejor = min(resultados, key=lambda r: min(r["history"]["val_loss"]))
    ep_mejor = int(np.argmin(mejor["history"]["val_loss"])) + 1
    axes[0].scatter([ep_mejor], [min(mejor["history"]["val_loss"])], s=90, zorder=4,
                    color=FAMILIA[familia(mejor)], edgecolor="white", linewidth=2)
    axes[0].annotate(
        f"mínimo global {min(mejor['history']['val_loss']):.4f}\n(Exp {mejor['id']}, época {ep_mejor})",
        (ep_mejor, min(mejor["history"]["val_loss"])),
        textcoords="offset points", xytext=(10, 18), fontsize=9, color=TINTA,
    )

    estilo_base(axes[0], "Pérdida de validación", "Pérdida", "Época")
    estilo_base(axes[1], "Exactitud de validación", "Accuracy", "Época")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Curvas de validación de las 8 configuraciones", fontsize=13, fontweight="bold", color=TINTA)
    save_fig(fig, "cmp_curvas_todas.png")


def plot_dos_modelos(resultados: list, ids=(7, 6)) -> None:
    """
    Curvas de los dos modelos que se comparan en la presentación: el mejor de las
    arquitecturas vistas en clase y el que se investigó aparte.
    """
    elegidos = [r for i in ids for r in resultados if r["id"] == i]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for r in elegidos:
        color = FAMILIA[familia(r)]
        h = r["history"]
        ep = range(1, len(h["loss"]) + 1)
        axes[0].plot(ep, h["val_loss"], color=color, lw=2.2, label=f"{etiqueta(r)} — validación")
        axes[0].plot(ep, h["loss"], color=color, lw=1.5, ls=":", label=f"{etiqueta(r)} — entrenamiento")
        axes[1].plot(ep, h["val_accuracy"], color=color, lw=2.2, label=etiqueta(r))
        axes[1].plot(ep, h["accuracy"], color=color, lw=1.5, ls=":")

        # mejor época de cada modelo
        mejor = r["mejor_epoca"]
        axes[0].scatter([mejor], [h["val_loss"][mejor - 1]], s=95, color=color, zorder=4,
                        edgecolor="white", linewidth=2)
        axes[0].annotate(
            f"mejor época {mejor}\n{h['val_loss'][mejor - 1]:.4f}",
            (mejor, h["val_loss"][mejor - 1]), textcoords="offset points",
            xytext=(10, 10), fontsize=9, color=TINTA,
        )

    estilo_base(axes[0], "Pérdida — validación (llena) vs entrenamiento (punteada)", "Pérdida", "Época")
    estilo_base(axes[1], "Exactitud — validación (llena) vs entrenamiento (punteada)", "Accuracy", "Época")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("El mejor modelo visto en clase vs. el investigado",
                 fontsize=13, fontweight="bold", color=TINTA)
    save_fig(fig, "cmp_dos_modelos.png")


# ---------------------------------------------------------------------------
# 3. Accuracy por conjunto (dot plot: diferencias de ~1 punto)
# ---------------------------------------------------------------------------
def plot_accuracy(resultados: list) -> None:
    y = np.arange(len(resultados))[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    for yi, r in zip(y, resultados):
        valores = [r["train_acc"], r["val_acc"], r["test_acc"]]
        ax.plot([min(valores), max(valores)], [yi, yi], color="#d8d8d3", lw=2, zorder=1)
        for color, v in zip(SERIES, valores):
            ax.scatter(v, yi, s=95, color=color, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{r['test_acc']:.4f}", (r["test_acc"], yi), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, fontweight="bold", color=TINTA)

    estilo_base(ax, "Exactitud por conjunto (etiqueta = test)", "", "Accuracy")
    ax.set_yticks(y, [etiqueta(r) for r in resultados], fontsize=9)
    manijas = [plt.Line2D([], [], marker="o", ls="", color=c, ms=9, label=n)
               for c, n in zip(SERIES, ["Entrenamiento", "Validación", "Test"])]
    ax.legend(handles=manijas, frameon=False, fontsize=9, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.16))
    save_fig(fig, "cmp_accuracy.png")


# ---------------------------------------------------------------------------
# 4. Costo vs beneficio
# ---------------------------------------------------------------------------
def plot_costo(resultados: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for ax, clave, xlab, titulo in (
        (axes[0], "params_entrenables", "Parámetros entrenables (log)", "Complejidad vs desempeño"),
        (axes[1], "tiempo_s", "Tiempo de entrenamiento en segundos (log)", "Costo de cómputo vs desempeño"),
    ):
        for r in resultados:
            ax.scatter(r[clave], r["test_acc"], s=130, color=FAMILIA[familia(r)], zorder=3,
                       edgecolor="white", linewidth=2)
            ax.annotate(f"Exp {r['id']}", (r[clave], r["test_acc"]), textcoords="offset points",
                        xytext=(9, -4), fontsize=9, color=TINTA)
        ax.set_xscale("log")
        estilo_base(ax, titulo, "Accuracy en test", xlab)

    manijas = [plt.Line2D([], [], marker="o", ls="", color=c, ms=9, label=f) for f, c in FAMILIA.items()]
    axes[0].legend(handles=manijas, frameon=False, fontsize=9, loc="lower right")
    save_fig(fig, "cmp_costo.png")


# ---------------------------------------------------------------------------
# 5. Sobreajuste
# ---------------------------------------------------------------------------
def plot_sobreajuste(resultados: list) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    etiquetas = [f"Exp {r['id']}" for r in resultados]
    colores = [FAMILIA[familia(r)] for r in resultados]

    gaps = [r["train_acc"] - r["val_acc"] for r in resultados]
    axes[0].bar(etiquetas, gaps, color=colores, width=0.62)
    for i, v in enumerate(gaps):
        axes[0].text(i, v, f"{v:+.4f}", ha="center", va="bottom", fontsize=9, color=TINTA)
    estilo_base(axes[0], "Gap de exactitud (train − validación)", "Δ accuracy")
    axes[0].margins(y=0.18)

    # épocas corridas vs época del mínimo de validación
    for i, r in enumerate(resultados):
        axes[1].plot([i, i], [r["mejor_epoca"], r["epocas_corridas"]], color="#d8d8d3", lw=3, zorder=1)
        axes[1].scatter(i, r["epocas_corridas"], s=90, color="#d8d8d3", edgecolor="white",
                        linewidth=1.5, zorder=3)
        axes[1].scatter(i, r["mejor_epoca"], s=110, color=colores[i], edgecolor="white",
                        linewidth=1.5, zorder=4)
    estilo_base(axes[1], "Mejor época vs épocas corridas", "Época")
    axes[1].set_xticks(range(len(resultados)), etiquetas)
    manijas = [
        plt.Line2D([], [], marker="o", ls="", color="#6b6b64", ms=9, label="Mejor época (mín. val_loss)"),
        plt.Line2D([], [], marker="o", ls="", color="#d8d8d3", ms=9, label="Épocas corridas"),
    ]
    axes[1].legend(handles=manijas, frameon=False, fontsize=9, ncol=2,
                   loc="lower center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Indicadores de sobreajuste", fontsize=13, fontweight="bold", color=TINTA)
    save_fig(fig, "cmp_sobreajuste.png")


# ---------------------------------------------------------------------------
# 6-7. Métricas por clase (requieren predecir con los 8 modelos)
# ---------------------------------------------------------------------------
def predicciones(resultados: list, X_test) -> dict:
    preds = {}
    for r in resultados:
        modelo = tf.keras.models.load_model(r["modelo_path"])
        preds[r["id"]] = modelo.predict(X_test, batch_size=512, verbose=0).argmax(axis=1)
        del modelo
        tf.keras.backend.clear_session()
        print(f"  predicciones Exp {r['id']} listas")
    return preds


def plot_recall_clases(resultados: list, preds: dict, y_test) -> np.ndarray:
    matriz = np.array(
        [[np.mean(preds[r["id"]][y_test == c] == c) for c in range(len(CLASS_NAMES))] for r in resultados]
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        matriz, annot=True, fmt=".3f", cmap=RAMPA_AZUL, vmin=0.6, vmax=1.0, ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=[etiqueta(r) for r in resultados],
        annot_kws={"size": 8}, linewidths=1.5, linecolor="white",
        cbar_kws={"label": "Recall"},
    )
    ax.set_title("Recall por clase y experimento (test)", fontsize=13, fontweight="bold", color=TINTA, pad=12)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_fig(fig, "cmp_recall_clases.png")
    return matriz


def plot_confusiones(resultados: list, preds: dict, y_test) -> None:
    """Matriz normalizada del mejor exponente de cada familia."""
    mejores = {}
    for r in resultados:
        fam = familia(r)
        if fam not in mejores or r["test_acc"] > mejores[fam]["test_acc"]:
            mejores[fam] = r
    elegidos = [mejores[f] for f in FAMILIA if f in mejores]

    fig, axes = plt.subplots(1, len(elegidos), figsize=(6.4 * len(elegidos), 6))
    for ax, r in zip(np.atleast_1d(axes), elegidos):
        cm = confusion_matrix(y_test, preds[r["id"]])
        sns.heatmap(
            cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap=RAMPA_AZUL,
            vmin=0, vmax=1, cbar=False, ax=ax, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"size": 6.5}, linewidths=0.8, linecolor="white",
        )
        ax.set_title(f"{etiqueta(r)}\ntest acc = {r['test_acc']:.4f}", fontsize=11,
                     fontweight="bold", color=TINTA)
        ax.set_xlabel("Predicción", fontsize=9)
        ax.set_ylabel("Clase real", fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.suptitle("Mejor modelo de cada familia — matriz de confusión normalizada",
                 fontsize=13, fontweight="bold", color=TINTA)
    save_fig(fig, "cmp_confusiones_familias.png")


# ---------------------------------------------------------------------------
# Tabla
# ---------------------------------------------------------------------------
def tabla(resultados: list) -> str:
    filas = [
        "| # | Exp | Arquitectura | Regularización | Train | Val | Test | Params | Épocas | Mejor ép. | Tiempo |",
        "|:---:|:---:|---|---|---:|---:|---:|---:|:---:|:---:|---:|",
    ]
    for orden, r in enumerate(resultados, start=1):
        reg = []
        if r["dropout"] != "No":
            reg.append(f"Dropout {r['dropout'].split('(')[-1].rstrip(')')}")
        if r["data_aug"] != "No":
            reg.append(r["data_aug"].split("(")[-1].rstrip(")"))
        filas.append(
            f"| {orden} | {r['id']} | {r['nombre']} | {' + '.join(reg) or '—'} | "
            f"{r['train_acc']:.4f} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | "
            f"{r['params_entrenables']:,} | {r['epocas_corridas']} | {r['mejor_epoca']} | "
            f"{r['tiempo_s']:.0f} s |"
        )
    return "\n".join(filas)


def main() -> None:
    set_seeds()
    sns.set_theme(style="white", context="notebook")
    plt.rcParams["figure.facecolor"] = "white"

    resultados = cargar()
    (_, _), _, (X_test, y_test) = load_data()

    print("Comparación global de los 8 experimentos...")
    plot_evolucion(resultados)
    plot_curvas(resultados)
    plot_dos_modelos(resultados)
    plot_accuracy(resultados)
    plot_costo(resultados)
    plot_sobreajuste(resultados)

    preds = predicciones(resultados, X_test)
    recall = plot_recall_clases(resultados, preds, y_test)
    plot_confusiones(resultados, preds, y_test)

    md = tabla(resultados)
    (RESULTS_DIR / "cmp_tabla.md").write_text(md + "\n")
    print("\n" + md)

    peor_clase = int(np.argmin(recall.mean(axis=0)))
    print(f"\nClase más difícil en promedio: {CLASS_NAMES[peor_clase]} "
          f"(recall medio {recall[:, peor_clase].mean():.4f}, "
          f"de {recall[:, peor_clase].min():.4f} a {recall[:, peor_clase].max():.4f})")
    base, final = resultados[0], max(resultados, key=lambda r: r["test_acc"])
    print(f"Mejora total: {base['test_acc']:.4f} -> {final['test_acc']:.4f} "
          f"({(final['test_acc'] - base['test_acc']) * 100:+.2f} puntos), "
          f"con {final['params_entrenables'] / base['params_entrenables']:.0f}x parámetros "
          f"y {final['tiempo_s'] / base['tiempo_s']:.0f}x tiempo.")


if __name__ == "__main__":
    main()
