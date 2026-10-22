"""
Análisis exploratorio del dataset candy-data.csv (TP2).

Genera:
  - eda_freq_binarios.png     : frecuencia de cada feature binario
  - eda_hist_continuas.png    : histogramas de sugarpercent/pricepercent/winpercent
  - eda_correlacion.png       : matriz de correlación entre features
  - eda_winpercent.png        : winpercent vs sugarpercent y pricepercent
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import (
    BINARY_COLS,
    CONT_COLS,
    WIN_COL,
    load_data,
    save_fig,
)


def main() -> None:
    df = load_data()
    print(f"Dataset: {df.shape[0]} golosinas, {df.shape[1]} columnas")
    print(f"Nulos totales: {int(df.isna().sum().sum())}")
    print("\nPrimeras filas:")
    print(df.head(3).to_string(index=False))

    # ------------------------------------------------------------------
    # 1. Frecuencia de features binarios
    # ------------------------------------------------------------------
    freq = df[BINARY_COLS].mean().sort_values(ascending=False)
    print("\nFrecuencia de features binarios (% de golosinas):")
    print((freq * 100).round(1).to_string())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=freq.index, y=freq.values, color="steelblue", ax=ax)
    ax.set_title("Frecuencia de features binarios (proporción de golosinas con el atributo)")
    ax.set_ylabel("Proporción")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(freq.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, freq.max() + 0.08)
    save_fig(fig, "eda_freq_binarios.png")

    # ------------------------------------------------------------------
    # 2. Histogramas de las continuas
    # ------------------------------------------------------------------
    cont = CONT_COLS + [WIN_COL]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, cont):
        sns.histplot(df[col], kde=True, ax=ax, color="darkorange")
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Distribución de las variables continuas (percentiles)", y=1.02)
    save_fig(fig, "eda_hist_continuas.png")

    print("\nDescriptivos de continuas:")
    print(df[cont].describe().round(3).to_string())

    # ------------------------------------------------------------------
    # 3. Matriz de correlación
    # ------------------------------------------------------------------
    corr = df[BINARY_COLS + cont].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Matriz de correlación entre features")
    save_fig(fig, "eda_correlacion.png")

    # Pares con mayor correlación absoluta (triángulo superior)
    tri = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(tri).unstack().dropna()
    top = pairs.reindex(pairs.abs().sort_values(ascending=False).index)
    print("\nTop 8 pares por |correlación|:")
    print(top.head(8).round(3).to_string())

    # ------------------------------------------------------------------
    # 4. winpercent vs sugarpercent / pricepercent
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.regplot(
        data=df, x="sugarpercent", y=WIN_COL, ax=axes[0],
        scatter_kws={"alpha": 0.7}, line_kws={"color": "red"},
    )
    axes[0].set_title("winpercent vs sugarpercent")
    sns.regplot(
        data=df, x="pricepercent", y=WIN_COL, ax=axes[1],
        scatter_kws={"alpha": 0.7, "color": "seagreen"}, line_kws={"color": "red"},
    )
    axes[1].set_title("winpercent vs pricepercent")
    save_fig(fig, "eda_winpercent.png")

    r_sugar = df["sugarpercent"].corr(df[WIN_COL])
    r_price = df["pricepercent"].corr(df[WIN_COL])
    print(f"\ncorr(winpercent, sugarpercent) = {r_sugar:.3f}")
    print(f"corr(winpercent, pricepercent) = {r_price:.3f}")


if __name__ == "__main__":
    main()
