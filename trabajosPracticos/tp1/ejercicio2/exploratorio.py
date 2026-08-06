"""
(1) Análisis exploratorio — TP1 Ejercicio 2 (UCI Credit Card, default)
Balance de clases, exploración visual por clase y matriz de correlación.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA = "archive/UCI_Credit_Card.csv"
df = pd.read_csv(DATA)
y = df["default.payment.next.month"]

print("=" * 72)
print("ANÁLISIS EXPLORATORIO — default.payment.next.month")
print("=" * 72)

# --- Balance de clases ---
counts = y.value_counts()
rate = y.mean()
print(f"\n[BALANCE] n={len(y)} | no default (0): {counts[0]} ({1-rate:.1%}) | "
      f"default (1): {counts[1]} ({rate:.1%}) | razón ~1:{counts[0]/counts[1]:.1f}")
print("Dataset DESBALANCEADO: la clase positiva (default) es la minoría (~22%).")
print("Implicancia: accuracy engañosa (baseline mayoritario ~78%); hay que usar")
print("precision/recall/F1, curvas ROC y sobre todo Precision-Recall (PR-AUC).")

fig, ax = plt.subplots(figsize=(5, 4))
counts.plot(kind="bar", color=["#4C72B0", "#C44E52"], ax=ax)
ax.set_xticklabels(["No default (0)", "Default (1)"], rotation=0)
ax.set_ylabel("Cantidad de clientes")
ax.set_title(f"Balance de clases (default = {rate:.1%})")
plt.tight_layout()
plt.savefig("figs/ej2_balance.png", dpi=130)
plt.close()

# --- Numéricas por clase ---
num_cols = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "BILL_AMT1", "PAY_AMT1"]
print("\n[NUM vs CLASE] media ± ds por clase:")
grp = df.groupby("default.payment.next.month")[num_cols]
print(grp.agg(lambda s: f"{s.mean():.0f}±{s.std():.0f}").T.to_string())

sel = ["LIMIT_BAL", "AGE", "PAY_0", "BILL_AMT1", "PAY_AMT1"]
fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 4))
for ax, c in zip(axes, sel):
    data = [df.loc[y == 0, c], df.loc[y == 1, c]]
    ax.boxplot(data, tick_labels=["No default", "Default"])
    ax.set_title(c, fontsize=10)
    ax.tick_params(labelsize=8)
plt.suptitle("Predictores numéricos según ocurra default", y=1.02)
plt.tight_layout()
plt.savefig("figs/ej2_numericas_por_clase.png", dpi=130, bbox_inches="tight")
plt.close()

# --- Categóricas por clase (tasa de default) ---
print("\n[CAT vs CLASE] tasa de default por nivel:")
for c in ["SEX", "EDUCATION", "MARRIAGE"]:
    print(df.groupby(c)["default.payment.next.month"].mean().round(3).to_string())
    print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, ["SEX", "EDUCATION", "MARRIAGE"]):
    df.groupby(c)["default.payment.next.month"].mean().plot(
        kind="bar", color="#C44E52", ax=ax)
    ax.set_title(f"Tasa de default por {c}")
    ax.set_ylabel("P(default)")
    ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig("figs/ej2_categoricas_por_clase.png", dpi=130)
plt.close()

# --- Matriz de correlación numéricas ---
num_all = df.select_dtypes(include=np.number).drop(
    columns=["ID", "default.payment.next.month", "SEX", "EDUCATION", "MARRIAGE"])
cm = num_all.corr()
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(cm, dtype=bool))
sns.heatmap(cm, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.3, cbar_kws={"shrink": 0.7})
plt.title("Correlación entre predictores numéricos")
plt.tight_layout()
plt.savefig("figs/ej2_correlacion.png", dpi=130)
plt.close()

s = cm.abs().unstack()
s = s[s.index.get_level_values(0) != s.index.get_level_values(1)].sort_values(
    ascending=False)
print("[CORR] Pares más correlacionados:")
for i, v in s.head(8).items():
    print(f"   {i[0]} ~ {i[1]}: {v:.3f}")
print("\n[FIG] figs/ej2_balance.png, ej2_numericas_por_clase.png, "
      "ej2_categoricas_por_clase.png, ej2_correlacion.png")