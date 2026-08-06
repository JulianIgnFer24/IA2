"""
(3) Regresión Lineal Multivariable (OLS) — TP1 Ejercicio 1
Ajusta OLS sobre todos los predictores, reporta coeficientes (β̂), VIF, métricas
en test y analiza supuestos de Gauss-Markov (residuos vs ajustados, Q-Q).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from common import load_data, metrics

d = load_data()
X_train, X_test = d["X_train"], d["X_test"]
y_train, y_test = d["y_train"], d["y_test"]
feature_names = d["feature_names"]
num_cols = d["num_cols"]

Xtr_aug = sm.add_constant(X_train.values, has_constant="add")
Xte_aug = sm.add_constant(X_test.values, has_constant="add")
model = sm.OLS(y_train, Xtr_aug).fit()

intercept = model.params.iloc[0]
coefs = pd.Series(model.params.iloc[1:].values, index=feature_names)
pvals = pd.Series(model.pvalues.iloc[1:].values, index=feature_names)

print("=" * 72)
print("MODELO OLS — variable respuesta log(SalePrice)")
print("=" * 72)

print(f"\n[R] R² = {model.rsquared:.4f} | R² ajustado = {model.rsquared_adj:.4f} (train)")
print(f"[β0] Intercepto (log) = {intercept:.4f}  -> precio base de la categoría "
      f"de referencia: ${np.exp(intercept):,.0f}")

num_coef = coefs.loc[num_cols].abs().sort_values(ascending=False).head(12)
print("\n[+] Coeficientes NUMÉRICOS más influyentes por |β̂|:")
for feat in num_coef.index:
    sig = "***" if pvals[feat] < 0.001 else ("**" if pvals[feat] < 0.01 else
          ("*" if pvals[feat] < 0.05 else "n.s."))
    print(f"   {feat:16s} β̂={coefs.loc[feat]:+.4f}  p={pvals[feat]:.3g}  {sig}")

# Interpretación de dummies (magnitud, signo y categoría de referencia)
dummy_cols = [c for c in feature_names if c not in num_cols]
dummy_top = coefs.loc[dummy_cols].abs().sort_values(ascending=False).head(10)
print("\n[>] Dummies con mayor efecto (vs. categoría de referencia = primer nivel "
      "alfabético, con β=0):")
for feat in dummy_top.index:
    cat, lvl = feat.rsplit("_", 1)
    print(f"   {cat}:{lvl:12s} β̂={coefs.loc[feat]:+.4f}  "
          f"(exp(β̂)={np.exp(coefs.loc[feat]):.2f} => multiplica el precio)")

# --- VIF de cada predictor ---
print("\n[VIF] Multicolinealidad (VIF > 10 => colinealidad fuerte; VIF = inf => perfecta):")
vif_df = pd.DataFrame({
    "predictor": feature_names,
    "VIF": [variance_inflation_factor(Xtr_aug, i + 1) for i in range(len(feature_names))]
}).sort_values("VIF", ascending=False)
inf_cnt = int(np.isinf(vif_df["VIF"]).sum())
finite = vif_df[np.isfinite(vif_df["VIF"])]
high = finite[finite["VIF"] >= 10]
print(f"Con VIF=inf (relación lineal perfecta): {inf_cnt} de {len(vif_df)}")
print(f"Con VIF >= 10 (colinealidad fuerte): {len(high)}")
print("\nTop VIF finitos:")
print(high.head(12).to_string(index=False))

print("\n[CANDIDATAS] Grupos de variables con colinealidad perfecta (VIF=inf):")
print(" - TotalBsmtSF  = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF  (suma directa)")
print(" - GrLivArea    = 1stFlrSF + 2ndFlrSF + LowQualFinSF  (suma directa)")
print(" - Dummies '*_NaN' de garaje (GarageType, GarageFinish, GarageQual,")
print("   GarageCond) son 1 en las mismas 81 casas sin garaje => idénticas")
print(" - Dummies '*_NaN' de sótano (BsmtQual, BsmtCond, BsmtExposure,")
print("   BsmtFinType1, BsmtFinType2) son 1 en las mismas ~37 casas")
print("Candidatas a ELIMINAR (redundantes): componentes de áreas y dummies NaN.")
print("Candidatas a COMBINAR: p.ej. un único indicador 'tiene garaje'.")

# --- Evaluación en test ---
pred_test = model.predict(Xte_aug)
m = metrics(y_test, pred_test)
print("\n[TEST] Evaluación sobre set de prueba:")
print(f"   R² test             = {m['r2']:.4f}")
print(f"   RMSE (log)          = {m['rmse_log']:.4f}")
print(f"   MAE  (log)          = {m['mae_log']:.4f}")
print(f"   RMSE ($)            = ${m['rmse_dol']:,.0f}")
print(f"   MAE  ($)            = ${m['mae_dol']:,.0f}")

# --- Análisis de residuos ---
resid = y_train - model.predict(Xtr_aug)
fitted = model.predict(Xtr_aug)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(fitted, resid, alpha=0.4, s=10)
ax[0].axhline(0, color="red", lw=1)
ax[0].set_xlabel("Valores ajustados (log)")
ax[0].set_ylabel("Residuos")
ax[0].set_title("Residuos vs. ajustados (OLS)")
stats.probplot(resid, dist="norm", plot=ax[1])
ax[1].set_title("Gráfico Q-Q de residuos (OLS)")
plt.tight_layout()
plt.savefig("figs/ols_residuos.png", dpi=130)
plt.close()
print("\n[RES] Figuras guardadas en figs/ols_residuos.png")
sw = stats.shapiro(resid)
print(f"[RES] Shapiro-Wilk sobre residuos: W={sw.statistic:.4f}, p={sw.pvalue:.3g} "
      "(p<0.05 => se rechaza normalidad de residuos)")
print("Asimetría de residuos:", round(float(stats.skew(resid)), 3))