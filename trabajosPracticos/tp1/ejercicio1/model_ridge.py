"""
(4) Regresión Ridge — TP1 Ejercicio 1
Búsqueda de λ con validación cruzada (5 folds), camino de regularización,
comparación de coeficientes vs. OLS y evaluación en test.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import cross_val_score
from common import load_data, metrics

d = load_data()
X_train, X_test = d["X_train"], d["X_test"]
y_train, y_test = d["y_train"], d["y_test"]
feature_names = d["feature_names"]

alphas = np.logspace(-3, 3, 100)
cv_ridge = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
cv_ridge.fit(X_train, y_train)
best_alpha = cv_ridge.alpha_

print("=" * 72)
print("REGRESIÓN RIDGE — variable respuesta log(SalePrice)")
print("=" * 72)
print(f"[λ] Mejor alpha por CV (5 folds): {best_alpha:.6g}")
print(f"[R] R² (train, alpha óptimo) = {cv_ridge.score(X_train, y_train):.4f}")

# --- Camino de regularización ---
coef_path = []
for a in np.logspace(-4, 4, 60):
    m = Ridge(alpha=a)
    m.fit(X_train, y_train)
    coef_path.append(m.coef_)
coef_path = np.array(coef_path)  # (n_alphas, n_features)

fig, ax = plt.subplots(figsize=(9, 5))
ls = np.logspace(-4, 4, 60)
for j in range(coef_path.shape[1]):
    ax.plot(ls, coef_path[:, j], linewidth=0.6, alpha=0.6)
ax.set_xscale("log")
ax.axvline(best_alpha, color="red", ls="--", lw=1, label=f"λ*={best_alpha:.3g}")
ax.set_xlabel("λ (log)")
ax.set_ylabel("Coeficientes")
ax.set_title("Camino de regularización Ridge")
ax.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig("figs/ridge_camino.png", dpi=130)
plt.close()
print("[FIG] figs/ridge_camino.png")

# --- Evaluación en test ---
pred = cv_ridge.predict(X_test)
m = metrics(y_test, pred)
print("\n[TEST] Ridge con α óptimo:")
print(f"   R² test      = {m['r2']:.4f}")
print(f"   RMSE ($)     = ${m['rmse_dol']:,.0f}")
print(f"   MAE  ($)     = ${m['mae_dol']:,.0f}")
print(f"   RMSE (log)   = {m['rmse_log']:.4f}")

# --- Comparación Ridge vs OLS ---
from statsmodels.api import add_constant, OLS
Xtr_aug = add_constant(X_train.values, has_constant="add")
ols_coef = OLS(y_train, Xtr_aug).fit().params[1:].values
comp = pd.DataFrame({"OLS": ols_coef, "Ridge": cv_ridge.coef_}, index=feature_names)
comp["|OLS-Ridge|"] = (comp["OLS"] - comp["Ridge"]).abs()
print("\n[COMP] Mayor diferencia |β_OLS − β_Ridge| (los más encogidos):")
print(comp["|OLS-Ridge|"].sort_values(ascending=False).head(10).round(3).to_string())

# Correlacionados: cómo Ridge reparte el peso entre ellos
pairs = [("GarageArea", "GarageCars"), ("GrLivArea", "TotRmsAbvGrd"),
         ("TotalBsmtSF", "1stFlrSF"), ("YearBuilt", "GarageYrBlt")]
print("\n[COMP] Coeficientes en pares correlacionados (OLS vs Ridge):")
for a, b in pairs:
    print(f"   {a:14s} OLS={ols_coef[feature_names.index(a)]:+.3f} Ridge={cv_ridge.coef_[feature_names.index(a)]:+.3f} "
          f"| {b:14s} OLS={ols_coef[feature_names.index(b)]:+.3f} Ridge={cv_ridge.coef_[feature_names.index(b)]:+.3f}")