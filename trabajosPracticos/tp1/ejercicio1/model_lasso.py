"""
(5) Regresión Lasso — TP1 Ejercicio 1
Búsqueda de λ con validación cruzada, identificación de coeficientes en cero
(selección de variables) y evaluación en test.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from common import load_data, metrics

d = load_data()
X_train, X_test = d["X_train"], d["X_test"]
y_train, y_test = d["y_train"], d["y_test"]
feature_names = d["feature_names"]

n_alphas = 200
lasso = LassoCV(cv=5, max_iter=50000, random_state=42)
lasso.fit(X_train, y_train)

print("=" * 72)
print("REGRESIÓN LASSO — variable respuesta log(SalePrice)")
print("=" * 72)
print(f"[λ] Mejor alpha por CV: {lasso.alpha_:.6g}")
print(f"[R] R² (train) = {lasso.score(X_train, y_train):.4f}")

coefs = lasso.coef_
nonzero = (np.abs(coefs) > 1e-8)
n_zero = int((~nonzero).sum())
n_sel = int(nonzero.sum())

print(f"\n[+] Predictores totales: {len(coefs)}")
print(f"[+] Llevados a cero exactamente por Lasso: {n_zero}")
print(f"[+] Variables que 'sobrevivieron' (coef ≠ 0): {n_sel}")

surv = pd.Series(coefs, index=feature_names)
print("\n[SEL] Variables sobrevivientes con mayor |β̂| (selección de variables):")
top = surv[np.abs(surv) > 1e-8].abs().sort_values(ascending=False).head(20)
for feat in top.index:
    print(f"   {feat:22s} β̂={surv[feat]:+.4f}")

# --- Evaluación en test ---
pred = lasso.predict(X_test)
m = metrics(y_test, pred)
print("\n[TEST] Lasso con α óptimo:")
print(f"   R² test      = {m['r2']:.4f}")
print(f"   RMSE ($)     = ${m['rmse_dol']:,.0f}")
print(f"   MAE  ($)     = ${m['mae_dol']:,.0f}")
print(f"   RMSE (log)   = {m['rmse_log']:.4f}")

# Guardar resultados para comparison.py
with open("lasso_results.pkl", "wb") as f:
    import pickle
    pickle.dump(dict(coef=surv.to_dict(), alpha=lasso.alpha_,
                     n_zero=n_zero, n_selected=n_sel), f)
print("\n[SAVE] Resultados guardados en lasso_results.pkl")