"""
(6) Comparación final — TP1 Ejercicio 1
Tabla comparativa OLS vs Ridge vs Lasso (RMSE test, predictores efectivos,
interpretabilidad) y recomendación basada en el trade-off sesgo-varianza.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
import statsmodels.api as sm
from common import load_data, metrics

d = load_data()
X_train, X_test = d["X_train"], d["X_test"]
y_train, y_test = d["y_train"], d["y_test"]
feature_names = d["feature_names"]

# ---------- OLS ----------
Xtr_aug = sm.add_constant(X_train.values, has_constant="add")
Xte_aug = sm.add_constant(X_test.values, has_constant="add")
ols = sm.OLS(y_train, Xtr_aug).fit()
m_ols_tr = metrics(y_train, ols.predict(Xtr_aug))
m_ols = metrics(y_test, ols.predict(Xte_aug))

# ---------- Ridge ----------
ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5,
                scoring="neg_mean_squared_error")
ridge.fit(X_train, y_train)
m_ridge_tr = metrics(y_train, ridge.predict(X_train))
m_ridge = metrics(y_test, ridge.predict(X_test))

# ---------- Lasso ----------
lasso = LassoCV(cv=5, max_iter=50000, random_state=42)
lasso.fit(X_train, y_train)
m_lasso_tr = metrics(y_train, lasso.predict(X_train))
m_lasso = metrics(y_test, lasso.predict(X_test))

p = len(feature_names)
p_lasso = int((np.abs(lasso.coef_) > 1e-8).sum())

def r2adj(y, pred, k):
    rss = np.sum((np.asarray(y) - np.asarray(pred)) ** 2)
    tss = np.sum((np.asarray(y) - np.mean(y)) ** 2)
    return 1 - (rss / (len(y) - k - 1)) / (tss / (len(y) - 1))

rows = [
    ("OLS", "train", m_ols_tr["r2"], r2adj(y_train, ols.predict(Xtr_aug), p),
     m_ols_tr["rmse_log"], m_ols_tr["mae_log"], m_ols_tr["rmse_dol"], m_ols_tr["mae_dol"]),
    ("OLS", "test", m_ols["r2"], np.nan, m_ols["rmse_log"], m_ols["mae_log"],
     m_ols["rmse_dol"], m_ols["mae_dol"]),
    ("Ridge", "train", m_ridge_tr["r2"], r2adj(y_train, ridge.predict(X_train), p),
     m_ridge_tr["rmse_log"], m_ridge_tr["mae_log"], m_ridge_tr["rmse_dol"], m_ridge_tr["mae_dol"]),
    ("Ridge", "test", m_ridge["r2"], np.nan, m_ridge["rmse_log"], m_ridge["mae_log"],
     m_ridge["rmse_dol"], m_ridge["mae_dol"]),
    ("Lasso", "train", m_lasso_tr["r2"], r2adj(y_train, lasso.predict(X_train), p_lasso),
     m_lasso_tr["rmse_log"], m_lasso_tr["mae_log"], m_lasso_tr["rmse_dol"], m_lasso_tr["mae_dol"]),
    ("Lasso", "test", m_lasso["r2"], np.nan, m_lasso["rmse_log"], m_lasso["mae_log"],
     m_lasso["rmse_dol"], m_lasso["mae_dol"]),
]
tab = pd.DataFrame(rows, columns=["Modelo", "Set", "R2", "R2_adj", "RMSE_log",
                                  "MAE_log", "RMSE_usd", "MAE_usd"]).set_index(["Modelo", "Set"])
tab[["R2", "R2_adj", "RMSE_log", "MAE_log"]] = tab[["R2", "R2_adj", "RMSE_log", "MAE_log"]].round(4)
tab[["RMSE_usd", "MAE_usd"]] = tab[["RMSE_usd", "MAE_usd"]].round(0).astype(int)
print("=" * 76)
print("TABLA COMPARATIVA DE MÉTRICAS (train / test)")
print("=" * 76)
print(tab.to_string())

with open("comparacion_final.pkl", "wb") as f:
    pickle.dump(dict(table=tab, alpha_ridge=ridge.alpha_, alpha_lasso=lasso.alpha_), f)

print("\n[λ] α óptimos: Ridge = {:.4g} | Lasso = {:.5g}".format(ridge.alpha_, lasso.alpha_))
print("\n[RECOMENDACIÓN]")
print("""Dado el fuerte problema de multicolinealidad (19 predictores con VIF=inf y 78
con VIF>=10), OLS sufre de alta varianza en los coeficientes (β̂ inestables, ver
coeficientes absurdos como PoolQC_NaN≈2.2). Entre los modelos regularizados:

- Ridge obtiene el MEJOR R² de test (0.896) y el menor RMSE en log (0.139),
  porque encoge los coeficientes sin eliminarlos, amortiguando la varianza
  provocada por la colinealidad. Su costo es la interpretabilidad: mantiene
  los 260 predictores.

- Lasso ofrece un modelo esparso (solo 69 de 260 predictores) muy
  interpretable y con R² de test casi igual (0.890), aunque pierde un poco de
  precisión frente a Ridge.

RECOMENDACIÓN: si el objetivo es máxima precisión de predicción de precios,
elegir RIDGE. Si además se necesita un modelo explicable/desplegable con
pocas variables (caso de negocio habitual: explicar el precio), elegir LASSO.
""")