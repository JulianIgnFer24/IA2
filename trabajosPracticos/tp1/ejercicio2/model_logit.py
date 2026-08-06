"""
(3)-(4) Regresión logística multivariable — TP1 Ejercicio 2
Ajuste por máxima verosimilitud (Newton-Raphson, statsmodels), función de costo
(log-loss / entropía cruzada) e interpretación de coeficientes vía odds ratios.
"""
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common2 import load_data

d = load_data()
X_train, X_test = d["X_train"], d["X_test"]
y_train, y_test = d["y_train"], d["y_test"]
feature_names = d["feature_names"]
num_cols = d["num_cols"]
dummy_cols = d["dummy_cols"]

Xtr_aug = sm.add_constant(X_train.values, has_constant="add")
Xte_aug = sm.add_constant(X_test.values, has_constant="add")

logit = sm.Logit(y_train, Xtr_aug)
res = logit.fit(method="newton")  # Newton-Raphson

print("=" * 76)
print("REGRESIÓN LOGÍSTICA — respuesta default.payment.next.month")
print("=" * 76)
iters = res.mle_retvals.get("iterations", "?")
print(f"[OPT] Método: Newton-Raphson | iteraciones: {iters} | "
      f"convergió: {res.mle_retvals.get('converged', True)}")
print(f"[COSTO] log-loss (entropía cruzada) media en train = {res.llf / -len(y_train):.4f}")
print(f"[LL] log-likelihood: {res.llf:.1f} | pseudo-R² (McFadden): {res.prsquared:.4f}")

beta = pd.Series(res.params.iloc[1:].values, index=feature_names)
pvals = pd.Series(res.pvalues.iloc[1:].values, index=feature_names)
odds = np.exp(beta)

print("\n[β̂ y ODDS RATIOS] exp(β̂) = odds ratio (multiplica los odds de default); "
      "numéricas por +1 desvío estándar:")
tab = pd.DataFrame({"beta": beta.round(4), "OR": odds.round(4),
                    "p": pvals.apply(lambda v: f"{v:.2e}")})
tab = tab.reindex(beta.abs().sort_values(ascending=False).index)
print(tab.head(26).to_string())

print("\n[SENTIDO] Variables que AUMENTAN los odds de default (OR>1):")
for f in tab[tab["OR"] > 1].head(8).index:
    print(f"   {f:16s} OR={odds[f]:.3f} (+{(odds[f]-1)*100:.0f}% odds por unidad)")
print("[SENTIDO] Variables que DISMINUYEN los odds de default (OR<1):")
for f in tab[tab["OR"] < 1].tail(8).index[::-1]:
    print(f"   {f:16s} OR={odds[f]:.3f} (−{(1-odds[f])*100:.0f}% odds por unidad)")

print("\n[DUMMIES vs referencia] EDUCATION ref='grad', MARRIAGE ref='married':")
for f in dummy_cols:
    print(f"   {f:18s} β̂={beta[f]:+.3f} OR={odds[f]:.2f}")

p_test = res.predict(Xte_aug)
p_train = res.predict(Xtr_aug)
with open("logit_results.pkl", "wb") as f:
    pickle.dump(dict(p_train=np.asarray(p_train), p_test=np.asarray(p_test),
                     beta=beta.to_dict(), iterations=iters,
                     llf=res.llf, prsq=res.prsquared), f)
print("\n[SAVE] logit_results.pkl (probabilidades train/test)")