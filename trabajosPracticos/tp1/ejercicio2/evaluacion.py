"""
(5)-(6) Evaluación del modelo logístico — TP1 Ejercicio 2
Umbral 0.5, matriz de confusión, métricas, ROC, PR, log-loss.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc,
                             precision_recall_curve, log_loss)

# Cargar datos y probabilidades
d = pickle.load(open("processed_data.pkl", "rb"))
X_test, y_test = d["X_test"], d["y_test"]
logit_d = pickle.load(open("logit_results.pkl", "rb"))
p_test = logit_d["p_test"]

print("=" * 76)
print("EVALUACIÓN DEL MODELO LOGÍSTICO (test set)")
print("=" * 76)

# --- 5. Umbral 0.5 ---
th = 0.5
y_pred = (p_test >= th).astype(int)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n[UMBRAL 0.5] Matriz de confusión:")
print(f"             Pred 0   Pred 1")
print(f"Real 0 (no)    {tn:6d}    {fp:6d}")
print(f"Real 1 (sí)    {fn:6d}    {tp:6d}")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ll = log_loss(y_test, p_test)

print(f"\n[MÉTRICAS @ th=0.5]")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Log-loss : {ll:.4f}")

# --- 6. ROC y AUC ---
fpr, tpr, thresholds_roc = roc_curve(y_test, p_test)
roc_auc = auc(fpr, tpr)

# --- Precision-Recall y AUC ---
precision_c, recall_c, thresholds_pr = precision_recall_curve(y_test, p_test)
pr_auc = auc(recall_c, precision_c)

print(f"\n[ROC] AUC = {roc_auc:.4f}")
print(f"[PR]  AUC = {pr_auc:.4f}")

# --- Gráficos ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Confusión
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["Real 0", "Real 1"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
ax.set_title(f"Matriz de confusión (th=0.5)")
plt.colorbar(im, ax=ax)

# ROC
ax = axes[1]
ax.plot(fpr, tpr, label=f"Logístico (AUC = {roc_auc:.3f})", color="#C44E52", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("Tasa falsos positivos (FPR)"); ax.set_ylabel("Tasa verdaderos positivos (TPR)")
ax.set_title("Curva ROC"); ax.legend(loc="lower right")

# Precision-Recall
ax = axes[2]
ax.plot(recall_c, precision_c, label=f"Logístico (AUC = {pr_auc:.3f})", color="#55A868", lw=2)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Curva Precision-Recall"); ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig("figs/ej2_evaluacion.png", dpi=130)
plt.close()

print("\n[FIG] figs/ej2_evaluacion.png")
print("""
[UMBRALES] Si se BAJA el umbral (p.ej. 0.3):
  - Se predice 'default' a más clientes → RECALL sube (atrapar más morosos)
  - PRECISION baja (más falsas alarmas)
  - Útil si el costo de NO detectar un moroso (falso negativo) es muy alto.
  Si se SUBE el umbral (p.ej. 0.7):
  - Se predice 'default' solo a casos muy seguros → PRECISION sube
  - RECALL baja (se dejan escapar morosos)
  - Útil si el costo de una falsa alarma (rechazar buen cliente) es alto.
""")

# --- Tabla de métricas por umbral ---
print("\n[MÉTRICAS POR UMBRAL] th={0.3, 0.5, 0.7}:")
for t in [0.3, 0.5, 0.7]:
    yp = (p_test >= t).astype(int)
    print(f"  th={t}: Acc={accuracy_score(y_test, yp):.3f} "
          f"Prec={precision_score(y_test, yp):.3f} "
          f"Rec={recall_score(y_test, yp):.3f} "
          f"F1={f1_score(y_test, yp):.3f}")