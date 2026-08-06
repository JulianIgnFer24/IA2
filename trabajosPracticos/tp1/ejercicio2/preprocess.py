"""
(2) Preparación de datos — TP1 Ejercicio 2 (UCI Credit Card)
Recodificación de categóricas, dummies k-1, estandarización y split
estratificado 70/30 que mantiene la proporción de clases.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = "archive/UCI_Credit_Card.csv"
OUT = "processed_data.pkl"
RANDOM_STATE = 42

df = pd.read_csv(DATA)
df = df.drop(columns=["ID"])

y = df["default.payment.next.month"]  # (a) variable respuesta binaria
X = df.drop(columns=["default.payment.next.month"])

# (b) numéricas continuas
num_cols = ["LIMIT_BAL", "AGE"] + [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]] + \
           [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
# (c) binaria 0/1: SEX (1=hombre, 2=mujer) -> 0/1
X["SEX"] = (X["SEX"] == 2).astype(int)
# (d) categóricas con >2 niveles: EDUCATION y MARRIAGE
#     (los códigos raros 0/5/6 de EDUCATION y 0 de MARRIAGE se agrupan en "otros")
X["EDUCATION"] = X["EDUCATION"].map(lambda v: v if v in (1, 2, 3) else 4).map(
    {1: "grad", 2: "univ", 3: "hs", 4: "other"})
X["MARRIAGE"] = X["MARRIAGE"].map(lambda v: v if v in (1, 2) else 3).map(
    {1: "married", 2: "single", 3: "other"})

X = pd.get_dummies(X, columns=["EDUCATION", "MARRIAGE"], drop_first=True, dtype=float)
dummy_cols = [c for c in X.columns if c.startswith(("EDUCATION_", "MARRIAGE_"))]

# Split estratificado 70/30 (mantiene la proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

# Estandarización de numéricas (ajustada solo en train)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

with open(OUT, "wb") as f:
    pickle.dump(dict(X_train=X_train, X_test=X_test, y_train=y_train,
                     y_test=y_test, feature_names=list(X_train.columns),
                     num_cols=num_cols, dummy_cols=dummy_cols, scaler=scaler), f)

print("Predictores:", len(X_train.columns),
      f"({len(num_cols)} numéricas estandarizadas + 1 binaria SEX + {len(dummy_cols)} dummies)")
print(f"Tasa default train={y_train.mean():.3f} | test={y_test.mean():.3f} "
      f"(global={y.mean():.3f}) -> split estratificado")
print("Shapes:", X_train.shape, X_test.shape)
print("Dummies (k-1):", dummy_cols)