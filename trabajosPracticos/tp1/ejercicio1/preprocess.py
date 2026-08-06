"""
Preparación de datos — TP1 Ejercicio 1 (House Prices)
Codifica categóricas como dummies (k-1), estandariza numéricas y divide train/test.
Guarda el dataset procesado en processed_data.pkl para el resto de los modelos.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = "house-prices-advanced-regression-techniques/train.csv"
OUT = "processed_data.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(DATA)
df = df.drop(columns=["Id"])
y = np.log(df["SalePrice"])  # variable respuesta: log(SalePrice), según EDA
X = df.drop(columns=["SalePrice"])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# Imputación de faltantes: numéricas -> mediana; categóricas -> "None"
for c in num_cols:
    X[c] = X[c].fillna(X[c].median())
for c in cat_cols:
    X[c] = X[c].astype(str).fillna("NaN")

# Codificación dummy con k-1 columnas por categoría (drop_first=True).
#   Se suelta una columna por categoría para que actúe como referencia y se
#   evita la colinealidad perfecta con el intercepto (trampa de la dummy).
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
dummy_cols = [c for c in X.columns if c not in num_cols]
# Solo las numéricas continuas originales se estandarizan; las dummies (0/1) no.

# --- Split train/test (escalador y dummies ajustados SOLO con train) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

feature_names = list(X_train.columns)

with open(OUT, "wb") as f:
    pickle.dump(dict(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        feature_names=feature_names, num_cols=num_cols, dummy_cols=dummy_cols,
        scaler=scaler), f)

# --- Resumen para el informe ---
n_cat_orig = len(cat_cols)
n_num = len(num_cols)
n_dummy = len(dummy_cols)
n_features = len(feature_names)
info = pd.DataFrame([
    ("Filas (train)", X_train.shape[0]),
    ("Filas (test)", X_test.shape[0]),
    ("Categorías originales", n_cat_orig),
    ("Variables numéricas (estandarizadas)", n_num),
    ("Dummies generadas (k-1)", n_dummy),
    ("Total de predictores", n_features),
])
print(info.to_string(index=False))
print("\nEscalado: StandardScaler ajustado en train; aplicado a train y test.")
print(f"Valores de train/test: {X_train.shape} / {X_test.shape}")