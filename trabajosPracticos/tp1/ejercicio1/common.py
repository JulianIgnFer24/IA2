"""
Utilidades compartidas por los modelos del TP1 Ejercicio 1.
Carga del dataset procesado y métricas de evaluación en escala log y original ($).
"""
import pickle
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def load_data(path="processed_data.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def metrics(y_log_true, y_log_pred):
    """Métricas: en escala logarítmica y en dólares originales (RMSE/MAE)."""
    r2 = r2_score(y_log_true, y_log_pred)
    rmse_log = root_mean_squared_error(y_log_true, y_log_pred)
    mae_log = mean_absolute_error(y_log_true, y_log_pred)
    true_dol = np.exp(np.asarray(y_log_true))
    pred_dol = np.exp(np.asarray(y_log_pred))
    rmse_dol = root_mean_squared_error(true_dol, pred_dol)
    mae_dol = mean_absolute_error(true_dol, pred_dol)
    return dict(r2=r2, rmse_log=rmse_log, mae_log=mae_log,
                rmse_dol=rmse_dol, mae_dol=mae_dol)