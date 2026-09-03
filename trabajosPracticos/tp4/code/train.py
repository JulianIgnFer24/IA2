"""
Entrenamiento automatizado de los 4 experimentos del TP4.

Para cada configuración de models.EXPERIMENTS:
  - construye y compila el modelo (Adam lr=1e-3, sparse_categorical_crossentropy),
  - entrena 20 épocas con batch 64 y EarlyStopping(val_loss, patience=5),
  - evalúa en train / val / test,
  - guarda el modelo (.keras) y las curvas de aprendizaje.

Salidas: results/<key>.keras, results/curvas_<key>.png, results/experimentos.pkl
"""
import pickle
import sys
import time

import matplotlib.pyplot as plt

from common import (
    BATCH_SIZE,
    EPOCHS,
    RESULTS_DIR,
    compile_model,
    count_params,
    early_stopping,
    gpu_info,
    load_data,
    save_fig,
    set_seeds,
)
from models import EXPERIMENTS

RESULTS_PATH = RESULTS_DIR / "experimentos.pkl"


def plot_curvas(history: dict, exp: dict) -> None:
    """Pérdida y exactitud de entrenamiento vs validación por época."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["loss"], marker="o", ms=3, label="Entrenamiento")
    axes[0].plot(epochs, history["val_loss"], marker="o", ms=3, label="Validación")
    axes[0].set_title("Pérdida (sparse categorical crossentropy)")
    axes[0].set_ylabel("Pérdida")

    axes[1].plot(epochs, history["accuracy"], marker="o", ms=3, label="Entrenamiento")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", ms=3, label="Validación")
    axes[1].set_title("Exactitud (accuracy)")
    axes[1].set_ylabel("Accuracy")

    best = int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])) + 1
    for ax in axes:
        ax.axvline(best, color="gray", ls="--", lw=1, label=f"Mejor época ({best})")
        ax.set_xlabel("Época")
        ax.legend()

    fig.suptitle(f"Exp {exp['id']} — {exp['nombre']}", fontsize=13, fontweight="bold")
    save_fig(fig, f"curvas_{exp['key']}.png")


def run_experiment(exp: dict, data: tuple) -> dict:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    set_seeds()  # misma inicialización para todos los experimentos

    model = compile_model(exp["builder"]())
    trainable, non_trainable = count_params(model)
    print(f"\n{'=' * 70}\nExp {exp['id']} — {exp['nombre']}  ({trainable:,} parámetros entrenables)\n{'=' * 70}")
    model.summary()

    t0 = time.perf_counter()
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping()],
        shuffle=True,
        verbose=2,
    )
    elapsed = time.perf_counter() - t0

    # métricas con los pesos de la mejor época (restore_best_weights=True)
    train_loss, train_acc = model.evaluate(X_train, y_train, batch_size=512, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=512, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=512, verbose=0)

    model_path = RESULTS_DIR / f"{exp['key']}.keras"
    model.save(model_path)

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    plot_curvas(history, exp)

    print(
        f"Exp {exp['id']}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
        f"test_acc={test_acc:.4f}  ({len(history['loss'])} épocas, {elapsed:.1f}s)"
    )

    return {
        **{k: v for k, v in exp.items() if k != "builder"},
        "history": history,
        "epocas_corridas": len(history["loss"]),
        "mejor_epoca": int(min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])) + 1,
        "params_entrenables": trainable,
        "params_no_entrenables": non_trainable,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "tiempo_s": float(elapsed),
        "modelo_path": str(model_path),
    }


def main() -> None:
    """Corre todos los experimentos, o sólo los ids pasados por línea de comandos."""
    ids = {int(a) for a in sys.argv[1:]}
    seleccion = [e for e in EXPERIMENTS if not ids or e["id"] in ids]
    if not seleccion:
        raise SystemExit(f"Ids válidos: {[e['id'] for e in EXPERIMENTS]}")

    set_seeds()
    print(f"Dispositivo: {gpu_info()}")
    print(f"Experimentos a correr: {[e['id'] for e in seleccion]}")

    data = load_data()
    (X_train, _), (X_val, _), (X_test, _) = data
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # se conservan los resultados previos de los experimentos que no se vuelven a correr
    previos = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "rb") as f:
            previos = {r["key"]: r for r in pickle.load(f)["resultados"]}

    for exp in seleccion:
        previos[exp["key"]] = run_experiment(exp, data)

    resultados = [previos[e["key"]] for e in EXPERIMENTS if e["key"] in previos]
    mejor = max(resultados, key=lambda r: r["test_acc"])
    print(f"\nMejor modelo global: Exp {mejor['id']} — {mejor['nombre']} (test_acc={mejor['test_acc']:.4f})")

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({"resultados": resultados, "mejor_key": mejor["key"]}, f)
    print(f"Resultados -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
