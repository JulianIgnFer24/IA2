"""
Utilidades compartidas del TP4 — CNNs sobre Fashion-MNIST.

Semillas, carga y partición del dataset (90/10 estratificado), normalización,
aumento de datos, callbacks de entrenamiento y helpers de graficado.
"""
import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# ptxas del paquete nvidia-cuda-nvcc (12.9) antes que el del sistema (12.0),
# que no soporta la compute capability 12.0 de la GPU. Debe hacerse antes de
# importar TensorFlow.
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
TP4_DIR = CODE_DIR.parent
_PTXAS_DIR = TP4_DIR / ".venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin"
if _PTXAS_DIR.is_dir():
    os.environ["PATH"] = f"{_PTXAS_DIR}:{os.environ.get('PATH', '')}"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
import numpy as np
import seaborn as sns
import tensorflow as tf

RESULTS_DIR = TP4_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
VAL_FRACTION = 0.10
INPUT_SHAPE = (28, 28, 1)
N_CLASSES = 10
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Hiperparámetros de entrenamiento (comunes a los 4 experimentos)
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 5
ROTATION_FACTOR = 0.05  # aumento de datos "ligero": ±0.05 * 2π ≈ ±18°
ERASING_PROB = 0.5      # Random Erasing: probabilidad de ocluir una imagen
ERASING_AREA = (0.02, 0.4)    # fracción del área de la imagen que se borra
ERASING_ASPECT = (0.3, 3.3)   # aspect ratio del rectángulo borrado

sns.set_theme(style="whitegrid", context="notebook")


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def set_seeds(seed: int = SEED) -> None:
    """Fija todas las semillas para que los experimentos sean reproducibles."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def gpu_info() -> str:
    """Descripción de los dispositivos GPU visibles (o CPU si no hay)."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "sin GPU (ejecución en CPU)"
    detalles = []
    for g in gpus:
        d = tf.config.experimental.get_device_details(g)
        cc = d.get("compute_capability")
        detalles.append(f"{d.get('device_name', g.name)} (CC {cc[0]}.{cc[1]})" if cc else g.name)
    return ", ".join(detalles)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def stratified_split(y: np.ndarray, fraction: float, seed: int = SEED):
    """Índices (train, val) con la misma proporción de clases en ambos lados."""
    rng = np.random.default_rng(seed)
    idx_train, idx_val = [], []
    for c in np.unique(y):
        idx_c = np.flatnonzero(y == c)
        rng.shuffle(idx_c)
        n_val = int(round(fraction * len(idx_c)))
        idx_val.append(idx_c[:n_val])
        idx_train.append(idx_c[n_val:])
    idx_train = np.concatenate(idx_train)
    idx_val = np.concatenate(idx_val)
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    return idx_train, idx_val


def load_data(val_fraction: float = VAL_FRACTION, seed: int = SEED):
    """
    Carga Fashion-MNIST, separa validación de forma balanceada y normaliza.

    Devuelve (X_train, y_train), (X_val, y_val), (X_test, y_test) con las
    imágenes en float32 [0, 1] y forma (N, 28, 28, 1).
    """
    (X_full, y_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    idx_train, idx_val = stratified_split(y_full, val_fraction, seed)
    X_train, y_train = X_full[idx_train], y_full[idx_train]
    X_val, y_val = X_full[idx_val], y_full[idx_val]

    def prep(X):
        # normalización uniforme a [0, 1] + canal de color explícito
        return (X.astype("float32") / 255.0)[..., np.newaxis]

    return (
        (prep(X_train), y_train.astype("int32")),
        (prep(X_val), y_val.astype("int32")),
        (prep(X_test), y_test.astype("int32")),
    )


def augmentation_layers(factor: float = ROTATION_FACTOR, seed: int = SEED):
    """Bloque de aumento de datos ligero (rotaciones leves), activo solo en train."""
    return tf.keras.Sequential(
        [tf.keras.layers.RandomRotation(factor, fill_mode="constant", fill_value=0.0, seed=seed)],
        name="data_augmentation",
    )


@keras.saving.register_keras_serializable(package="tp4")
class RandomErasing(keras.layers.Layer):
    """
    Random Erasing (Zhong et al., 2017), aplicado solo durante el entrenamiento.

    Para cada imagen del batch, con probabilidad `probability`:
      1. se sortea el área del rectángulo dentro de `area_range` (fracción del área total),
      2. se sortea su aspect ratio dentro de `aspect_range`,
      3. se sortea la posición dentro de la imagen,
      4. se rellena esa región con ruido uniforme en [0, 1] (el rango de los píxeles ya
         normalizados).

    Simula oclusiones parciales, de modo que la red no pueda apoyarse en una única región
    de la imagen para clasificar.
    """

    def __init__(
        self,
        probability: float = ERASING_PROB,
        area_range: tuple[float, float] = ERASING_AREA,
        aspect_range: tuple[float, float] = ERASING_ASPECT,
        seed: int = SEED,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.probability = probability
        self.area_range = tuple(area_range)
        self.aspect_range = tuple(aspect_range)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        if not training:
            return inputs

        shape = tf.shape(inputs)
        n, alto, ancho = shape[0], shape[1], shape[2]
        alto_f, ancho_f = tf.cast(alto, "float32"), tf.cast(ancho, "float32")

        def sortear(minimo, maximo, forma):
            return keras.random.uniform(forma, minimo, maximo, seed=self.seed_generator)

        # 1-2. área y aspect ratio del rectángulo
        area = sortear(*self.area_range, (n, 1)) * alto_f * ancho_f
        ratio = sortear(*self.aspect_range, (n, 1))
        alto_r = tf.clip_by_value(tf.round(tf.sqrt(area * ratio)), 1.0, alto_f)
        ancho_r = tf.clip_by_value(tf.round(tf.sqrt(area / ratio)), 1.0, ancho_f)

        # 3. posición (esquina superior izquierda)
        y0 = tf.floor(sortear(0.0, 1.0, (n, 1)) * (alto_f - alto_r + 1.0))
        x0 = tf.floor(sortear(0.0, 1.0, (n, 1)) * (ancho_f - ancho_r + 1.0))

        filas = tf.reshape(tf.range(alto_f), (1, -1))
        columnas = tf.reshape(tf.range(ancho_f), (1, -1))
        en_filas = (filas >= y0) & (filas < y0 + alto_r)
        en_columnas = (columnas >= x0) & (columnas < x0 + ancho_r)
        mascara = en_filas[:, :, None] & en_columnas[:, None, :]

        # aplicar solo a una fracción `probability` de las imágenes del batch
        aplicar = sortear(0.0, 1.0, (n, 1, 1)) < self.probability
        mascara = (mascara & aplicar)[..., None]

        # 4. relleno con ruido uniforme
        ruido = sortear(0.0, 1.0, shape)
        return tf.where(mascara, ruido, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            **super().get_config(),
            "probability": self.probability,
            "area_range": self.area_range,
            "aspect_range": self.aspect_range,
            "seed": self.seed,
        }


def random_erasing_layers(seed: int = SEED):
    """Bloque de aumento de datos por oclusión (Random Erasing), activo solo en train."""
    return tf.keras.Sequential([RandomErasing(seed=seed)], name="random_erasing")


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def compile_model(model: tf.keras.Model, lr: float = LEARNING_RATE) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def early_stopping(patience: int = PATIENCE) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )


def count_params(model: tf.keras.Model) -> tuple[int, int]:
    """(parámetros entrenables, no entrenables)."""
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return trainable, non_trainable


# ---------------------------------------------------------------------------
# Graficado
# ---------------------------------------------------------------------------
def save_fig(fig, name: str) -> Path:
    path = RESULTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {path.relative_to(TP4_DIR)}")
    return path
