"""
Configuración del VAE sobre rostros de Los Simpson.

Rutas, hiperparámetros, dispositivo y semillas. Las rutas se derivan de la ubicación
de este archivo, así los scripts corren desde cualquier directorio de trabajo.
"""
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

DATA_DIR = PROJECT_DIR / "archive" / "cropped"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOSS_CSV = OUTPUT_DIR / "loss_history.csv"

for _d in (OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------
IMG_SIZE = 64
LATENT_DIM = 128
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 200
CHECKPOINT_EPOCHS = [30, 200]
NUM_WORKERS = 4
VAL_SPLIT = 0.05
SEED = 42

# Tamaños fijos de las visualizaciones (deben coincidir entre épocas para comparar)
N_RECON = 8     # imágenes de validación reconstruidas
N_SAMPLES = 64  # grilla 8x8 de caras generadas desde z ~ N(0, I)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int = SEED) -> None:
    """Fija las semillas de random, NumPy y PyTorch (CPU y GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_device() -> None:
    """Imprime el dispositivo en uso y, si hay GPU, su nombre y compute capability."""
    print(f"Dispositivo: {DEVICE}")
    if DEVICE.type == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"(compute capability {major}.{minor}, {total_gb:.1f} GB)")
        print(f"PyTorch {torch.__version__} / CUDA {torch.version.cuda}")


def as_dict() -> dict:
    """Hiperparámetros serializables, para guardar dentro del checkpoint."""
    return {
        "img_size": IMG_SIZE,
        "latent_dim": LATENT_DIM,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "num_workers": NUM_WORKERS,
        "val_split": VAL_SPLIT,
        "seed": SEED,
        "data_dir": str(DATA_DIR),
    }
