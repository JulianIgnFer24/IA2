"""
Repite los dos entrenamientos (VAE y AE baseline) sobre archive/simplified.

No modifica ni un archivo del experimento original: redirige las salidas reescribiendo
las rutas de `config` ANTES de importar train.py y train_ae.py. Eso importa porque esos
módulos capturan algunas rutas como valores por defecto al importarse, así que el parcheo
tiene que ocurrir primero.

Uso:
    python code/simplified/run_experimentos.py            # VAE y AE, 200 épocas
    python code/simplified/run_experimentos.py --epochs 2 # prueba rápida
    python code/simplified/run_experimentos.py --solo vae
"""
import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "baseline"))

import config

# --- Redirección de rutas: todo lo de simplified vive aparte -----------------
SIMPLIFIED_OUT = config.PROJECT_DIR / "outputs" / "simplified"

config.DATA_DIR = config.PROJECT_DIR / "archive" / "simplified"
config.OUTPUT_DIR = SIMPLIFIED_OUT
config.CHECKPOINT_DIR = SIMPLIFIED_OUT / "checkpoints"
config.FIGURE_DIR = SIMPLIFIED_OUT / "figures"
config.LOSS_CSV = SIMPLIFIED_OUT / "loss_history.csv"

for _d in (config.OUTPUT_DIR, config.CHECKPOINT_DIR, config.FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Recién ahora: estos módulos leen config al importarse.
import train            # noqa: E402
import train_ae         # noqa: E402


def correr(modulo, nombre, epochs):
    print(f"\n{'=' * 70}\n{nombre} sobre {config.DATA_DIR.name}\n{'=' * 70}")
    sys.argv = ["x", "--data_dir", str(config.DATA_DIR), "--epochs", str(epochs)]
    modulo.main()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--solo", choices=["vae", "ae"], default=None)
    args = p.parse_args()

    assert config.DATA_DIR.is_dir(), f"No existe {config.DATA_DIR}"
    print(f"Datos:    {config.DATA_DIR}")
    print(f"Salidas:  {SIMPLIFIED_OUT}")

    if args.solo in (None, "vae"):
        correr(train, "VAE", args.epochs)
    if args.solo in (None, "ae"):
        correr(train_ae, "AE baseline", args.epochs)

    print(f"\nListo. Todo en {SIMPLIFIED_OUT}")


if __name__ == "__main__":
    main()
