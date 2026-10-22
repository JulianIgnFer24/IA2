"""
Genera el mismo juego de figuras y resultados.json, pero para el experimento simplified.

Mismo truco que run_experimentos.py: se reescriben las rutas de `config` antes de
importar make_figures, que las captura al importarse.
"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "baseline"))
sys.path.insert(0, str(CODE_DIR / "presentacion"))

import config

SIMPLIFIED_OUT = config.PROJECT_DIR / "outputs" / "simplified"
config.DATA_DIR = config.PROJECT_DIR / "archive" / "simplified"
config.OUTPUT_DIR = SIMPLIFIED_OUT
config.CHECKPOINT_DIR = SIMPLIFIED_OUT / "checkpoints"
config.FIGURE_DIR = SIMPLIFIED_OUT / "figures"
config.LOSS_CSV = SIMPLIFIED_OUT / "loss_history.csv"

import make_figures  # noqa: E402

if __name__ == "__main__":
    print(f"Figuras de simplified -> {make_figures.FIGURAS_DIR}")
    make_figures.main()
