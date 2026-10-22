"""
Entrena UN solo VAE con los dos tipos de imágenes juntas: cropped + simplified.

19.754 imágenes en total (las mismas 9.877 caras, cada una en sus dos versiones). Misma
arquitectura, mismos hiperparámetros y mismas 200 épocas que los experimentos anteriores;
lo único que cambia es el conjunto de datos.

No modifica nada de lo que ya existe: reescribe las rutas de `config` y sustituye
`data.build_dataloaders` ANTES de importar train.py, que captura esa función al
importarse. Las salidas van a outputs/combinado/.

Uso:
    python code/combinado/run_combinado.py                 # VAE, 200 épocas
    python code/combinado/run_combinado.py --epochs 2      # prueba rápida
    python code/combinado/run_combinado.py --con-ae        # además el AE baseline
"""
import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "baseline"))

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

import config
import data as data_mod
from data import SimpsonsFacesDataset

COMBINADO_OUT = config.PROJECT_DIR / "outputs" / "combinado"
CARPETAS = [config.PROJECT_DIR / "archive" / "cropped",
            config.PROJECT_DIR / "archive" / "simplified"]

config.OUTPUT_DIR = COMBINADO_OUT
config.CHECKPOINT_DIR = COMBINADO_OUT / "checkpoints"
config.FIGURE_DIR = COMBINADO_OUT / "figures"
config.LOSS_CSV = COMBINADO_OUT / "loss_history.csv"
for _d in (config.OUTPUT_DIR, config.CHECKPOINT_DIR, config.FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def build_dataloaders_combinado(data_dir=None, batch_size=config.BATCH_SIZE,
                                num_workers=config.NUM_WORKERS,
                                val_split=config.VAL_SPLIT, seed=config.SEED,
                                verbose=True):
    """Mismo contrato que data.build_dataloaders, pero sobre las dos carpetas.

    Se concatenan los dos datasets y recién ahí se hace el split, así que la partición
    mezcla los dos estilos en lugar de dejar uno entero de un lado.
    """
    partes = [SimpsonsFacesDataset(c) for c in CARPETAS]
    dataset = ConcatDataset(partes)

    n_total = len(dataset)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    pin = config.DEVICE.type == "cuda"
    persistent = num_workers > 0
    comun = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin,
                 persistent_workers=persistent)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **comun)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **comun)

    if verbose:
        for carpeta, parte in zip(CARPETAS, partes):
            print(f"  {carpeta.name}: {len(parte)} imágenes")
        print(f"Total combinado: {n_total}")
        print(f"  train: {n_train} ({len(train_loader)} batches de {batch_size})")
        print(f"  val:   {n_val} ({len(val_loader)} batches)")

    return train_loader, val_loader, val_ds


# Sustitución antes de importar train/train_ae, que hacen `from data import ...`.
data_mod.build_dataloaders = build_dataloaders_combinado

import train       # noqa: E402
import train_ae    # noqa: E402

train.build_dataloaders = build_dataloaders_combinado
train_ae.build_dataloaders = build_dataloaders_combinado


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--con-ae", action="store_true",
                   help="entrenar también el AE determinístico baseline")
    args = p.parse_args()

    for c in CARPETAS:
        assert c.is_dir(), f"No existe {c}"
    print(f"Datos:   {' + '.join(c.name for c in CARPETAS)}")
    print(f"Salidas: {COMBINADO_OUT}\n")

    sys.argv = ["x", "--epochs", str(args.epochs)]
    print("=" * 70, "\nVAE sobre cropped + simplified\n", "=" * 70, sep="")
    train.main()

    if args.con_ae:
        sys.argv = ["x", "--epochs", str(args.epochs)]
        print("\n" + "=" * 70, "\nAE baseline sobre cropped + simplified\n",
              "=" * 70, sep="")
        train_ae.main()

    print(f"\nListo. Todo en {COMBINADO_OUT}")


if __name__ == "__main__":
    main()
