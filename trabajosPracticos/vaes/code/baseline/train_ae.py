"""
Entrenamiento del autoencoder determinístico baseline.

Única diferencia de fondo con train.py: la pérdida es SOLO reconstrucción (BCE sumada),
sin término KL. Mismos datos, mismo split, mismo optimizador y mismo batch que el VAE,
para que la comparación sea limpia.

Uso:
    python code/baseline/train_ae.py --epochs 200
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from data import build_dataloaders
from model import count_params
from model_ae import AE

BASELINE_DIR = config.OUTPUT_DIR / "baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = BASELINE_DIR / "ae_final.pt"
LOSS_CSV = BASELINE_DIR / "ae_loss_history.csv"

CAMPOS_CSV = ["epoch", "train_recon", "val_recon", "seconds"]


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el autoencoder determinístico baseline.")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--lr", type=float, default=config.LR)
    return p.parse_args()


def run_epoch(model, loader, optimizer=None, desc=""):
    """Una pasada completa. Devuelve la BCE promediada por imagen."""
    entrenando = optimizer is not None
    model.train(entrenando)

    suma = 0.0
    n_imgs = 0

    barra = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.set_grad_enabled(entrenando):
        for x in barra:
            x = x.to(config.DEVICE, non_blocking=True)
            x_recon = model(x)
            # Sin KL: el AE solo optimiza la reconstrucción.
            recon = F.binary_cross_entropy(x_recon, x, reduction="sum")

            if entrenando:
                optimizer.zero_grad(set_to_none=True)
                recon.backward()
                optimizer.step()

            suma += recon.item()
            n_imgs += x.size(0)
            barra.set_postfix(recon=f"{recon.item() / x.size(0):.1f}")

    return suma / n_imgs


def main():
    args = parse_args()

    config.set_seeds()
    config.print_device()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.DATA_DIR
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = AE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Parámetros entrenables: {count_params(model):,}")
    print(f"\nEntrenando el AE baseline {args.epochs} épocas\n")

    history = []
    t_inicio = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer,
                       desc=f"época {epoch}/{args.epochs} [train]")
        va = run_epoch(model, val_loader, None,
                       desc=f"época {epoch}/{args.epochs} [val]")
        dt = time.time() - t0

        history.append({"epoch": epoch, "train_recon": tr, "val_recon": va, "seconds": dt})
        with LOSS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
            writer.writeheader()
            writer.writerows(history)

        print(f"época {epoch:3d}/{args.epochs} | {dt:5.1f}s | "
              f"train recon {tr:8.2f} | val recon {va:8.2f}")

    minutos = (time.time() - t_inicio) / 60
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {**config.as_dict(), "lr": args.lr, "batch_size": args.batch_size,
                       "epochs": args.epochs, "modelo": "AE determinístico (sin KL)"},
            "history": history,
            "minutos": minutos,
        },
        CHECKPOINT_PATH,
    )
    print(f"\nEntrenamiento terminado en {minutos:.1f} min.")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Historial:  {LOSS_CSV}")


if __name__ == "__main__":
    main()
