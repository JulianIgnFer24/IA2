"""
Entrenamiento del VAE sobre rostros de Los Simpson.

Adam (lr=1e-3), 200 épocas, batch 128, pérdida BCE + KL en float32 (sin AMP).
Reporta total / reconstrucción / KL promediados POR IMAGEN (nats/imagen) en train y
validación, guarda el historial en outputs/loss_history.csv y checkpoints en
outputs/checkpoints/.

Uso:
    python code/train.py
    python code/train.py --data_dir archive/cropped --epochs 2
    python code/train.py --resume outputs/checkpoints/vae_last.pt
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from tqdm import tqdm

import config
from data import build_dataloaders
from loss import vae_loss
from model import VAE, count_params
from visualize import plot_loss_curves, plot_reconstructions, plot_samples

CAMPOS_CSV = [
    "epoch",
    "train_total", "train_recon", "train_kl",
    "val_total", "val_recon", "val_kl",
    "seconds",
]


def parse_args():
    p = argparse.ArgumentParser(description="Entrena el VAE de rostros de Los Simpson.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="carpeta plana de imágenes (default: archive/cropped)")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint .pt desde el cual continuar")
    return p.parse_args()


def run_epoch(model, loader, optimizer=None, desc=""):
    """Una pasada completa. Con optimizer entrena; sin él, evalúa.

    Devuelve (total, recon, kl) promediados por imagen.
    """
    entrenando = optimizer is not None
    model.train(entrenando)

    sum_total = sum_recon = sum_kl = 0.0
    n_imgs = 0

    barra = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.set_grad_enabled(entrenando):
        for x in barra:
            x = x.to(config.DEVICE, non_blocking=True)
            x_recon, mu, logvar = model(x)
            total, recon, kl = vae_loss(x_recon, x, mu, logvar)

            if entrenando:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()

            b = x.size(0)
            sum_total += total.item()
            sum_recon += recon.item()
            sum_kl += kl.item()
            n_imgs += b
            barra.set_postfix(loss=f"{total.item() / b:.1f}")

    return sum_total / n_imgs, sum_recon / n_imgs, sum_kl / n_imgs


def save_checkpoint(ruta, model, optimizer, epoch, history, x_fixed, z_fixed, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {**config.as_dict(), "lr": args.lr, "batch_size": args.batch_size},
            "history": history,
            "x_fixed": x_fixed.cpu(),
            "z_fixed": z_fixed.cpu(),
        },
        ruta,
    )
    print(f"  checkpoint: {ruta}")


def write_history_csv(history, ruta=config.LOSS_CSV):
    """Reescribe el CSV completo cada época: una interrupción no pierde el log."""
    with Path(ruta).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()

    config.set_seeds()
    config.print_device()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else config.DATA_DIR
    train_loader, val_loader, val_ds = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = VAE(latent_dim=config.LATENT_DIM).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Parámetros entrenables: {count_params(model):,}")

    history = []
    epoca_inicial = 1

    # Tensores fijos para que las figuras de distintas épocas sean comparables.
    x_fixed = torch.stack([val_ds[i] for i in range(config.N_RECON)])
    g = torch.Generator().manual_seed(config.SEED)
    z_fixed = torch.randn(config.N_SAMPLES, config.LATENT_DIM, generator=g)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        history = ckpt.get("history", [])
        epoca_inicial = ckpt["epoch"] + 1
        # Se reusan los tensores fijos originales para no romper la comparación.
        if ckpt.get("x_fixed") is not None:
            x_fixed = ckpt["x_fixed"]
        if ckpt.get("z_fixed") is not None:
            z_fixed = ckpt["z_fixed"]
        print(f"Reanudando desde {args.resume}: próxima época {epoca_inicial}")

    if epoca_inicial > args.epochs:
        print(f"El checkpoint ya alcanzó la época {args.epochs}. Nada que hacer.")
        return

    print(f"\nEntrenando épocas {epoca_inicial}..{args.epochs}\n")
    t_inicio = time.time()

    for epoch in range(epoca_inicial, args.epochs + 1):
        t0 = time.time()
        tr_total, tr_recon, tr_kl = run_epoch(
            model, train_loader, optimizer, desc=f"época {epoch}/{args.epochs} [train]"
        )
        va_total, va_recon, va_kl = run_epoch(
            model, val_loader, None, desc=f"época {epoch}/{args.epochs} [val]"
        )
        dt = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_total": tr_total, "train_recon": tr_recon, "train_kl": tr_kl,
            "val_total": va_total, "val_recon": va_recon, "val_kl": va_kl,
            "seconds": dt,
        })
        write_history_csv(history)

        print(
            f"época {epoch:3d}/{args.epochs} | {dt:5.1f}s | "
            f"train: total {tr_total:8.2f}  recon {tr_recon:8.2f}  kl {tr_kl:7.2f} | "
            f"val: total {va_total:8.2f}  recon {va_recon:8.2f}  kl {va_kl:7.2f}"
        )

        if epoch in config.CHECKPOINT_EPOCHS or epoch == args.epochs:
            save_checkpoint(
                config.CHECKPOINT_DIR / f"vae_epoch_{epoch}.pt",
                model, optimizer, epoch, history, x_fixed, z_fixed, args,
            )
            plot_reconstructions(model, x_fixed, epoch)
            plot_samples(model, z_fixed, epoch)

        if epoch % 10 == 0 or epoch == args.epochs:
            save_checkpoint(
                config.CHECKPOINT_DIR / "vae_last.pt",
                model, optimizer, epoch, history, x_fixed, z_fixed, args,
            )

    minutos = (time.time() - t_inicio) / 60
    print(f"\nEntrenamiento terminado en {minutos:.1f} min.")
    plot_loss_curves(history)
    print(f"Historial: {config.LOSS_CSV}")


if __name__ == "__main__":
    main()
