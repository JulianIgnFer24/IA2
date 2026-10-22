"""
Dataset y DataLoaders de los rostros de Los Simpson.

Las imágenes están sueltas en una carpeta plana (sin subcarpetas por clase), por eso
no se usa ImageFolder sino un Dataset propio. Son PNG de 200x200 RGB que se reducen a
64x64 con interpolación LANCZOS y se normalizan a [0, 1].
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config

EXTENSIONES = (".png", ".jpg", ".jpeg")


class SimpsonsFacesDataset(Dataset):
    """Lee todas las imágenes de una carpeta plana y devuelve tensores (3, 64, 64)."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio de datos: {self.data_dir.resolve()}"
            )

        self.paths = sorted(
            p for p in self.data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES
        )
        if not self.paths:
            raise FileNotFoundError(
                f"No se encontraron imágenes {EXTENSIONES} en {self.data_dir.resolve()}"
            )

        self.transform = transform if transform is not None else build_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Sin etiqueta: el VAE es no supervisado, solo necesita la imagen.
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_transform(img_size: int = config.IMG_SIZE):
    """200x200 -> img_size x img_size con LANCZOS, y tensor float en [0, 1]."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])


def build_dataloaders(
    data_dir=config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.SEED,
    verbose: bool = True,
):
    """Arma el split 95/5 reproducible y sus DataLoaders.

    Devuelve (train_loader, val_loader, val_dataset).
    """
    dataset = SimpsonsFacesDataset(data_dir)

    n_total = len(dataset)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val

    # Generador propio con semilla fija: el mismo split en cada corrida y al reanudar.
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    pin = config.DEVICE.type == "cuda"
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=persistent,
    )

    if verbose:
        print(f"Imágenes encontradas en {Path(data_dir).resolve()}: {n_total}")
        print(f"  train: {n_train} imágenes ({len(train_loader)} batches de {batch_size})")
        print(f"  val:   {n_val} imágenes ({len(val_loader)} batches)")

    return train_loader, val_loader, val_ds


if __name__ == "__main__":
    train_loader, val_loader, val_ds = build_dataloaders(num_workers=0)
    x = next(iter(train_loader))
    print(f"\nBatch de train: {tuple(x.shape)}  dtype={x.dtype}")
    print(f"Rango de píxeles: [{x.min():.4f}, {x.max():.4f}]  media={x.mean():.4f}")
    assert x.shape[1:] == (3, config.IMG_SIZE, config.IMG_SIZE)
    assert 0.0 <= x.min() and x.max() <= 1.0, "ToTensor debe dejar los píxeles en [0, 1]"
    print("OK")
