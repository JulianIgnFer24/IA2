"""
Arquitecturas del TP4: LeNet-5 adaptada y AlexNet adaptada.

Ambas reciben imágenes (28, 28, 1), usan ReLU como activación principal y
terminan en una capa Dense de 10 unidades con softmax.
"""
import tensorflow as tf
from tensorflow.keras import layers

from common import INPUT_SHAPE, N_CLASSES, augmentation_layers, random_erasing_layers


def _stem(augment: bool, seed_layers: list) -> list:
    """Capas iniciales: aumento de datos opcional (inactivo en inferencia)."""
    return [layers.Input(shape=INPUT_SHAPE)] + ([augmentation_layers()] if augment else []) + seed_layers


def build_lenet(
    kernel_size: int = 5,
    pooling: str = "avg",
    padding: str = "valid",
    augment: bool = False,
    name: str = "LeNet5",
) -> tf.keras.Model:
    """
    LeNet-5 adaptada a 28x28x1.

    Conv1(6, k) -> Pool -> Conv2(16, k) -> Pool -> Flatten -> Dense(120) ->
    Dense(84) -> Dense(10, softmax).

    Con kernels 5x5 y padding 'valid' el mapa cae a 4x4 tras el segundo
    pooling, por lo que la capa C5 original (120 filtros de 5x5) no entra:
    se usa la variante Dense(120) prevista en la consigna.
    """
    pool = layers.AveragePooling2D if pooling == "avg" else layers.MaxPooling2D
    return tf.keras.Sequential(
        _stem(
            augment,
            [
                layers.Conv2D(6, kernel_size, padding=padding, activation="relu", name="conv1"),
                pool(2, name="pool1"),
                layers.Conv2D(16, kernel_size, padding=padding, activation="relu", name="conv2"),
                pool(2, name="pool2"),
                layers.Flatten(name="flatten"),
                layers.Dense(120, activation="relu", name="fc1"),
                layers.Dense(84, activation="relu", name="fc2"),
                layers.Dense(N_CLASSES, activation="softmax", name="output"),
            ],
        ),
        name=name,
    )


def build_alexnet(
    filtros: tuple[int, ...] = (32, 64, 128, 512, 256),
    dropout: float = 0.0,
    augment: bool = False,
    name: str = "AlexNet",
) -> tf.keras.Model:
    """
    AlexNet adaptada a 28x28x1: 5 bloques convolucionales, 3 max-poolings y dos capas
    densas de 512, con Dropout opcional entre ellas.

    Las dimensiones espaciales bajan 28 -> 14 -> 7 -> 3, de modo que el mapa final
    (3x3 x filtros[-1]) nunca se anula. `filtros` controla el ancho de los cinco
    bloques: (32, 64, 128, 512, 256) ensancha la representación que llega al Flatten,
    (32, 64, 128, 128, 64) es la variante angosta original.
    """
    f1, f2, f3, f4, f5 = filtros
    cuerpo = [
        layers.Conv2D(f1, 3, padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D(2, strides=2, name="pool1"),
        layers.Conv2D(f2, 3, padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(2, strides=2, name="pool2"),
        layers.Conv2D(f3, 3, padding="same", activation="relu", name="conv3"),
        layers.Conv2D(f4, 3, padding="same", activation="relu", name="conv4"),
        layers.Conv2D(f5, 3, padding="same", activation="relu", name="conv5"),
        layers.MaxPooling2D(2, strides=2, name="pool3"),
        layers.Flatten(name="flatten"),
        layers.Dense(512, activation="relu", name="fc6"),
        *([layers.Dropout(dropout, name="drop6")] if dropout > 0 else []),
        layers.Dense(512, activation="relu", name="fc7"),
        *([layers.Dropout(dropout, name="drop7")] if dropout > 0 else []),
        layers.Dense(N_CLASSES, activation="softmax", name="output"),
    ]
    return tf.keras.Sequential(_stem(augment, cuerpo), name=name)


# ---------------------------------------------------------------------------
# Wide ResNet (WRN-28-10)
# ---------------------------------------------------------------------------
def _wrn_block(x, filtros: int, stride: int, dropout: float, nombre: str):
    """
    Bloque residual pre-activado: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Dropout -> Conv3x3,
    con conexión skip (identidad, o proyección 1x1 cuando cambian las dimensiones).
    """
    pre = layers.Activation("relu", name=f"{nombre}_relu1")(
        layers.BatchNormalization(name=f"{nombre}_bn1")(x)
    )
    cambia_dim = stride != 1 or x.shape[-1] != filtros
    atajo = (
        layers.Conv2D(filtros, 1, strides=stride, padding="same", use_bias=False, name=f"{nombre}_proj")(pre)
        if cambia_dim
        else x
    )

    o = layers.Conv2D(filtros, 3, strides=stride, padding="same", use_bias=False, name=f"{nombre}_conv1")(pre)
    o = layers.Activation("relu", name=f"{nombre}_relu2")(
        layers.BatchNormalization(name=f"{nombre}_bn2")(o)
    )
    if dropout > 0:
        o = layers.Dropout(dropout, name=f"{nombre}_drop")(o)
    o = layers.Conv2D(filtros, 3, strides=1, padding="same", use_bias=False, name=f"{nombre}_conv2")(o)
    return layers.Add(name=f"{nombre}_add")([o, atajo])


def build_wrn(
    depth: int = 28,
    k: int = 10,
    dropout: float = 0.3,
    random_erasing: bool = False,
    name: str = "WRN_28_10",
) -> tf.keras.Model:
    """
    Wide ResNet WRN-`depth`-`k` adaptada a 28x28x1.

    depth = 28 -> n = (28 - 4) / 6 = 4 bloques residuales por grupo; k = 10 multiplica
    por 10 los canales del ResNet base (16 -> 160, 32 -> 320, 64 -> 640):

        Conv inicial (16) -> Grupo 1 (160, stride 1, 28x28)
                          -> Grupo 2 (320, stride 2, 14x14)
                          -> Grupo 3 (640, stride 2, 7x7)
                          -> BN + ReLU -> GlobalAveragePooling -> Dense(10, softmax)

    El Dropout(0.3) dentro del bloque residual es el del paper original de Wide ResNets.
    """
    n = (depth - 4) // 6
    anchos = [16 * k, 32 * k, 64 * k]

    entrada = layers.Input(shape=INPUT_SHAPE, name="input")
    x = random_erasing_layers()(entrada) if random_erasing else entrada
    x = layers.Conv2D(16, 3, padding="same", use_bias=False, name="conv0")(x)

    for grupo, (filtros, stride) in enumerate(zip(anchos, [1, 2, 2]), start=1):
        for bloque in range(n):
            x = _wrn_block(
                x,
                filtros,
                stride if bloque == 0 else 1,
                dropout,
                nombre=f"g{grupo}b{bloque + 1}",
            )

    x = layers.Activation("relu", name="relu_final")(layers.BatchNormalization(name="bn_final")(x))
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    salida = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)
    return tf.keras.Model(entrada, salida, name=name)


# ---------------------------------------------------------------------------
# Matriz de experimentos obligatoria
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "id": 1,
        "key": "exp1_lenet_clasica",
        "nombre": "LeNet-5 Clásica",
        "arquitectura": "LeNet-5",
        "pooling": "Average",
        "kernel": "5x5",
        "padding": "valid",
        "dropout": "No",
        "data_aug": "No",
        "builder": lambda: build_lenet(5, "avg", "valid", False, "LeNet5_clasica"),
    },
    {
        "id": 2,
        "key": "exp2_lenet_optimizada",
        "nombre": "LeNet-5 Optimizada",
        "arquitectura": "LeNet-5",
        "pooling": "Max",
        "kernel": "3x3",
        "padding": "same",
        "dropout": "No",
        "data_aug": "No",
        "builder": lambda: build_lenet(3, "max", "same", False, "LeNet5_optimizada"),
    },
    {
        "id": 3,
        "key": "exp3_alexnet",
        "nombre": "AlexNet Adaptada",
        "arquitectura": "AlexNet",
        "pooling": "Max",
        "kernel": "3x3",
        "padding": "same",
        "dropout": "No",
        "data_aug": "No",
        "builder": lambda: build_alexnet(augment=False, name="AlexNet_adaptada"),
    },
    {
        "id": 4,
        "key": "exp4_alexnet_dataaug",
        "nombre": "AlexNet + DataAug",
        "arquitectura": "AlexNet",
        "pooling": "Max",
        "kernel": "3x3",
        "padding": "same",
        "dropout": "No",
        "data_aug": "Sí (Rot ±0.05)",
        "builder": lambda: build_alexnet(augment=True, name="AlexNet_dataaug"),
    },
    {
        "id": 5,
        "key": "exp5_wrn2810",
        "nombre": "WRN-28-10",
        "arquitectura": "WRN-28-10",
        "pooling": "GlobalAvg",
        "kernel": "3x3",
        "padding": "same",
        "dropout": "Sí (0.3 en bloque)",
        "data_aug": "No",
        "builder": lambda: build_wrn(28, 10, 0.3, False, "WRN_28_10"),
    },
    {
        "id": 6,
        "key": "exp6_wrn2810_erasing",
        "nombre": "WRN-28-10 + RandomErasing",
        "arquitectura": "WRN-28-10",
        "pooling": "GlobalAvg",
        "kernel": "3x3",
        "padding": "same",
        "dropout": "Sí (0.3 en bloque)",
        "data_aug": "Sí (Random Erasing)",
        "builder": lambda: build_wrn(28, 10, 0.3, True, "WRN_28_10_erasing"),
    },
    {
        "id": 7,
        "key": "exp7_alexnet_angosta_dropout",
        "nombre": "AlexNet angosta + Dropout",
        "arquitectura": "AlexNet",
        "pooling": "Max",
        "kernel": "3x3",
        "padding": "same",
        "dropout": "Sí (0.5)",
        "data_aug": "No",
        "builder": lambda: build_alexnet((32, 64, 128, 128, 64), 0.5, False, "AlexNet_angosta_dropout"),
    },
    {
        "id": 8,
        "key": "exp8_alexnet_angosta_dropout_dataaug",
        "nombre": "AlexNet angosta + Dropout + DataAug",
        "arquitectura": "AlexNet",
        "pooling": "Max",
        "kernel": "3x3",
        "padding": "same",
        "dropout": "Sí (0.5)",
        "data_aug": "Sí (Rot ±0.05)",
        "builder": lambda: build_alexnet((32, 64, 128, 128, 64), 0.5, True, "AlexNet_angosta_dropout_dataaug"),
    },
]
