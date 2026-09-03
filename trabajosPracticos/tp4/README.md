# TP4 — Redes Neuronales Convolucionales sobre Fashion-MNIST

Comparación experimental entre **LeNet-5 adaptada** (liviana), **AlexNet adaptada**
(profunda y ancha) y **WRN-28-10** (Wide ResNet residual), midiendo el efecto del tipo de
pooling, el tamaño de kernel, el padding, la cantidad de mapas de características, las
conexiones residuales y dos esquemas de aumento de datos (rotación leve y Random Erasing)
sobre la capacidad de generalización.

- Dataset: `tf.keras.datasets.fashion_mnist` — 60.000 imágenes de entrenamiento y 10.000 de test, 28×28 en escala de grises, 10 clases.
- Partición: **54.000 train / 6.000 validación** (10 % estratificado: exactamente 600 imágenes por clase) + 10.000 test.
- Normalización: uniforme, `X / 255.0` → rango [0, 1]; se agrega el canal explícito → `(N, 28, 28, 1)`.
- Entrenamiento: Adam (lr = 0.001), `sparse_categorical_crossentropy`, 20 épocas, batch 64, `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`.
- Reproducibilidad: `set_seeds(42)` fija `random`, `PYTHONHASHSEED`, NumPy y TensorFlow, y activa `tf.config.experimental.enable_op_determinism()`. Verificado: al reentrenar, los Exp 1 y 2 reprodujeron sus métricas dígito a dígito.
- Hardware: NVIDIA GeForce RTX 5060 Ti (CUDA, compute capability 12.0). Exp 1–4: ~3,5 min en total; Exp 5–6 (WRN-28-10): ~64 min.

## Resumen de resultados

| Exp ID | Arquitectura | Pooling | Tamaño Kernel | Dropout | Data Aug. | Train Acc | Val Acc | Test Acc | Param. Totales |
|:---:|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| 1 | LeNet-5 Clásica | Average | 5x5 | No | No | 0.9187 | 0.8952 | 0.8926 | 44.426 |
| 2 | LeNet-5 Optimizada | Max | 3x3 | No | No | 0.9234 | 0.8977 | 0.8943 | 106.154 |
| 3 | AlexNet Adaptada | Max | 3x3 | No | No | 0.9255 | 0.9025 | 0.9049 | 3.310.858 |
| 4 | AlexNet + DataAug | Max | 3x3 | No | Sí (Rot ±0.05) | 0.9221 | 0.9030 | 0.9021 | 3.310.858 |
| 5 | WRN-28-10 | GlobalAvg | 3x3 | Sí (0.3 en bloque) | No | 0.9345 | 0.9157 | 0.9169 | 36.478.906 |
| 6 | **WRN-28-10 + RandomErasing** ⭐ | GlobalAvg | 3x3 | Sí (0.3 en bloque) | Sí (Random Erasing) | 0.9623 | 0.9267 | **0.9305** | 36.478.906 |
| 7 | AlexNet angosta + Dropout | Max | 3x3 | Sí (0.5) | No | 0.9388 | 0.9110 | 0.9124 | 877.258 |
| 8 | AlexNet angosta + Dropout + DataAug | Max | 3x3 | Sí (0.5) | Sí (Rot ±0.05) | 0.9314 | 0.9098 | 0.9099 | 877.258 |

⭐ = mejor modelo global. Las métricas se calculan con los pesos de la mejor época
(`restore_best_weights=True`), en modo inferencia. Los Exp 1–6 son la matriz pedida por la
consigna; los Exp 7–8 son la variante angosta de AlexNet (conv4=128, conv5=64) con
Dropout(0.5), que se probó primero y se conserva para poder comparar contra la versión
ancha sin Dropout.

Detalle de la corrida:

| Exp | Épocas corridas | Mejor época (val_loss) | Val loss mín. | Test loss | Gap train−val (acc) | Tiempo |
|:---:|:---:|:---:|---:|---:|---:|---:|
| 1 | 20 / 20 | 17 | 0.2936 | 0.3171 | +0.0235 | 55,7 s |
| 2 | 12 (early stop) | 7 | 0.2660 | 0.2789 | +0.0257 | 30,1 s |
| 3 | 9 (early stop) | 4 | 0.2738 | 0.2804 | +0.0230 | 46,0 s |
| 4 | 12 (early stop) | 7 | 0.2680 | 0.2779 | +0.0191 | 65,7 s |
| 5 | 10 (early stop) | 5 | 0.2355 | 0.2398 | +0.0188 | 1326,1 s |
| 6 | 19 (early stop) | 14 | **0.2123** | **0.2200** | +0.0356 | 2518,7 s |
| 7 | 11 (early stop) | 6 | 0.2499 | 0.2512 | +0.0278 | 48,1 s |
| 8 | 20 / 20 | 17 | 0.2498 | 0.2704 | +0.0216 | 77,6 s |

## Arquitecturas

### LeNet-5 adaptada (Exp 1 y 2)

| Capa | Exp 1 (clásica) | Exp 2 (optimizada) |
|---|---|---|
| Conv1 | 6 filtros 5×5, `valid` → 24×24×6 | 6 filtros 3×3, `same` → 28×28×6 |
| Pool1 | AveragePooling 2×2 → 12×12×6 | MaxPooling 2×2 → 14×14×6 |
| Conv2 | 16 filtros 5×5, `valid` → 8×8×16 | 16 filtros 3×3, `same` → 14×14×16 |
| Pool2 | AveragePooling 2×2 → 4×4×16 | MaxPooling 2×2 → 7×7×16 |
| Flatten | 256 | 784 |
| FC1 / FC2 / salida | Dense 120 → Dense 84 → Dense 10 (softmax) | ídem |
| **Parámetros** | **44.426** | **106.154** |

Con kernels de 5×5 y `padding='valid'` el mapa queda en 4×4 después del segundo pooling,
así que la capa C5 original de LeNet (120 filtros de 5×5, pensada para entradas de 32×32)
no entra: se usa la variante `Dense(120)` prevista en la consigna. La diferencia de
parámetros entre ambas variantes viene casi toda del `Flatten`: con `padding='same'` el
mapa se conserva más grande (784 contra 256) y eso triplica el costo de FC1.

### AlexNet adaptada (Exp 3 y 4)

Cinco bloques convolucionales, tres max-poolings y dos densas de 512, **sin Dropout**.
Los dos últimos bloques ensanchan fuerte la representación (512 y 256 mapas), de modo que
al `Flatten` llegan 3×3×256 = 2.304 activaciones.

| Capa | Salida | Parámetros |
|---|---|---:|
| *data_augmentation: RandomRotation(0.05)* — **solo Exp 4** | 28×28×1 | 0 |
| Conv1: 32 filtros 3×3 `same` + ReLU | 28×28×32 | 320 |
| MaxPool1 2×2 (stride 2) | 14×14×32 | 0 |
| Conv2: 64 filtros 3×3 `same` + ReLU | 14×14×64 | 18.496 |
| MaxPool2 2×2 (stride 2) | 7×7×64 | 0 |
| Conv3: 128 filtros 3×3 `same` + ReLU | 7×7×128 | 73.856 |
| Conv4: **512** filtros 3×3 `same` + ReLU | 7×7×512 | 590.336 |
| Conv5: **256** filtros 3×3 `same` + ReLU | 7×7×256 | 1.179.904 |
| MaxPool3 2×2 (stride 2) | 3×3×256 | 0 |
| Flatten | 2304 | 0 |
| FC6: Dense 512 + ReLU | 512 | 1.180.160 |
| FC7: Dense 512 + ReLU | 512 | 262.656 |
| Salida: Dense 10 + softmax | 10 | 5.130 |
| **Total** | | **3.310.858** |

### WRN-28-10 (Exp 5 y 6)

Wide ResNet: en vez de apilar profundidad, ensancha los canales por un factor k. Con
`depth=28` → n = (28 − 4) / 6 = **4 bloques residuales por grupo**, y con k = 10 los
canales del ResNet base se multiplican por 10 (16 → 160, 32 → 320, 64 → 640).

| Etapa | Salida | Detalle |
|---|---|---|
| *random_erasing* — **solo Exp 6** | 28×28×1 | p=0.5, área 2–40 %, ruido uniforme |
| Conv inicial | 28×28×16 | Conv 3×3, sin bias |
| Grupo 1 | 28×28×**160** | 4 bloques residuales, stride 1 |
| Grupo 2 | 14×14×**320** | 4 bloques residuales, stride 2 en el primero |
| Grupo 3 | 7×7×**640** | 4 bloques residuales, stride 2 en el primero |
| BN + ReLU | 7×7×640 | pre-activación final |
| GlobalAveragePooling | 640 | reemplaza al `Flatten` — no aporta parámetros |
| Dense + softmax | 10 | 6.410 |
| **Total** | | **36.478.906** entrenables (+17.952 no entrenables de BN) |

Cada bloque residual sigue el patrón pre-activado
**BN → ReLU → Conv3×3 → BN → ReLU → Dropout(0.3) → Conv3×3**, con conexión skip identidad,
o proyección 1×1 cuando cambian las dimensiones (primer bloque de cada grupo). Son 28 capas
convolucionales en total. El Dropout(0.3) dentro del bloque es el del paper original de
Wide ResNets; el optimizador se mantiene en Adam(1e-3) como en el resto del TP, en lugar
del SGD con momentum del paper, para que la comparación entre arquitecturas sea limpia.

**Random Erasing** (Zhong et al., 2017) está implementado como capa propia en
`common.py` — activa solo en entrenamiento. Para cada imagen, con probabilidad 0.5: sortea
el área del rectángulo (2–40 % de la imagen), su aspect ratio (0.3–3.3) y su posición, y
rellena esa región con ruido uniforme en [0, 1]. Se implementó a mano en vez de usar
`keras.layers.RandomErasing` porque la capa incorporada interpreta `factor` como "una
probabilidad muestreada entre 0 y factor" y no expone el rango de aspect ratio. Verificado
sobre 2.000 imágenes: 50,2 % ocluidas, área efectiva entre 1,8 % y 41,5 %, e identidad
exacta en inferencia.

## Análisis

**La profundidad convolucional compra exactitud; el ancho de las capas densas, no.**
Del Exp 1 al Exp 2 los parámetros se multiplican por 2,4 sin agregar una sola capa (todo
va al `Flatten`/FC1) y el test accuracy sube apenas 0,17 puntos. Del Exp 2 al Exp 3, tres
bloques convolucionales adicionales rinden 1,1 puntos. Y del Exp 3 al Exp 5, pasar a una
arquitectura residual sube otros 1,2 puntos.

**Sin regularización, la capacidad extra se gasta en memorizar.** La AlexNet ancha (3,31 M
parámetros, sin Dropout) es el experimento que antes sobreajusta: alcanza su mínimo de
validación en la **época 4** y después la pérdida de validación sube (0.2738 → 0.3035)
mientras la de entrenamiento cae a 0.109. Tiene 3,8 veces más parámetros que la variante angosta
con Dropout(0.5) (Exp 7: 877 k, test 0.9124) y rinde *menos* (0.9049).

**El WRN-28-10 mejora pese a tener 11 veces más parámetros que la AlexNet.** Con 36,5 M
parámetros contra 3,3 M, la intuición diría sobreajuste seguro sobre 54.000 imágenes de
28×28. Pasa lo contrario: el Exp 5 tiene el gap train−val más chico de todos salvo el Exp 4
(+0.0188), la mejor pérdida de validación hasta ese punto (0.2355) y 1,2 puntos más de test
accuracy que la AlexNet. La diferencia no es la cantidad de parámetros sino **cómo están
regularizados y conectados**: las conexiones residuales facilitan la optimización, el
BatchNorm de cada bloque normaliza las activaciones, el Dropout(0.3) intra-bloque actúa
sobre los mapas de características (no sobre una capa densa final), y el
GlobalAveragePooling elimina de raíz la capa densa gigante que en la AlexNet concentraba el
36 % de los pesos: acá la capa de salida son 6.410 parámetros contra 1,18 M de FC6.

**Random Erasing es el mayor salto individual de todo el TP: +1,36 puntos.** El Exp 6
llega a **0.9305** en test contra 0.9169 del Exp 5, con exactamente la misma arquitectura y
la misma cantidad de parámetros. También logra la mejor pérdida de validación (0.2123) y de
test (0.2200) de los ocho experimentos, y es el único que aprovecha casi todo el
presupuesto de épocas: corre 19 de 20 con la mejor en la 14, mientras que sin erasing el
EarlyStopping corta en la 10. Al ocluir un rectángulo aleatorio, la red no puede apoyarse en
una sola región de la prenda y se ve forzada a distribuir la evidencia — exactamente el
efecto buscado, y el que explica la mejora en las clases que se distinguen por detalles
locales.

**El gap más grande no siempre es la peor señal.** El Exp 6 tiene el gap train−val más alto
(+0.0356), pero eso no indica sobreajuste dañino: su accuracy de validación (0.9267) y de
test (0.9305) son las mejores del TP, y su val_loss la más baja. El gap crece simplemente
porque el modelo entrena 19 épocas y aprende una tarea más difícil que la que se evalúa:
las imágenes de entrenamiento están ocluidas la mitad de las veces. La diferencia con el
Exp 3 está en la forma de la curva de validación: allí la val_loss sube de forma sostenida
apenas pasa el mínimo (0.2738 → 0.3035 en cinco épocas), mientras que acá se estabiliza en
una meseta entre 0.212 y 0.252 desde la época 11 hasta la 19, sin tendencia clara al alza —
el EarlyStopping corta por falta de mejora, no por deterioro.

**Costo.** El WRN-28-10 cuesta ~132 s por época contra ~5 s de la AlexNet: 25 veces más por
1,2–2,6 puntos de accuracy. Es el compromiso clásico de este tipo de arquitecturas — vale
la pena si el objetivo es exactitud, no si el objetivo es iterar rápido.

### Comparación global de las 8 configuraciones

`code/comparacion.py` genera la vista de conjunto de todo el trabajo, en orden cronológico
(Exp 1 → 2 → 7 → 8 → 3 → 4 → 5 → 6). En todas las figuras el **color codifica la familia de
arquitectura** — LeNet-5 azul, AlexNet naranja, WRN verde — y el estilo de línea separa las
variantes dentro de cada familia; las magnitudes (recall, matrices de confusión) usan una
rampa secuencial de un solo tono.

| Paso | Exp | Cambio introducido | Test Acc | Δ vs. anterior |
|:---:|:---:|---|---:|---:|
| 1 | 1 | punto de partida | 0.8926 | — |
| 2 | 2 | kernels 3×3, max pooling, padding `same` | 0.8943 | +0.0017 |
| 3 | 7 | profundidad (5 conv) + Dropout 0.5 | 0.9124 | +0.0181 |
| 4 | 8 | + rotación leve | 0.9099 | −0.0025 |
| 5 | 3 | conv4/conv5 más anchas (512/256), sin Dropout | 0.9049 | −0.0050 |
| 6 | 4 | + rotación leve | 0.9021 | −0.0028 |
| 7 | 5 | arquitectura residual ancha (WRN-28-10) | 0.9169 | +0.0120 |
| 8 | 6 | + Random Erasing | **0.9305** | **+0.0136** |

De punta a punta: **+3,79 puntos de accuracy, a cambio de 821× parámetros y 45× tiempo de
entrenamiento**. La progresión no es monótona, y ahí está lo interesante: los dos únicos
saltos grandes son *profundidad + Dropout* (+1,8) y *Random Erasing* (+1,4). Los pasos que
sólo agregan parámetros (Exp 3 respecto del 7: 3,8× más pesos y ninguna regularización)
directamente empeoran el resultado.

**Rotación leve vs. Random Erasing.** El mismo TP probó dos aumentos de datos con
resultados opuestos: la rotación de ±0.05 restó en las tres veces que se aplicó
(−0.0025 en la AlexNet angosta, −0.0028 en la ancha), mientras que Random Erasing sumó
+0.0136. La rotación produce imágenes que se parecen mucho a las originales — Fashion-MNIST
ya viene centrado y alineado, así que rotar no agrega información nueva y sólo introduce
bordes interpolados. Random Erasing, en cambio, cambia cualitativamente la tarea: obliga a
clasificar con parte de la prenda tapada.

**El costo no compra desempeño por sí solo** (`cmp_costo.png`). Ordenados por parámetros,
los ocho modelos no forman una recta: la AlexNet angosta con Dropout (877 k) le gana a la
ancha sin Dropout (3,3 M), y el WRN sin Random Erasing (36,5 M, 22 min) sólo saca 0,45
puntos sobre esa AlexNet angosta que entrena en 48 segundos. El salto real del WRN aparece
recién cuando se lo combina con el aumento de datos adecuado.

**Todas las arquitecturas fallan en la misma clase** (`cmp_recall_clases.png`). Shirt tiene
un recall promedio de 0.678 entre los ocho modelos, contra 0.98–0.99 de Trouser, Bag o Ankle
boot. Su rango va de 0.609 (AlexNet ancha + rotación) a 0.791 (WRN + Random Erasing): ningún
cambio de arquitectura la arregla, y el único que mueve la aguja de verdad es el aumento de
datos por oclusión. La segunda clase más difícil, Coat, sigue el mismo patrón. Es el techo
del dataset: a 28×28 en escala de grises, camisa, remera, pullover y abrigo comparten
silueta.

## Desempeño del mejor modelo (Exp 6 — WRN-28-10 + Random Erasing)

| Clase | Precision | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| T-shirt/top | 0.8914 | 0.8780 | 0.8846 | 1000 |
| Trouser | 0.9842 | 0.9960 | 0.9901 | 1000 |
| Pullover | 0.8641 | 0.9350 | 0.8982 | 1000 |
| Dress | 0.9163 | 0.9520 | 0.9338 | 1000 |
| Coat | 0.9259 | 0.8620 | 0.8928 | 1000 |
| Sandal | 0.9899 | 0.9810 | 0.9854 | 1000 |
| Shirt | 0.8248 | 0.7910 | 0.8076 | 1000 |
| Sneaker | 0.9761 | 0.9400 | 0.9577 | 1000 |
| Bag | 0.9959 | 0.9810 | 0.9884 | 1000 |
| Ankle boot | 0.9392 | 0.9890 | 0.9635 | 1000 |
| **Promedio macro** | 0.9308 | 0.9305 | 0.9302 | 10000 |

**Shirt** sigue siendo la clase crítica, pero su recall salta de 0.654 (mejor modelo
anterior) a **0.791**, y sus confusiones con T-shirt/top bajan de 160 a 91 casos. Es
justamente la clase que más se beneficia de Random Erasing: distinguir camisa de remera
depende de detalles locales (botones, cuello, mangas), y forzar a la red a clasificar con
parte de la prenda tapada la obliga a usar todas las pistas disponibles en vez de una sola.
El resto de las confusiones sigue la lógica semántica esperable (Coat ↔ Pullover,
Sneaker ↔ Ankle boot), y las clases de silueta inconfundible —Trouser, Bag, Sandal— superan
0.98 de F1.

## Gráficos (`results/`)

| Figura | Contenido |
|---|---|
| `curvas_exp1_lenet_clasica.png` … `curvas_exp6_wrn2810_erasing.png` | Curvas de pérdida y exactitud train vs. validación por experimento, con la mejor época marcada |
| `comparacion_curvas.png` | Las 6 corridas superpuestas (validación sólida, entrenamiento punteada) |
| `comparacion_accuracy.png` | Barras de accuracy train / validación / test por experimento |
| `gap_sobreajuste.png` | Gap train−val de accuracy y de pérdida como indicador de sobreajuste |
| `parametros_vs_accuracy.png` | Parámetros entrenables por modelo y su relación con el accuracy de test |
| `matriz_confusion_mejor.png` | Matriz de confusión del mejor modelo en test (conteos y normalizada por clase) |
| `accuracy_por_clase.png` | Recall por clase del mejor modelo, ordenado |
| `errores_mejor_modelo.png` | Los 15 errores con mayor confianza del mejor modelo |
| `dataset_muestras.png` | 3 muestras por clase del conjunto de entrenamiento |
| `augmentacion_ejemplos.png` | Efecto de la rotación aleatoria (factor 0.05) usada en el Exp 4 |
| `random_erasing_ejemplos.png` | Efecto de Random Erasing (p=0.5, área 2–40 %) usado en el Exp 6 |

Comparación global de las 8 configuraciones (`code/comparacion.py`):

| Figura | Contenido |
|---|---|
| `cmp_evolucion.png` | Recorrido cronológico del accuracy en test, anotado con el cambio que introduce cada paso |
| `cmp_curvas_todas.png` | Pérdida y exactitud de validación de las 8 corridas, color por familia |
| `cmp_accuracy.png` | Exactitud train / validación / test de cada configuración (dot plot) |
| `cmp_costo.png` | Parámetros y tiempo de entrenamiento (escala log) frente al accuracy de test |
| `cmp_sobreajuste.png` | Gap train−validación y mejor época vs. épocas corridas |
| `cmp_recall_clases.png` | Heatmap de recall por clase × experimento |
| `cmp_confusiones_familias.png` | Matriz de confusión normalizada del mejor modelo de cada familia |

También se guardan `experimentos.pkl` (historiales y métricas de las 6 corridas),
`matriz_confusion_mejor.npy`, `tabla_resumen.md`, `metricas_por_clase.md`, los logs
`train_log.txt` / `train_log_wrn.txt`, la tabla `cmp_tabla.md` y los modelos entrenados `exp*.keras`.

## Código

| Archivo | Rol |
|---|---|
| `code/common.py` | Semillas, carga y partición estratificada, normalización, capa `RandomErasing`, aumento de datos, callbacks y helpers de graficado |
| `code/models.py` | `build_lenet()`, `build_alexnet()`, `build_wrn()` y la matriz `EXPERIMENTS` con las 6 configuraciones |
| `code/train.py` | Entrena los experimentos, guarda modelos, historiales y curvas de aprendizaje |
| `code/evaluate.py` | Gráficos por experimento, matriz de confusión del mejor modelo y tablas Markdown |
| `code/comparacion.py` | Vista de conjunto de las 8 configuraciones: evolución, curvas, costo, recall por clase |

### Reproducir

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # tensorflow[and-cuda]==2.21.0
cd code
../.venv/bin/python train.py         # los 6 experimentos (~68 min en GPU)
../.venv/bin/python train.py 5 6     # o sólo algunos; se fusionan con los resultados previos
../.venv/bin/python evaluate.py
../.venv/bin/python comparacion.py   # comparación global (requiere los 8 experimentos)
```

`common.py` antepone al `PATH` el `ptxas` que trae el paquete `nvidia-cuda-nvcc` (12.9),
porque el del sistema (12.0) no soporta la compute capability de la GPU usada.
