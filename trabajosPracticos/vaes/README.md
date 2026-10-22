# VAE — Autoencoder Variacional sobre rostros de Los Simpson

Autoencoder Variacional convolucional en PyTorch que comprime rostros de Los Simpson a un
espacio latente de **128 dimensiones** y los reconstruye en **64×64×3**. Entrenado 200
épocas en GPU local.

- **Dataset**: 9.877 imágenes PNG de 200×200 RGB, 400 MB (`archive/cropped`, carpeta plana sin
  etiquetas), redimensionadas a 64×64 con interpolación LANCZOS y normalizadas a [0, 1].
- **Partición**: 9.383 train / 494 validación (95 % / 5 %, `random_split` con semilla 42).
- **Pérdida**: ELBO negativo = BCE píxel a píxel (`reduction="sum"`) + KL analítica contra
  N(0, I). Todo en float32, sin precisión mixta (`binary_cross_entropy` no es segura con AMP).
- **Entrenamiento**: Adam (lr = 1e-3), batch 128, 200 épocas, **12,6 min** en total
  (3,19 s/época).
- **Hardware**: NVIDIA GeForce RTX 5060 Ti 16 GB (compute capability 12.0),
  PyTorch 2.11.0+cu128.

## Arquitectura

Encoder y decoder son espejos: cuatro etapas que dividen (o multiplican) la resolución por
2 y duplican (o reducen a la mitad) la cantidad de canales.

### Encoder — (3, 64, 64) → (mu, logvar) ∈ R¹²⁸ × R¹²⁸

| Capa | Salida | Detalle |
|---|---|---|
| `Conv2d(3, 32, k=4, s=2, p=1)` + BN + LeakyReLU(0.2) | (32, 32, 32) | |
| `Conv2d(32, 64, k=4, s=2, p=1)` + BN + LeakyReLU(0.2) | (64, 16, 16) | |
| `Conv2d(64, 128, k=4, s=2, p=1)` + BN + LeakyReLU(0.2) | (128, 8, 8) | |
| `Conv2d(128, 256, k=4, s=2, p=1)` + BN + LeakyReLU(0.2) | (256, 4, 4) | |
| `Flatten` | 4096 | |
| `Linear(4096, 128)` ×2 | mu, logvar | dos cabezas **independientes** |

### Reparametrización

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)   # ε ~ N(0, I), sin gradiente
z = mu + std * eps
```

El truco mueve toda la estocasticidad a `eps`, que no lleva gradiente. Así el muestreo
queda dentro del grafo y `mu` / `logvar` se pueden entrenar por retropropagación, que es lo
que no permitiría muestrear directo de N(mu, sigma²).

### Decoder — z ∈ R¹²⁸ → (3, 64, 64)

| Capa | Salida | Detalle |
|---|---|---|
| `Linear(128, 4096)` + ReLU → reshape | (256, 4, 4) | |
| `ConvTranspose2d(256, 128, k=4, s=2, p=1)` + BN + ReLU | (128, 8, 8) | |
| `ConvTranspose2d(128, 64, k=4, s=2, p=1)` + BN + ReLU | (64, 16, 16) | |
| `ConvTranspose2d(64, 32, k=4, s=2, p=1)` + BN + ReLU | (32, 32, 32) | |
| `ConvTranspose2d(32, 3, k=4, s=2, p=1)` + **Sigmoid** | (3, 64, 64) | sin BatchNorm: la Sigmoid entrega los píxeles en [0, 1] |

**Parámetros**: encoder 1.739.936 + decoder 1.218.723 = **2.958.659** entrenables.

## Resultados

Todas las pérdidas están **promediadas por imagen** (nats/imagen, sumadas sobre los 12.288
píxeles-canal y las 128 dimensiones latentes).

| Época | train total | train recon | train KL | val total | val recon | val KL | gap val−train |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7705,67 | 7655,30 | 50,37 | 7490,58 | 7427,84 | 62,74 | −215,09 |
| 10 | 6773,61 | 6646,19 | 127,42 | 6741,03 | 6616,03 | 125,00 | −32,58 |
| **30** | 6675,12 | 6537,31 | 137,81 | 6681,14 | 6544,42 | 136,72 | +6,02 |
| 62 | 6630,31 | 6488,81 | 141,50 | **6668,29** | 6528,53 | 139,75 | +37,98 |
| 100 | 6604,70 | 6461,65 | 143,06 | 6677,79 | 6534,70 | 143,08 | +73,08 |
| **200** | 6576,28 | 6431,79 | 144,49 | 6682,84 | 6539,77 | 143,07 | +106,56 |

Lectura de las curvas (`outputs/figures/curvas_perdida.png`):

- **Sin posterior collapse.** El KL sube de 50 a ~144 nats en las primeras 50 épocas y se
  estabiliza ahí: 1,13 nats por dimensión latente. Si el KL hubiera caído a ~0, el decoder
  estaría ignorando `z` y el modelo habría degenerado en un generador incondicional.
- **La reconstrucción domina el ELBO**: 6.432 de 6.576 nats (97,8 %). Son 0,52 nats por
  píxel-canal, el costo irreducible de modelar un píxel continuo con una Bernoulli.
- **Convergencia a la época ~60.** El mejor `val_total` es de la **época 62** (6668,29);
  de ahí en adelante la validación se amesetó mientras el train seguía bajando, así que
  las ~140 épocas finales solo agrandan el gap (+107 nats a la 200) sin mejorar la
  generalización. En nats/imagen el sobreajuste es leve (1,6 % del total) y no se nota a
  simple vista en las figuras, pero está.

### Figuras

Todas en `outputs/figures/`, generadas con el modelo en `eval()` (imprescindible: hay
BatchNorm, y con las estadísticas del batch las muestras salen distorsionadas).

| Figura | Qué muestra |
|---|---|
| `reconstrucciones_epoca_{30,200}.png` | 8 imágenes fijas de validación, original arriba y reconstrucción abajo |
| `muestras_epoca_{30,200}.png` | grilla 8×8 de caras nuevas decodificadas desde un z ~ N(0, I) fijo |
| `interpolaciones_epoca_200.png` | 3 pares de validación con 10 pasos de interpolación lineal entre sus `mu` |
| `curvas_perdida.png` | total / reconstrucción / KL, train vs validación, por época |

**Reconstrucciones**: a la época 2 son manchas amarillas con el color y la posición
correctos; a la 30 ya se distinguen Lisa, Homer y Bart; a la 200 se recuperan la silueta
del pelo, los ojos, la boca y el fondo, pero no las líneas negras del contorno ni los
detalles finos.

**Muestreo desde el prior**: las 64 caras generadas tienen la paleta y la estructura
correctas (mancha amarilla centrada, ojos blancos, fondo de color), y varias son
reconocibles como Lisa o Homer; otras son rostros plausibles pero difusos.

**Interpolación**: las transiciones entre `mu_A` y `mu_B` son continuas y todos los pasos
intermedios son caras válidas — el latente no tiene agujeros entre ejemplos del dataset,
que es justamente lo que la regularización KL fuerza y lo que un autoencoder común no
garantiza. Se usa `mu` y no un `z` muestreado para que la transición sea determinística.

### Sobre el desenfoque

Las salidas son notoriamente borrosas. No es un error de implementación: es la limitación
conocida de los VAE con verosimilitud factorizada píxel a píxel. La BCE independiente por
píxel penaliza el error cuadrático promedio, de modo que ante la ambigüedad sobre dónde cae
exactamente un borde el decoder minimiza la pérdida promediando todas las posiciones
posibles en vez de comprometerse con una. El resultado es el promedio de las caras
compatibles con ese `z`, que es borroso por construcción. Mitigarlo exigiría cambiar el
objetivo (pérdida perceptual, VAE-GAN, un decoder autorregresivo), fuera del alcance de
este TP.

## Baseline: autoencoder determinístico

Para mostrar *por qué* hace falta el término KL se entrenó un autoencoder común con la
**misma arquitectura, los mismos datos y las mismas 200 épocas**. La única diferencia: el
encoder termina en una sola `Linear(4096, 128)` que devuelve `z` directo —sin `mu`, sin
`logvar`, sin reparametrización— y la pérdida es solo BCE, sin KL.

| | AE determinístico | VAE |
|---|---:|---:|
| Parámetros | 2.434.243 | 2.958.659 |
| Pérdida | BCE | BCE + KL |
| Reconstrucción final (train) | **6.327,85** | 6.431,79 |
| Reconstrucción final (val) | 6.504,56 | 6.539,77 |
| Mejor val (época) | 6.453,15 (33) | 6.528,53 (62) |
| Tiempo (200 épocas) | 11,4 min | 12,6 min |
| ¿Genera desde z ~ N(0, I)? | **No** | Sí |

El AE reconstruye **103,94 nats mejor** que el VAE, exactamente lo esperable: no paga el
costo de la regularización. Pero no sirve para generar, y el PCA del latente explica por
qué: los `z` del AE se reparten en un rango de **[−57,9, 54,9]** en la primera componente,
mientras que las muestras de N(0, I) caen en un punto diminuto del origen. Decodificar un
`z` aleatorio en el AE devuelve siempre la misma mancha verdosa —el promedio del dataset—
porque todas las muestras del prior caen en la misma región mínima que el decoder nunca
aprendió a usar. En el VAE el rango es **[−4,3, 5,6]** y prior y datos se superponen.

Dos diagnósticos adicionales del VAE entrenado, medidos sobre validación:

- **101 de las 128 dimensiones latentes están activas** (σ medio < 0,5); solo 3 quedaron
  pegadas al prior. No hay colapso ni global ni por dimensión.
- El σ medio es **0,38**, o sea que las nubes son angostas: muestrear `z = μ + σ·ε` cambia
  la imagen apenas 0,023 por píxel, y la variación se concentra en los **contornos**, no en
  las zonas planas de color. Hay que llegar a `k = 6` en `z = μ + k·σ·ε` para que la
  identidad de la cara se rompa.

Reproducir:

```bash
.venv/bin/python code/baseline/model_ae.py            # test de formas y parámetros
.venv/bin/python code/baseline/train_ae.py --epochs 200
```

## Segundo experimento: cropped contra simplified

El archivo de Kaggle trae las mismas 9.877 caras en dos versiones: `archive/cropped` (el
fotograma real, ~9.500 colores únicos por imagen) y `archive/simplified` (la misma cara con
colores planos, **3 colores únicos por imagen**, seis veces menos peso). Se repitieron los
dos entrenamientos completos —VAE y AE baseline, 200 épocas cada uno— sobre `simplified`,
**sin tocar nada del experimento original**: `code/simplified/run_experimentos.py` reescribe
las rutas de `config` antes de importar `train.py` y `train_ae.py`, así las salidas van a
`outputs/simplified/`.

| Métrica | cropped | simplified |
|---|---:|---:|
| Pérdida total VAE (train) | 6576.28 | 5559.30 |
| Reconstrucción (train) | 6431.79 | 5452.40 |
| Divergencia KL | 144.49 | 106.90 |
| **Brecha validación − train** | **106.56** | **18.06** |
| **Dimensiones latentes activas** | **101 / 128** | **60 / 128** |
| **Nitidez conservada (reconstrucción)** | **6.2 %** | **39.8 %** |
| **Muestras: distancia a los datos** | **0.90 ×** | **1.14 ×** |
| Tiempo (200 épocas) | 10.6 min | 7.7 min |

**Las dos primeras filas no se pueden comparar.** `simplified` tiene colores planos, así que
su BCE mínima alcanzable es mucho más baja: un número menor significa tarea más fácil, no
modelo mejor. Las cuatro filas en negrita sí son comparables.

### El resultado no apunta todo para el mismo lado

- **Reconstruir: gana simplified, por lejos.** Conserva el **39.8 %**
  de la nitidez del original contra el **6.2 %** de cropped
  —más de seis veces— y casi no sobreajusta (brecha 18.06 contra
  106.56). Visualmente la reconstrucción es casi exacta.
- **Generar: gana cropped.** Para medirlo se calcula, para cada cara generada desde
  z ~ N(0, I), la distancia a la imagen real más parecida, normalizada por lo que se parecen
  entre sí dos caras reales del mismo dataset. En cropped da **0.90×**
  (las muestras caen *dentro* de la nube de datos) y en simplified **1.14×**
  (caen *afuera*). Se ve en `comp_generacion.png`: las de cropped son borrosas pero son
  caras; las de simplified son nítidas y están rotas, manchas amarillas fragmentadas sin
  estructura facial.

La explicación más plausible tiene dos partes. El decoder de simplified aprendió a producir
colores casi binarios, así que un error no sale como desenfoque sino como un pedazo de color
en el lugar equivocado —y el ojo perdona el desenfoque, no la rotura—. Y simplified usa solo
**60 de las 128** dimensiones contra
**101** de cropped, así que hay más dimensiones donde el prior
no está bien cubierto por los datos.

**Conclusión:** para el objetivo del trabajo, que es generar caras nuevas, conviene `cropped`.
Una tarea más fácil de reconstruir no es automáticamente más fácil de generar: son dos
capacidades distintas y conviene medirlas por separado.

Reproducir:

```bash
.venv/bin/python code/simplified/run_experimentos.py --epochs 200   # VAE + AE (~17 min)
.venv/bin/python code/simplified/make_figures_simplified.py         # figuras de simplified
.venv/bin/python code/presentacion/comparar_datasets.py             # comp_*.png + comparacion.json
```

## Modelo combinado: un VAE entrenado con los dos estilos

Además de los dos experimentos por separado, se entrenó **un único VAE con las
19,754 imágenes juntas** (las mismas 9.877 caras, cada una en sus dos versiones):
misma arquitectura, mismos hiperparámetros, 200 épocas, 2,958,659 parámetros.
`code/combinado/run_combinado.py` sustituye `data.build_dataloaders` por una versión que
concatena las dos carpetas antes de hacer el split —así la partición mezcla los dos
estilos— sin tocar `train.py` ni ningún otro archivo existente.

| | cropped | simplified | **combinado** |
|---|---:|---:|---:|
| Imágenes | 9.877 | 9.877 | **19.754** |
| Pérdida total (train) | 6.576,3 | 5.559,3 | **6.090,3** |
| Divergencia KL | 144,5 | 106,9 | **129,1** |
| Brecha val − train | +106,6 | +18,1 | **+75,7** |
| Mejor validación | época 62 | época 49 | **época 117** |
| Tiempo | 10,6 min | 7,7 min | **17,9 min** |

El modelo combinado queda, como era de esperar, entre los dos en pérdida y KL. Lo
interesante aparece al mirar qué hizo con el espacio latente.

### 1. No separa los estilos: organiza por cara, no por estilo

La distancia entre los centroides de las dos nubes es **2.418**,
contra una dispersión interna media de **9.551**: un cociente de
**0.253**. Las dos nubes están completamente entremezcladas
(`comb_latente.png`). En el t-SNE se ven pares de puntos pegados de distinto color: son la
misma cara en sus dos versiones, que caen juntas. El modelo organizó el latente por *qué
cara es*, no por *en qué estilo está dibujada*.

### 2. «Simplificar» quedó codificado como una dirección fija

Si se calcula `v = media de ( μ(simplified) − μ(cropped) )` sobre 1.500 pares, resulta que
**el 100 % de los pares tiene coseno positivo con esa
dirección** (coseno medio 0.283). La dirección es consistente en signo
aunque ruidosa en magnitud.

Y sirve: sumarle `α · v` al `μ` de una cara **que no entró en el cálculo de v** la
simplifica progresivamente, y en `α = 1` el resultado se parece mucho a su versión
simplificada real (`comb_direccion_estilo.png`). El modelo nunca supo que las imágenes
venían en pares; la correspondencia la dedujo solo.

### 3. Genera los dos estilos, mitad y mitad

Proyectando sobre la dirección de estilo, las caras reales caen en
-1.086 (cropped) y 1.332 (simplified). Las muestreadas de
z ~ N(0, I) caen en 0.004, repartidas
**46 % / 54 %**
entre los dos lados. En la grilla de `comb_generacion.png` se ven las tres cosas: caras en
estilo plano, caras con fondo texturado, e híbridos.

> **Una medida que hubo que descartar.** El primer intento midió el estilo de lo generado
> con la textura por píxel (variación total), y daba «100 % simplified». Era un artefacto:
> el desenfoque del VAE baja la variación total igual que la simplificación, así que una
> cara cropped borrosa se confunde con una simplified. Medido en el latente —donde el
> modelo efectivamente codificó el estilo— el reparto es parejo y coincide con lo que se
> ve. El `analisis.json` conserva las dos medidas con la aclaración.

Reproducir:

```bash
.venv/bin/python code/combinado/run_combinado.py --epochs 200   # ~18 min
.venv/bin/python code/combinado/analizar_combinado.py           # 6 figuras + analisis.json
```

Salidas en `outputs/combinado/` (checkpoints, curvas) y `outputs/combinado/analisis/`.

## ¿Mejoró el modelo al agregar simplified?

La pregunta hay que responderla **sobre imágenes cropped**, que es la tarea original, y con
los dos modelos evaluados **sobre exactamente las mismas imágenes**.

### El cuidado metodológico

Los dos modelos tienen particiones distintas. De las **494** imágenes de
validación del modelo base, el combinado vio **468** durante su
entrenamiento. Comparar sobre esas 494 da una «mejora» de
**128 nats** (6540 → 6411)
que es falsa: el combinado está reconstruyendo imágenes que memorizó.

La única comparación limpia es sobre las que quedaron fuera del entrenamiento de los dos:
son **26**. Con ese n, en vez de comparar dos promedios se compara
**imagen por imagen** (la misma cara en los dos modelos) con un intervalo de confianza por
bootstrap, que es mucho más potente.

### El resultado

| | solo cropped | cropped + simplified |
|---|---:|---:|
| Reconstrucción (BCE, 26 caras limpias) | 6712.14 | **6694.88** |
| Divergencia KL | 142.00 | 142.75 |
| Nitidez conservada | 7.0 % | **7.8 %** |
| Generación: distancia a las caras cropped | **0.90** | 0.96 |

- **Reconstruyendo: sí mejora, y la mejora es real pero mínima.** La diferencia pareada es
  de **-17.3 nats** por imagen, con IC 95 %
  [-33, -2]. No incluye el cero, así que es
  estadísticamente significativa, y el combinado gana en
  **19/26** imágenes. Pero son
  17.3 nats sobre 6712: un **0,26 %**.
  Significativo y prácticamente irrelevante.
- **Generando: empeora.** Las muestras del combinado caen a
  0.96 de las caras cropped contra
  0.90 del base, porque la mitad las dibuja en el otro estilo.
  Para generar caras con aspecto de fotograma, el modelo entrenado solo con cropped sigue
  siendo mejor.

> **Confundidor.** El combinado hizo el **doble de pasos de gradiente** (146 batches por
> época contra 73), aunque cada imagen cropped la vio las mismas 200 veces. Así que ese
> 0,26 % ni siquiera es todo mérito de los datos extra.

**Conclusión:** agregar simplified como datos extra no mejora de forma útil la tarea
original. Lo que sí aporta es otra cosa —la estructura de estilo en el latente, de la
sección anterior— que no es una mejora de rendimiento sino una capacidad nueva.

```bash
.venv/bin/python code/combinado/comparar_con_base.py
```

## Presentación

`code/presentacion/` genera una presentación de **35 diapositivas** en 16:9 con notas del
orador completas, armada como **una historia en tres actos**:

| Acto | Qué pasa | Con qué se cierra |
|---|---|---|
| 1 · El caso base | El autoencoder comprime y descomprime | No puede generar: islas y huecos |
| 2 · La mejora | Se agrega la divergencia KL | Caras nuevas desde z ~ N(0, I) |
| 3 · Todos los datos | Un solo VAE con los dos estilos | «Simplificar» es una dirección del latente |

**Ambientación.** El diseño está ambientado en la serie: la portada usa el cielo con nubes
de la apertura, cada acto arranca con el gag del pizarrón de Bart escribiendo la lección
del acto, y el cierre es el sofá vacío. Esos motivos se **dibujan** en
`ambiente_simpsons.py` en vez de bajar capturas de la apertura, que son material con
copyright; así además entran en la paleta del resto y rinden a cualquier resolución. Lo
único descargado es `donut.svg` de Wikimedia Commons, que está en **dominio público sin
restricciones**. Las caras son las del propio dataset.

El sistema de diseño vive en `diseno.py` (estructura) y `tema_simpsons.py` (paleta de la
serie y tipos de diapositiva propios), separados del contenido.

Ningún número está escrito a mano: todos se leen de `dataset_stats.json`,
`resultados.json`, `comparacion.json` y `combinado/analisis/analisis.json` al generar el
.pptx, así que la presentación no se puede desincronizar de los experimentos.

```bash
.venv/bin/python code/presentacion/dataset_stats.py        # estadísticas + 2 figuras
.venv/bin/python code/presentacion/make_figures.py         # 16 figuras + resultados.json
.venv/bin/python code/presentacion/comparar_datasets.py    # figuras comparativas
.venv/bin/python code/presentacion/ambiente_simpsons.py    # cielo, pizarrones, sofá
.venv/bin/python code/presentacion/make_presentation.py    # el .pptx
.venv/bin/python code/presentacion/make_guion.py           # el guion del orador
```

### Guion del orador

`make_guion.py` escribe `outputs/presentacion/guion.md`: el texto completo de lo que hay
que decir, diapositiva por diapositiva, con una tabla de ritmo (duración de cada una y
tiempo acumulado) para saber sobre la marcha si uno va adelantado o atrasado. Son
**~3.150 palabras, unos 25 minutos** a 125 palabras por minuto.

No se escribe a mano: reconstruye el mazo en memoria y lee el índice que el `Deck` va
registrando mientras lo arma, así que sale de la misma fuente que la presentación y no se
puede desincronizar.

El recorrido de la charla es el argumento del baseline: se construye el autoencoder, se
muestra que reconstruye bien, se intenta generar y falla, se diagnostica el latente con PCA
y t-SNE, y recién ahí aparece el VAE como la solución a ese problema concreto.

Salidas en `outputs/presentacion/`: `presentacion_vae_simpsons.pptx`, `figuras/`,
`dataset_stats.json` y `resultados.json`.

## Estructura

```
vaes/
├── code/
│   ├── config.py      # rutas, hiperparámetros, semillas, dispositivo
│   ├── data.py        # SimpsonsFacesDataset + DataLoaders (split 95/5)
│   ├── model.py       # Encoder, Decoder, VAE
│   ├── loss.py        # BCE + KL
│   ├── train.py       # loop de entrenamiento, checkpoints, CSV de pérdidas
│   ├── visualize.py   # reconstrucciones, muestreo, interpolación, curvas
│   ├── baseline/
│   │   ├── model_ae.py        # EncoderAE + AE (reusa el Decoder del VAE)
│   │   └── train_ae.py        # entrenamiento sin KL
│   ├── simplified/
│   │   ├── run_experimentos.py        # repite VAE y AE sobre archive/simplified
│   │   └── make_figures_simplified.py
│   ├── combinado/
│   │   ├── run_combinado.py           # un VAE con cropped + simplified juntas
│   │   ├── analizar_combinado.py      # separación de estilos, dirección de estilo
│   │   └── comparar_con_base.py       # ¿mejoró con más datos? (pareado, sin fuga)
│   └── presentacion/
│       ├── diseno.py                  # estructura del sistema de diseño
│       ├── tema_simpsons.py           # paleta y diapositivas de la serie
│       ├── ambiente_simpsons.py       # cielo, pizarrón y sofá (dibujados)
│       ├── dataset_stats.py
│       ├── make_figures.py
│       ├── comparar_datasets.py       # cropped vs simplified
│       ├── make_presentation.py       # la historia en tres actos
│       └── make_guion.py              # guion del orador en markdown
├── requirements.txt
├── archive/cropped/   # dataset (no versionado)
└── outputs/           # no versionado
    ├── loss_history.csv
    ├── train_200.log
    ├── checkpoints/   # vae_epoch_30.pt, vae_epoch_200.pt, vae_last.pt
    ├── figures/       # figuras del VAE
    ├── baseline/      # ae_final.pt, ae_loss_history.csv, ae_train.log
    ├── simplified/    # el experimento completo sobre archive/simplified
    ├── combinado/     # el VAE entrenado con los dos estilos + analisis/
    └── presentacion/  # .pptx, figuras/, dataset_stats.json, resultados.json,
                       # comparacion.json
```

Los checkpoints guardan `model_state_dict`, `optimizer_state_dict`, `epoch`, los
hiperparámetros, el historial completo y los tensores fijos `x_fixed` / `z_fixed`, así las
figuras de distintas épocas son comparables entre sí.

## Reproducir

La GPU es Blackwell (compute capability 12.0) y necesita wheels de PyTorch compilados con
CUDA 12.8+, que no están en PyPI:

```bash
cd trabajosPracticos/vaes
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install pillow matplotlib numpy tqdm
uv pip install scikit-learn python-pptx   # baseline y presentación
```

```bash
.venv/bin/python code/model.py                  # test de formas y conteo de parámetros
.venv/bin/python code/data.py                   # verifica el dataset y el split
.venv/bin/python code/train.py                  # 200 épocas (~13 min)
.venv/bin/python code/train.py --epochs 2       # prueba corta de punta a punta
.venv/bin/python code/train.py --resume outputs/checkpoints/vae_last.pt
.venv/bin/python code/visualize.py --checkpoint outputs/checkpoints/vae_epoch_200.pt
```

`train.py` acepta además `--data_dir`, `--batch_size`, `--num_workers` y `--lr`. Las rutas
por defecto se derivan de la ubicación del código, así que los scripts corren desde
cualquier directorio.
