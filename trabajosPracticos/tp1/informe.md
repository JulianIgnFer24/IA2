# Informe TP1 — Regresión Lineal Multivariable (Ejercicio 1)

**Dataset:** *House Prices — Advanced Regression Techniques* (Kaggle).
**Archivo:** `train.csv` con 1460 observaciones y 81 columnas (1460 filas × 81 atributos + *Id*).

Este informe se redacta por partes según la consigna. Cada sección se completa a medida que
se resuelve cada ítem.

---

## Identificación de variables (según consigna)

- **(a) Variable respuesta:** `SalePrice` (precio de venta de la vivienda, numérica continua).
  - El enunciado permite usar el precio o una transformación de este (p. ej. `log(SalePrice)`),
    que se decide más abajo en el análisis exploratorio.
- **(b) Predictores numéricos continuos** (ejemplos relevantes): `LotArea`, `GrLivArea`
  (superficie habitable), `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `LotFrontage`, `YearBuilt`,
  `YearRemodAdd`, así como puntajes de calidad ordinal (`OverallQual`, `OverallCond`).
- **(c) Predictores binarios (0/1):** en el dataset la presencia/ausencia suele estar codificada
  como categóricas (`CentralAir` Y/N, `PavedDrive` Y/N/P), o en columnas numéricas que actúan como
  indicadores (p. ej. `GarageCars`). Para el ejercicio se considera el patrón de ejemplo del
  enunciado (garaje / pileta / aire acondicionado).
- **(d) Categóricas con más de 2 niveles:** `Neighborhood` (25 niveles), `MSZoning` (5),
  `HouseStyle` (8), `RoofStyle` (6), `BldgType` (5), `Functional` (7), `SaleType` (9), etc.

> A lo largo de todo el ejercicio se usan estos nombres reales de columnas.

---

## 1. Análisis exploratorio

### 1.1 Análisis descriptivo — Variables numéricas

| Variable | Media | Desv. estándar | Min | Q1 | Mediana | Q3 | Max | % missings |
|---|---|---|---|---|---|---|---|---|
| `SalePrice` | 180 921 | 79 442 | 34 900 | 129 975 | 163 000 | 214 000 | 755 000 | 0 % |
| `GrLivArea` | 1 515 | 525 | 334 | 1 129 | 1 464 | 1 777 | 5 642 | 0 % |
| `LotArea` | 10 517 | 9 981 | 1 300 | 7 553 | 9 478 | 11 601 | 215 245 | 0 % |
| `TotalBsmtSF` | 1 057 | 439 | 0 | 796 | 991 | 1 298 | 6 110 | 0 % |
| `1stFlrSF` | 1 163 | 387 | 334 | 882 | 1087 | 1 391 | 4 692 | 0 % |
| `OverallQual` | 6.1 | 1.4 | 1 | 5 | 6 | 7 | 10 | 0 % |
| `YearBuilt` | 1971 | 30 | 1872 | 1954 | 1973 | 2000 | 2010 | 0 % |

**Observaciones generales:**

- Hay **38 variables numéricas** y **43 categóricas**.
- Hay **datos faltantes** en varias columnas: `LotFrontage` (259), `GarageYrBlt` (81),
  `MasVnrArea` (8), además de muchas categorías con NA que representan *"no tiene esa
  característica"* (p. ej. `Alley` con 1369, `PoolQC` con 1453, `MiscFeature` con 1406).
  Estas últimas son NA con significado semántico y deberán tratarse en preparación de datos.
- Varias variables numéricas muestran **asimetría positiva** fuerte: `LotArea` (12.2),
  `PoolArea` (14.8), `MiscVal` (24.5), `3SsnPorch` (10.3), `LowQualFinSF` (9.0), `KitchenAbvGr`
  (4.5), `TotalBsmtSF`-relacionadas, etc. Esto sugiere que **la estandarización será necesaria**
  antes de Ridge/Lasso.

### 1.2 Variables categóricas

La mayoría de las categóricas no tienen faltantes reales. Las categorías predominantes son:

- `Neighborhood` (25 niveles): los más frecuentes `NAmes` (225), `CollgCr` (150), `OldTown` (113).
- `MSZoning` (5): `RL` (1151), `RM` (218), `FV` (65).
- `BldgType` (5): `1Fam` (1220) domina; seguido de `TwnhsE` (114), `Duplex` (52).
- `HouseStyle` (8): `1Story` (726), `2Story` (445), `1.5Fin` (154).
- `RoofStyle` (6): `Gable` (1141), `Hip` (286).

La columna `Neighborhood` concentrará gran parte del poder predictivo y, al tener 25 niveles,
generará **24 variables dummy** en la codificación (k−1). Otras de gran número de niveles:
`Exterior1st` (15), `Exterior2nd` (16).

Algunas categóricas son casi constantes (poco informativas): `Street` (Pave 1454 / Grvl 6),
`Utilities` (AllPub 1459), `Condition2` (Norm 1445), `RoofMatl` (CompShg 1434), `Heating`
(GasA 1428). Candidatas a ser reducidas o eliminadas.

### (c) 1.3 Distribución de la variable respuesta (`SalePrice`)

**Figura 1 — Histograma de `SalePrice` vs. `log(SalePrice)`:**
![Distribución de SalePrice](figs/fig_dist_precio.png)

- **Asimetría (skewness):** `SalePrice` presenta **sesgo positivo (asimétrico a la derecha)**
  de **1.88**. Su **kurtosis** es 6.54, indicando **colas pesadas** con valores atípicos
  (pocas viviendas muy caras).
- El test de normalidad (D'Agostino-Pearson) rechaza la normalidad (p ≈ 1e-33).
- Esto es esperable: precios son concentrados en un rango medio con una cola hacia arriba
  (valores extremos muy caros).
- **Transformación logarítmica:** al aplicar `log(SalePrice)` la asimetría cae a **0.12** y el
  histograma se vuelve prácticamente simétrico (aunque el test sigue rechazando Gaussianidad
  estricta, la escala mejora notablemente).

**¿Convendría trabajar con el logaritmo?**
Sí. Motivación conforme a la teoría (Normalidad, para inferencia):

1. **Normalidad de los errores:** al estabilizar la distribución de la variable respuesta se
   aproxima a que los residuos sean más normales, requisito para la inferencia válida (valores p
   e intervalos de confianza fiables) — ver supuestos de Gauss-Markov.
2. **Estabiliza la varianza (homocedasticidad):** con precios el error absoluto crece con el
   precio; bajo log las variaciones se interpretan en términos relativos (%), reduciendo la
   heterocedasticidad.
3. **Efecto multiplicativo más realista:** un incremento de $10 000 no pesa igual en una casa de
   $50 000 que en una de $500 000; el log describe mejor el efecto que sobre esta variable.
Como desventaja, los coeficientes y errores se interpretan en términos de **% de cambio del
precio** y el RMSE pasa a medirse en escala logarítmica.

Por lo tanto, se trabajará con **`log(SalePrice)` como variable respuesta** en el resto
del ejercicio.

### 1.4 Matriz de correlación entre predictores numéricos

![Matriz de correlación](figs/correlacion_predictores.png)

![Correlaciones relevantes](figs/correlaciones_detalle.png)

**Relaciones más fuertes entre predictores (|corr| ≥ 0.5):** 28 pares; con |corr| ≥ 0.7 son:

| Par | Pearson r |
|---|---|
| `GarageArea` ~ `GarageCars` | **0.88** |
| `GarageYrBlt` ~ `YearBuild` | **0.83** |
| `GrLivArea` ~ `TotRmsAbvGrd` | **0.83** |
| `TotalBsmtSF` ~ `1stFlrSF` | **0.82** |
| `2ndFlrSF` ~ `GrLivArea` | 0.69 |
| `BedroomAbvGr` ~ `TotRmsAbvGrd` | 0.68 |
| `BsmtFullBath` ~ `BsmtFinSF1` | 0.65 |

**Correlación de los predictores con `SalePrice`:**

| Variable | corr |
|---|---|
| `OverallQual` | 0.79 |
| `GrLivArea` | 0.71 |
| `GarageCars` | 0.64 |
| `GarageArea` | 0.62 |
| `TotalBsmtSF` | 0.61 |
| `1stFlrSF` | 0.61 |

**Interpretación y consecuencias:**

- **Multicolinealidad esperable:** pares como `GarageArea–GarageCars`, `GrLivArea–TotRmsAbvGrd`
  y `TotalBsmtSF–1stFlrSF` son **muy redundantes** (miden ruido el mismo concepto en distintas
  unidades). Esto de acuerdo con el supuesto de **"no multicolinealidad perfecta"** de la teoría;
  aunque no hay colinearidad perfecta, la fuerte redundancia infla la varianza de los
  estimadores OLS (VIF alto).
- **Consecuencia para el modelado:** en la parte 3 se medirá con VIF y será necesario
  selección/regularización (Ridge/Lasso) que la varianza inflada.
- Tendencia natural del precio: atributos de **calidad general, superficie vivitable y garaje**.

**Nota de interpretación visual:** en el heatmap, los bloques rojos en la diagonal de bloques
de «superficie» y «calidad/años» se refuerzan esta conclusión.

---

## 2. Preparación de datos

Se implementa en el script reproducible `ejercicio1/preprocess.py`:

1. **Imputación de faltantes:** numéricas → mediana; categóricas → nivel `"NaN"`
   (los NA en columnas como `PoolQC`, `GarageType`, `Alley` significan *"no tiene esa
   característica"* y se conservan como un nivel propio).
2. **Codificación dummy con categoría de referencia (k−1):** cada categórica con más de
   2 niveles se codifica con `pd.get_dummies(..., drop_first=True)`, generando **k−1**
   columnas. Por ejemplo, `Neighborhood` (25 niveles) → 24 dummies; la categoría
   **referencia es la primera por orden alfabético** (para `Neighborhood`, `Blmngtn`),
   que queda representada por el intercepto.
3. **Estandarización:** `StandardScaler` ajustado **solo en train** y aplicado a train y
   test (evita *data leakage*). Se estandarizan únicamente las numéricas continuas; las
   dummies (0/1) no se estandarizan.
4. **División 80/20** con `random_state=42`: **1168 train / 292 test**.

**Resultado:** 36 numéricas estandarizadas + 224 dummies = **260 predictores**.

**¿Por qué no incluir las k dummies completas (trampa de la variable dummy)?**
Con el intercepto y las k columnas dummies de una misma categoría, la suma de las k
dummies es exactamente el vector de unos → **colinealidad perfecta**: `D₁+D₂+…+Dₖ = 1ₙ`,
y `(XᵀX)` se vuelve singular (no invertible), por lo que OLS no tiene solución única.
Usar **k−1** columnas elimina esa redundancia y hace que cada coeficiente dummy se
interprete como el **efecto diferencial respecto a la categoría de referencia**.

---

## 3. Regresión lineal multivariable (OLS)

Script: `ejercicio1/model_ols.py`.

**Ajuste sobre todos los predictores** (respuesta `log(SalePrice)`):

- **R² = 0.9449**, **R² ajustado = 0.9301** (train).
- Intercepto β₀ = 7.43 → precio base de la categoría de referencia ≈ $1 685 (log).

**Interpretación de coeficientes β̂ (magnitud y signo):**

| Predictor | β̂ | Interpretación (con el resto fijo) |
|---|---|---|
| `GrLivArea` | +0.065 | +1 ds de superficie habitable → +6.5 % precio |
| `YearBuilt` | +0.060 | +1 ds de antigüedad (más nuevo) → +6.0 % |
| `OverallQual` | +0.056 | +1 ds de calidad general → +5.6 % |
| `OverallCond` | +0.039 | +1 ds de condición → +3.9 % |
| `TotalBsmtSF` | +0.038 | +1 ds de sótano → +3.8 % |
| `2ndFlrSF`, `1stFlrSF`, `GarageArea` | + | más superficie → mayor precio |

- **Dummies** (interpretación respecto a la referencia): el coeficiente dummy `D_j` indica
  el **cambio logarítmico respecto a la categoría de referencia**; `exp(β̂)` es el factor
  multiplicativo del precio. Ej.: `RoofMatl_CompShg` β̂≈+1.74 → las casas con techo de
  compuesto valen ≈ **e^1.74 ≈ 5.7×** más que la referencia `ClyTile`.
- **Señales de inestabilidad:** dummies casi constantes (p. ej. `PoolQC_NaN` con β̂≈+2.2,
  por la casi ausencia de piscinas) muestran β̂ irreales → síntoma de la alta varianza
  que provoca la multicolinealidad.

**VIF (multicolinealidad):** de los 260 predictores:
- **19 con VIF = ∞** (relación lineal perfecta): `TotalBsmtSF` es suma exacta de sus
  componentes; `GrLivArea` es suma de las superficies de piso; las dummies `*_NaN` de
  garaje y de sótano son **idénticas** (marcan las mismas casas sin garaje/sótano).
- **78 con VIF ≥ 10** (colinealidad fuerte), incl. `PoolArea` (3651) y `RoofMatl_*`.
- **Conclusión:** hay **fuerte evidencia de multicolinealidad**.
- **Candidatas a eliminar:** las componentes de área redundantes (`BsmtFinSF1/SF2/UnfSF`
  frente a `TotalBsmtSF`; `1stFlrSF`/`2ndFlrSF`/`LowQualFinSF` frente a `GrLivArea`) y las
  dummies `*_NaN` duplicadas.
- **Candidatas a combinar:** un único indicador *"tiene garaje"* en vez de 4 dummies `NaN`.

**Evaluación en test:**

| Métrica | Valor |
|---|---|
| R² | 0.8383 |
| RMSE (log) | 0.1737 |
| MAE (log) | 0.0958 |
| RMSE | **$25 457** |
| MAE | $15 522 |

**Análisis de residuos:**

![Residuos OLS](ejercicio1/figs/ols_residuos.png)

- Residuos vs. ajustados: patrón de embudo (la dispersión cambia con el nivel) →
  **heterocedasticidad** (Breusch-Pagan p≈0).
- Q-Q: puntos se apartan de la diagonal en las colas → **no-normalidad** (Shapiro-Wilk
  p≈8e-27; asimetría −0.95, kurtosis 11). Hay **45 residuos >2σ** (12 >3σ), típico de
  precios extremos.
- **Supuestos de Gauss-Markov:** se violan **independencia/normalidad** y
  **homocedasticidad** (mejoradas por el log, pero no eliminadas) y **no-multicolinealidad**.
  Los β̂ siguen siendo **insesgados**, pero los errores estándar y p-valores no son fiables;
  la varianza está inflada. Esto justifica recurrir a la regularización.

---

## 4. Regresión Ridge

Script: `ejercicio1/model_ridge.py`.

- **Búsqueda de λ por CV de 5 folds:** λ\* = **20.09**.
- **Camino de regularización** (coefs vs. λ): a λ creciente todos los coeficientes
  **encogen hacia 0 sin anularse** (regularización L2).

![Camino Ridge](ejercicio1/figs/ridge_camino.png)
- **Comparación con OLS:** Ridge encoge fuertemente los β̂ inestables del OLS. Los más
  encogidos son precisamente los que tenían β̂ irreales en OLS (`PoolQC_NaN`: 2.23→~0.01;
  `RoofMatl_*`: 1.7→~0). 

**¿Qué ocurre con los predictores correlacionados?** Ridge reparte el peso entre los
miembros de cada grupo colineal en lugar de asignarlo arbitrariamente a uno:

| Par | OLS | Ridge |
|---|---|---|
| `GarageArea` | +0.033 | +0.008 |
| `GarageCars` | +0.006 | +0.042 |
| `GrLivArea` | +0.065 | +0.056 |
| `TotRmsAbvGrd` | +0.005 | +0.022 |

**Evaluación en test:** R² = **0.8959**, RMSE(log) = 0.1394, RMSE = $25 786, MAE = $16 750.

---

## 5. Regresión Lasso

Script: `ejercicio1/model_lasso.py`.

- **λ\* por CV de 5 folds = 0.00137.**
- **Coeficientes llevados exactamente a cero: 191 de 260.** "Sobreviven" **69 variables**
  (selección de variables L1).
- **Variables sobrevivientes más influyentes:** `GrLivArea` (+0.101), `OverallQual`
  (+0.096), dummies de barrio de alta gama (`Neighborhood_NridgHt` +0.074, `Crawfor`
  +0.072, `StoneBr` +0.062), `MSZoning_RM` (−0.056), `GarageCars` (+0.053),
  `YearBuilt` (+0.046), `OverallCond` (+0.043), `CentralAir_Y` (+0.043).
- **Interpretación:** Lasso eliminó las dummies ruidosas/casi constantes (piscina,
  materiales de techo, etc.) y las columnas redundantes; retuvo los predictores que
  realmente importan (calidad, superficie, barrio, garaje, aire acondicionado). Es una
  **selección de variables automática** basada en los datos.

**Evaluación en test:** R² = 0.8899, RMSE(log) = 0.1433, RMSE = $27 290, MAE = $17 325.

---

## 6. Comparación final

Script: `ejercicio1/comparison.py`. Resultados sobre el set de **test**:

| Modelo | R² test | RMSE (log) | RMSE ($) | MAE ($) | Predictores efectivos | Interpretabilidad |
|---|---|---|---|---|---|---|
| **OLS** | 0.8383 | 0.1737 | $25 457 | $15 522 | 260 (todos) | Media — alta varianza por multicolinealidad |
| **Ridge** | **0.8959** | **0.1394** | $25 786 | $16 750 | 260 (todos, encogidos) | Baja-media (no selecciona) |
| **Lasso** | 0.8899 | 0.1433 | $27 290 | $17 325 | **69 de 260** | **Alta** (modelo esparso) |

**Recomendación (trade-off sesgo-varianza):**
Con 19 predictores con VIF=∞ y 78 con VIF≥10, OLS tiene **varianza máxima** (β̂
inestables, mal R² de test a pesar del buen R² de train → claro sobreajuste, alta
varianza). La regularización introduce un poco de **sesgo** (encogen β̂) a cambio de
**reducir drásticamente la varianza**:

- **Ridge** logra el mejor equilibrio predictivo (mayor R² de test y menor RMSE en log):
  como todas las variables son informativas aunque estén correlacionadas, encogerlas
  (sin eliminarlas) aprovecha toda la información. 
- **Lasso** logra casi el mismo R² con solo 69 variables: ideal cuando se necesita un
  **modelo explicable y desplegable** (menos costo de recolección y mayor claridad).

**Conclusión del caso de negocio:** si el objetivo es **precisión de predicción de
precios**, recomiendo **Ridge**; si se prioriza **interpretabilidad/despliegue con pocas
variables**, recomiendo **Lasso**. OLS queda descartado por la inestabilidad que genera la
multicolinealidad.