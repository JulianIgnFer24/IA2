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

*(Secciones 2 a 6 del ejercicio se completarán en pasos siguientes.)*