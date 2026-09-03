# Guion de la presentación — TP4

Guion para las 13 láminas de la presentación. El texto en *cursiva* es lo que se dice;
lo demás son referencias para vos. Duración estimada: **11 a 13 minutos**.

Enlace de la presentación: https://claude.ai/code/artifact/5564eb13-2b36-4097-ace0-9d06f7b80f8e

**Antes de empezar:** abrí la presentación a pantalla completa. Se navega con las flechas
del teclado (← →) o la barra espaciadora. Cualquier gráfico se agranda haciéndole clic, y
adentro se puede acercar con la rueda del mouse y arrastrar para moverse; se cierra con
Escape. Si la sala tiene mucha luz, el botón "Modo claro" de arriba a la derecha cambia el
tema.

---

## Lámina 1 — Portada y ¿qué es Fashion-MNIST? · ~50 s

*Buenas. Soy Julián y les voy a presentar el trabajo práctico 4 de Inteligencia
Artificial 2: redes neuronales convolucionales aplicadas a Fashion-MNIST.*

*Fashion-MNIST es un dataset de imágenes de prendas de vestir que publicó Zalando en 2017.
La idea con la que lo crearon es que fuera un reemplazo directo del MNIST clásico, el de
los dígitos escritos a mano. Mantiene exactamente el mismo formato y el mismo tamaño, así
que es lo que se llama un dataset "drop-in": si vos ya tenías un pipeline andando con
MNIST, le cambiás la línea de la carga de datos y funciona igual, sin tocar nada más.*

*La diferencia está en la dificultad. Reconocer dígitos escritos a mano hoy es un problema
resuelto —cualquier red simple pasa el 99 %—, mientras que distinguir una camisa de una
remera resulta bastante más complicado, y eso es justamente lo que hace interesante al
dataset.*

> Señalá la tira de imágenes de la derecha: son tres muestras de cada una de las diez
> categorías, tal cual entran a la red.

---

## Lámina 2 — Estructura y variables · ~1 min 20 s

*Veamos qué contiene exactamente.*

*En cuanto al tamaño: son 70.000 imágenes en total, divididas en 60.000 de entrenamiento y
10.000 de test. Y están perfectamente balanceadas: hay exactamente 6.000 imágenes de cada
categoría en entrenamiento y 1.000 de cada una en test. Esto es importante porque significa
que la exactitud, el accuracy, es una métrica honesta acá: no hay una clase mayoritaria que
infle el resultado.*

*En cuanto a las variables o features: cada imagen es de 28 por 28 píxeles, en escala de
grises, o sea un solo canal. Cada píxel es un número entero de 0 a 255 que representa la
intensidad del gris, donde 0 es negro y 255 es blanco. Si uno aplana la imagen, cada
ejemplo es un vector de 784 valores. Nosotros no la aplanamos, porque justamente el punto
de las convolucionales es aprovechar la estructura espacial de la imagen, así que trabajamos
con tensores de 28 por 28 por 1.*

*Y la variable objetivo, el label, es una sola etiqueta categórica por imagen: un entero
del 0 al 9 que identifica qué prenda es. Es un problema de clasificación multiclase de
etiqueta única — cada imagen pertenece a una y solo una categoría.*

*Estas son las diez clases: remera, pantalón, pulóver, vestido, abrigo, sandalia, camisa,
zapatilla, bolso y botineta.*

*Y acá quiero que se fijen en algo que va a explicar prácticamente todo lo que viene
después: cuatro de las diez clases son ropa de torso. Remera, pulóver, abrigo y camisa.
A 28 por 28 píxeles y en blanco y negro, las cuatro son básicamente el mismo contorno con
mangas. Ahí se concentra casi todo el error de todos los modelos que probé.*

---

## Lámina 3 — Preparación y protocolo · ~1 min

*Antes de entrenar nada, tres decisiones que tomé una sola vez y apliqué idénticas a las
ocho configuraciones.*

*Primero, la partición. Del conjunto de entrenamiento separé el 10 % para validación, pero
de forma estratificada: respetando la proporción de clases. Quedaron 54.000 imágenes para
entrenar y 6.000 para validar, con exactamente 600 de cada categoría. El test de 10.000 lo
toco una sola vez, al final de todo, para no contaminar las decisiones.*

*Segundo, la normalización: divido todos los píxeles por 255 para llevarlos al rango 0 a 1,
y agrego el canal explícito.*

*Tercero, y esto lo quiero remarcar: la reproducibilidad. Fijo las semillas de random, de
NumPy, de TensorFlow, la variable PYTHONHASHSEED, y además activo el modo determinista de
las operaciones de GPU. Y no lo asumí: cuando reentrené todo desde cero para agregar
experimentos nuevos, los primeros dos reprodujeron sus métricas dígito a dígito, hasta el
cuarto decimal.*

*El protocolo de entrenamiento es idéntico para las ocho: Adam con learning rate de 0.001,
entropía cruzada categórica, 20 épocas, lotes de 64, y parada temprana monitoreando la
pérdida de validación con paciencia de 5 épocas, quedándome siempre con los pesos de la
mejor época. Mantener esto fijo es lo que me permite decir que cualquier diferencia de
resultado viene de la arquitectura y no de haber tuneado hiperparámetros.*

---

## Lámina 4 — Las ocho combinaciones · ~1 min 15 s

*Estas son las ocho configuraciones que entrené. Tres familias de arquitectura y dos
esquemas de aumento de datos.*

*Las dos primeras son LeNet-5, la red de LeCun de 1998: la versión clásica con kernels de
5 por 5 y average pooling, y una versión optimizada con kernels de 3 por 3, max pooling y
padding same. Están en el orden de 44 mil y 106 mil parámetros, y dan 0.8926 y 0.8943.*

*Después vienen cuatro variantes de AlexNet adaptada, con cinco capas convolucionales. La
resaltada en naranja, el experimento 7, es la mejor de todas las arquitecturas que vimos en
clase: 0.9124 con 877 mil parámetros.*

*Y las dos últimas son la arquitectura que investigué por fuera: una Wide ResNet 28-10, con
36 millones y medio de parámetros. Sola da 0.9169, y combinada con Random Erasing —que es
una técnica de aumento de datos que ahora explico— llega a 0.9305, que es el mejor
resultado de todo el trabajo.*

> Si preguntan por la numeración: los experimentos están numerados por orden de definición,
> pero el orden cronológico en que los corrí fue 1, 2, 7, 8, 3, 4, 5, 6. La lámina siguiente
> los muestra en ese orden real.

---

## Lámina 5 — El recorrido · ~1 min

*Este gráfico muestra cómo se fue moviendo el accuracy a lo largo del trabajo, en el orden
real en que hice los experimentos. Cada punto es una configuración, y el color indica la
familia: azul LeNet, naranja AlexNet, verde la Wide ResNet.*

*De punta a punta la mejora es de 3,79 puntos. Pero lo interesante no es el número final,
es la forma de la curva: no sube de manera monótona, tiene dos retrocesos claros.*

*Si miran bien, sólo dos pasos dan saltos grandes. Uno es agregar profundidad con Dropout,
que suma 1,8 puntos. El otro es agregar Random Erasing, que suma 1,4. Todo el resto se
mueve dentro del ruido o directamente empeora.*

*Y el retroceso del medio es el más aleccionador: ahí lo que hice fue ensanchar las últimas
dos capas convolucionales a 512 y 256 filtros y sacar el Dropout. Multipliqué los
parámetros por 3,8 y perdí tres cuartos de punto. Más parámetros, peor resultado.*

---

## Lámina 6 — AlexNet adaptada · ~1 min 30 s

*Vamos a la mejor de las arquitecturas que vimos en clase.*

*El problema con AlexNet es que la original está diseñada para imágenes de 227 por 227 en
color. Si uno la aplica tal cual a imágenes de 28 por 28, las dimensiones espaciales se
achican hasta anularse antes de llegar al final. Entonces hay que adaptarla conservando su
esencia: cinco bloques convolucionales, más filtros a medida que baja la resolución, y dos
capas densas anchas con Dropout, que es el elemento distintivo de AlexNet.*

*Acá está el detalle capa por capa. Arranca con 32 filtros de 3 por 3 sobre la imagen de
28 por 28, y va alternando convoluciones con max pooling: 28, 14, 7, y termina en un mapa
de 3 por 3 por 64. Eso se aplana en 576 valores que entran a dos capas densas de 512, cada
una seguida de un Dropout del 50 %, y finalmente la capa de salida de 10 neuronas con
softmax.*

*En total, 877.258 parámetros entrenables.*

*Ahora, el dato que a mí me pareció más interesante es cómo se reparten. El 64 % de los
pesos —558 mil de los 877 mil— está en las dos capas densas, FC6 y FC7. Las cinco
convoluciones juntas suman 313 mil. O sea que la parte más pesada de la red es también la
menos eficiente por parámetro, y es exactamente la que protege el Dropout.*

*Y que el Dropout ahí es imprescindible lo comprobé por las malas: la variante ancha sin
Dropout, que tiene 3,3 millones de parámetros, toca su mínimo de validación en la época 4 y
después se pone a memorizar. Termina en 0.9049, por debajo de esta que tiene cuatro veces
menos parámetros.*

---

## Lámina 7 — WRN-28-10 · ~1 min 45 s

*Y esta es la arquitectura que investigué aparte: una Wide ResNet.*

*La idea de las Wide ResNet es un poco a contramano de la intuición habitual. En vez de
hacer las redes más profundas, que es lo que hicieron las ResNet clásicas llegando a 50, a
101, a 152 capas, las hace más anchas: menos capas, pero muchos más canales por capa.*

*La notación 28-10 se lee así: 28 es la profundidad total, o sea la cantidad de capas
convolucionales, y 10 es el factor de ensanchamiento k, que multiplica por diez la cantidad
de canales de cada bloque respecto de una ResNet base. Con profundidad 28 salen cuatro
bloques residuales por grupo, y con k igual a 10 los tres grupos pasan a tener 160, 320 y
640 canales.*

*La estructura es: una convolución inicial de 16 filtros, después los tres grupos de bloques
residuales, donde el primer bloque de cada grupo baja la resolución a la mitad con stride 2,
y al final BatchNorm, ReLU, un Global Average Pooling y la capa densa de salida.*

*Cada bloque residual sigue este patrón que ven en el diagrama: BatchNorm, ReLU,
convolución de 3 por 3, otra vez BatchNorm y ReLU, Dropout, y una segunda convolución. Y
lo que lo define como residual es la conexión de atajo: la entrada se suma directamente a la
salida, así que la red aprende el residuo en vez de la transformación completa. Cuando
cambian las dimensiones, ese atajo pasa por una convolución de 1 por 1 para poder sumarse.*

*El total son 36 millones y medio de parámetros, unas 42 veces la AlexNet, y el grupo 3
solo se lleva el 76 % de todos ellos.*

*Con 36 millones de parámetros sobre 54.000 imágenes chiquitas, uno esperaría que sobreajuste
de entrada. Y pasa exactamente lo contrario: tiene uno de los gaps entre entrenamiento y
validación más chicos de todos los experimentos. ¿Por qué? Por cuatro cosas: las conexiones
residuales facilitan la optimización y evitan que los gradientes se desvanezcan; el
BatchNorm normaliza las activaciones en cada bloque; el Dropout actúa acá sobre mapas de
características en vez de sobre una capa densa final; y el Global Average Pooling elimina de
raíz la capa densa gigante. Fíjense en el contraste: la capa de salida de esta red tiene
6.410 parámetros, contra 1 millón 180 mil de la FC6 de AlexNet.*

---

## Lámina 8 — Random Erasing · ~1 min 15 s

*El otro ingrediente es Random Erasing, una técnica de aumento de datos que se publicó
en 2017 y que se usa mucho como baseline fuerte junto con las Wide ResNet.*

*El algoritmo es simple: para cada imagen del lote, con probabilidad 0.5 se decide si
intervenirla. Si se la interviene, se sortea el área de un rectángulo —entre el 2 % y el
40 % de la imagen—, se sortea su relación de aspecto, se sortea la posición, y esa región
se rellena con ruido. Arriba ven las imágenes originales y abajo las mismas después de
pasar por la capa.*

*Lo importante es que sólo actúa durante el entrenamiento: en validación y en test la capa
es la identidad exacta, no toca nada.*

*¿Por qué ayuda? Porque simula oclusiones parciales, como si la prenda estuviera tapada.
Al no poder confiar en que una región concreta vaya a estar visible, la red se ve forzada a
distribuir la evidencia por toda la imagen en lugar de apoyarse en un único detalle.*

*El resultado: 1,36 puntos más de accuracy, con exactamente la misma arquitectura y la misma
cantidad de parámetros. Es el mayor salto individual de todo el trabajo.*

*Y quiero contrastarlo con el otro aumento de datos que probé, que fue la rotación leve.
La rotación restó las tres veces que la apliqué. La explicación es que Fashion-MNIST ya
viene centrado y alineado, entonces rotar no agrega información nueva, sólo mete bordes
interpolados. Ocluir, en cambio, cambia cualitativamente la tarea. La moraleja es que el
aumento de datos no es gratis ni siempre positivo: tiene que atacar una variación que
realmente exista en el problema.*

---

## Lámina 9 — Quién sobreajusta y quién aguanta · ~1 min

*Acá comparo las curvas de los dos modelos que nos interesan: en naranja la AlexNet, en
verde la Wide ResNet con Random Erasing. La línea llena es validación y la punteada
entrenamiento.*

*La AlexNet toca su mínimo de validación en la época 6, con una pérdida de 0.2499, y a
partir de ahí se da vuelta: la pérdida de validación sube hasta 0.3254 mientras la de
entrenamiento sigue bajando hasta 0.139. Eso es sobreajuste de manual —la red dejó de
aprender el problema y empezó a memorizar el conjunto de entrenamiento—. La parada temprana
la corta en la época 11.*

*La Wide ResNet con Random Erasing se comporta distinto. Arranca más ruidosa, porque con la
mitad de las imágenes ocluidas la validación salta bastante, pero sostiene la mejora hasta
la época 14, donde toca 0.2123, que es la pérdida más baja de todo el trabajo. Y después se
queda en una meseta entre 0.212 y 0.252, sin tendencia clara a subir. Acá la parada temprana
corta por falta de mejora, no por deterioro. Es una diferencia importante.*

---

## Lámina 10 — Costo contra desempeño · ~50 s

*Estos dos gráficos ponen los ocho modelos contra lo que cuestan: a la izquierda contra la
cantidad de parámetros, a la derecha contra el tiempo de entrenamiento, los dos en escala
logarítmica. Abajo tienen la referencia de qué modelo es cada punto.*

*Lo que quiero mostrar es que no forman una recta. La AlexNet de 877 mil parámetros con
Dropout le gana a la de 3,3 millones sin Dropout. Y la Wide ResNet sin Random Erasing, con
36 millones de parámetros y 22 minutos de entrenamiento, le saca apenas 0,45 puntos a esa
AlexNet que entrena en 48 segundos.*

*O sea que el salto real de la Wide ResNet no viene de su tamaño: aparece recién cuando se
la combina con el aumento de datos adecuado. La capacidad sin regularización se gasta en
memorizar; con la regularización correcta, esa misma capacidad se convierte en 1,4 puntos.*

---

## Lámina 11 — Dónde falla · ~1 min

*Este mapa de calor tiene el recall por clase de cada uno de los ocho modelos. Cada fila es
un experimento, cada columna una categoría, y cuanto más oscuro mejor.*

*Y hay una columna que canta: la de Shirt, camisa. El recall promedio entre los ocho modelos
es 0.678, cuando pantalón, bolso y botineta están en 0.98 o 0.99. El rango va de 0.609 en el
peor modelo a 0.791 en el mejor.*

*Lo que me parece más significativo es que ningún cambio de arquitectura la arregla. Podés
pasar de 44 mil a 36 millones de parámetros y la camisa sigue siendo el agujero. El único
cambio que mueve la aguja de verdad es el aumento de datos por oclusión: sube el recall de
camisa de 0.654 a 0.791 y baja las confusiones con remera de 160 a 91 casos.*

*Y tiene todo el sentido: distinguir una camisa de una remera depende de los botones, del
cuello, de los puños. Son detalles locales que a 28 por 28 en escala de grises casi no
sobreviven. Forzar a la red a clasificar con parte de la prenda tapada es exactamente lo que
la obliga a exprimir todas las pistas disponibles.*

---

## Lámina 12 — Conclusiones · ~1 min 15 s

*Cinco conclusiones.*

*Primero: la profundidad convolucional compra exactitud, el ancho de las capas densas no.
Pasar de LeNet a AlexNet, o sea agregar capas convolucionales, rinde 1,8 puntos. Duplicar
los parámetros de la LeNet ensanchando solamente el Flatten rinde 0,17.*

*Segundo: la regularización rinde más que la capacidad. Con 3,8 veces menos parámetros, la
AlexNet con Dropout le gana por tres cuartos de punto a la versión ancha sin él.*

*Tercero: la arquitectura residual escala donde la clásica no. La Wide ResNet sostiene 36
millones y medio de parámetros sin degradarse, gracias a los atajos, al BatchNorm y al
Global Average Pooling.*

*Cuarto, y para mí el más interesante: el aumento de datos correcto vale más que la
arquitectura. Random Erasing es el mayor salto individual de todo el trabajo, y no tocó ni
un solo parámetro del modelo.*

*Y quinto: hay un techo que es del dataset y no del modelo. Cuatro clases de ropa de torso
comparten silueta a 28 por 28 píxeles. Ahí se queda el error que ningún experimento logró
eliminar.*

*Los dos números finales: 0.9124 con la AlexNet adaptada, que fue la mejor de las
arquitecturas vistas en clase, y 0.9305 con la Wide ResNet más Random Erasing, que fue la
mejor global.*

---

## Lámina 13 — Cierre · ~10 s

*Eso es todo. Muchas gracias, y quedo a disposición para las preguntas.*

---

## Preguntas probables y cómo responderlas

**¿Por qué usaste Adam y no SGD con momentum, que es lo que usa el paper de Wide ResNet?**
Justamente para que la comparación entre arquitecturas fuera limpia. Si le cambiaba el
optimizador solo a la Wide ResNet, después no podía saber si la mejora venía de la
arquitectura o del optimizador. Con SGD con momentum y un schedule de learning rate, el WRN
probablemente daría bastante más.

**¿Por qué 20 épocas y no más?**
Porque era el presupuesto fijado para todos por igual. Y de hecho se nota: cinco de los
ocho experimentos cortaron por parada temprana antes de llegar a 20, así que para esos el
límite no fue el problema. El único que claramente quedó cortado antes de converger fue el
WRN con Random Erasing, que llegó a la época 19 todavía mejorando.

**¿No es sospechoso que el modelo con Random Erasing tenga el gap train-validación más
grande, de 0.0356?**
Es una buena observación, pero no indica sobreajuste dañino. El gap crece porque durante el
entrenamiento la tarea es más difícil que la que se evalúa: la mitad de las imágenes están
ocluidas. La prueba es que su pérdida de validación es la más baja de las ocho y su accuracy
de test la más alta. Lo que hay que mirar no es el gap sino la forma de la curva de
validación.

**¿Cuánto tardó entrenar todo?**
Los seis modelos chicos, unos tres minutos y medio en total sobre una RTX 5060 Ti. Los dos
Wide ResNet, unos 64 minutos entre los dos: 132 segundos por época.

**¿Por qué el WRN tiene 28 capas si el paper habla de bloques?**
La fórmula es n = (profundidad − 4) / 6, que para 28 da 4 bloques residuales por grupo. Los
4 que se restan son la convolución inicial, la capa de salida y las proyecciones. Con tres
grupos de 4 bloques y 2 convoluciones por bloque llegás a las 28 capas convolucionales.

**¿Se puede mejorar todavía más?**
Sí, y hay varios caminos claros: entrenar más épocas con un schedule de learning rate,
combinar Random Erasing con volteo horizontal —que en ropa sí es una variación realista, a
diferencia de la rotación—, o usar un ensamble de varios modelos. El estado del arte en
Fashion-MNIST anda alrededor del 96 %.
