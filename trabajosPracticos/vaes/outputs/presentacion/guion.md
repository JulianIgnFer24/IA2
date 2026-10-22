# Guion del orador

> Generado por `code/presentacion/make_guion.py` desde la misma fuente que la presentación.
> Si cambia una diapositiva, volver a correrlo en vez de editar este archivo a mano.

**30 diapositivas** · 3.151 palabras · **~25 minutos** a 125 palabras por minuto, pausas incluidas.

## Ritmo

| # | Diapositiva | Dura | Acumulado |
|---:|---|---:|---:|
| 1 | Generando caras de Los Simpson | 59 s | 0:59 |
| 2 | Tres actos | 40 s | 1:38 |
| 3 | 9.877 caras, cero etiquetas | 39 s | 2:17 |
| 4 | De 12.288 números a 128 | 49 s | 3:06 |
| 5 | Encoder, espacio latente, decoder | 49 s | 3:55 |
| 6 | El objetivo: reconstruir lo mejor posible | 58 s | 4:53 |
| 7 | Encoder y decoder espejados | 30 s | 5:24 |
| 8 | Y aprende | 36 s | 6:00 |
| 9 | La idea tentadora | 37 s | 6:37 |
| 10 | La moraleja del acto | 13 s | 6:51 |
| 11 | Por qué falla: islas y huecos | 72 s | 8:03 |
| 12 | La divergencia KL | 4 s | 8:07 |
| 13 | Codificar una nube, no un punto | 62 s | 9:09 |
| 14 | El resorte: la divergencia KL | 64 s | 10:13 |
| 15 | La nube de una cara | 67 s | 11:20 |
| 16 | ¿Y si agrandamos la nube? | 63 s | 12:23 |
| 17 | Esta vez la pérdida sí informa | 69 s | 13:32 |
| 18 | Reconstrucción: época 30 contra 200 | 47 s | 14:19 |
| 19 | El pago: caras que no existen | 46 s | 15:05 |
| 20 | Caminar entre dos caras | 43 s | 15:48 |
| 21 | Un solo modelo, los dos estilos | 4 s | 15:52 |
| 22 | Cada cara viene en dos versiones | 40 s | 16:33 |
| 23 | Un VAE con 19.754 imágenes | 45 s | 17:17 |
| 24 | «Simplificar» es una dirección | 107 s | 19:04 |
| 25 | Caminar de un estilo al otro | 36 s | 19:39 |
| 26 | ¿Y qué estilo inventa? | 58 s | 20:37 |
| 27 | Cómo medimos en qué estilo dibuja | 115 s | 22:32 |
| 28 | Lo que aprendimos | 47 s | 23:19 |
| 29 | Por qué sale borroso | 83 s | 24:41 |
| 30 | ¿Preguntas? | 31 s | 25:12 |


---

# Apertura

## 1. Generando caras de Los Simpson
<small>Trabajo práctico · Inteligencia Artificial 2 · ~59 s</small>

Buenos días. Vamos a ver cómo una red aprende a inventar caras de Los Simpson que no existen, partiendo de 9.877 fotogramas de la serie sin ninguna etiqueta. La charla está armada como una historia en tres actos, y cada acto arranca con el pizarrón de Bart escribiendo lo que aprendimos. Acto uno: construimos un autoencoder común y descubrimos que no sirve para generar. Vamos a llegar a caras inventadas que no están en el dataset, pero recién en el acto dos: hoy empezamos por el principio. Acto dos: lo arreglamos con un solo término en la función de pérdida. Acto tres: entrenamos un único modelo con los dos tipos de imagen que trae el dataset, y aparece algo que no esperábamos.

## 2. Tres actos
<small>La historia · ~40 s</small>

Acto uno, el caso base: un autoencoder comprime y descomprime. Funciona para reconstruir, y falla para generar. Vamos a ver por qué falla, que es lo más importante de la charla. Acto dos, la mejora: agregamos la divergencia KL. Con eso el espacio latente se ordena y el modelo se vuelve generativo. Acto tres, todos los datos: el dataset trae cada cara en dos versiones. Entrenamos un solo modelo con las dos juntas y aparece un resultado que no estaba en el plan.

## 3. 9.877 caras, cero etiquetas
<small>El dataset · ~39 s</small>

Son 9.877 rostros recortados de capturas de la serie, de 200x200 píxeles en RGB, unos 400 MB. Lo bajamos de Kaggle, del dataset «Simpsons Faces» del usuario kostastokis. El punto importante: no hay etiquetas. No sabemos qué personaje es cada cara. Esto es aprendizaje no supervisado puro. Y alcanza, porque en un autoencoder el dato es su propia etiqueta: la respuesta correcta para cada cara es esa misma cara. En la grilla se ve la variedad: personajes distintos, poses, fondos, encuadres.

## 4. De 12.288 números a 128
<small>El dataset · ~49 s</small>

Reducimos de 200×200 a 64×64 con interpolación LANCZOS, que preserva los bordes mejor que una bilineal. Perdemos las líneas finas del contorno, como se ve en la figura, pero entra cómodo en memoria y entrena en minutos. Los píxeles quedan en [0, 1], algo que va a importar para la función de pérdida. Y el número clave: cada imagen son 12.288 valores y los comprimimos a 128. 96 veces menos. Casi cualquier combinación de 12.288 valores es ruido: las caras ocupan un rincón muy chico de ese espacio, y la apuesta es que 128 números alcanzan para describirlo. El split es 95/5.


---

# Acto 1

## 5. Encoder, espacio latente, decoder
<small>Acto 1 · El caso base · ~49 s</small>

Dos mitades. El encoder toma la imagen y la comprime a un vector de 128 números: eso es el espacio latente. El decoder hace el camino inverso. El truco está en el cuello de botella. Como 128 números no alcanzan para guardar 12.288, la red está obligada a quedarse con lo esencial: la forma de la cabeza, el color, dónde están los ojos. Lo que no pasa por el cuello, no se recupera. Y nadie le dice qué guardar. Esas 128 dimensiones las organiza el entrenamiento solo. En el mapa de calor del medio está el vector latente real de esa cara.

## 6. El objetivo: reconstruir lo mejor posible
<small>Acto 1 · El caso base · ~58 s</small>

¿Cómo medimos si la reconstrucción es buena? Con entropía cruzada binaria, píxel a píxel. ¿Por qué BCE y no error cuadrático? Porque la última capa del decoder es una sigmoide, que devuelve valores entre 0 y 1, y los píxeles de entrada también están en [0, 1]. Tratamos cada píxel como una probabilidad. Formalmente, suponemos que cada píxel sigue una Bernoulli. La sumamos sobre los 12.288 valores de cada imagen, así que los números que vamos a ver son del orden de los 6.000 nats por imagen. Un detalle de implementación: por esto mismo no se puede usar precisión mixta, la BCE es inestable en float16. Todo corre en float32. Y la idea más linda: el dato es su propia etiqueta.

## 7. Encoder y decoder espejados
<small>Acto 1 · El caso base · ~30 s</small>

El encoder son cuatro convoluciones con stride 2: cada una divide la resolución a la mitad y duplica los canales. De 64×64×3 a 4×4×256, que aplanados son 4.096, y una lineal baja a 128. El decoder es el espejo exacto. Cada bloque lleva BatchNorm; el encoder usa LeakyReLU y el decoder ReLU, salvo la última capa, que usa sigmoide. En total 2.434.243 parámetros.

## 8. Y aprende
<small>Acto 1 · El caso base · ~36 s</small>

Entrenamos 200 épocas con Adam, solo con la pérdida de reconstrucción. Funciona: la fila de abajo son caras que el modelo nunca vio, del conjunto de validación. Recupera la silueta del pelo, el color de piel, la posición de los ojos, el fondo. La pérdida final de reconstrucción es 6.327,85 nats por imagen. Lo que se pierde es el detalle fino: las líneas negras del contorno desaparecen. Hasta acá, todo bien. El autoencoder comprime y descomprime.

## 9. La idea tentadora
<small>Acto 1 · El caso base · ~37 s</small>

Si el decoder sabe convertir vectores de 128 números en caras, ¿por qué no le damos un vector cualquiera, sorteado de una normal estándar? Tendríamos un generador gratis. Esto que ven son 16 vectores sorteados de N(0, I) pasados por el decoder. No son caras: es la misma mancha verdosa dieciséis veces. Fíjense en ese detalle, porque es la pista: no solo falla, falla devolviendo siempre lo mismo. Eso ya dice algo sobre dónde cayeron esos dieciséis vectores.

## 10. La moraleja del acto
<small>Acto 1 · El caso base · ~13 s</small>

Entonces: el autoencoder comprime y descomprime bien, pero no puede inventar una cara. Dejemos eso escrito antes de pasar a la pregunta que importa, que es por qué.

## 11. Por qué falla: islas y huecos
<small>Acto 1 · El caso base · ~72 s</small>

Esta es la diapositiva más importante de la charla. Proyectamos con PCA los vectores latentes de todas las caras de validación, en azul, y encima 500 muestras de N(0, I) en naranja. Panel izquierdo, el autoencoder: las caras están desparramadas en un rango de −58 a 54, mientras que las muestras de la normal se concentran en un puntito en el origen. Miren los números de los ejes: difieren en un factor de diez. Ahí está la explicación de la mancha repetida. Los dieciséis vectores caen todos en la misma región minúscula, donde el autoencoder nunca puso ninguna cara. El decoder nunca aprendió qué hacer ahí. ¿Y por qué quedó así? Porque nunca se lo pedimos. La pérdida solo premia reconstruir; en ningún término dice «y que el espacio quede ordenado». Panel derecho, adelanto del acto dos: rango de −4 a 6, y el prior se superpone con los datos.


---

# Acto 2

## 12. La divergencia KL
<small>Acto 2 · La mejora · ~4 s</small>

Segundo acto. Arreglamos el problema con un solo término.

## 13. Codificar una nube, no un punto
<small>Acto 2 · La mejora · ~62 s</small>

La solución: en vez de mandar cada cara a un punto, el encoder devuelve una nube. Dos vectores, μ y log σ², que definen una gaussiana por cada imagen. ¿Por qué ayuda? Porque el punto que le llega al decoder cambia en cada pasada. El decoder tiene que reconstruir bien desde cualquier lugar de la nube, así que aprende regiones en vez de puntos sueltos. Las nubes vecinas se pisan y los huecos se llenan con mezclas. Respecto del autoencoder cambian dos cosas: dos cabezas en vez de una, y un sorteo en el medio. El resto son los mismos bloques. Predecimos log σ² y no σ para que la red pueda devolver cualquier real y la exponencial garantice varianza positiva. 2.958.659 parámetros, apenas 524.416 más que el autoencoder.

## 14. El resorte: la divergencia KL
<small>Acto 2 · La mejora · ~64 s</small>

Pero codificar nubes no alcanza, porque la red tiene dos trampas: achicar las nubes hasta que vuelvan a ser puntos, o separarlas tanto que no se toquen. En los dos casos el ruido deja de molestar y volvemos a tener un autoencoder. Por eso agregamos la KL contra N(0, I). Y lo lindo es que la fórmula tiene exactamente dos términos, uno contra cada trampa. μ² es un resorte hacia el origen: impide separar las nubes. σ² − log σ² − 1 vale cero justo en σ = 1 y crece si la nube se achica o se agranda: impide achicarlas. Entre los dos obligan a que las nubes queden centradas, de tamaño razonable y superpuestas. Los huecos se llenan. La pérdida total es la suma. Son dos fuerzas que tiran en direcciones opuestas.

## 15. La nube de una cara
<small>Acto 2 · La mejora · ~67 s</small>

Qué significa concretamente que una cara sea una nube. Arriba a la izquierda, la original y su reconstrucción usando μ sin ruido: el centro de la nube. Las siete de la derecha son muestras de esa misma nube. Y acá hay que ser honesto: a simple vista parecen idénticas. No es un error. La nube es angosta, σ medio 0,38, así que las muestras difieren milésimas por píxel. Por eso la fila de abajo: la diferencia contra decode(μ) multiplicada por seis. Ahí aparece el dato interesante: la variación está concentrada en los contornos —el borde de la cabeza, los ojos, la boca— y en el fondo. Las zonas planas son negras, idénticas. La incertidumbre del modelo es sobre dónde cae exactamente cada borde. Y esa es la razón de que las imágenes salgan borrosas, algo que retomamos al final.

## 16. ¿Y si agrandamos la nube?
<small>Acto 2 · La mejora · ~63 s</small>

Para medir el tamaño de la nube la agrandamos a propósito, multiplicando σ por un factor k. Cada columna usa el mismo ε, así que lo único que cambia entre filas es el factor. Con k = 0 no hay ruido. Con k = 1, 2 y hasta 3 sigue siendo la misma cara: puse la diferencia medida por píxel al lado de cada fila para que se vea que crece, de 0,023 a 0,079, pero el ojo casi no la registra. Recién en k = 6 la cara se desarma. Cada cara ocupa una región chica y bien definida. Agregué la fila de k = 6 justamente porque con las cuatro primeras la conclusión no se veía. Es más honesto mostrar dónde se rompe que afirmar que se rompe antes.

## 17. Esta vez la pérdida sí informa
<small>Acto 2 · La mejora · ~69 s</small>

Adam con lr 0,001, batch 128, 200 épocas, en una NVIDIA GeForce RTX 5060 Ti. Tardó 12,6 minutos. El panel izquierdo es la reconstrucción; en verde la del autoencoder común, que termina más abajo — no paga el costo de la regularización. El del medio es la KL, el panel de diagnóstico. Sube de 50 a unos 144,49 nats y se estabiliza. Si hubiera caído a cero tendríamos «posterior collapse»: el decoder ignoraría z. No pasó, y lo confirmé dimensión por dimensión: 101 de las 128 tienen σ por debajo de 0,5. Solo 3 quedaron pegadas al prior. El derecho es la suma. La punteada marca la época 62, el mínimo de validación; de ahí en adelante el train sigue bajando y la validación no: sobreajuste leve. A diferencia de una GAN, hay una sola pérdida que baja de forma monótona y que informa.

## 18. Reconstrucción: época 30 contra 200
<small>Acto 2 · La mejora · ~47 s</small>

Arriba las originales, en el medio el VAE a la época 30, abajo a la 200. A la 30 ya se reconocen los personajes; la validación era 6.681,14 nats. Y algo que conviene señalar antes de que lo pregunten: a la 200 la validación queda en 6.682,84, prácticamente igual, de hecho un poquito peor. Y sin embargo las imágenes se ven mejor: ojos más definidos, fondo más limpio. No es contradicción. La pérdida promedia 12.288 valores, así que una mejora concentrada en los bordes casi no la mueve. La métrica y la calidad percibida no son lo mismo.

## 19. El pago: caras que no existen
<small>Acto 2 · La mejora · ~46 s</small>

Y acá está la respuesta a la pregunta del acto uno. A la izquierda, el autoencoder decodificando vectores aleatorios: la mancha de siempre. A la derecha, el VAE decodificando exactamente los mismos vectores. Son caras. Borrosas, pero con estructura. Y ninguna está en el dataset. Vale subrayarlo: la arquitectura es casi idéntica, los datos son los mismos, el optimizador es el mismo, los vectores son literalmente los mismos. Lo único que cambia es que el encoder devuelve una distribución y que la pérdida tiene el término KL. Eso solo convierte un compresor en un generador.

## 20. Caminar entre dos caras
<small>Acto 2 · La mejora · ~43 s</small>

Última evidencia del acto, y la más convincente. Tomamos dos caras de validación, las codificamos, y recorremos la recta que las une en el latente, con diez pasos. Si el espacio tuviera huecos, en algún punto del camino caeríamos en una zona muerta. Y no pasa: toda la transición son caras. El pelo se transforma gradualmente, el fondo cambia de color de forma continua, la pose rota de a poco. La región entre dos caras conocidas también está poblada de caras. Es la continuidad que pedíamos en el acto uno.


---

# Acto 3

## 21. Un solo modelo, los dos estilos
<small>Acto 3 · Todos los datos · ~4 s</small>

Tercer acto. Hasta acá usamos la mitad del dataset.

## 22. Cada cara viene en dos versiones
<small>Acto 3 · Todos los datos · ~40 s</small>

El archivo de Kaggle trae dos carpetas con las mismas 9.877 caras. Arriba «cropped»: el fotograma real, con sombreados, texturas y fondos. Unos 9.500 colores distintos por imagen. Abajo «simplified»: la misma cara con colores planos. Tres colores únicos por imagen. Hasta ahora entrenamos solo con cropped. Ahora entrenamos UN modelo con las dos juntas: 19.754 imágenes. La pregunta no es cuál anda mejor. Es qué hace el modelo cuando tiene que acomodar las dos cosas en un mismo espacio latente de 128 dimensiones.

## 23. Un VAE con 19.754 imágenes
<small>Acto 3 · Todos los datos · ~45 s</small>

Mismo modelo, mismos hiperparámetros, 200 épocas. Lo único que cambia es que ve las dos carpetas, y el split mezcla los dos estilos en lugar de dejar uno entero de un lado. Tardó 17,9 minutos, casi el doble, porque son el doble de imágenes. Los números quedan donde uno esperaría: entre los dos modelos individuales. La pérdida total 6.090,26, contra 6.576,28 del de cropped y 5.559,30 del de simplified. Eso no es interesante en sí mismo. Lo interesante es lo que hizo con el espacio latente, que es lo que vemos a continuación.

## 24. «Simplificar» es una dirección
<small>Acto 3 · Todos los datos · ~107 s</small>

Esta es la diapositiva que justifica el acto entero. Lo primero que miramos fue si el modelo había armado dos nubes separadas, una por estilo. No: los centroides están a 2,42 con una dispersión interna de 9,55, o sea completamente entremezcladas. El modelo organizó el latente por qué cara es, no por en qué estilo está dibujada. Entonces, ¿dónde quedó el estilo? Probamos esto: tomamos 1.500 pares y calculamos v, el promedio de μ de la versión simplificada menos μ de la versión cropped. Si el estilo fuera ruido, esas restas apuntarían para cualquier lado y el promedio daría casi cero. Pero el 100 por ciento de los pares tiene coseno positivo con esa dirección. Todos. La dirección es consistente. Y sirve. En la figura, la primera columna es una cara cropped, la segunda su reconstrucción, y después le vamos sumando α por v. En α = 1 el resultado se parece mucho a su versión simplificada real, que está en la última columna. El detalle importante: estas cinco caras NO entraron en el cálculo de v. Y el modelo nunca supo que las imágenes venían en pares — la correspondencia la dedujo solo, nada más que por tener que acomodar los dos estilos en el mismo espacio. Esto es aritmética en el espacio latente: sumar un vector constante aplica un concepto.

## 25. Caminar de un estilo al otro
<small>Acto 3 · Todos los datos · ~36 s</small>

Otra forma de verlo: interpolar entre una cara y su propia versión simplificada. La transición es suave y, sobre todo, la identidad se mantiene todo el camino. Es siempre la misma cara, en la misma pose, cambiando de estilo. Eso confirma lo de la diapositiva anterior desde otro ángulo: identidad y estilo quedaron codificados de forma separable. El modelo puede mover uno dejando el otro quieto. Y lo hizo sin supervisión de ningún tipo.

## 26. ¿Y qué estilo inventa?
<small>Acto 3 · Todos los datos · ~58 s</small>

Última pregunta del acto: cuando le pedimos una cara nueva, ¿en qué estilo la dibuja? Las dos cosas, mitad y mitad. Proyectando sobre la dirección de estilo, las caras reales caen en -1,09 las cropped y 1,33 las simplified; las generadas caen en 0,00, repartidas 46 / 54. En la grilla, ordenada por textura, se ve el recorrido completo: a la izquierda caras de color plano con fondo celeste, a la derecha caras con fondo texturado, y en el medio híbridos que no existen en ninguna de las dos carpetas. Eso es lo que uno querría de un modelo generativo entrenado con dos fuentes: no eligió una y descartó la otra, cubrió las dos y lo que hay entre medio.

## 27. Cómo medimos en qué estilo dibuja
<small>Acto 3 · Todos los datos · ~115 s</small>

Vale la pena contar cómo se mide esto, porque el primer intento estuvo mal y el error es instructivo. Primer intento: la textura por píxel, o sea cuánto difiere cada píxel de sus vecinos. Las simplified reales dan 0,0218 y las cropped 0,0520: bien separadas. Las generadas daban 0,0274, del lado de simplified, y la conclusión salía «100 % simplified». Era falso. El desenfoque del VAE baja la textura exactamente igual que la simplificación, así que una cara cropped borrosa se confunde con una simplified nítida. La medida no estaba midiendo estilo: estaba midiendo desenfoque. Lo correcto es medirlo donde el modelo efectivamente codificó el estilo, que es el latente. El procedimiento es el del gráfico, en tres pasos. Uno: tomamos la dirección de estilo v de la diapositiva anterior y la normalizamos. Dos: proyectamos sobre ella los μ de las caras reales. Las cropped caen en -1,09 y las simplified en 1,33: son las dos campanas del gráfico. Que estén bien separadas es lo que valida que la dirección sirve para distinguir estilos. Tres: proyectamos los z sorteados del prior —la línea negra— y vemos de qué lado del punto medio cae cada uno. Resultado: 46 por ciento del lado simplified, 54 del lado cropped. Y eso sí coincide con lo que se ve en la grilla. La moraleja vale para todo el trabajo: cuando una métrica y los ojos no coinciden, lo primero a revisar es la métrica.


---

# Cierre

## 28. Lo que aprendimos
<small>Cierre · ~47 s</small>

Cierre. El pizarrón resume los cinco puntos del trabajo. Uno: un autoencoder comprime y descomprime, y el dato es su propia etiqueta. Dos: un autoencoder no genera, porque su latente tiene islas y huecos. Tres: la KL ata las nubes a N(0, I) y eso vuelve el latente continuo y completo. Cuatro: la reparametrización z = μ + σ·ε permite entrenarlo con gradiente. Cinco: con los dos estilos juntos, simplificar resultó ser una dirección del espacio, aprendida sin que nadie le dijera que las imágenes venían en pares. Queda una limitación, que es la que vemos ahora.

## 29. Por qué sale borroso
<small>Cierre · ~83 s</small>

La limitación honesta. Las caras son borrosas, y no es un bug. La BCE penaliza cada píxel por separado. Cuando el modelo no está seguro de dónde cae un borde, lo que minimiza la pérdida es promediar todas las posiciones posibles en vez de jugarse por una. El promedio de muchas caras es, inevitablemente, una cara borrosa. Lo confirmamos dos veces: en los mapas de diferencia del acto dos, donde la variación estaba toda en los contornos, y en que la métrica no mejora aunque los bordes sí. Una GAN no promedia, porque el discriminador castiga cualquier cosa que no parezca real, y una cara promedio no parece real. Por eso da más nítido, a cambio de un entrenamiento inestable y riesgo de mode collapse. Abajo vuelvo a poner la figura del acto dos, porque es la prueba más directa: la variación del modelo está concentrada en los contornos, que es exactamente donde no sabe dónde cae el borde. Ahí es donde promedia, y por eso es ahí donde se ve borroso.

## 30. ¿Preguntas?
<small>Cierre · ~31 s</small>

Gracias. Dejo el sofá vacío porque, como en la serie, nunca se sabe quién se va a sentar. Si hay preguntas, arranco por la que siempre aparece: por qué no usamos una GAN. La respuesta corta está en la diapositiva anterior; la larga es que el VAE nos daba un encoder, que es lo que permitió todo el análisis del espacio latente del acto tres.
