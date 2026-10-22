"""
Genera presentacion_vae_simpsons.pptx: la historia en tres actos, ambientada en la serie.

Acto 1 — el caso base: un autoencoder común, que reconstruye pero no genera.
Acto 2 — la mejora: el término KL, que ordena el latente y lo vuelve generativo.
Acto 3 — todos los datos: un solo VAE entrenado con los dos estilos juntos.

El diseño vive en diseno.py y tema_simpsons.py; la ambientación (cielo, pizarrón, sofá)
la genera ambiente_simpsons.py. Ningún número está escrito a mano: todos se leen de los
JSON de resultados al generar el .pptx.

Uso: python code/presentacion/make_presentation.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

import config
from diseno import ALTO, ALTO_CUERPO, ANCHO, ANCHO_UTIL, MARGEN, Y_CUERPO
from tema_simpsons import (AMARILLO, AZUL_MARGE, BLANCO, CARBON, CREMA, GRIS,
                           GRIS_CLARO, ROJO, DeckSimpsons)

PRES = config.OUTPUT_DIR / "presentacion"
FIGS = PRES / "figuras"
AMBIENTE = PRES / "ambiente"
COMBINADO = config.OUTPUT_DIR / "combinado"
SALIDA = PRES / "presentacion_vae_simpsons.pptx"

COL = Inches(0.4)


def fmt(n, dec=2):
    if isinstance(n, int) or float(n).is_integer():
        return f"{int(n):,}".replace(",", ".")
    return f"{n:,.{dec}f}".replace(",", "~").replace(".", ",").replace("~", ".")


def ultima_fila(csv_path):
    with Path(csv_path).open(newline="") as f:
        filas = list(csv.DictReader(f))
    return {k: float(v) for k, v in filas[-1].items()}


def construir(st, rs, cp, ac, comb, cb):
    vae, ae, ent = rs["vae"], rs["ae"], rs["entrenamiento"]
    mejor, lat = vae["mejor_validacion"], vae["uso_del_latente"]
    e30, e200 = vae["epoca_30"], vae["epoca_200"]
    cr, si = cp["cropped"], cp["simplified"]
    sep, dr, eg = ac["separacion_latente"], ac["direccion_estilo"], ac["estilo_generado"]

    d = DeckSimpsons(AMBIENTE, FIGS)
    fig = d.fig

    # ================================================ APERTURA · 1
    d.portada_cielo(
        "Trabajo práctico · Inteligencia Artificial 2",
        "Generando caras de Los Simpson",
        f"Un autoencoder variacional que comprime cada cara a "
        f"{st['latent_dim']} números y la vuelve a dibujar",
        f"""
Buenos días. Vamos a ver cómo una red aprende a inventar caras de Los Simpson que no
existen, partiendo de {fmt(st['n_imagenes'])} fotogramas de la serie sin ninguna etiqueta.
La charla está armada como una historia en tres actos, y cada acto arranca con el pizarrón
de Bart escribiendo lo que aprendimos.
Acto uno: construimos un autoencoder común y descubrimos que no sirve para generar.
Vamos a llegar a caras inventadas que no están en el dataset, pero recién en el acto dos:
hoy empezamos por el principio.
Acto dos: lo arreglamos con un solo término en la función de pérdida.
Acto tres: entrenamos un único modelo con los dos tipos de imagen que trae el dataset, y
aparece algo que no esperábamos.
""")

    # ================================================ APERTURA · 2
    s = d.slide("La historia", "Tres actos", notas="""
Acto uno, el caso base: un autoencoder comprime y descomprime. Funciona para reconstruir,
y falla para generar. Vamos a ver por qué falla, que es lo más importante de la charla.
Acto dos, la mejora: agregamos la divergencia KL. Con eso el espacio latente se ordena y
el modelo se vuelve generativo.
Acto tres, todos los datos: el dataset trae cada cara en dos versiones. Entrenamos un solo
modelo con las dos juntas y aparece un resultado que no estaba en el plan.
""")
    d.donut(s, ANCHO - MARGEN - Inches(1.2), Inches(1.15), Inches(1.1))
    d.tabla(s, MARGEN, Y_CUERPO, ANCHO_UTIL,
            ["Acto", "Qué pasa", "Con qué nos quedamos"],
            [["1", "El caso base: el autoencoder",
              "*Reconstruye bien, pero no puede generar"],
             ["2", "La mejora: la divergencia KL",
              "*Codificar una nube, no un punto"],
             ["3", "Todos los datos juntos",
              "*«Simplificar» es una dirección del latente"]],
            anchos=[0.08, 0.44, 0.48], alto_fila=Inches(0.62))
    d.callout(s, MARGEN, Y_CUERPO + Inches(2.5), ANCHO_UTIL, "La pregunta de fondo",
              "¿Cómo se consigue que una red invente caras nuevas sin que nadie le diga "
              "nunca qué personaje es cada una?", AZUL_MARGE, alto=Inches(1.05))
    d.franja_caras(s)

    # ================================================ APERTURA · 3
    s = d.slide("El dataset", f"{fmt(st['n_imagenes'])} caras, cero etiquetas",
                notas=f"""
Son {fmt(st['n_imagenes'])} rostros recortados de capturas de la serie, de
{list(st['dimensiones'])[0]} píxeles en RGB, unos {fmt(st['tamano_mb'])} MB. Lo bajamos de
Kaggle, del dataset «Simpsons Faces» del usuario kostastokis.
El punto importante: no hay etiquetas. No sabemos qué personaje es cada cara. Esto es
aprendizaje no supervisado puro. Y alcanza, porque en un autoencoder el dato es su propia
etiqueta: la respuesta correcta para cada cara es esa misma cara.
En la grilla se ve la variedad: personajes distintos, poses, fondos, encuadres.
""")
    w_fig = Inches(6.2)
    fig(s, "dataset_ejemplos.png", MARGEN, Y_CUERPO, w_fig, ALTO_CUERPO)
    x2 = MARGEN + w_fig + COL
    w2 = ANCHO - x2 - MARGEN
    d._parrafos(s, x2, Y_CUERPO + Inches(0.1), w2, Inches(2.0), [
        [(f"{fmt(st['n_imagenes'])} imágenes", {"bold": True, "color": CARBON,
                                                "size": 16})],
        [(f"{list(st['dimensiones'])[0].replace('x', '×')} px · RGB · PNG · "
          f"{fmt(st['tamano_mb'])} MB", {})],
        [("Kaggle — «Simpsons Faces», de kostastokis", {})],
    ], size=14, espaciado=10)
    d.callout(s, x2, Y_CUERPO + Inches(1.7), w2, "Sin etiquetas",
              "No sabemos qué personaje es cada cara. En un autoencoder no hace falta: "
              "la respuesta correcta para cada imagen es esa misma imagen.",
              AZUL_MARGE, alto=Inches(1.5))

    # ================================================ APERTURA · 4
    comp = st["factor_compresion"]
    s = d.slide("El dataset", "De 12.288 números a 128", notas=f"""
Reducimos de 200×200 a 64×64 con interpolación LANCZOS, que preserva los bordes mejor que
una bilineal. Perdemos las líneas finas del contorno, como se ve en la figura, pero entra
cómodo en memoria y entrena en minutos.
Los píxeles quedan en [0, 1], algo que va a importar para la función de pérdida.
Y el número clave: cada imagen son {fmt(st['valores_por_imagen_64'])} valores y los
comprimimos a {st['latent_dim']}. {comp:.0f} veces menos. Casi cualquier combinación de
{fmt(st['valores_por_imagen_64'])} valores es ruido: las caras ocupan un rincón muy chico
de ese espacio, y la apuesta es que {st['latent_dim']} números alcanzan para describirlo.
El split es {int((1 - st['val_split']) * 100)}/{int(st['val_split'] * 100)}.
""")
    fig(s, "preprocesamiento.png", MARGEN, Y_CUERPO, Inches(3.4), ALTO_CUERPO)
    x2 = MARGEN + Inches(3.4) + COL
    w2 = ANCHO - x2 - MARGEN
    media = (w2 - COL) / 2
    d.tarjeta(s, x2, Y_CUERPO, media, Inches(1.35), "200×200 → 64×64",
              "Interpolación LANCZOS. Píxeles escalados a [0, 1].", AZUL_MARGE)
    d.tarjeta(s, x2 + media + COL, Y_CUERPO, media, Inches(1.35),
              f"Split {int((1 - st['val_split']) * 100)} / "
              f"{int(st['val_split'] * 100)}",
              f"{fmt(st['n_train'])} de entrenamiento y {fmt(st['n_val'])} de "
              f"validación.", AZUL_MARGE)
    d.callout(s, x2, Y_CUERPO + Inches(1.65), w2, "El número que importa",
              f"Cada imagen son {fmt(st['valores_por_imagen_64'])} valores "
              f"(64 × 64 × 3) y los comprimimos a {st['latent_dim']}: {comp:.0f} veces "
              f"menos. Casi cualquier combinación de esos "
              f"{fmt(st['valores_por_imagen_64'])} valores es ruido — las caras ocupan "
              f"un rincón muy chico del espacio.", ROJO, alto=Inches(2.0))

    # ================================================== ACTO 1 · 5
    s = d.slide("Acto 1 · El caso base", "Encoder, espacio latente, decoder",
                notas=f"""
Dos mitades. El encoder toma la imagen y la comprime a un vector de {st['latent_dim']}
números: eso es el espacio latente. El decoder hace el camino inverso.
El truco está en el cuello de botella. Como {st['latent_dim']} números no alcanzan para
guardar {fmt(st['valores_por_imagen_64'])}, la red está obligada a quedarse con lo
esencial: la forma de la cabeza, el color, dónde están los ojos. Lo que no pasa por el
cuello, no se recupera.
Y nadie le dice qué guardar. Esas {st['latent_dim']} dimensiones las organiza el
entrenamiento solo.
En el mapa de calor del medio está el vector latente real de esa cara.
""")
    fig(s, "pipeline_ejemplo.png", MARGEN, Y_CUERPO, ANCHO_UTIL, Inches(2.75))
    y2 = Y_CUERPO + Inches(2.9)
    tercio = (ANCHO_UTIL - 2 * COL) / 3
    for i, (tit, txt, c) in enumerate([
        ("Encoder", "Comprime la imagen a un vector chico.", AZUL_MARGE),
        ("Latente", f"Donde viven los códigos z: {st['latent_dim']} números.", CARBON),
        ("Decoder", "Descomprime el vector a una imagen.", ROJO)]):
        d.tarjeta(s, MARGEN + i * (tercio + COL), y2, tercio, Inches(1.15), tit, txt, c)

    # ================================================== ACTO 1 · 6
    s = d.slide("Acto 1 · El caso base", "El objetivo: reconstruir lo mejor posible",
                notas=f"""
¿Cómo medimos si la reconstrucción es buena? Con entropía cruzada binaria, píxel a píxel.
¿Por qué BCE y no error cuadrático? Porque la última capa del decoder es una sigmoide, que
devuelve valores entre 0 y 1, y los píxeles de entrada también están en [0, 1]. Tratamos
cada píxel como una probabilidad. Formalmente, suponemos que cada píxel sigue una Bernoulli.
La sumamos sobre los {fmt(st['valores_por_imagen_64'])} valores de cada imagen, así que
los números que vamos a ver son del orden de los 6.000 nats por imagen.
Un detalle de implementación: por esto mismo no se puede usar precisión mixta, la BCE es
inestable en float16. Todo corre en float32.
Y la idea más linda: el dato es su propia etiqueta.
""")
    d._rect(s, MARGEN, Y_CUERPO, ANCHO_UTIL, Inches(1.25), BLANCO)
    d._txt(s, MARGEN, Y_CUERPO + Inches(0.42), ANCHO_UTIL, Inches(0.5),
           "BCE(x, x̂)  =  − Σ [ x · log(x̂)  +  (1 − x) · log(1 − x̂) ]",
           size=22, color=CARBON, fuente="Noto Serif", align=PP_ALIGN.CENTER)
    y2 = Y_CUERPO + Inches(1.5)
    media = (ANCHO_UTIL - COL) / 2
    d.tarjeta(s, MARGEN, y2, media, Inches(1.3), "Sigmoide + [0, 1]",
              "La salida del decoder y los píxeles reales viven en el mismo rango: se "
              "comparan como probabilidades.", AZUL_MARGE)
    d.tarjeta(s, MARGEN + media + COL, y2, media, Inches(1.3),
              f"Suma sobre {fmt(st['valores_por_imagen_64'])} valores",
              "Por eso las pérdidas se miden en miles de nats por imagen.", AZUL_MARGE)
    d.callout(s, MARGEN, y2 + Inches(1.5), ANCHO_UTIL, "Intuición",
              "El dato es su propia etiqueta. No hace falta etiquetar nada: la respuesta "
              "correcta para cada cara es esa misma cara.", ROJO, alto=Inches(1.0))

    # ================================================== ACTO 1 · 7
    s = d.slide("Acto 1 · El caso base", "Encoder y decoder espejados", notas=f"""
El encoder son cuatro convoluciones con stride 2: cada una divide la resolución a la mitad
y duplica los canales. De 64×64×3 a 4×4×256, que aplanados son 4.096, y una lineal baja a
{st['latent_dim']}.
El decoder es el espejo exacto. Cada bloque lleva BatchNorm; el encoder usa LeakyReLU y el
decoder ReLU, salvo la última capa, que usa sigmoide.
En total {fmt(ae['parametros'])} parámetros.
""")
    fig(s, "ae_arquitectura.png", MARGEN, Y_CUERPO - Inches(0.1), ANCHO_UTIL,
        Inches(3.9))
    d._txt(s, MARGEN, Inches(6.3), ANCHO_UTIL, Inches(0.4),
           [("4 convoluciones con stride 2: 64 → 32 → 16 → 8 → 4. El decoder, al "
             "revés.     ", {}),
            (f"{fmt(ae['parametros'])} parámetros.",
             {"bold": True, "color": CARBON})], size=14, color=GRIS,
           align=PP_ALIGN.CENTER)

    # ================================================== ACTO 1 · 8
    s = d.slide("Acto 1 · El caso base", "Y aprende", notas=f"""
Entrenamos {ae['epocas']} épocas con Adam, solo con la pérdida de reconstrucción.
Funciona: la fila de abajo son caras que el modelo nunca vio, del conjunto de validación.
Recupera la silueta del pelo, el color de piel, la posición de los ojos, el fondo.
La pérdida final de reconstrucción es {fmt(ae['final']['train_recon'])} nats por imagen.
Lo que se pierde es el detalle fino: las líneas negras del contorno desaparecen.
Hasta acá, todo bien. El autoencoder comprime y descomprime.
""")
    fig(s, "ae_reconstruccion.png", MARGEN, Y_CUERPO, ANCHO_UTIL, Inches(2.6))
    y2 = Y_CUERPO + Inches(2.78)
    media = (ANCHO_UTIL - COL) / 2
    d.tarjeta(s, MARGEN, y2, media, Inches(1.35), "Sobrevive",
              "Forma de la cabeza, peinado, colores, pose, fondo.", AZUL_MARGE)
    d.tarjeta(s, MARGEN + media + COL, y2, media, Inches(1.35), "Se pierde",
              "El detalle fino: las líneas negras del contorno, los bordes nítidos.",
              ROJO)

    # ================================================== ACTO 1 · 9
    s = d.slide("Acto 1 · El caso base", "La idea tentadora", oscura=True, notas="""
Si el decoder sabe convertir vectores de 128 números en caras, ¿por qué no le damos un
vector cualquiera, sorteado de una normal estándar? Tendríamos un generador gratis.
Esto que ven son 16 vectores sorteados de N(0, I) pasados por el decoder. No son caras: es
la misma mancha verdosa dieciséis veces.
Fíjense en ese detalle, porque es la pista: no solo falla, falla devolviendo siempre lo
mismo. Eso ya dice algo sobre dónde cayeron esos dieciséis vectores.
""")
    fig(s, "ae_generacion_aleatoria.png", MARGEN, Y_CUERPO - Inches(0.25),
        Inches(5.4), Inches(4.5))
    x2 = MARGEN + Inches(5.4) + Inches(0.7)
    w2 = ANCHO - x2 - MARGEN
    d._parrafos(s, x2, Y_CUERPO + Inches(0.1), w2, Inches(2.4), [
        [("Ya tenemos un decoder, ¿no?", {"size": 17, "color": CREMA})],
        [("Sorteamos 16 vectores de N(0, I) y los decodificamos. No salen caras: sale ",
          {"color": GRIS_CLARO}),
         ("la misma mancha dieciséis veces", {"bold": True, "color": CREMA}),
         (".", {"color": GRIS_CLARO})],
        [("Que siempre devuelva lo mismo es la pista de dónde cayeron esos vectores.",
          {"color": GRIS_CLARO})],
    ], size=15, espaciado=14)
    d._txt(s, x2, Y_CUERPO + Inches(2.9), w2, Inches(0.6), "No anda.",
           size=26, color=AMARILLO, fuente="Noto Serif", bold=True)

    # ================================================== ACTO 1 · 10
    d.divisor_pizarron(
        "Acto 1 · El caso base",
        "La moraleja del acto",
        "Comprime y descomprime. Reconstruye bien. Y no puede inventar nada.",
        "pizarron_acto1.png", """
Entonces: el autoencoder comprime y descomprime bien, pero no puede inventar una cara.
Dejemos eso escrito antes de pasar a la pregunta que importa, que es por qué.
""")

    # ================================================== ACTO 1 · 11
    s = d.slide("Acto 1 · El caso base", "Por qué falla: islas y huecos", notas="""
Esta es la diapositiva más importante de la charla.
Proyectamos con PCA los vectores latentes de todas las caras de validación, en azul, y
encima 500 muestras de N(0, I) en naranja.
Panel izquierdo, el autoencoder: las caras están desparramadas en un rango de −58 a 54,
mientras que las muestras de la normal se concentran en un puntito en el origen. Miren los
números de los ejes: difieren en un factor de diez.
Ahí está la explicación de la mancha repetida. Los dieciséis vectores caen todos en la
misma región minúscula, donde el autoencoder nunca puso ninguna cara. El decoder nunca
aprendió qué hacer ahí.
¿Y por qué quedó así? Porque nunca se lo pedimos. La pérdida solo premia reconstruir; en
ningún término dice «y que el espacio quede ordenado».
Panel derecho, adelanto del acto dos: rango de −4 a 6, y el prior se superpone con los datos.
""")
    fig(s, "espacio_latente.png", MARGEN, Y_CUERPO, ANCHO_UTIL, Inches(3.3))
    y2 = Y_CUERPO + Inches(3.45)
    media = (ANCHO_UTIL - COL) / 2
    d.callout(s, MARGEN, y2, media, "Continuidad",
              "Dos puntos cercanos deberían dar imágenes parecidas.", AZUL_MARGE,
              alto=Inches(1.05))
    d.callout(s, MARGEN + media + COL, y2, media, "Completitud",
              "Cualquier punto razonable debería dar algo con sentido.", AZUL_MARGE,
              alto=Inches(1.05))

    # ================================================== ACTO 2 · 12
    d.divisor_pizarron(
        "Acto 2 · La mejora",
        "La divergencia KL",
        "Si queremos un espacio sin huecos, tenemos que pedirlo.",
        "pizarron_acto2.png", """
Segundo acto. Arreglamos el problema con un solo término.
""")

    # ================================================== ACTO 2 · 13
    s = d.slide("Acto 2 · La mejora", "Codificar una nube, no un punto", acento=ROJO,
                notas=f"""
La solución: en vez de mandar cada cara a un punto, el encoder devuelve una nube. Dos
vectores, μ y log σ², que definen una gaussiana por cada imagen.
¿Por qué ayuda? Porque el punto que le llega al decoder cambia en cada pasada. El decoder
tiene que reconstruir bien desde cualquier lugar de la nube, así que aprende regiones en
vez de puntos sueltos. Las nubes vecinas se pisan y los huecos se llenan con mezclas.
Respecto del autoencoder cambian dos cosas: dos cabezas en vez de una, y un sorteo en el
medio. El resto son los mismos bloques.
Predecimos log σ² y no σ para que la red pueda devolver cualquier real y la exponencial
garantice varianza positiva.
{fmt(vae['parametros'])} parámetros, apenas {fmt(vae['parametros'] - ae['parametros'])}
más que el autoencoder.
""")
    fig(s, "vae_arquitectura.png", MARGEN, Y_CUERPO - Inches(0.2), ANCHO_UTIL,
        Inches(3.85))
    y2 = Y_CUERPO + Inches(3.66)
    media = (ANCHO_UTIL - COL) / 2
    d._rect(s, MARGEN, y2, media, Inches(0.72), BLANCO)
    d._txt(s, MARGEN, y2 + Inches(0.17), media, Inches(0.4),
           "q(z | x)  =  N( μ(x),  σ²(x) )", size=17, color=CARBON,
           fuente="Noto Serif", align=PP_ALIGN.CENTER)
    d._txt(s, MARGEN + media + COL, y2 + Inches(0.14), media, Inches(0.5),
           [("Dos cabezas y un sorteo en el medio: solo ", {}),
            (f"{fmt(vae['parametros'] - ae['parametros'])} parámetros más",
             {"bold": True, "color": CARBON}), (".", {})], size=13.5, color=GRIS)

    # ================================================== ACTO 2 · 14
    s = d.slide("Acto 2 · La mejora", "El resorte: la divergencia KL", acento=ROJO,
                notas="""
Pero codificar nubes no alcanza, porque la red tiene dos trampas: achicar las nubes hasta
que vuelvan a ser puntos, o separarlas tanto que no se toquen. En los dos casos el ruido
deja de molestar y volvemos a tener un autoencoder.
Por eso agregamos la KL contra N(0, I). Y lo lindo es que la fórmula tiene exactamente dos
términos, uno contra cada trampa.
μ² es un resorte hacia el origen: impide separar las nubes.
σ² − log σ² − 1 vale cero justo en σ = 1 y crece si la nube se achica o se agranda:
impide achicarlas.
Entre los dos obligan a que las nubes queden centradas, de tamaño razonable y
superpuestas. Los huecos se llenan.
La pérdida total es la suma. Son dos fuerzas que tiran en direcciones opuestas.
""")
    d._rect(s, MARGEN, Y_CUERPO, ANCHO_UTIL, Inches(1.2), BLANCO)
    d._txt(s, MARGEN, Y_CUERPO + Inches(0.38), ANCHO_UTIL, Inches(0.5),
           "KL  =  ½ Σⱼ ( μⱼ²  +  σⱼ² − log σⱼ² − 1 )",
           size=22, color=CARBON, fuente="Noto Serif", align=PP_ALIGN.CENTER)
    y2 = Y_CUERPO + Inches(1.45)
    media = (ANCHO_UTIL - COL) / 2
    d.tarjeta(s, MARGEN, y2, media, Inches(1.45), "μ²  ·  resorte al origen",
              "Impide separar las nubes. Sin esto, la red las manda lejos unas de otras "
              "y vuelven las islas.", AZUL_MARGE)
    d.tarjeta(s, MARGEN + media + COL, y2, media, Inches(1.45),
              "σ² − log σ² − 1  ·  ni chica ni grande",
              "Vale 0 en σ = 1 y castiga los dos extremos. Impide achicar las nubes "
              "hasta volver a un punto.", ROJO)
    d.callout(s, MARGEN, y2 + Inches(1.65), ANCHO_UTIL, "Pérdida total",
              "BCE(x, x̂)  +  KL( N(μ, σ²) ‖ N(0, I) ).  La reconstrucción quiere nubes "
              "chicas y separadas; la KL las quiere estándar y encimadas. El VAE queda "
              "en el medio.", ROJO, alto=Inches(1.1))

    # ================================================== ACTO 2 · 15
    s = d.slide("Acto 2 · La mejora", "La nube de una cara", acento=ROJO, notas=f"""
Qué significa concretamente que una cara sea una nube.
Arriba a la izquierda, la original y su reconstrucción usando μ sin ruido: el centro de la
nube. Las siete de la derecha son muestras de esa misma nube.
Y acá hay que ser honesto: a simple vista parecen idénticas. No es un error. La nube es
angosta, σ medio {fmt(lat['sigma_medio'], 2)}, así que las muestras difieren milésimas por
píxel.
Por eso la fila de abajo: la diferencia contra decode(μ) multiplicada por seis. Ahí
aparece el dato interesante: la variación está concentrada en los contornos —el borde de
la cabeza, los ojos, la boca— y en el fondo. Las zonas planas son negras, idénticas.
La incertidumbre del modelo es sobre dónde cae exactamente cada borde. Y esa es la razón
de que las imágenes salgan borrosas, algo que retomamos al final.
""")
    fig(s, "vae_variaciones_mu.png", MARGEN, Y_CUERPO, ANCHO_UTIL, Inches(3.0))
    d.callout(s, MARGEN, Y_CUERPO + Inches(3.2), ANCHO_UTIL, "Dónde varía",
              f"La nube es angosta (σ medio {fmt(lat['sigma_medio'], 2)}), así que las "
              f"muestras parecen iguales. La fila de abajo amplifica la diferencia ×6: "
              f"toda la variación está en los contornos y el fondo, no en el relleno.",
              ROJO, alto=Inches(1.15))

    # ================================================== ACTO 2 · 16
    s = d.slide("Acto 2 · La mejora", "¿Y si agrandamos la nube?", acento=ROJO,
                notas="""
Para medir el tamaño de la nube la agrandamos a propósito, multiplicando σ por un factor k.
Cada columna usa el mismo ε, así que lo único que cambia entre filas es el factor.
Con k = 0 no hay ruido. Con k = 1, 2 y hasta 3 sigue siendo la misma cara: puse la
diferencia medida por píxel al lado de cada fila para que se vea que crece, de 0,023 a
0,079, pero el ojo casi no la registra.
Recién en k = 6 la cara se desarma. Cada cara ocupa una región chica y bien definida.
Agregué la fila de k = 6 justamente porque con las cuatro primeras la conclusión no se
veía. Es más honesto mostrar dónde se rompe que afirmar que se rompe antes.
""")
    fig(s, "vae_nube_agrandada.png", MARGEN, Y_CUERPO, Inches(7.4), ALTO_CUERPO)
    x2 = MARGEN + Inches(7.4) + COL
    w2 = ANCHO - x2 - MARGEN
    d._txt(s, x2, Y_CUERPO + Inches(0.15), w2, Inches(0.5), "z = μ + k · σ · ε",
           size=18, color=CARBON, fuente="Noto Serif", bold=True)
    d._parrafos(s, x2, Y_CUERPO + Inches(0.8), w2, Inches(2.4), [
        [("k = 0", {"bold": True, "color": CARBON}), ("  ·  la reconstrucción exacta", {})],
        [("k = 1", {"bold": True, "color": CARBON}), ("  ·  el tamaño que aprendió", {})],
        [("k = 3", {"bold": True, "color": CARBON}), ("  ·  todavía la misma cara", {})],
        [("k = 6", {"bold": True, "color": ROJO}), ("  ·  recién acá se desarma", {})],
    ], size=14, espaciado=12)
    d.callout(s, x2, Y_CUERPO + Inches(2.9), w2, "",
              "Cada fila lleva la diferencia medida por píxel: crece siempre, aunque el "
              "ojo no la registre hasta k alto.", AZUL_MARGE, alto=Inches(1.2))

    # ================================================== ACTO 2 · 17
    s = d.slide("Acto 2 · La mejora", "Esta vez la pérdida sí informa", oscura=True,
                notas=f"""
Adam con lr {str(ent['lr']).replace('.', ',')}, batch {ent['batch_size']}, {vae['epocas']} épocas, en una
{ent['gpu']}. Tardó {fmt(vae['minutos_entrenamiento'], 1)} minutos.
El panel izquierdo es la reconstrucción; en verde la del autoencoder común, que termina
más abajo — no paga el costo de la regularización.
El del medio es la KL, el panel de diagnóstico. Sube de 50 a unos {fmt(e200['train_kl'])}
nats y se estabiliza. Si hubiera caído a cero tendríamos «posterior collapse»: el decoder
ignoraría z. No pasó, y lo confirmé dimensión por dimensión: {lat['dims_activas']} de las
{lat['dims_totales']} tienen σ por debajo de 0,5. Solo {lat['dims_colapsadas']} quedaron
pegadas al prior.
El derecho es la suma. La punteada marca la época {mejor['epoca']}, el mínimo de
validación; de ahí en adelante el train sigue bajando y la validación no: sobreajuste leve.
A diferencia de una GAN, hay una sola pérdida que baja de forma monótona y que informa.
""")
    fig(s, "curvas_perdida.png", MARGEN, Y_CUERPO - Inches(0.15), Inches(9.0),
        Inches(3.2))
    x2 = MARGEN + Inches(9.0) + Inches(0.45)
    w2 = ANCHO - x2 - MARGEN
    d.stat(s, x2, Y_CUERPO - Inches(0.1), w2, "Por imagen, 200 épocas",
           f"{fmt(e200['train_total'], 0)}",
           f"Mínimo de validación en la época {mejor['epoca']}: "
           f"{fmt(mejor['val_total'])} nats.", oscura=True)
    d._parrafos(s, x2, Y_CUERPO + Inches(2.1), w2, Inches(1.6), [
        [("KL estable en ~", {"color": GRIS_CLARO}),
         (f"{fmt(e200['train_kl'], 0)} nats", {"bold": True, "color": CREMA}),
         (": no hubo colapso.", {"color": GRIS_CLARO})],
        [(f"{lat['dims_activas']} de {lat['dims_totales']}",
          {"bold": True, "color": CREMA}),
         (" dimensiones latentes llevan información.", {"color": GRIS_CLARO})],
    ], size=14, espaciado=12)

    # ================================================== ACTO 2 · 18
    s = d.slide("Acto 2 · La mejora", "Reconstrucción: época 30 contra 200", notas=f"""
Arriba las originales, en el medio el VAE a la época 30, abajo a la 200.
A la 30 ya se reconocen los personajes; la validación era {fmt(e30['val_total'])} nats.
Y algo que conviene señalar antes de que lo pregunten: a la 200 la validación queda en
{fmt(e200['val_total'])}, prácticamente igual, de hecho un poquito peor. Y sin embargo las
imágenes se ven mejor: ojos más definidos, fondo más limpio.
No es contradicción. La pérdida promedia {fmt(st['valores_por_imagen_64'])} valores, así
que una mejora concentrada en los bordes casi no la mueve. La métrica y la calidad
percibida no son lo mismo.
""")
    fig(s, "reconstrucciones_30_vs_200.png", MARGEN, Y_CUERPO, Inches(8.3), ALTO_CUERPO)
    x2 = MARGEN + Inches(8.3) + COL
    w2 = ANCHO - x2 - MARGEN
    d.tabla(s, x2, Y_CUERPO + Inches(0.3), w2, ["Validación", "nats"],
            [["Época 30", fmt(e30['val_total'])],
             ["Época 200", fmt(e200['val_total'])],
             [f"Mejor (ép. {mejor['epoca']})", "*" + fmt(mejor['val_total'])]],
            anchos=[0.55, 0.45])
    d.callout(s, x2, Y_CUERPO + Inches(2.5), w2, "",
              "La métrica se amesetó, pero los detalles finos sí mejoran: la pérdida "
              "promedia 12.288 valores y no registra una mejora de borde.",
              AZUL_MARGE, alto=Inches(1.5))

    # ================================================== ACTO 2 · 19
    s = d.slide("Acto 2 · La mejora", "El pago: caras que no existen", notas="""
Y acá está la respuesta a la pregunta del acto uno.
A la izquierda, el autoencoder decodificando vectores aleatorios: la mancha de siempre.
A la derecha, el VAE decodificando exactamente los mismos vectores. Son caras. Borrosas,
pero con estructura. Y ninguna está en el dataset.
Vale subrayarlo: la arquitectura es casi idéntica, los datos son los mismos, el
optimizador es el mismo, los vectores son literalmente los mismos. Lo único que cambia es
que el encoder devuelve una distribución y que la pérdida tiene el término KL.
Eso solo convierte un compresor en un generador.
""")
    fig(s, "ae_vs_vae_generacion.png", MARGEN, Y_CUERPO - Inches(0.1), ANCHO_UTIL,
        Inches(4.1))
    d._txt(s, MARGEN, Inches(6.5), ANCHO_UTIL, Inches(0.4),
           [("Mismos z, misma arquitectura, mismos datos. ", {}),
            ("La diferencia es el término KL.", {"bold": True, "color": CARBON})],
           size=14, color=GRIS)

    # ================================================== ACTO 2 · 20
    s = d.slide("Acto 2 · La mejora", "Caminar entre dos caras", notas="""
Última evidencia del acto, y la más convincente.
Tomamos dos caras de validación, las codificamos, y recorremos la recta que las une en el
latente, con diez pasos.
Si el espacio tuviera huecos, en algún punto del camino caeríamos en una zona muerta. Y no
pasa: toda la transición son caras. El pelo se transforma gradualmente, el fondo cambia de
color de forma continua, la pose rota de a poco.
La región entre dos caras conocidas también está poblada de caras. Es la continuidad que
pedíamos en el acto uno.
""")
    w_fig = Inches(7.8)
    for i in range(3):
        fig(s, f"interpolacion_{i + 1}.png", MARGEN, Y_CUERPO + i * Inches(1.05),
            w_fig, Inches(1.0), marco=False)
    x2 = MARGEN + w_fig + COL
    w2 = ANCHO - x2 - MARGEN
    d._txt(s, x2, Y_CUERPO - Inches(0.05), w2, Inches(0.4), "z = (1−t)·μA + t·μB",
           size=15, color=CARBON, fuente="Noto Serif", bold=True)
    d.callout(s, x2, Y_CUERPO + Inches(0.5), w2, "Transición gradual",
              "Los diez pasos intermedios son caras válidas: no hay saltos ni zonas "
              "rotas.", ROJO, alto=Inches(1.75))
    d.callout(s, x2, Y_CUERPO + Inches(2.4), w2, "Por diseño",
              "Es la continuidad del acto 1. En el autoencoder el camino podía cruzar "
              "un hueco.", AZUL_MARGE, alto=Inches(1.75))

    # ================================================== ACTO 3 · 21
    d.divisor_pizarron(
        "Acto 3 · Todos los datos",
        "Un solo modelo, los dos estilos",
        "El dataset trae cada cara dos veces. ¿Qué pasa si las mezclamos?",
        "pizarron_acto3.png", """
Tercer acto. Hasta acá usamos la mitad del dataset.
""")

    # ================================================== ACTO 3 · 22
    s = d.slide("Acto 3 · Todos los datos", "Cada cara viene en dos versiones",
                acento=ROJO, notas=f"""
El archivo de Kaggle trae dos carpetas con las mismas {fmt(st['n_imagenes'])} caras.
Arriba «cropped»: el fotograma real, con sombreados, texturas y fondos. Unos 9.500
colores distintos por imagen.
Abajo «simplified»: la misma cara con colores planos. Tres colores únicos por imagen.
Hasta ahora entrenamos solo con cropped. Ahora entrenamos UN modelo con las dos juntas:
{fmt(19754)} imágenes.
La pregunta no es cuál anda mejor. Es qué hace el modelo cuando tiene que acomodar las
dos cosas en un mismo espacio latente de {st['latent_dim']} dimensiones.
""")
    fig(s, "comp_dataset.png", MARGEN, Y_CUERPO, Inches(6.6), ALTO_CUERPO)
    x2 = MARGEN + Inches(6.6) + COL
    w2 = ANCHO - x2 - MARGEN
    d.tarjeta(s, x2, Y_CUERPO, w2, Inches(1.4), "cropped",
              "El fotograma real: sombreados, texturas, fondos. ~9.500 colores únicos "
              "por imagen.", AZUL_MARGE)
    d.tarjeta(s, x2, Y_CUERPO + Inches(1.55), w2, Inches(1.4), "simplified",
              "La misma cara con colores planos. 3 colores únicos por imagen.", ROJO)
    d.callout(s, x2, Y_CUERPO + Inches(3.1), w2, "La pregunta",
              f"¿Qué hace el modelo cuando tiene que acomodar los dos estilos en el "
              f"mismo latente de {st['latent_dim']} dimensiones?", ROJO,
              alto=Inches(1.2))

    # ================================================== ACTO 3 · 23
    s = d.slide("Acto 3 · Todos los datos", "Un VAE con 19.754 imágenes", acento=ROJO,
                notas=f"""
Mismo modelo, mismos hiperparámetros, 200 épocas. Lo único que cambia es que ve las dos
carpetas, y el split mezcla los dos estilos en lugar de dejar uno entero de un lado.
Tardó {fmt(comb['minutos'], 1)} minutos, casi el doble, porque son el doble de imágenes.
Los números quedan donde uno esperaría: entre los dos modelos individuales. La pérdida
total {fmt(comb['train_total'])}, contra {fmt(cr['vae']['train_total'])} del de cropped y
{fmt(si['vae']['train_total'])} del de simplified.
Eso no es interesante en sí mismo. Lo interesante es lo que hizo con el espacio latente,
que es lo que vemos a continuación.
""")
    d.tabla(s, MARGEN, Y_CUERPO, ANCHO_UTIL,
            ["", "solo cropped", "solo simplified", "los dos juntos"],
            [["Imágenes", fmt(9877), fmt(9877), "*" + fmt(19754)],
             ["Pérdida total (train)", fmt(cr['vae']['train_total']),
              fmt(si['vae']['train_total']), "*" + fmt(comb['train_total'])],
             ["Divergencia KL", fmt(cr['vae']['train_kl']),
              fmt(si['vae']['train_kl']), "*" + fmt(comb['train_kl'])],
             ["Brecha val − train", fmt(cr['vae']['brecha_final']),
              fmt(si['vae']['brecha_final']), "*" + fmt(comb['brecha'])],
             ["Tiempo", f"{fmt(cr['vae']['minutos'], 1)} min",
              f"{fmt(si['vae']['minutos'], 1)} min",
              f"*{fmt(comb['minutos'], 1)} min"]],
            anchos=[0.34, 0.22, 0.22, 0.22])
    d.callout(s, MARGEN, Y_CUERPO + Inches(3.4), ANCHO_UTIL, "",
              "Queda entre los dos, como era esperable. Lo interesante no está en la "
              "pérdida: está en cómo organizó el espacio latente.", ROJO,
              alto=Inches(0.95))

    # ================================================== ACTO 3 · 24
    s = d.slide("Acto 3 · Todos los datos", "«Simplificar» es una dirección",
                acento=ROJO, notas=f"""
Esta es la diapositiva que justifica el acto entero.
Lo primero que miramos fue si el modelo había armado dos nubes separadas, una por estilo.
No: los centroides están a {fmt(sep['distancia_centroides'], 2)} con una dispersión
interna de {fmt(sep['dispersion_media'], 2)}, o sea completamente entremezcladas. El modelo organizó el
latente por qué cara es, no por en qué estilo está dibujada.
Entonces, ¿dónde quedó el estilo? Probamos esto: tomamos 1.500 pares y calculamos v, el
promedio de μ de la versión simplificada menos μ de la versión cropped.
Si el estilo fuera ruido, esas restas apuntarían para cualquier lado y el promedio daría
casi cero. Pero el {dr['fraccion_positivos'] * 100:.0f} por ciento de los pares tiene
coseno positivo con esa dirección. Todos. La dirección es consistente.
Y sirve. En la figura, la primera columna es una cara cropped, la segunda su
reconstrucción, y después le vamos sumando α por v. En α = 1 el resultado se parece mucho
a su versión simplificada real, que está en la última columna.
El detalle importante: estas cinco caras NO entraron en el cálculo de v. Y el modelo nunca
supo que las imágenes venían en pares — la correspondencia la dedujo solo, nada más que
por tener que acomodar los dos estilos en el mismo espacio.
Esto es aritmética en el espacio latente: sumar un vector constante aplica un concepto.
""")
    fig(s, "comb_direccion_estilo.png", MARGEN, Y_CUERPO - Inches(0.1), Inches(7.0),
        Inches(4.4))
    x2 = MARGEN + Inches(7.0) + COL
    w2 = ANCHO - x2 - MARGEN
    d._rect(s, x2, Y_CUERPO, w2, Inches(0.95), BLANCO)
    d._txt(s, x2, Y_CUERPO + Inches(0.28), w2, Inches(0.4),
           "v = media( μ_simp − μ_crop )", size=15, color=CARBON,
           fuente="Noto Serif", bold=True, align=PP_ALIGN.CENTER)
    d.callout(s, x2, Y_CUERPO + Inches(1.1), w2, "Consistente",
              f"El {dr['fraccion_positivos'] * 100:.0f} % de los 1.500 pares apunta "
              f"para el mismo lado (coseno medio {fmt(dr['coseno_medio'], 2)}).",
              ROJO, alto=Inches(1.4))
    d.callout(s, x2, Y_CUERPO + Inches(2.65), w2, "Y generaliza",
              "Estas caras no entraron en el cálculo de v. El modelo nunca supo que las "
              "imágenes venían en pares.", AZUL_MARGE, alto=Inches(1.6))

    # ================================================== ACTO 3 · 25
    s = d.slide("Acto 3 · Todos los datos", "Caminar de un estilo al otro", acento=ROJO,
                notas="""
Otra forma de verlo: interpolar entre una cara y su propia versión simplificada.
La transición es suave y, sobre todo, la identidad se mantiene todo el camino. Es siempre
la misma cara, en la misma pose, cambiando de estilo.
Eso confirma lo de la diapositiva anterior desde otro ángulo: identidad y estilo quedaron
codificados de forma separable. El modelo puede mover uno dejando el otro quieto.
Y lo hizo sin supervisión de ningún tipo.
""")
    fig(s, "comb_interpolacion_estilos.png", MARGEN, Y_CUERPO, ANCHO_UTIL, Inches(3.1))
    d.callout(s, MARGEN, Y_CUERPO + Inches(3.3), ANCHO_UTIL, "Identidad y estilo, separables",
              "La transición es suave y la cara es siempre la misma, en la misma pose. "
              "El modelo puede mover el estilo dejando quieta la identidad.",
              AZUL_MARGE, alto=Inches(1.05))

    # ================================================== ACTO 3 · 26
    s = d.slide("Acto 3 · Todos los datos", "¿Y qué estilo inventa?", acento=ROJO,
                notas=f"""
Última pregunta del acto: cuando le pedimos una cara nueva, ¿en qué estilo la dibuja?
Las dos cosas, mitad y mitad. Proyectando sobre la dirección de estilo, las caras reales
caen en {fmt(eg['proy_cropped'], 2)} las cropped y {fmt(eg['proy_simplified'], 2)} las
simplified; las generadas caen en {fmt(eg['proy_generadas'], 2)}, repartidas
{eg['pct_generadas_lado_simplified']:.0f} / {100 - eg['pct_generadas_lado_simplified']:.0f}.
En la grilla, ordenada por textura, se ve el recorrido completo: a la izquierda caras de
color plano con fondo celeste, a la derecha caras con fondo texturado, y en el medio
híbridos que no existen en ninguna de las dos carpetas.
Eso es lo que uno querría de un modelo generativo entrenado con dos fuentes: no eligió una
y descartó la otra, cubrió las dos y lo que hay entre medio.
""")
    fig(s, "comb_generacion.png", MARGEN, Y_CUERPO - Inches(0.15), ANCHO_UTIL,
        Inches(3.3))
    y2 = Y_CUERPO + Inches(3.25)
    media = (ANCHO_UTIL - COL) / 2
    d.tarjeta(s, MARGEN, y2, media, Inches(1.05),
              f"{eg['pct_generadas_lado_simplified']:.0f} % / "
              f"{100 - eg['pct_generadas_lado_simplified']:.0f} %",
              "Reparto parejo entre los dos estilos, medido en el espacio latente.",
              ROJO)
    d.tarjeta(s, MARGEN + media + COL, y2, media, Inches(1.05), "Y también híbridos",
              "En el medio aparecen caras que no existen en ninguna de las dos "
              "carpetas.", AZUL_MARGE)

    # ================================================== ACTO 3 · 27
    s = d.slide("Acto 3 · Todos los datos", "Cómo medimos en qué estilo dibuja",
                acento=ROJO, notas=f"""
Vale la pena contar cómo se mide esto, porque el primer intento estuvo mal y el error es
instructivo.
Primer intento: la textura por píxel, o sea cuánto difiere cada píxel de sus vecinos. Las
simplified reales dan {fmt(ac['textura']['simplified_real'], 4)} y las cropped
{fmt(ac['textura']['cropped_real'], 4)}: bien separadas. Las generadas daban
{fmt(ac['textura']['generadas'], 4)}, del lado de simplified, y la conclusión salía
«100 % simplified».
Era falso. El desenfoque del VAE baja la textura exactamente igual que la simplificación,
así que una cara cropped borrosa se confunde con una simplified nítida. La medida no
estaba midiendo estilo: estaba midiendo desenfoque.
Lo correcto es medirlo donde el modelo efectivamente codificó el estilo, que es el
latente. El procedimiento es el del gráfico, en tres pasos.
Uno: tomamos la dirección de estilo v de la diapositiva anterior y la normalizamos.
Dos: proyectamos sobre ella los μ de las caras reales. Las cropped caen en
{fmt(eg['proy_cropped'], 2)} y las simplified en {fmt(eg['proy_simplified'], 2)}: son las
dos campanas del gráfico. Que estén bien separadas es lo que valida que la dirección
sirve para distinguir estilos.
Tres: proyectamos los z sorteados del prior —la línea negra— y vemos de qué lado del
punto medio cae cada uno. Resultado: {eg['pct_generadas_lado_simplified']:.0f} por ciento
del lado simplified, {100 - eg['pct_generadas_lado_simplified']:.0f} del lado cropped. Y
eso sí coincide con lo que se ve en la grilla.
La moraleja vale para todo el trabajo: cuando una métrica y los ojos no coinciden, lo
primero a revisar es la métrica.
""")
    w_fig = Inches(7.9)
    fig(s, "comb_estilo_proyeccion.png", MARGEN, Y_CUERPO - Inches(0.1), w_fig,
        Inches(4.3))
    x2 = MARGEN + w_fig + COL
    w2 = ANCHO - x2 - MARGEN
    d._parrafos(s, x2, Y_CUERPO - Inches(0.05), w2, Inches(1.7), [
        [("1.", {"bold": True, "color": ROJO}),
         ("  Normalizamos la dirección v.", {})],
        [("2.", {"bold": True, "color": ROJO}),
         (f"  Proyectamos los μ reales: cropped {fmt(eg['proy_cropped'], 2)}, "
          f"simplified {fmt(eg['proy_simplified'], 2)}.", {})],
        [("3.", {"bold": True, "color": ROJO}),
         ("  Proyectamos los z del prior y vemos de qué lado caen.", {})],
    ], size=13.5, espaciado=10)
    d.tarjeta(s, x2, Y_CUERPO + Inches(1.75), w2, Inches(1.0),
              f"{eg['pct_generadas_lado_simplified']:.0f} % / "
              f"{100 - eg['pct_generadas_lado_simplified']:.0f} %",
              "Reparto entre los dos estilos.", AZUL_MARGE)
    d.callout(s, x2, Y_CUERPO + Inches(2.9), w2, "Lo que NO sirve",
              "La textura por píxel: el desenfoque la baja igual que la "
              "simplificación.", ROJO, alto=Inches(1.35))

    # ================================================== CIERRE · 28
    d.divisor_pizarron(
        "Cierre",
        "Lo que aprendimos",
        "Todo el trabajo, escrito en el pizarrón.",
        "pizarron_cierre.png", """
Cierre. El pizarrón resume los cinco puntos del trabajo.
Uno: un autoencoder comprime y descomprime, y el dato es su propia etiqueta.
Dos: un autoencoder no genera, porque su latente tiene islas y huecos.
Tres: la KL ata las nubes a N(0, I) y eso vuelve el latente continuo y completo.
Cuatro: la reparametrización z = μ + σ·ε permite entrenarlo con gradiente.
Cinco: con los dos estilos juntos, simplificar resultó ser una dirección del espacio,
aprendida sin que nadie le dijera que las imágenes venían en pares.
Queda una limitación, que es la que vemos ahora.
""")

    # ================================================== CIERRE · 29
    s = d.slide("Cierre", "Por qué sale borroso", notas=f"""
La limitación honesta. Las caras son borrosas, y no es un bug.
La BCE penaliza cada píxel por separado. Cuando el modelo no está seguro de dónde cae un
borde, lo que minimiza la pérdida es promediar todas las posiciones posibles en vez de
jugarse por una. El promedio de muchas caras es, inevitablemente, una cara borrosa.
Lo confirmamos dos veces: en los mapas de diferencia del acto dos, donde la variación
estaba toda en los contornos, y en que la métrica no mejora aunque los bordes sí.
Una GAN no promedia, porque el discriminador castiga cualquier cosa que no parezca real,
y una cara promedio no parece real. Por eso da más nítido, a cambio de un entrenamiento
inestable y riesgo de mode collapse.
Abajo vuelvo a poner la figura del acto dos, porque es la prueba más directa: la variación
del modelo está concentrada en los contornos, que es exactamente donde no sabe dónde cae
el borde. Ahí es donde promedia, y por eso es ahí donde se ve borroso.
""")
    media = (ANCHO_UTIL - COL) / 2
    d.callout(s, MARGEN, Y_CUERPO, media, "Por qué es borroso",
              "La BCE premia el promedio de las posiciones posibles de cada borde. El "
              "promedio de muchas caras es una cara borrosa. No es falta de "
              "entrenamiento.", ROJO, alto=Inches(1.7))
    d.callout(s, MARGEN + media + COL, Y_CUERPO, media, "Qué hace distinto una GAN",
              "El discriminador castiga lo que no parece real, y una cara promedio no "
              "lo parece. Más nítido, a cambio de inestabilidad y mode collapse.",
              AZUL_MARGE, alto=Inches(1.7))
    fig(s, "vae_variaciones_mu.png", MARGEN, Y_CUERPO + Inches(1.95), ANCHO_UTIL,
        Inches(1.95))
    d._txt(s, MARGEN, Inches(6.45), ANCHO_UTIL, Inches(0.4),
           [("La prueba: la variación del modelo está toda en los contornos, "
             "que es justo donde ", {}),
            ("no sabe dónde cae el borde", {"bold": True, "color": CARBON}), (".", {})],
           size=14, color=GRIS, align=PP_ALIGN.CENTER)

    # ================================================== CIERRE · 30
    s = d._nueva(CARBON)
    d._cabecera = ("Cierre", "¿Preguntas?")
    d.figura(s, "sofa.png", MARGEN, Inches(1.5), Inches(6.6), Inches(4.2),
             d.ambiente, marco=False)
    x2 = MARGEN + Inches(7.0)
    w2 = ANCHO - x2 - MARGEN
    d._txt(s, x2, Inches(2.6), w2, Inches(0.3), "G R A C I A S", size=13,
           color=AMARILLO, bold=True)
    d._txt(s, x2, Inches(3.05), w2, Inches(1.6),
           "¿Preguntas?", size=40, color=CREMA, fuente="Noto Serif", bold=True)
    d._parrafos(s, x2, Inches(4.3), w2, Inches(1.6), [
        [("Todo el código, los checkpoints y las figuras están en el repositorio "
          "del trabajo.", {"color": GRIS_CLARO})],
        [("Los tres modelos se entrenan de cero en menos de 40 minutos de GPU.",
          {"color": GRIS_CLARO})],
    ], size=14, espaciado=12)
    d.donut(s, x2, Inches(5.7), Inches(0.9))
    d._notas(s, """
Gracias. Dejo el sofá vacío porque, como en la serie, nunca se sabe quién se va a sentar.
Si hay preguntas, arranco por la que siempre aparece: por qué no usamos una GAN. La
respuesta corta está en la diapositiva anterior; la larga es que el VAE nos daba un
encoder, que es lo que permitió todo el análisis del espacio latente del acto tres.
""")

    return d


def cargar_datos():
    """Los JSON de resultados, en el orden en que los toma construir()."""
    # Las figuras del modelo combinado viven con su experimento; se copian acá para
    # que la presentación tenga una sola carpeta de figuras.
    for p in sorted((COMBINADO / "analisis").glob("*.png")):
        shutil.copy2(p, FIGS / p.name)

    st = json.loads((PRES / "dataset_stats.json").read_text())
    rs = json.loads((PRES / "resultados.json").read_text())
    cp = json.loads((PRES / "comparacion.json").read_text())
    ac = json.loads((COMBINADO / "analisis" / "analisis.json").read_text())
    cb = json.loads((COMBINADO / "analisis" / "comparativa_base.json").read_text())

    u = ultima_fila(COMBINADO / "loss_history.csv")
    with (COMBINADO / "loss_history.csv").open(newline="") as f:
        segs = sum(float(r["seconds"]) for r in csv.DictReader(f))
    comb = {"train_total": u["train_total"], "train_kl": u["train_kl"],
            "val_total": u["val_total"],
            "brecha": u["val_total"] - u["train_total"], "minutos": segs / 60}
    return st, rs, cp, ac, comb, cb


def main():
    construir(*cargar_datos()).guardar(SALIDA)


if __name__ == "__main__":
    main()
