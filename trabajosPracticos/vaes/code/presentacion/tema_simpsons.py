"""
Tema visual Simpsons: extiende el sistema de diseño con la paleta y los recursos
de la serie.

Mantiene la estructura del sistema original (eyebrow + título serif + pie numerado,
tarjetas con barra lateral, tablas con encabezado oscuro) pero cambia la paleta al
amarillo y el azul cielo de la serie, y agrega tipos de diapositiva propios: portada
sobre el cielo de la apertura, divisores con el pizarrón de Bart, y un cierre con el
sofá.

Las imágenes de ambiente las genera ambiente_simpsons.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

import diseno
from diseno import ALTO, ANCHO, ANCHO_UTIL, MARGEN, Deck, _esp

# --- Paleta de la serie -----------------------------------------------------
AMARILLO = RGBColor(0xFF, 0xD9, 0x0F)
AMARILLO_HONDO = RGBColor(0xE8, 0xB6, 0x00)
CIELO = RGBColor(0x6F, 0xA8, 0xDC)
CIELO_HONDO = RGBColor(0x3E, 0x6F, 0xA0)
CREMA = RGBColor(0xFD, 0xF7, 0xE3)
CARBON = RGBColor(0x2B, 0x2B, 0x2B)
PIZARRON = RGBColor(0x2C, 0x4A, 0x3E)
MADERA = RGBColor(0x8A, 0x5A, 0x3B)
ROJO = RGBColor(0xD9, 0x4F, 0x30)
AZUL_MARGE = RGBColor(0x10, 0x9D, 0xD3)
TIZA = RGBColor(0xF2, 0xF0, 0xE6)
GRIS = RGBColor(0x5A, 0x55, 0x4C)
GRIS_CLARO = RGBColor(0x9B, 0x94, 0x87)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
AMARILLO_BG = RGBColor(0xFD, 0xF2, 0xC9)
CIELO_BG = RGBColor(0xE2, 0xED, 0xF8)
ROJO_BG = RGBColor(0xFA, 0xE3, 0xDC)


class DeckSimpsons(Deck):
    """Deck con la paleta y los recursos de la serie."""

    def __init__(self, ambiente: Path, figuras: Path, pie="VAEs sobre Los Simpson"):
        # El módulo base lee los colores de sus globals, así que se reemplazan ahí.
        diseno.CREMA = CREMA
        diseno.NAVY = CARBON
        diseno.TEAL = AZUL_MARGE
        diseno.NARANJA = ROJO
        diseno.TEAL_BG = CIELO_BG
        diseno.NARANJA_BG = ROJO_BG
        diseno.GRIS = GRIS
        diseno.GRIS_CLARO = GRIS_CLARO
        diseno.FILA_ALT = AMARILLO_BG
        diseno.TEAL_CLARO = AMARILLO
        super().__init__(pie=pie)
        self.ambiente = Path(ambiente)
        self.figuras = Path(figuras)

    # ------------------------------------------------------------- recursos
    def fondo_imagen(self, slide, nombre):
        """Imagen a sangre completa, detrás de todo."""
        pic = slide.shapes.add_picture(str(self.ambiente / nombre), 0, 0,
                                       width=ANCHO, height=ALTO)
        slide.shapes._spTree.remove(pic._element)
        slide.shapes._spTree.insert(2, pic._element)
        return pic

    def donut(self, slide, x, y, lado=Inches(0.6)):
        return slide.shapes.add_picture(str(self.ambiente / "donut.png"), x, y,
                                        width=lado, height=lado)

    def fig(self, slide, nombre, x, y, w, h, marco=True):
        return self.figura(slide, nombre, x, y, w, h, self.figuras, marco)

    # ------------------------------------------------- tipos de diapositiva
    def portada_cielo(self, eyebrow, titulo, subtitulo, notas):
        """Portada sobre el cielo de la apertura."""
        slide = self._nueva(CIELO)
        self._cabecera = (eyebrow, titulo)
        self.fondo_imagen(slide, "cielo.png")

        # Panel translúcido para que el título siempre se lea sobre las nubes
        panel = self._rect(slide, 0, Inches(2.5), ANCHO, Inches(2.9), CARBON)
        panel.fill.fore_color.rgb = CARBON
        panel.fill.transparency = 0.12

        self._txt(slide, MARGEN, Inches(2.78), ANCHO_UTIL, Inches(0.3),
                  _esp(eyebrow), size=12, color=AMARILLO, bold=True)
        # 38 pt entra en una línea en 10"; el donut va en lo que queda a la derecha.
        self._txt(slide, MARGEN, Inches(3.25), Inches(10.0), Inches(0.9),
                  titulo, size=38, color=CREMA, fuente=diseno.SERIF, bold=True)
        self._txt(slide, MARGEN, Inches(4.35), Inches(10.0), Inches(0.5),
                  subtitulo, size=16, color=RGBColor(0xEC, 0xE4, 0xD0),
                  fuente=diseno.SERIF, italic=True)
        self.donut(slide, ANCHO - MARGEN - Inches(1.15), Inches(3.3), Inches(1.15))
        self._notas(slide, notas)
        return slide

    def divisor_pizarron(self, acto, titulo, subtitulo, imagen, notas=""):
        """Divisor de acto: el gag del pizarrón a la izquierda, el título a la derecha."""
        slide = self._nueva(CARBON)
        self._cabecera = (acto, titulo)
        # El pizarrón se achica para que el título entre cómodo a la derecha, incluso
        # cuando ocupa dos líneas.
        self.figura(slide, imagen, MARGEN, Inches(1.6), Inches(6.4), Inches(4.1),
                    self.ambiente, marco=False)
        x = MARGEN + Inches(6.8)
        w = ANCHO - x - MARGEN
        self._txt(slide, x, Inches(2.15), w, Inches(0.3), _esp(acto), size=12,
                  color=AMARILLO, bold=True)
        self._txt(slide, x, Inches(2.6), w, Inches(1.5), titulo, size=31,
                  color=CREMA, fuente=diseno.SERIF, bold=True, interlineado=1.05)
        self._txt(slide, x, Inches(4.3), w, Inches(1.2), subtitulo, size=15,
                  color=RGBColor(0xC9, 0xC2, 0xB2), italic=True)
        self._notas(slide, notas)
        return slide

    def slide(self, eyebrow, titulo, notas="", oscura=False, acento=None):
        return super().slide(eyebrow, titulo, notas=notas, oscura=oscura,
                             acento=acento or AZUL_MARGE)

    def franja_caras(self, slide, alto=Inches(0.5)):
        """Tira decorativa de caras del dataset, pegada al borde inferior.

        Corre el pie hacia arriba para que no quede tapado.
        """
        for caja in getattr(self, "_cajas_pie", ()):
            caja.top = int(ALTO - alto - Inches(0.42))
        return slide.shapes.add_picture(str(self.ambiente / "tira_caras.png"),
                                        0, int(ALTO - alto), width=ANCHO, height=alto)
