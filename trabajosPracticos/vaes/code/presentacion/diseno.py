"""
Sistema de diseño de la presentación, calcado del PDF de la clase teórica.

Fondo crema, títulos serif, "eyebrow" en versalitas espaciadas, divisores de sección a
sangre completa, tarjetas con barra lateral de color, tablas con encabezado navy y pie
con el número de diapositiva.

Fuentes: Noto Serif y Liberation Sans, que están instaladas en el sistema (así el PDF de
verificación muestra exactamente lo que se va a ver). Si se abre en otra máquina, caen
en el serif y el sans por defecto sin romper el maquetado.
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from pptx.util import Emu, Inches, Pt

# --- Paleta ----------------------------------------------------------------
CREMA = RGBColor(0xF8, 0xF6, 0xF0)
NAVY = RGBColor(0x1D, 0x2E, 0x3A)
TEAL = RGBColor(0x1F, 0x7A, 0x6E)
NARANJA = RGBColor(0xCE, 0x6A, 0x28)
TEAL_BG = RGBColor(0xE3, 0xEF, 0xED)
NARANJA_BG = RGBColor(0xFB, 0xE8, 0xDA)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
GRIS = RGBColor(0x55, 0x60, 0x6A)
GRIS_CLARO = RGBColor(0x9A, 0xA3, 0xAA)
FILA_ALT = RGBColor(0xEF, 0xED, 0xE8)
TEAL_CLARO = RGBColor(0x6F, 0xBF, 0xB2)   # sobre navy

SERIF = "Noto Serif"
SANS = "Liberation Sans"

# --- Geometría -------------------------------------------------------------
ANCHO = Inches(13.333)
ALTO = Inches(7.5)
MARGEN = Inches(0.9)
ANCHO_UTIL = ANCHO - 2 * MARGEN

Y_EYEBROW = Inches(0.85)
Y_TITULO = Inches(1.25)
Y_CUERPO = Inches(2.35)
ALTO_CUERPO = ALTO - Y_CUERPO - Inches(0.85)

PIE = "VAEs sobre Los Simpson"


def _sin_sombra(forma):
    """Quita la sombra de una autoforma.

    `shape.shadow.inherit = False` no alcanza: LibreOffice igual dibuja la sombra del
    tema. Hay que insertar una lista de efectos vacía en las propiedades de la forma.
    """
    spPr = forma._element.spPr
    for viejo in spPr.findall(qn("a:effectLst")):
        spPr.remove(viejo)
    spPr.append(parse_xml(
        '<a:effectLst xmlns:a="http://schemas.openxmlformats.org/'
        'drawingml/2006/main"/>'))


def _esp(texto: str) -> str:
    """Versalitas espaciadas: 'BLOQUE 1' -> 'B L O Q U E  1' (letter-spacing a mano,
    porque python-pptx no expone el atributo de espaciado entre caracteres)."""
    return " ".join(texto.upper())


class Deck:
    def __init__(self, pie: str = PIE):
        self.prs = Presentation()
        self.prs.slide_width = ANCHO
        self.prs.slide_height = ALTO
        self.n = 0
        self.pie = pie
        # Índice (n, epígrafe, título, notas) para poder generar el guion del orador
        # desde la misma fuente que la presentación.
        self.indice = []
        self._cabecera = ("", "")

    # ---------------------------------------------------------------- base
    def _nueva(self, fondo=CREMA):
        self.n += 1
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = fondo
        return slide

    def _rect(self, slide, x, y, w, h, color, radio=False):
        forma = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if radio else MSO_SHAPE.RECTANGLE, x, y, w, h)
        forma.fill.solid()
        forma.fill.fore_color.rgb = color
        forma.line.fill.background()
        _sin_sombra(forma)
        if radio:
            forma.adjustments[0] = 0.06
        return forma

    def _txt(self, slide, x, y, w, h, partes, size=14, color=GRIS, fuente=SANS,
             bold=False, italic=False, align=PP_ALIGN.LEFT, espaciado=0,
             interlineado=None, anchor=MSO_ANCHOR.TOP):
        """`partes` es un str o una lista de (texto, {bold, italic, color, size, font})."""
        caja = slide.shapes.add_textbox(x, y, w, h)
        tf = caja.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        p = tf.paragraphs[0]
        p.alignment = align
        if interlineado:
            p.line_spacing = interlineado
        if espaciado:
            p.space_after = Pt(espaciado)

        if isinstance(partes, str):
            partes = [(partes, {})]
        for texto, est in partes:
            run = p.add_run()
            run.text = texto
            f = run.font
            f.size = Pt(est.get("size", size))
            f.bold = est.get("bold", bold)
            f.italic = est.get("italic", italic)
            f.color.rgb = est.get("color", color)
            f.name = est.get("font", fuente)
        return caja

    def _parrafos(self, slide, x, y, w, h, lineas, size=14, color=GRIS,
                  espaciado=10, interlineado=1.25, vinetas=False):
        """Varias líneas en una sola caja. Cada línea: str o lista de (texto, estilo)."""
        caja = slide.shapes.add_textbox(x, y, w, h)
        tf = caja.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, linea in enumerate(lineas):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(espaciado)
            p.line_spacing = interlineado
            partes = [(linea, {})] if isinstance(linea, str) else linea
            if vinetas:
                partes = [("•   ", {"color": GRIS_CLARO})] + list(partes)
            for texto, est in partes:
                run = p.add_run()
                run.text = texto
                f = run.font
                f.size = Pt(est.get("size", size))
                f.bold = est.get("bold", False)
                f.italic = est.get("italic", False)
                f.color.rgb = est.get("color", color)
                f.name = est.get("font", SANS)
        return caja

    def _pie(self, slide, claro=False):
        c = GRIS_CLARO if not claro else RGBColor(0x6B, 0x7A, 0x86)
        izq = self._txt(slide, MARGEN, ALTO - Inches(0.62), Inches(5), Inches(0.3),
                        self.pie, size=11, color=c)
        der = self._txt(slide, ANCHO - MARGEN - Inches(1), ALTO - Inches(0.62),
                        Inches(1), Inches(0.3), str(self.n), size=11, color=c,
                        align=PP_ALIGN.RIGHT)
        # Se guardan para que una franja al pie pueda correrlas hacia arriba.
        self._cajas_pie = (izq, der)
        return izq, der

    def _notas(self, slide, texto):
        slide.notes_slide.notes_text_frame.text = texto.strip()
        epigrafe, titulo = self._cabecera
        self.indice.append({"n": self.n, "epigrafe": epigrafe, "titulo": titulo,
                            "notas": texto.strip()})
        self._cabecera = ("", "")

    # ------------------------------------------------------- tipos de slide
    def portada(self, eyebrow, titulo, subtitulo, notas):
        slide = self._nueva(NAVY)
        self._cabecera = (eyebrow, titulo)
        self._txt(slide, MARGEN, Inches(1.9), ANCHO_UTIL, Inches(0.35),
                  _esp(eyebrow), size=13, color=TEAL_CLARO, bold=True)
        self._txt(slide, MARGEN, Inches(2.45), Inches(10.5), Inches(2.4),
                  titulo, size=50, color=CREMA, fuente=SERIF, bold=True,
                  interlineado=1.05)
        self._txt(slide, MARGEN, Inches(5.0), Inches(10), Inches(0.6),
                  subtitulo, size=20, color=RGBColor(0xB9, 0xC4, 0xCB),
                  fuente=SERIF, italic=True)
        self._notas(slide, notas)
        return slide

    def divisor(self, bloque, titulo, subtitulo, color=TEAL, notas=""):
        slide = self._nueva(color)
        self._cabecera = (bloque, titulo)
        claro = CREMA
        self._txt(slide, MARGEN, Inches(3.3), ANCHO_UTIL, Inches(0.35),
                  _esp(bloque), size=13, color=RGBColor(0xDC, 0xE8, 0xE5), bold=True)
        self._txt(slide, MARGEN, Inches(3.85), Inches(11), Inches(1.1),
                  titulo, size=44, color=claro, fuente=SERIF, bold=True)
        self._txt(slide, MARGEN, Inches(5.0), Inches(11), Inches(0.6),
                  subtitulo, size=19, color=RGBColor(0xE3, 0xEC, 0xEA),
                  fuente=SERIF, italic=True)
        self._notas(slide, notas)
        return slide

    def slide(self, eyebrow, titulo, notas="", oscura=False, acento=TEAL):
        """Cabecera estándar. Devuelve (slide, y_cuerpo) para ir apilando bloques."""
        slide = self._nueva(NAVY if oscura else CREMA)
        self._cabecera = (eyebrow, titulo)
        col_eye = TEAL_CLARO if oscura else acento
        col_tit = CREMA if oscura else NAVY
        self._txt(slide, MARGEN, Y_EYEBROW, ANCHO_UTIL, Inches(0.3),
                  _esp(eyebrow), size=12, color=col_eye, bold=True)
        self._txt(slide, MARGEN, Y_TITULO, Inches(11.6), Inches(1.0),
                  titulo, size=36, color=col_tit, fuente=SERIF, bold=True)
        self._pie(slide, claro=oscura)
        if notas:
            self._notas(slide, notas)
        return slide

    # ----------------------------------------------------------- componentes
    def figura(self, slide, nombre, x, y, w, h, dir_figuras, marco=True):
        """Inserta la figura centrada en el rectángulo, con tarjeta blanca detrás."""
        ruta = Path(dir_figuras) / nombre
        if not ruta.is_file():
            raise FileNotFoundError(f"Falta la figura {ruta}")
        with Image.open(ruta) as im:
            pw, ph = im.size
        pad = Inches(0.16) if marco else 0
        esc = min((w - 2 * pad) / pw, (h - 2 * pad) / ph)
        iw, ih = int(pw * esc), int(ph * esc)
        ix, iy = int(x + (w - iw) / 2), int(y + (h - ih) / 2)
        if marco:
            self._rect(slide, ix - pad, iy - pad, iw + 2 * pad, ih + 2 * pad, BLANCO)
        slide.shapes.add_picture(str(ruta), ix, iy, width=iw, height=ih)
        return iw, ih

    def callout(self, slide, x, y, w, etiqueta, texto, color=TEAL, alto=None):
        """Caja tintada con barra lateral de color, estilo «INTUICIÓN»."""
        fondo = TEAL_BG if color == TEAL else NARANJA_BG
        h = alto or Inches(1.15)
        self._rect(slide, x, y, w, h, fondo)
        self._rect(slide, x, y, Inches(0.055), h, color)
        dy = Inches(0.17)
        if etiqueta:
            self._txt(slide, x + Inches(0.33), y + dy, w - Inches(0.6), Inches(0.26),
                      _esp(etiqueta), size=11, color=color, bold=True)
            dy += Inches(0.36)
        self._parrafos(slide, x + Inches(0.33), y + dy, w - Inches(0.66),
                       h - dy - Inches(0.1), [texto], size=13.5, espaciado=0)
        return h

    def tarjeta(self, slide, x, y, w, h, titulo, texto, color=None, fondo=BLANCO):
        """Tarjeta con título en color y texto; opcionalmente barra lateral."""
        self._rect(slide, x, y, w, h, fondo, radio=(fondo == BLANCO))
        if color:
            self._rect(slide, x, y, Inches(0.05), h, color)
        px = x + (Inches(0.3) if color else Inches(0.28))
        self._txt(slide, px, y + Inches(0.2), w - Inches(0.6), Inches(0.3),
                  titulo, size=15, color=color or NAVY, bold=True)
        self._parrafos(slide, px, y + Inches(0.6), w - Inches(0.6), h - Inches(0.7),
                       [texto], size=13, espaciado=0)

    def tabla(self, slide, x, y, w, encabezados, filas, anchos=None, alto_fila=None):
        """Tabla con encabezado navy y filas alternadas."""
        n_col = len(encabezados)
        anchos = anchos or [1 / n_col] * n_col
        hf = alto_fila or Inches(0.52)
        hh = Inches(0.56)

        self._rect(slide, x, y, w, hh, NAVY)
        cx = x
        for i, (titulo, frac) in enumerate(zip(encabezados, anchos)):
            cw = int(w * frac)
            al = PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT
            self._txt(slide, cx + Inches(0.22), y + Inches(0.15),
                      cw - Inches(0.44), Inches(0.3), titulo, size=13.5,
                      color=BLANCO, bold=True, align=al)
            cx += cw

        yy = y + hh
        for j, fila in enumerate(filas):
            if j % 2 == 0:
                self._rect(slide, x, yy, w, hf, FILA_ALT)
            cx = x
            for i, (celda, frac) in enumerate(zip(fila, anchos)):
                cw = int(w * frac)
                al = PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT
                destacar = celda.startswith("*")
                self._txt(slide, cx + Inches(0.22), yy + Inches(0.14),
                          cw - Inches(0.44), Inches(0.3), celda.lstrip("*"),
                          size=13.5, color=NAVY if destacar else GRIS,
                          bold=destacar, align=al)
                cx += cw
            yy += hf
        return yy - y

    def lista_numerada(self, slide, x, y, w, items, colores=None):
        """Tarjetas blancas con un número grande en color, estilo «En cinco líneas»."""
        h = Inches(0.76)
        hueco = Inches(0.15)
        colores = colores or [TEAL, TEAL, NARANJA, NARANJA, NAVY]
        for i, item in enumerate(items):
            yy = y + i * (h + hueco)
            self._rect(slide, x, yy, w, h, BLANCO, radio=True)
            self._txt(slide, x + Inches(0.35), yy + Inches(0.2), Inches(0.5),
                      Inches(0.4), str(i + 1), size=19,
                      color=colores[i % len(colores)], fuente=SERIF, bold=True)
            partes = [(item, {})] if isinstance(item, str) else item
            self._txt(slide, x + Inches(1.0), yy + Inches(0.24), w - Inches(1.4),
                      Inches(0.4), partes, size=14, color=GRIS)
        return len(items) * (h + hueco)

    def stat(self, slide, x, y, w, etiqueta, valor, detalle, oscura=False):
        """Bloque de número grande, estilo «4475 → 3323»."""
        col_et = TEAL_CLARO if oscura else TEAL
        col_val = CREMA if oscura else NAVY
        col_det = RGBColor(0xB9, 0xC4, 0xCB) if oscura else GRIS
        self._txt(slide, x, y, w, Inches(0.3), _esp(etiqueta), size=11,
                  color=col_et, bold=True)
        self._txt(slide, x, y + Inches(0.38), w, Inches(0.8), valor, size=38,
                  color=col_val, fuente=SERIF, bold=True)
        self._parrafos(slide, x, y + Inches(1.3), w, Inches(1.2), [detalle],
                       size=13.5, color=col_det, espaciado=6)

    def guardar(self, ruta):
        self.prs.save(str(ruta))
        print(f"{self.n} diapositivas -> {ruta}")
