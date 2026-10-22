"""
Genera el guion del orador a partir de la MISMA fuente que la presentación.

Reconstruye el mazo en memoria (sin volver a guardar el .pptx) y lee el índice que el
Deck va registrando: epígrafe, título y notas de cada diapositiva. Así el guion no puede
desincronizarse: si cambia una diapositiva, cambia el guion.

Salida: outputs/presentacion/guion.md

Uso: python code/presentacion/make_guion.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import make_presentation as mp

SALIDA = config.OUTPUT_DIR / "presentacion" / "guion.md"

# Ritmo de una exposición técnica en español, contando las pausas.
PALABRAS_POR_MINUTO = 125


def parrafos(nota):
    """El texto fuente viene cortado a 90 columnas; se rearma en párrafos corridos."""
    out = []
    for bloque in re.split(r"\n\s*\n", nota.strip()):
        linea = " ".join(l.strip() for l in bloque.split("\n") if l.strip())
        if linea:
            out.append(linea)
    return out


def mmss(minutos):
    seg = int(round(minutos * 60))
    return f"{seg // 60}:{seg % 60:02d}"


def main():
    d = mp.construir(*mp.cargar_datos())
    filas = []
    for e in d.indice:
        ps = parrafos(e["notas"])
        palabras = sum(len(p.split()) for p in ps)
        filas.append({**e, "parrafos": ps, "palabras": palabras})

    total = sum(f["palabras"] for f in filas)
    minutos = total / PALABRAS_POR_MINUTO

    out = ["# Guion del orador\n"]
    out.append("> Generado por `code/presentacion/make_guion.py` desde la misma fuente "
               "que la presentación.\n> Si cambia una diapositiva, volver a correrlo en "
               "vez de editar este archivo a mano.\n")
    miles = f"{total:,}".replace(",", ".")
    out.append(f"**{len(filas)} diapositivas** · {miles} palabras · "
               f"**~{minutos:.0f} minutos** a {PALABRAS_POR_MINUTO} palabras por "
               f"minuto, pausas incluidas.\n")

    out.append("## Ritmo\n")
    out.append("| # | Diapositiva | Dura | Acumulado |")
    out.append("|---:|---|---:|---:|")
    acum = 0.0
    for f in filas:
        dur = f["palabras"] / PALABRAS_POR_MINUTO
        acum += dur
        out.append(f"| {f['n']} | {f['titulo']} | {dur * 60:.0f} s | {mmss(acum)} |")
    out.append("")

    def seccion(epigrafe):
        """El epígrafe trae «Acto 2 · La mejora» o «Cierre»; lo previo es la apertura."""
        cab = epigrafe.split("·")[0].strip()
        return cab if cab.startswith(("Acto", "Cierre")) else "Apertura"

    acto = None
    for f in filas:
        bloque = seccion(f["epigrafe"])
        if bloque != acto:
            acto = bloque
            out.append(f"\n---\n\n# {acto}\n")
        out.append(f"## {f['n']}. {f['titulo']}")
        meta = [f["epigrafe"]] if f["epigrafe"] else []
        meta.append(f"~{f['palabras'] / PALABRAS_POR_MINUTO * 60:.0f} s")
        out.append(f"<small>{' · '.join(meta)}</small>\n")
        out.extend(p + "\n" for p in f["parrafos"] or
                   ["_(sin notas: la diapositiva habla sola)_"])

    SALIDA.write_text("\n".join(out))
    print(f"{len(filas)} diapositivas · {total} palabras · ~{minutos:.0f} min")
    print(f"guion -> {SALIDA}")


if __name__ == "__main__":
    main()
