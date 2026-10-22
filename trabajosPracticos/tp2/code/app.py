"""
Interfaz web del predictor de clusters — TP2 (candy-data).

Formulario para cargar el vector de una golosina nueva y ver a qué
cluster cae según los tres métodos del TP (k-means, jerárquico y
DBSCAN), con proyección PCA, distancias, perfil y miembros del cluster.

Uso:

    cd code
    ../.venv/bin/python app.py                  # solo localhost (ideal con túnel SSH)
    ../.venv/bin/python app.py --host 0.0.0.0   # accesible desde la red

Si se accede por SSH, lo más seguro es un túnel desde la máquina local:

    ssh -L 5000:localhost:5000 usuario@este-equipo

y abrir http://127.0.0.1:5000 en el navegador local.
"""
import argparse

from flask import Flask, render_template, request

import predict as pred
from common import BINARY_COLS, CONT_COLS, WIN_COL

app = Flask(__name__)

FEATURE_LABELS = {
    "chocolate": "¿Contiene chocolate?",
    "fruity": "¿Es frutal?",
    "caramel": "¿Tiene caramelo?",
    "peanutyalmondy": "¿Maní/almendras?",
    "nougat": "¿Tiene turrón (nougat)?",
    "crispedricewafer": "¿Arroz crocante / oblea?",
    "hard": "¿Es dura?",
    "bar": "¿Es una barra?",
    "pluribus": "¿Viene en bolsa/varios?",
    "sugarpercent": "Percentil de azúcar (0–1)",
    "pricepercent": "Percentil de precio (0–1)",
    "winpercent": "Popularidad, % de duelos ganados (0–100)",
}


def _parse_form(form) -> tuple[str, dict]:
    """Valida el formulario y devuelve (escenario, {feature: valor})."""
    scen = form.get("scenario", "comp")
    if scen not in ("comp", "full"):
        raise ValueError("Escenario inválido.")

    vals = {c: 1.0 if form.get(c) == "on" else 0.0 for c in BINARY_COLS}
    for c in CONT_COLS:
        try:
            v = float(form.get(c, ""))
        except ValueError:
            raise ValueError(f"«{c}» debe ser numérico.") from None
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"«{c}» debe estar en [0, 1] (es un percentil).")
        vals[c] = v

    if scen == "full":
        try:
            w = float(form.get(WIN_COL, ""))
        except ValueError:
            raise ValueError(f"«{WIN_COL}» es obligatorio en el escenario "
                             "con winpercent y debe ser numérico.") from None
        if not 0.0 <= w <= 100.0:
            raise ValueError(f"«{WIN_COL}» debe estar en [0, 100].")
        vals[WIN_COL] = w
    return scen, vals


@app.get("/")
def index():
    return render_template(
        "index.html",
        binary=BINARY_COLS,
        cont=CONT_COLS,
        win=WIN_COL,
        labels=FEATURE_LABELS,
        form={},
        result=None,
        error=None,
    )


@app.post("/predict")
def do_predict():
    ctx = {"binary": BINARY_COLS, "cont": CONT_COLS, "win": WIN_COL,
           "labels": FEATURE_LABELS, "result": None, "error": None}
    try:
        scen, vals = _parse_form(request.form)
        vector = [vals[c] for c in pred.feature_cols(scen)]
        result = pred.predict(vector, scen)
        result["plot"] = pred.scatter_png_b64(result)
        ctx["result"] = result
        ctx["form"] = {"scenario": scen, **vals}
    except ValueError as e:
        ctx["error"] = str(e)
        ctx["form"] = request.form.to_dict()
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interfaz web del predictor — TP2")
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 (default, usar con túnel SSH) o "
                             "0.0.0.0 (visible en la red)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    # El debugger de Werkzeug permite ejecutar código: solo con localhost
    debug = args.host == "127.0.0.1"
    app.run(host=args.host, port=args.port, debug=debug)
