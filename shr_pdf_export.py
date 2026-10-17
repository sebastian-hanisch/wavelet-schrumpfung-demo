"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import shr_constants as C
from shr_evaluation import Analysis, de, wavelet_name

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "β": "beta", "π": "pi",
            "≥": ">=", "λ": "lambda", "ν": "nu"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Wavelet-Schrumpfung: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    fixed = f", feste Schwelle {de(s.fixed, 1)}" if s.method == "fest" else ""
    pdf.multi_cell(0, 5, _s(f"{s.signal}, n = {s.n}, Signal-Rausch-Verhältnis {de(s.snr, 1)}, {wavelet_name(s.p)}, {s.levels} Stufen, Verfahren {C.METHOD_LABELS[s.method]}{fixed}, {s.rule}e Schwelle, "
                            f"Rauschpegel {s.sigma_mode}, Cycle Spinning {s.cycle}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [f"Fehler (mittlere quadratische Abweichung je Abtastwert gegen die Wahrheit): verrauscht {de(a.mse_noisy, 3)}, nach der Schrumpfung {de(a.mse_est, 3)}.",
           f"Rauschpegel sigma = {de(a.sigma, 3)}; universale Schwelle {de(a.universal, 2)}.",
           "Schwellen je Stufe (Stufe 1 bis J): " + ", ".join(de(t, 2) for t in a.thresholds) + ".",
           f"Detailkoeffizienten ungleich null nach der Schwelle: {a.kept} von {a.total}."]
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: wavelet-schrumpfung-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
