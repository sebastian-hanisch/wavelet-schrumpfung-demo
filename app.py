"""Wavelet-Schrumpfung: Schwellwerte entscheiden – bei Rauschabstand 7 ist die universale Schwelle zu hoch, SURE und Cycle Spinning senken den Fehler - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 13 der Reihe „Spektral- und Wavelet-Analyse“. Die vier Testsignale von Donoho und Johnstone (Blocks, Bumps, HeaviSine, Doppler) werden verrauscht und per Wavelet-Schrumpfung entrauscht
(DWT, Schwelle auf den Details, Rücktransformation); gezeigt werden universale Schwelle, SURE, Hybrid, feste Schwelle, Cycle Spinning und fünf Experimente zu Schwellwerten, Fehlerkurve,
Cycle Spinning, Kompression und Rauschpegel. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import shr_constants as C
import shr_evaluation as ev
from shr_evaluation import Settings, analyse, de, wavelet_name
from shr_pdf_export import build_pdf
from shr_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from shr_visualization import (
    build_coefficients,
    build_curve,
    build_cycle,
    build_error,
    build_methods,
    build_nterm,
    build_signals,
    build_sigma,
    build_sorted,
    build_thresholds,
    build_wrong_sigma,
)

st.set_page_config(page_title="Wavelet-Schrumpfung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("✂️ Wavelet-Schrumpfung: Schwellwerte entscheiden")
st.markdown(
    """
Wavelet-Schrumpfung entrauscht in drei Schritten: Die Reihe wird mit der **DWT** zerlegt, die Detailkoeffizienten werden mit einer **Schwelle** geschrumpft (klein gleich Rauschen, groß gleich Signal), und die Rücktransformation
liefert die Schätzung. Das funktioniert, weil glatte Signale und Sprünge in der Wavelet-Basis **dünn** sind: Wenige große Koeffizienten tragen das Signal, das Rauschen verteilt sich gleichmäßig auf alle. Die **weiche** Schwelle
zieht große Koeffizienten um $\\lambda$ herunter, die **harte** lässt sie unverändert. Die Frage ist die Höhe der Schwelle: die **universale** $\\sigma\\sqrt{2\\ln n}$, **SURE** (aus den Daten je Stufe) oder ein Hybrid.
**Cycle Spinning** mittelt über Verschiebungen und gleicht die Verschiebungsvarianz der DWT aus.

Dies ist die dreizehnte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf die **diskrete Wavelet-Transformation** aufbauend (die Koeffizienten, ihre Sparsität und die Verschiebungsvarianz). Alles ist **simuliert**: Signal und
Rauschen sind bekannt, der Fehler wird gegen die Wahrheit gemessen. Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So entsteht die Schrumpfung", expanded=True):
    st.markdown(
        """
1. **Zerlegen:** DWT mit $J$ Stufen; die Glättung (die gröbste Stufe) bleibt unverändert, nur die Details werden geschwellt.
2. **Rauschpegel:** $\\hat\\sigma = \\operatorname{median}(|d_1|)/0{,}6745$ aus der feinsten Stufe (oder bekannt).
3. **Schwelle:** weich $\\operatorname{sign}(d)(|d| - \\lambda)_+$ oder hart $d\\,\\mathbf 1\\{|d| > \\lambda\\}$; $\\lambda$ ist universal $\\hat\\sigma\\sqrt{2\\ln n}$, SURE je Stufe, Hybrid oder fest.
4. **Zurückrechnen:** Die geschwellten Details und die Glättung ergeben die Schätzung; der Fehler gegen die Wahrheit ist die mittlere quadratische Abweichung.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    kind = st.selectbox("Signal", list(C.SIGNALS), key="signal_select", help="Blocks: Stufen. Bumps: Spitzen. HeaviSine: glatt mit zwei Sprüngen. Doppler: schnell oszillierend am Anfang.")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Zahl der Abtastwerte (Zweierpotenz).")
    snr = st.slider("Signal-Rausch-Verhältnis", C.SNR_MIN, C.SNR_MAX, float(st.session_state["snr_slider"]), step=C.SNR_STEP, key="snr_slider",
                    help="Standardabweichung des Signals bei Rauschen N(0, 1): 1 heißt Rauschen so stark wie das Signal, 7 das Rauschen klein dagegen.")
    st.markdown("**Wavelet und Schwelle**")
    p = st.select_slider("Wavelet (Daubechies)", options=list(C.WAVELET_P), key="p_select", format_func=wavelet_name, help="p verschwindende Momente, Filterlänge 2p; Haar ist die einfachste.")
    max_levels = min(C.MAX_LEVELS, int(math.log2(n)) - 1)
    if st.session_state["levels_select"] > max_levels:
        st.session_state["levels_select"] = max_levels
    levels = st.select_slider("Stufen J", options=list(range(C.LEVEL_MIN, max_levels + 1)), key="levels_select", help="Zahl der Zerlegungsstufen; höchstens log₂(n) − 1.")
    method = st.radio("Verfahren", list(C.METHODS), key="method_select", format_func=lambda m: C.METHOD_LABELS[m],
                      help="Universal: eine Schwelle σ√(2 ln n) für alle Stufen. SURE: je Stufe die Schwelle, die Steins Risikoschätzer minimiert. Hybrid (SureShrink): SURE, bei dünnen Stufen die universale. Fest: eine selbst gewählte Schwelle.")
    rule = st.radio("Schwelle", list(C.RULES), key="rule_select", horizontal=True, help="Weich zieht große Koeffizienten um λ herunter, hart lässt sie unverändert.")
    if method == "fest":
        fixed = st.slider("Feste Schwelle λ", C.FIXED_MIN, C.FIXED_MAX, float(st.session_state["fixed_kept"]), step=C.FIXED_STEP, help="Schwelle in Einheiten des Signals (Rauschen N(0, 1) hat σ = 1).")
        st.session_state["fixed_kept"] = float(fixed)
    else:
        fixed = float(st.session_state["fixed_kept"])
    sigma_mode = st.radio("Rauschpegel", list(C.SIGMA_MODES), key="sigma_select", help="Median der Stufe 1 geteilt durch 0,6745 oder das wahre σ = 1.")
    cycle = st.radio("Cycle Spinning", list(C.CYCLE_MODES), key="cycle_select", horizontal=True, help=f"Mittelt über bis zu {C.MAX_SHIFTS} zirkuläre Verschiebungen (verschieben, schwellen, zurückschieben).")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "signal_select": kind, "snr_slider": float(snr), "p_select": int(p), "levels_select": int(levels), "method_select": method, "rule_select": rule, "fixed_kept": float(fixed),
                   "sigma_select": sigma_mode, "cycle_select": cycle, "seed_input": int(seed)})

settings = Settings(int(n), kind, float(snr), int(p), int(levels), method, rule, float(fixed), sigma_mode, cycle, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Fehler der Schätzung", de(a.mse_est, 3), delta=f"verrauscht {de(a.mse_noisy, 3)}", delta_color="off", help="Mittlere quadratische Abweichung je Abtastwert gegen die Wahrheit.")
m2.metric("Verbesserung", f"Faktor {de(a.mse_noisy / a.mse_est, 1)}" if a.mse_est > 0 else "–", delta=f"{de(10 * math.log10(a.mse_noisy / a.mse_est), 1)} dB" if a.mse_est > 0 else "", delta_color="off",
          help="Fehler des verrauschten Signals geteilt durch den Fehler der Schätzung; unter 1: das Verfahren verschlechtert das Signal.")
m3.metric("Behaltene Koeffizienten", f"{a.kept} von {a.total}", delta=f"{de(100 * a.kept / a.total, 1)} % der Details", delta_color="off", help="Detailkoeffizienten ungleich null nach der Schwelle (ohne Verschiebung).")
m4.metric("Rauschpegel σ", de(a.sigma, 3), delta=f"universale Schwelle {de(a.universal, 2)}", delta_color="off", help="Verwendeter Pegel (MAD der Stufe 1 oder bekannt) und die daraus folgende universale Schwelle σ√(2 ln n).")
st.plotly_chart(build_signals(a), width="stretch", key="signals")
st.plotly_chart(build_error(a), width="stretch", key="error")
st.caption("Fehler der Schätzung (grün) gegen das Rauschen (grau): Die Schwelle nimmt das Rauschen weg und lässt an den Sprüngen und Spitzen Abweichungen zurück (Gibbs-Artefakte der DWT; Cycle Spinning verkleinert sie).")
st.plotly_chart(build_coefficients(a), width="stretch", key="coefficients")
st.caption("Beträge der Detailkoeffizienten je Stufe vor und nach der Schwelle (dB gegen den größten Betrag); die Koeffizienten der Stufe j liegen im Abstand von 2^j Abtastwerten (ihr Träger ist länger). Wenige große Koeffizienten bleiben.")
st.plotly_chart(build_sorted(a), width="stretch", key="sorted")
st.caption("Alle Detailkoeffizienten nach Größe geordnet: Das Signal sitzt in wenigen großen, das Rauschen im flachen Rest; die universale Schwelle liegt am oberen Ende des Rauschens.")
st.plotly_chart(build_thresholds(a), width="stretch", key="thresholds")
st.caption("Schwelle je Stufe: Universal und fest sind gleich für alle Stufen, SURE und Hybrid wählen je Stufe (die gröbste Stufe mit wenigen Koeffizienten bekommt eine kleine oder keine Schwelle). Bei Cycle Spinning ist es die Schwelle ohne Verschiebung.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="wavelet-schrumpfung.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie hoch soll die Schwelle sein?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption(f"Feste Zufallsströme, unabhängig von den Reglern. n = {C.EXP_N}, Signal-Rausch-Verhältnis {C.EXP_SNR:g}, Rauschen N(0, 1), D8, {C.EXP_J} Stufen; Fehler = mittlere quadratische Abweichung je Abtastwert gegen die Wahrheit, "
           "„verrauscht“ ist der Fehler ohne Schwelle (etwa 1).")

st.markdown("#### Schwellwerte im Vergleich")
methods = _experiment("methods")
st.caption(f"{methods[0]['runs']} Läufe je Signal, σ aus dem Median der Stufe 1; „beste globale Schwelle“ ist das Orakel mit der besten gleichen Schwelle für alle Stufen je Lauf (Gitter 0 bis 5).")
st.dataframe(pd.DataFrame([{"Signal": r["signal"], "verrauscht": de(r["noisy"], 3), "universal weich": de(r["universal_soft"], 3), "universal hart": de(r["universal_hard"], 3), "SURE": de(r["sure"], 3),
                            "Hybrid": de(r["hybrid"], 3), "beste globale Schwelle": de(r["oracle"], 3), "beste Schwelle λ": de(r["lam_oracle"], 2), "σ̂": f"{de(r['sigma_mean'], 3)} ± {de(r['sigma_sd'], 3)}"} for r in methods]),
             hide_index=True, width="stretch")
st.plotly_chart(build_methods(methods), width="stretch", key="exp_methods")
kind_v, text = ev.verdict_methods(methods)
getattr(st, kind_v)(text)

st.markdown("#### Fehler als Funktion der Schwelle")
curve = _experiment("curve")
st.caption(f"Globale weiche Schwelle λ, σ = 1 bekannt, {curve[0]['runs']} Läufe: Das Minimum und der Fehler bei der universalen Schwelle (nächster Gitterpunkt).")
st.dataframe(pd.DataFrame([{"Signal": r["signal"], "beste Schwelle": de(r["best_lam"], 1), "Fehler dort": de(r["best_error"], 3), "universale Schwelle": de(r["universal_lam"], 1), "Fehler dort ": de(r["universal_error"], 3),
                            "λ = 0 (Rauschen)": de(r["error_zero"], 3), "λ = 5": de(r["error_max"], 3)} for r in curve]), hide_index=True, width="stretch")
st.plotly_chart(build_curve(curve), width="stretch", key="exp_curve")
kind_v, text = ev.verdict_curve(curve)
getattr(st, kind_v)(text)

st.markdown("#### Cycle Spinning")
cycle_rows = _experiment("cycle")
st.caption(f"Universale Schwelle ohne und mit Mitteln über alle 2^J = {2 ** C.EXP_J} Verschiebungen, {cycle_rows[0]['runs']} Läufe; zusätzlich der größte Fehler im Fenster ± {C.EXP_CYCLE_WINDOW} Abtastwerte um den Sprung "
           "(Blocks 0,1 n, HeaviSine 0,3 n; bei Doppler 0,5 n, ohne echten Sprung).")
st.dataframe(pd.DataFrame([{"Wavelet": wavelet_name(r["p"]), "Signal": r["signal"], "Schwelle": r["rule"], "DWT": de(r["plain"], 3), "Cycle Spinning": de(r["cycle"], 3), "Fenster: DWT": de(r["window_plain"], 2),
                            "Fenster: Cycle Spinning": de(r["window_cycle"], 2)} for r in cycle_rows]), hide_index=True, width="stretch")
st.plotly_chart(build_cycle(cycle_rows), width="stretch", key="exp_cycle")
kind_v, text = ev.verdict_cycle(cycle_rows)
getattr(st, kind_v)(text)

st.markdown("#### Kompression: beste k-Term-Näherung")
nterm = _experiment("nterm")
st.caption(f"Relativer L2-Fehler der besten Näherung mit k Koeffizienten (kein Rauschen, n = {C.EXP_N}): Haar, D8 und Fourier (k größte komplexe Koeffizienten, ein konjugiertes Paar zählt als zwei).")
st.dataframe(pd.DataFrame([{"Signal": r["signal"], "Basis": label, **{f"k = {k}": de(v, 4) for k, v in zip(r["ks"], r[key])}} for r in nterm for label, key in (("Haar", "haar"), ("D8", "d8"), ("Fourier", "fourier"))]),
             hide_index=True, width="stretch")
st.plotly_chart(build_nterm(nterm), width="stretch", key="exp_nterm")
st.dataframe(pd.DataFrame([{"Signal": r["signal"], "D8": r["count_d8"], "Fourier": r["count_fourier"]} for r in nterm]), hide_index=True, width="stretch")
st.caption("Zahl der Koeffizienten, die 99 % der Energie tragen.")
kind_v, text = ev.verdict_nterm(nterm)
getattr(st, kind_v)(text)

st.markdown("#### Rauschpegel und universale Schwelle")
noise = _experiment("noise")
st.caption(f"σ̂ = Median der Stufe 1 / 0,6745 ({noise['runs']} Läufe, D8). Darunter bei reinem Rauschen der Anteil der Läufe, in denen die universale Schwelle alle Details nullt (D4, 4 Stufen, {C.EXP_ZERO_RUNS} Läufe), und der "
           "Fehler bei Blocks mit falschem σ (universal weich).")
st.dataframe(pd.DataFrame([{"Signal": r["signal"], "σ̂ (wahr 1)": f"{de(r['mean'], 3)} ± {de(r['sd'], 3)}"} for r in noise["sigma"]]), hide_index=True, width="stretch")
st.plotly_chart(build_sigma(noise), width="stretch", key="exp_sigma")
st.dataframe(pd.DataFrame([{"n": r["n"], "alle Details null": f"{de(100 * r['share'], 1)} %", "Näherung 1 − 1/√(π ln n)": f"{de(100 * r['theory'], 1)} %"} for r in noise["zero"]]), hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"verwendetes σ / wahres σ": de(r["factor"], 2), "Fehler (Blocks)": de(r["error"], 3)} for r in noise["wrong"]]), hide_index=True, width="stretch")
st.plotly_chart(build_wrong_sigma(noise), width="stretch", key="exp_wrong")
kind_v, text = ev.verdict_noise(noise)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die universale Schwelle ist das Richtige** | Sie ist auf „kein Rauschen bleibt übrig“ gebaut, nicht auf kleinen Fehler: Bei Rauschabstand 7 liegt der Fehler der weichen universalen Schwelle bei Blocks und Bumps über dem des verrauschten Signals. | SURE, Hybrid, Minimax-Schwellen |
| **Das Rauschen ist weiß mit bekanntem σ** | Der Pegel aus der Stufe 1 ist bei Blocks und Bumps 5 bis 6 % zu hoch, bei HeaviSine nicht; farbiges Rauschen braucht eine Schwelle je Stufe. | stufenabhängiges σ_j |
| **Die DWT ist verschiebungsinvariant** | Die Schwelle auf der DWT hängt von der Lage der Sprünge ab (Artefakte am Sprung); Cycle Spinning mildert das (Fehler bei harter Schwelle etwa halbiert, Spitzenfehler am Sprung um bis zu einem Drittel kleiner). | Cycle Spinning, MODWT |
| **Die Reihe ist periodisch** | Die periodische Fortsetzung macht Randsprünge; Blocks hat einen. | Spiegeln (nicht gebaut) |
| **Das Signal ist dünn in der Wavelet-Basis** | Beim Signal HeaviSine braucht Fourier für 99 % der Energie weniger Koeffizienten als D8 (9 gegen 12); bei Doppler, einer glatten Schwingung, ist es umgekehrt (175 gegen 40). | Basiswahl, Wavelet-Pakete |
| **Ein Fehlermaß für alles** | Der Fehler ist die mittlere quadratische Abweichung; visuelle Güte (Artefakte) ist davon nicht gedeckt. | – |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $y_i = f_i + \sigma\,\varepsilon_i$ mit $\varepsilon_i \sim N(0, 1)$. Die orthogonale DWT $W$ liefert $d = W y = W f + \sigma\,W\varepsilon$ (das Rauschen bleibt weiß, $\sigma^2$ je Koeffizient).

**Schwellen.** $\eta_S(d, \lambda) = \operatorname{sign}(d)(|d| - \lambda)_+$ (weich) und $\eta_H(d, \lambda) = d\,\mathbf 1\{|d| > \lambda\}$ (hart); die Schätzung ist $\hat f = W^{\mathsf T}\,\eta(Wy, \lambda)$ (nur auf den Details).

**Universale Schwelle.** $\lambda_U = \sigma\sqrt{2\ln n}$: Mit großer Wahrscheinlichkeit übersteigt kein Rauschkoeffizient die Schwelle (bei reinem Rauschen etwa $1 - 1/\sqrt{\pi \ln n}$); das Risiko liegt höchstens um den Faktor $2\ln n + 1$ über dem idealen Risiko (damit nahe am Minimax).

**SURE.** Für Koeffizienten $z_i = d_i/\sigma$ mit Rauschvarianz 1 ist $\mathrm{SURE}(\lambda) = m - 2\,\#\{|z_i| \le \lambda\} + \sum_i \min(|z_i|, \lambda)^2$ ein unverzerrter Schätzer des Risikos der weichen Schwelle (Stein 1981).
SureShrink minimiert ihn je Stufe über $\lambda \in \{0\} \cup \{|z_i|\}$, $\lambda \le \sqrt{2\ln m}$; der **Hybrid** nimmt $\lambda_U$ bei Stufen mit $\frac1m\sum(z_i^2 - 1) \le (\log_2 m)^{3/2}/\sqrt m$.

**Rauschpegel.** $\hat\sigma = \operatorname{median}(|d_{1,k}|)/0{,}6745$.

**Cycle Spinning.** $\hat f = \frac1S \sum_{s} T_{-s}\,\mathcal S(T_s y)$ mit der zirkulären Verschiebung $T_s$ und dem Schrumpfungsoperator $\mathcal S$ über $S = \min(2^J, 64)$ Verschiebungen.

**n-Term-Näherung.** Bei einer orthogonalen Basis ist der Fehler der besten $k$-Term-Näherung $\sqrt{\sum_{i > k} c_{(i)}^2}$ mit den nach Größe geordneten Koeffizienten.

**Quellen.** Lehrbuchwissen und Aufsätze (Donoho und Johnstone 1994 und 1995; Coifman und Donoho 1995; Stein 1981). Formeln hier per Simulation, Handrechnung und gegen geschlossene Ausdrücke geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html)."
)
