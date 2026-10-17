"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import shr_constants as C
from shr_evaluation import Analysis


def _fixed(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def level_labels(levels: int) -> list[str]:
    return [f"Stufe {j}" for j in range(1, levels + 1)]


def build_signals(a: Analysis) -> go.Figure:
    """Verrauschtes Signal (grau), Wahrheit (rot gestrichelt) und Schätzung (grün)."""
    t = np.arange(len(a.truth))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.noisy, mode="lines", line=dict(color=C.COLOR_NOISY, width=1), name="verrauscht", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.truth, mode="lines", line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), name="Wahrheit", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.estimate, mode="lines", line=dict(color=C.COLOR_EST, width=2), name="Schätzung", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", range=[0, len(t) - 1])
    fig.update_yaxes(title_text="Signal")
    return _fixed(fig, height=380)


def build_error(a: Analysis) -> go.Figure:
    """Fehler der Schätzung (grün) und das Rauschen (grau) über der Zeit."""
    t = np.arange(len(a.truth))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.noisy - a.truth, mode="lines", line=dict(color=C.COLOR_NOISY, width=1), name="Rauschen", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=a.estimate - a.truth, mode="lines", line=dict(color=C.COLOR_EST, width=1.6), name="Fehler der Schätzung", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", range=[0, len(t) - 1])
    fig.update_yaxes(title_text="Abweichung von der Wahrheit")
    return _fixed(fig, height=240)


def coefficient_map(details) -> np.ndarray:
    """Beträge der Detailkoeffizienten je Stufe auf die Zeitachse gelegt (ein Koeffizient der Stufe j gilt für 2^j Abtastwerte); Zeilen Stufe 1 .. J."""
    return np.array([np.repeat(np.abs(d), 2 ** j) for j, d in enumerate(details, start=1)])


def build_coefficients(a: Analysis) -> go.Figure:
    """Beträge der Detailkoeffizienten je Stufe vor und nach der Schwelle (gleiche Farbskala, dB gegen den größten Betrag vor der Schwelle, bei −60 dB abgeschnitten)."""
    before, after = coefficient_map(a.before), coefficient_map(a.after)
    top = float(before.max())

    def db(z):
        return np.maximum(20.0 * np.log10(np.maximum(z, 1e-300) / top), -60.0) if top > 0 else np.full(z.shape, -60.0)
    labels = level_labels(len(a.before))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("vor der Schwelle", "nach der Schwelle"), vertical_spacing=0.12)
    for row, z in ((1, db(before)), (2, db(after))):
        fig.add_trace(go.Heatmap(x=np.arange(z.shape[1]), y=labels, z=z, colorscale="Viridis", zmin=-60.0, zmax=0.0, showscale=row == 1, colorbar=dict(title="dB", thickness=12, len=0.9),
                                 hovertemplate="t = %{x}, %{y}: %{z:.1f} dB<extra></extra>"), row=row, col=1)
    fig.update_xaxes(title_text="Zeit (Abtastwerte)", row=2, col=1)
    return _fixed(fig, height=520)


def build_sorted(a: Analysis) -> go.Figure:
    """Betragsgrößen aller Detailkoeffizienten absteigend (log) vor der Schwelle; waagerecht die universale Schwelle; die Zahl der behaltenen Koeffizienten markiert."""
    flat = np.sort(np.abs(np.concatenate([np.asarray(d) for d in a.before])))[::-1]
    fig = go.Figure(go.Scatter(x=np.arange(1, len(flat) + 1), y=np.maximum(flat, 1e-6), mode="lines", line=dict(color=C.COLOR_BLUE, width=1.6), name="Betrag vor der Schwelle",
                               hovertemplate="Rang %{x}: %{y:.3g}<extra></extra>"))
    fig.add_hline(y=a.universal, line=dict(color=C.COLOR_AUX, width=1.4, dash="dot"), annotation_text="universale Schwelle", annotation_position="top right")
    if a.kept:
        fig.add_vline(x=a.kept, line=dict(color=C.COLOR_EST, width=1.4, dash="dash"), annotation_text=f"{a.kept} bleiben", annotation_position="top")
    fig.update_xaxes(title_text="Rang der Koeffizienten", type="log")
    fig.update_yaxes(title_text="Betrag", type="log")
    return _fixed(fig, height=300)


def build_thresholds(a: Analysis) -> go.Figure:
    """Schwelle je Stufe (bei SURE und Hybrid je Stufe verschieden) neben der universalen Schwelle."""
    labels = level_labels(len(a.thresholds))
    fig = go.Figure(go.Bar(x=labels, y=a.thresholds, marker_color=C.COLOR_EST, hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=a.universal, line=dict(color=C.COLOR_AUX, width=1.4, dash="dot"), annotation_text="universal", annotation_position="top right")
    fig.update_yaxes(title_text="Schwelle", rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=260)


def build_methods(rows) -> go.Figure:
    fig = go.Figure()
    names = (("noisy", "verrauscht"), ("universal_soft", "universal weich"), ("universal_hard", "universal hart"), ("sure", "SURE"), ("hybrid", "Hybrid"), ("oracle", "beste globale Schwelle (Orakel)"))
    for key, label in names:
        fig.add_trace(go.Bar(x=[r["signal"] for r in rows], y=[r[key] for r in rows], name=label, hovertemplate=label + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Fehler (mittlere quadratische Abweichung)", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=360)


def build_curve(rows) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Scatter(x=r["lams"], y=r["errors"], mode="lines+markers", name=r["signal"], hovertemplate=r["signal"] + ", λ = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_vline(x=rows[0]["universal_lam"], line=dict(color="black", width=1, dash="dot"), annotation_text="universal", annotation_position="top")
    fig.update_xaxes(title_text="globale weiche Schwelle λ")
    fig.update_yaxes(title_text="Fehler", type="log")
    return _fixed(fig, height=340)


def build_cycle(rows) -> go.Figure:
    sub = [r for r in rows if r["p"] == 4]
    labels = [f"{r['signal']} {r['rule']}" for r in sub]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["plain"] for r in sub], name="DWT", marker_color=C.COLOR_BLUE, hovertemplate="%{x}: %{y:.3f}<extra>DWT</extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["cycle"] for r in sub], name="Cycle Spinning", marker_color=C.COLOR_EST, hovertemplate="%{x}: %{y:.3f}<extra>Cycle Spinning</extra>"))
    fig.update_yaxes(title_text="Fehler (D8, universale Schwelle)", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _fixed(fig, height=340)


def build_nterm(rows) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(rows), subplot_titles=[r["signal"] for r in rows], shared_yaxes=True)
    for i, r in enumerate(rows, start=1):
        for key, label in (("haar", "Haar"), ("d8", "D8"), ("fourier", "Fourier")):
            fig.add_trace(go.Scatter(x=r["ks"], y=[max(v, 1e-4) for v in r[key]], mode="lines+markers", name=label, showlegend=i == 1, legendgroup=label,
                                     hovertemplate=label + ", k = %{x}: %{y:.4f}<extra></extra>"), row=1, col=i)
    fig.update_xaxes(title_text="Zahl k der Terme", type="log")
    fig.update_yaxes(title_text="relativer Fehler (bis 1e-4 abgeschnitten)", type="log", row=1, col=1)
    return _fixed(fig, height=340)


def build_sigma(res) -> go.Figure:
    rows = res["sigma"]
    fig = go.Figure(go.Bar(x=[r["signal"] for r in rows], y=[r["mean"] for r in rows], error_y=dict(type="data", array=[r["sd"] for r in rows]), marker_color=C.COLOR_BLUE,
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), annotation_text="wahr", annotation_position="top left")
    fig.update_yaxes(title_text="geschätzter Rauschpegel σ̂", range=[0.8, 1.2])
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=300)


def build_wrong_sigma(res) -> go.Figure:
    rows = res["wrong"]
    fig = go.Figure(go.Scatter(x=[r["factor"] for r in rows], y=[r["error"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_EST), hovertemplate="σ-Faktor %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="verwendetes σ geteilt durch das wahre σ")
    fig.update_yaxes(title_text="Fehler (Blocks, universal weich)", rangemode="tozero")
    return _fixed(fig, height=300)
