"""Wavelet-Schrumpfung: Schwellen (weich, hart), universale Schwelle, SURE je Stufe, Hybrid (SureShrink), Rauschpegel aus dem Median, Cycle Spinning und beste n-Term-Näherung.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Alle Funktionen arbeiten auch auf mehreren Reihen zugleich (Form (..., n), σ je Reihe).
"""

import math

import numpy as np

import shr_wavelet as W

METHODS = ("universal", "SURE", "Hybrid", "fest")
RULES = ("weich", "hart")


def soft(d, lam):
    """Weiche Schwelle sign(d) (|d| - lam)_+."""
    d = np.asarray(d, dtype=float)
    return np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)


def hard(d, lam):
    """Harte Schwelle d 1{|d| > lam}."""
    d = np.asarray(d, dtype=float)
    return np.where(np.abs(d) > lam, d, 0.0)


def apply_rule(d, lam, rule: str):
    if rule == "weich":
        return soft(d, lam)
    if rule == "hart":
        return hard(d, lam)
    raise ValueError(f"unbekannte Schwelle: {rule}")


def universal_threshold(sigma, n: int):
    """Universale Schwelle sigma sqrt(2 ln n)."""
    return np.asarray(sigma, dtype=float) * math.sqrt(2.0 * math.log(n))


def mad_sigma(d1):
    """Rauschpegel aus dem Median der Beträge der Detailkoeffizienten der Stufe 1: median(|d|) / 0,6745 (je Reihe)."""
    return np.median(np.abs(np.asarray(d1, dtype=float)), axis=-1) / 0.6745


def sure_risk(z, lam) -> float:
    """Steins unverzerrter Risikoschätzer der weichen Schwelle für Koeffizienten mit Rauschvarianz 1: m - 2 #{|z| <= lam} + sum min(|z|, lam)^2 (eine Reihe)."""
    z = np.abs(np.asarray(z, dtype=float))
    return float(len(z) - 2 * np.sum(z <= lam) + np.sum(np.minimum(z, lam) ** 2))


def sure_threshold(z, lam_max):
    """Schwelle aus {0} ∪ {|z_k|} bis lam_max, die SURE minimiert (Rauschvarianz 1), je Reihe; z hat die Form (..., m)."""
    a = np.sort(np.abs(np.asarray(z, dtype=float)), axis=-1)
    m = a.shape[-1]
    k = np.arange(1, m + 1)
    risk = m - 2 * k + np.cumsum(a ** 2, axis=-1) + (m - k) * a ** 2
    risk = np.where(a <= np.asarray(lam_max)[..., None] if np.ndim(lam_max) else a <= lam_max, risk, np.inf)
    best = np.argmin(risk, axis=-1)
    lam = np.take_along_axis(a, best[..., None], axis=-1)[..., 0]
    return np.where(np.min(risk, axis=-1) < m, lam, 0.0)       # SURE(0) = m: nur eine echte Verbesserung hebt die Schwelle über null


def level_threshold(d, sigma, method: str, n: int, fixed: float = 0.0):
    """Schwelle für die Detailkoeffizienten einer Stufe (Form (..., m), sigma je Reihe): universal, SURE (je Stufe), Hybrid oder fest (fixed mal sigma... nein: fixed ist die Schwelle selbst)."""
    d = np.asarray(d, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    m = d.shape[-1]
    if method == "fest":
        return np.broadcast_to(np.asarray(fixed, dtype=float), sigma.shape).copy()
    universal = universal_threshold(sigma, n)
    if method == "universal":
        return universal
    z = d / sigma[..., None]
    sure = sigma * sure_threshold(z, math.sqrt(2.0 * math.log(m)))
    if method == "SURE":
        return sure
    if method == "Hybrid":
        eta = (np.sum(z ** 2, axis=-1) - m) / m
        return np.where(eta <= math.log2(m) ** 1.5 / math.sqrt(m), universal, sure)
    raise ValueError(f"unbekanntes Verfahren: {method}")


def shrink_details(details, sigma, method: str, rule: str, n: int, fixed: float = 0.0):
    """(geschwellte Details, Schwellen je Stufe); die Glättung bleibt unberührt."""
    out, thresholds = [], []
    for d in details:
        lam = level_threshold(d, sigma, method, n, fixed)
        thresholds.append(lam)
        out.append(apply_rule(d, np.asarray(lam)[..., None], rule))
    return out, thresholds


def shrink(y, h, levels: int, sigma, method: str, rule: str, fixed: float = 0.0):
    """Entrauschen: DWT, Schwelle auf den Details, Rücktransformation; ((..., n)-Schätzung, Schwellen je Stufe)."""
    y = np.asarray(y, dtype=float)
    details, smooth = W.dwt(y, h, levels)
    new, thresholds = shrink_details(details, sigma, method, rule, y.shape[-1], fixed)
    return W.idwt(new, smooth, h), thresholds


def cycle_spin(y, h, levels: int, sigma, method: str, rule: str, fixed: float = 0.0, max_shifts: int = 64) -> np.ndarray:
    """Cycle Spinning: Mittel über min(2^J, max_shifts) zirkuläre Verschiebungen (verschieben, schwellen, zurückschieben); eine Reihe y (n,), sigma Skalar."""
    y = np.asarray(y, dtype=float)
    count = min(2 ** levels, max_shifts)
    rolled = np.stack([np.roll(y, k) for k in range(count)])
    est, _ = shrink(rolled, h, levels, np.full(count, float(sigma)), method, rule, fixed)
    return np.mean([np.roll(est[k], -k) for k in range(count)], axis=0)


def best_n_term(coefs, k: int) -> np.ndarray:
    """Beste k-Term-Näherung: die k betragsgrößten Koeffizienten bleiben, der Rest wird null."""
    coefs = np.asarray(coefs)
    out = np.zeros_like(coefs)
    idx = np.argsort(np.abs(coefs))[::-1][:k]
    out[idx] = coefs[idx]
    return out
