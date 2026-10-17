"""Fehlermaße und Kennzahlen der Schrumpfung: mittlerer quadratischer Fehler, Fehler gegen die Schwelle (Orakel), Näherung mit k Termen. Reine Funktionen, nur numpy."""

import numpy as np

import shr_shrink as S
import shr_wavelet as W


def mse(a, b) -> np.ndarray:
    """Mittlerer quadratischer Fehler je Abtastwert über die letzte Achse."""
    return np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2, axis=-1)


def threshold_errors(y, f, h, levels: int, lams, rule: str = "weich") -> np.ndarray:
    """Wahrer Fehler der globalen Schwelle lam (gleich für alle Stufen) für jede Reihe von y: Form (Reihen, Schwellen)."""
    details, smooth = W.dwt(y, h, levels)
    return np.stack([mse(W.idwt([S.apply_rule(d, lam, rule) for d in details], smooth, h), f) for lam in lams], axis=-1)


def oracle_global(y, f, h, levels: int, lams, rule: str = "weich"):
    """Beste globale Schwelle je Reihe (kleinster wahrer Fehler auf dem Gitter): (Fehler je Reihe, Schwelle je Reihe)."""
    errs = threshold_errors(y, f, h, levels, lams, rule)
    best = np.argmin(errs, axis=-1)
    return np.min(errs, axis=-1), np.asarray(lams)[best]


def wavelet_coefficients(x, h, levels: int) -> np.ndarray:
    """Alle DWT-Koeffizienten (Details aller Stufen, dann die Glättung) in einem Vektor."""
    d, a = W.dwt(x, h, levels)
    return np.concatenate([np.asarray(v) for v in d] + [np.asarray(a)])


def n_term_error_wavelet(x, h, levels: int, ks) -> list[float]:
    """Relativer L2-Fehler der besten k-Term-Näherung in der DWT-Basis (orthogonal: Fehler aus den verworfenen Koeffizienten)."""
    flat = wavelet_coefficients(x, h, levels)
    order = np.sort(flat ** 2)[::-1]
    total = float(order.sum())
    return [float(np.sqrt(max(total - order[:k].sum(), 0.0) / total)) for k in ks]


def n_term_error_fourier(x, ks) -> list[float]:
    """Relativer L2-Fehler der besten k-Term-Näherung in der Fourier-Basis: die k betragsgrößten komplexen Koeffizienten bleiben (ein konjugiertes Paar zählt als zwei)."""
    x = np.asarray(x, dtype=float)
    spec = np.fft.fft(x)
    errs = []
    for k in ks:
        kept = S.best_n_term(spec, k)
        errs.append(float(np.linalg.norm(np.fft.ifft(kept).real - x) / np.linalg.norm(x)))
    return errs


def terms_for_energy(values, share: float = 0.99) -> int:
    """Kleinste Zahl betragsgrößter Koeffizienten, die `share` der Energie tragen."""
    e = np.sort(np.abs(np.asarray(values)) ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(e) / e.sum(), share)) + 1
