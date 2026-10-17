"""Daubechies-Filter und periodische DWT (Pyramide), auch für mehrere Reihen zugleich (Form (..., n)). Reine Funktionen, nur numpy."""

import math

import numpy as np


def daubechies(p: int) -> np.ndarray:
    """Skalierungsfilter h der Länge 2p (Summe sqrt 2, Summe der Quadrate 1, Extremalphase) der Daubechies-Wavelet mit p verschwindenden Momenten (p = 1: Haar), per Spektralfaktorisierung."""
    if p < 1:
        raise ValueError("p muss mindestens 1 sein")
    if p == 1:
        return np.array([1.0, 1.0]) / math.sqrt(2.0)
    coef = [math.comb(p - 1 + k, k) for k in range(p)]
    zs = []
    for y in np.roots(coef[::-1]):
        r = np.roots([1.0, -(2.0 - 4.0 * y), 1.0])
        zs.append(r[np.argmin(np.abs(r))])
    poly = np.array([1.0 + 0j])
    for _ in range(p):
        poly = np.convolve(poly, [1.0, 1.0])
    for z in zs:
        poly = np.convolve(poly, [1.0, -z])
    h = np.real(poly)
    h = h / h.sum() * math.sqrt(2.0)
    return h[::-1] if abs(h[0]) < abs(h[-1]) else h


def qmf(h) -> np.ndarray:
    """Wavelet-Filter g[l] = (-1)^l h[L - 1 - l] (Quadraturspiegelfilter)."""
    h = np.asarray(h, dtype=float)
    length = len(h)
    return np.array([(-1) ** l * h[length - 1 - l] for l in range(length)])


def check_levels(n: int, levels: int) -> None:
    if levels < 1:
        raise ValueError("Stufenzahl muss mindestens 1 sein")
    if n % (2 ** levels) != 0:
        raise ValueError("Die Länge muss durch 2^J teilbar sein")


def dwt(x, h, levels: int):
    """Periodische DWT per Pyramide: ([d_1, ..., d_J], a_J); c[k] = sum_l h[l] a[(2k + l) mod N], d[k] = sum_l g[l] a[(2k + l) mod N]; x hat die Form (..., n)."""
    a = np.asarray(x, dtype=float)
    check_levels(a.shape[-1], levels)
    h = np.asarray(h, dtype=float)
    g = qmf(h)
    length = len(h)
    details = []
    for _ in range(levels):
        size = a.shape[-1]
        idx = (2 * np.arange(size // 2)[:, None] + np.arange(length)[None, :]) % size
        window = a[..., idx]
        details.append(window @ g)
        a = window @ h
    return details, a


def idwt(details, a, h) -> np.ndarray:
    """Inverse der periodischen DWT (perfekte Rekonstruktion), Form (..., n)."""
    h = np.asarray(h, dtype=float)
    g = qmf(h)
    length = len(h)
    a = np.asarray(a, dtype=float)
    for d in reversed(details):
        d = np.asarray(d, dtype=float)
        m = d.shape[-1]
        out = np.zeros(d.shape[:-1] + (2 * m,))
        for l in range(length):
            idx = (2 * np.arange(m) + l) % (2 * m)
            out[..., idx] += a * h[l] + d * g[l]
        a = out
    return a
