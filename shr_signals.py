"""Die vier Testsignale von Donoho und Johnstone (1994): Blocks, Bumps, HeaviSine, Doppler. Reine Funktionen, kein Zufall.

Die Formen sind auf die Standardabweichung 1 skaliert; mit dem Rauschen N(0, 1) ist die Amplitude `snr` das Verhältnis der Standardabweichungen von Signal und Rauschen.
"""

import numpy as np

KNOTS = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])
BLOCKS_H = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 5.1, -4.2])
BUMPS_H = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
BUMPS_W = np.array([0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005])
NAMES = ("Blocks", "Bumps", "HeaviSine", "Doppler")


def _grid(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def blocks_raw(n: int) -> np.ndarray:
    """Summe von Stufen h_j (1 + sgn(t - t_j))/2."""
    t = _grid(n)
    return sum(h * (1.0 + np.sign(t - tj)) / 2.0 for h, tj in zip(BLOCKS_H, KNOTS))


def bumps_raw(n: int) -> np.ndarray:
    """Summe von Spitzen h_j (1 + |(t - t_j)/w_j|)^-4."""
    t = _grid(n)
    return sum(h * (1.0 + np.abs((t - tj) / w)) ** -4 for h, tj, w in zip(BUMPS_H, KNOTS, BUMPS_W))


def heavisine_raw(n: int) -> np.ndarray:
    """4 sin(4 pi t) - sgn(t - 0,3) - sgn(0,72 - t): glatt mit zwei Sprüngen."""
    t = _grid(n)
    return 4.0 * np.sin(4.0 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)


def doppler_raw(n: int) -> np.ndarray:
    """sqrt(t (1 - t)) sin(2 pi 1,05 / (t + 0,05)): schnell oszillierend am Anfang."""
    t = _grid(n)
    return np.sqrt(t * (1.0 - t)) * np.sin(2.0 * np.pi * 1.05 / (t + 0.05))


RAW = {"Blocks": blocks_raw, "Bumps": bumps_raw, "HeaviSine": heavisine_raw, "Doppler": doppler_raw}


def shape(name: str, n: int) -> np.ndarray:
    """Signalform auf die Standardabweichung 1 skaliert (der Mittelwert bleibt)."""
    if name not in RAW:
        raise ValueError(f"unbekanntes Signal: {name}")
    f = RAW[name](n)
    return f / f.std()


def signal(name: str, n: int, snr: float) -> np.ndarray:
    """Wahres Signal der Standardabweichung snr (Rauschen N(0, 1))."""
    return snr * shape(name, n)
