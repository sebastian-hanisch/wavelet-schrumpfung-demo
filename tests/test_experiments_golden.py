"""Festwerte der Experimente (feste Zufallsströme): Das Ergebnis ist bei jedem Lauf dasselbe; Abweichungen bedeuten eine geänderte Rechnung."""

import pytest

from sweeps_cache import get

METHODS = {'Blocks': (1.006948, 1.662208, 0.516118, 0.364867, 0.513474, 0.41918, 1.028, 1.049325, 0.049302), 'Bumps': (1.008111, 2.011019, 0.533736, 0.394656, 0.457424, 0.450306, 0.998, 1.053411, 0.054372), 'HeaviSine': (1.004827, 0.184851, 0.121913, 0.105438, 0.141515, 0.110993, 1.854, 1.002931, 0.054111), 'Doppler': (0.99791, 0.756942, 0.239002, 0.228925, 0.28484, 0.260001, 1.352, 1.027229, 0.057032)}
CURVE = {'Blocks': (1.0, 0.413713, 3.8, 1.586295, 0.99907, 2.349946), 'Bumps': (1.0, 0.443398, 3.8, 1.896652, 1.000764, 2.888471), 'HeaviSine': (1.8, 0.110516, 3.8, 0.186235, 0.997314, 0.228664), 'Doppler': (1.4, 0.268333, 3.8, 0.750288, 1.004876, 1.111159)}
CYCLE = {(1, 'Blocks', 'weich'): (0.724668, 0.533909, 2.737873, 2.547851), (1, 'Blocks', 'hart'): (0.098968, 0.052067, 0.887777, 0.872836), (1, 'HeaviSine', 'weich'): (0.721585, 0.127556, 2.797428, 1.976662), (1, 'HeaviSine', 'hart'): (0.337365, 0.083492, 3.039149, 1.475259), (1, 'Doppler', 'weich'): (2.089568, 1.084221, 3.012982, 0.448026), (1, 'Doppler', 'hart'): (0.963511, 0.276743, 1.968338, 0.819629), (4, 'Blocks', 'weich'): (1.675587, 1.340327, 5.266958, 4.750324), (4, 'Blocks', 'hart'): (0.521211, 0.230522, 3.259494, 2.118402), (4, 'HeaviSine', 'weich'): (0.189861, 0.143822, 2.492588, 2.35009), (4, 'HeaviSine', 'hart'): (0.125703, 0.058836, 2.65668, 2.206504), (4, 'Doppler', 'weich'): (0.755267, 0.546629, 0.601673, 0.300461), (4, 'Doppler', 'hart'): (0.248568, 0.118657, 0.40546, 0.298841)}
NTERM = {'Blocks': ([0.470956, 0.299741, 0.17077, 0.0, 0.0], [0.496012, 0.351589, 0.19729, 0.111053, 0.030977], [0.420573, 0.354097, 0.252152, 0.165935, 0.115934], 70, 168), 'Bumps': ([0.710047, 0.589217, 0.426413, 0.217294, 0.081448], [0.735923, 0.60818, 0.411124, 0.221685, 0.065427], [0.806555, 0.731628, 0.619339, 0.464472, 0.308262], 107, 410), 'HeaviSine': ([0.380287, 0.200238, 0.110442, 0.05127, 0.024847], [0.339995, 0.057445, 0.019475, 0.00184, 1.6e-05], [0.098243, 0.065435, 0.046056, 0.032711, 0.02307], 12, 9), 'Doppler': ([0.527587, 0.379129, 0.267191, 0.170111, 0.094322], [0.545984, 0.324351, 0.151131, 0.041536, 0.005725], [0.645932, 0.467503, 0.324021, 0.21271, 0.129339], 40, 175)}
SIGMA = {'Blocks': (1.051352, 0.059843), 'Bumps': (1.059518, 0.055783), 'HeaviSine': (1.002413, 0.049959), 'Doppler': (1.025434, 0.049919)}
ZERO = {256: (0.813333, 0.760411), 1024: (0.826667, 0.785705), 4096: (0.816667, 0.804376)}
WRONG = {0.5: 0.583472, 0.75: 1.006893, 1.0: 1.533592, 1.5: 2.727085, 2.0: 3.93584}


def test_methods_golden():
    rows = get("methods")
    assert [r["signal"] for r in rows] == list(METHODS)
    for r in rows:
        assert (r["noisy"], r["universal_soft"], r["universal_hard"], r["sure"], r["hybrid"], r["oracle"], r["lam_oracle"], r["sigma_mean"], r["sigma_sd"]) == pytest.approx(METHODS[r["signal"]], abs=1e-6) and r["runs"] == 100


def test_curve_golden():
    rows = get("curve")
    assert [r["signal"] for r in rows] == list(CURVE) and len(rows[0]["lams"]) == 26
    for r in rows:
        assert (r["best_lam"], r["best_error"], r["universal_lam"], r["universal_error"], r["error_zero"], r["error_max"]) == pytest.approx(CURVE[r["signal"]], abs=1e-6)


def test_cycle_golden():
    rows = get("cycle")
    assert len(rows) == len(CYCLE) == 12
    for r in rows:
        assert (r["plain"], r["cycle"], r["window_plain"], r["window_cycle"]) == pytest.approx(CYCLE[(r["p"], r["signal"], r["rule"])], abs=1e-6) and r["runs"] == 40


def test_nterm_golden():
    rows = get("nterm")
    assert [r["signal"] for r in rows] == list(NTERM)
    for r in rows:
        haar, d8, fourier, c8, cf = NTERM[r["signal"]]
        assert r["haar"] == pytest.approx(haar, abs=1e-6) and r["d8"] == pytest.approx(d8, abs=1e-6) and r["fourier"] == pytest.approx(fourier, abs=1e-6) and (r["count_d8"], r["count_fourier"]) == (c8, cf)


def test_noise_golden():
    res = get("noise")
    for r in res["sigma"]:
        assert (r["mean"], r["sd"]) == pytest.approx(SIGMA[r["signal"]], abs=1e-6)
    for r in res["zero"]:
        assert (r["share"], r["theory"]) == pytest.approx(ZERO[r["n"]], abs=1e-6)
    for r in res["wrong"]:
        assert r["error"] == pytest.approx(WRONG[r["factor"]], abs=1e-6)
