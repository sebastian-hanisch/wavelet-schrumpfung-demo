"""Auswertung: Einstellungen, Messung, Analyse und die Experimentfunktionen (kleine Instanzen von Hand, feste Werte)."""

import math
import pickle

import numpy as np
import pytest

import shr_constants as C
import shr_evaluation as ev
import shr_shrink as S
import shr_signals as SG
import shr_wavelet as W
import signal_generator as G


def test_settings_flags_names_and_the_decimal_comma():
    s = ev.Settings()
    assert (s.n, s.signal, s.snr, s.p, s.levels, s.method, s.rule, s.sigma_mode, s.cycle) == (1024, "Blocks", 7.0, 4, 6, "universal", "weich", "geschätzt (MAD)", "aus")
    assert s.cycle_on is False and s.sigma_known is False and ev.Settings(cycle="an").cycle_on is True and ev.Settings(sigma_mode="bekannt").sigma_known is True
    assert ev.wavelet_name(1) == "Haar" and ev.wavelet_name(4) == "D8" and ev.wavelet_name(8) == "D16"
    assert ev.de(0.12345, 3) == "0,123" and ev.de(2.0) == "2,0"


def test_measurement_is_the_scaled_signal_plus_unit_noise():
    s = ev.Settings(n=256, signal="Doppler", snr=3.0, seed=5)
    f, y = ev.build_measurement(s)
    assert f.tolist() == pytest.approx((3.0 * SG.shape("Doppler", 256)).tolist()) and y - f == pytest.approx(G.SplitMix64(5).normal(256))
    assert ev.build_measurement(ev.Settings(n=256, seed=6))[1].tolist() != ev.build_measurement(ev.Settings(n=256, seed=5))[1].tolist()


def test_analyse_the_five_presets_golden():
    def run(name, **over):
        p = C.PRESETS[name]
        base = dict(n=p["n"], signal=p["signal"], snr=p["snr"], p=p["p"], levels=p["levels"], method=p["method"], rule=p["rule"], fixed=p["fixed"], sigma_mode=p["sigma"], cycle=p["cycle"], seed=p["seed"])
        return ev.analyse(ev.Settings(**{**base, **over}))
    a = run("Universale Schwelle auf Blocks: schlechter als das Rauschen")
    assert (a.mse_noisy, a.mse_est) == pytest.approx((0.955242, 1.589853), abs=1e-6) and (a.kept, a.total) == (88, 1008) and a.sigma == pytest.approx(1.043088, abs=1e-6) and a.universal == pytest.approx(3.883726, abs=1e-6)
    assert a.thresholds == pytest.approx([3.8837] * 6, abs=1e-4)
    b = run("SURE auf Blocks: der Fehler sinkt auf ein Drittel")
    assert b.mse_est == pytest.approx(0.299722, abs=1e-6) and b.kept == 272 and b.thresholds == pytest.approx([1.6831, 1.4051, 0.7521, 0.3388, 0.4404, 0.0], abs=1e-4)
    c = run("Cycle Spinning: der Fehler halbiert sich")
    assert c.mse_est == pytest.approx(0.047993, abs=1e-6) and c.kept == 8 and run("Cycle Spinning: der Fehler halbiert sich", cycle="aus").mse_est == pytest.approx(0.101467, abs=1e-6)
    d = run("HeaviSine: wenige Koeffizienten tragen alles")
    assert d.mse_est == pytest.approx(0.101467, abs=1e-6) and d.kept == 8 and d.sigma == pytest.approx(0.96851, abs=1e-5)
    e = run("Niedriger Rauschabstand: die Schwelle zählt")
    assert e.mse_est == pytest.approx(0.157108, abs=1e-6) and e.kept == 6
    assert [run("Niedriger Rauschabstand: die Schwelle zählt", method=m, rule=r).mse_est for m, r in (("universal", "hart"), ("SURE", "weich"), ("Hybrid", "weich"), ("fest", "weich"))] == pytest.approx(
        [0.106247, 0.100538, 0.109105, 0.172133], abs=1e-6)


def test_analyse_with_known_sigma_and_the_structure_of_the_result():
    a = ev.analyse(ev.Settings(method="Hybrid", sigma_mode="bekannt"))
    assert a.sigma == 1.0 and a.mse_est == pytest.approx(0.446908, abs=1e-6) and a.thresholds == pytest.approx([3.723, 1.405, 0.752, 0.339, 0.44, 0.0], abs=1e-3)
    assert len(a.before) == len(a.after) == len(a.thresholds) == 6 and [len(d) for d in a.before] == [512, 256, 128, 64, 32, 16] and a.estimate.shape == a.truth.shape == a.noisy.shape == (1024,)
    assert a.kept == sum(int(np.count_nonzero(d)) for d in a.after) and a.total == 1008


def test_analyse_equals_the_shrink_functions():
    s = ev.Settings(n=256, signal="Bumps", snr=5.0, p=2, levels=4, method="SURE", rule="hart", seed=3)
    a = ev.analyse(s)
    f, y = ev.build_measurement(s)
    h = W.daubechies(2)
    sigma = float(S.mad_sigma(W.dwt(y, h, 1)[0][0]))
    assert a.sigma == pytest.approx(sigma) and a.estimate == pytest.approx(S.shrink(y, h, 4, sigma, "SURE", "hart")[0])
    cs = ev.analyse(ev.Settings(n=256, signal="Bumps", snr=5.0, p=2, levels=4, method="SURE", rule="hart", seed=3, cycle="an"))
    assert cs.estimate == pytest.approx(S.cycle_spin(y, h, 4, sigma, "SURE", "hart"), abs=1e-12) and cs.thresholds == a.thresholds and cs.kept == a.kept


def test_cycle_shifts_are_capped():
    s = ev.Settings(n=512, signal="HeaviSine", p=2, levels=8, rule="hart", cycle="an")
    a = ev.analyse(s)
    f, y = ev.build_measurement(s)
    h = W.daubechies(2)
    sigma = float(S.mad_sigma(W.dwt(y, h, 1)[0][0]))
    assert a.estimate == pytest.approx(S.cycle_spin(y, h, 8, sigma, "universal", "hart", 1.0, max_shifts=64), abs=1e-12) and C.MAX_SHIFTS == 64


def test_the_fixed_threshold_is_used_only_with_the_fixed_method():
    base = dict(n=256, signal="Doppler", p=2, levels=4)
    assert ev.analyse(ev.Settings(**base, method="fest", fixed=2.0)).thresholds == [2.0] * 4 and ev.analyse(ev.Settings(**base, method="fest", fixed=0.0)).kept == ev.analyse(ev.Settings(**base, method="fest", fixed=0.0)).total
    assert ev.analyse(ev.Settings(**base, method="universal", fixed=2.0)).thresholds == ev.analyse(ev.Settings(**base, method="universal", fixed=3.0)).thresholds


def test_analysis_is_picklable():
    a = ev.analyse(ev.Settings(n=256, levels=4, cycle="an"))
    b = pickle.loads(pickle.dumps(a))
    assert b.mse_est == a.mse_est and b.settings == a.settings and (b.estimate == a.estimate).all()


def test_sweep_methods_small_instance():
    rows = ev.sweep_methods(runs=6, n=256, levels=4, lams=(0.0, 1.0, 2.0, 3.0))
    assert [r["signal"] for r in rows] == list(SG.NAMES) and rows[0]["runs"] == 6
    b = rows[0]
    assert (b["noisy"], b["universal_soft"], b["universal_hard"], b["sure"], b["hybrid"], b["oracle"]) == pytest.approx((1.006385, 4.004525, 1.64117, 0.723576, 0.985455, 0.752758), abs=1e-6)
    assert b["lam_oracle"] == 1.0 and b["sigma_mean"] == pytest.approx(1.266084, abs=1e-6) and b["lam_universal"] == pytest.approx(math.sqrt(2 * math.log(256)))
    assert rows[1]["sure"] == pytest.approx(0.878915, abs=1e-6) and rows[1]["sigma_sd"] == pytest.approx(0.083507, abs=1e-6)


def test_sweep_curve_small_instance():
    rows = ev.sweep_curve(runs=6, n=256, levels=4, lams=(0.0, 1.0, 2.0, 3.0, 4.0))
    got = [(r["signal"], r["best_lam"], round(r["best_error"], 6), r["universal_lam"], round(r["universal_error"], 6), round(r["error_max"], 6)) for r in rows]
    assert got == [("Blocks", 1.0, 0.784338, 3.0, 2.549724, 3.772399), ("Bumps", 1.0, 0.771253, 3.0, 2.838233, 4.441348), ("HeaviSine", 2.0, 0.240518, 3.0, 0.278621, 0.299057),
                   ("Doppler", 1.0, 0.427202, 3.0, 1.093207, 1.671893)]
    assert rows[0]["lams"] == [0.0, 1.0, 2.0, 3.0, 4.0] and len(rows[0]["errors"]) == 5 and rows[0]["error_zero"] == rows[0]["errors"][0] and rows[0]["runs"] == 6


def test_sweep_cycle_small_instance():
    rows = ev.sweep_cycle(runs=4, ps=(1,), signals=("Blocks", "Doppler"), n=256, levels=4, window=10)
    got = [(r["signal"], r["rule"], round(r["plain"], 6), round(r["cycle"], 6), round(r["window_plain"], 6), round(r["window_cycle"], 6)) for r in rows]
    assert got == [("Blocks", "weich", 1.27281, 1.039456, 3.327881, 2.944628), ("Blocks", "hart", 0.241002, 0.155087, 1.533591, 1.263994), ("Doppler", "weich", 4.767018, 2.679772, 4.477827, 0.99078),
                   ("Doppler", "hart", 2.30794, 0.692921, 3.521928, 1.059114)] and rows[0]["p"] == 1 and rows[0]["runs"] == 4


def test_sweep_nterm_small_instance():
    rows = ev.sweep_nterm(ks=(4, 16), n=256, levels=4)
    got = [(r["signal"], [round(v, 6) for v in r["haar"]], [round(v, 6) for v in r["fourier"]], r["count_d8"], r["count_fourier"]) for r in rows]
    assert got == [("Blocks", [0.687298, 0.30846], [0.503265, 0.349923], 58, 113), ("Bumps", [0.797176, 0.566025], [0.865937, 0.736179], 71, 197), ("HeaviSine", [0.646193, 0.201245], [0.193226, 0.065139], 12, 9),
                   ("Doppler", [0.735606, 0.373994], [0.783227, 0.466796], 33, 144)] and rows[0]["ks"] == [4, 16] and len(rows[0]["d8"]) == 2


def test_sweep_noise_small_instance():
    r = ev.sweep_noise(runs=10, n=256, zero_ns=(256,), zero_runs=20, factors=(0.5, 1.0), factor_runs=4, levels=4)
    assert [(x["signal"], round(x["mean"], 6), round(x["sd"], 6)) for x in r["sigma"]] == [("Blocks", 1.146512, 0.140212), ("Bumps", 1.417107, 0.141313), ("HeaviSine", 1.081857, 0.079269), ("Doppler", 1.098560, 0.116291)]
    assert r["zero"][0]["share"] == 0.85 and r["zero"][0]["n"] == 256 and r["zero"][0]["theory"] == pytest.approx(1 - 1 / math.sqrt(math.pi * math.log(256))) and r["runs"] == 10
    assert [(x["factor"], round(x["error"], 6)) for x in r["wrong"]] == [(0.5, 1.173516), (1.0, 3.022769)]
