"""Randfälle: alle Signale, Längen, Wavelets, Stufen, Verfahren, Rauschpegel, Cycle Spinning, Abbildungen und PDF."""

import math

import numpy as np
import pytest

import shr_constants as C
import shr_evaluation as ev
import shr_visualization as V
from shr_pdf_export import build_pdf
from sweeps_cache import get


@pytest.mark.parametrize("signal", list(C.SIGNALS))
def test_every_signal_runs_for_the_shortest_and_the_longest_record_and_every_method(signal):
    for n in (min(C.N_CHOICES), max(C.N_CHOICES)):
        for p in (C.WAVELET_P[0], C.WAVELET_P[-1]):
            for method in C.METHODS:
                a = ev.analyse(ev.Settings(n=n, signal=signal, p=p, levels=min(C.MAX_LEVELS, int(math.log2(n)) - 1), method=method, fixed=2.0))
                assert np.isfinite(a.estimate).all() and a.mse_est >= 0 and a.kept <= a.total and len(a.thresholds) == len(a.before)


@pytest.mark.parametrize("rule", list(C.RULES))
def test_every_rule_sigma_mode_cycle_and_extreme_snr_runs(rule):
    for sigma_mode in C.SIGMA_MODES:
        for cycle in C.CYCLE_MODES:
            for snr in (C.SNR_MIN, C.SNR_MAX):
                a = ev.analyse(ev.Settings(n=256, signal="Doppler", snr=snr, p=2, levels=4, rule=rule, sigma_mode=sigma_mode, cycle=cycle, method="Hybrid"))
                assert np.isfinite(a.estimate).all() and a.sigma > 0


def test_extreme_thresholds_and_seeds_run():
    for fixed in (C.FIXED_MIN, C.FIXED_MAX):
        for seed in (0, C.SEED_MAX):
            a = ev.analyse(ev.Settings(n=256, signal="Bumps", levels=4, method="fest", fixed=fixed, seed=seed))
            assert np.isfinite(a.estimate).all()
    assert ev.analyse(ev.Settings(n=256, levels=4, method="fest", fixed=0.0)).kept == ev.analyse(ev.Settings(n=256, levels=4, method="fest", fixed=0.0)).total
    assert ev.analyse(ev.Settings(n=256, levels=4, method="fest", fixed=C.FIXED_MAX, snr=1.0, rule="hart")).kept == 0


def test_a_single_level_and_the_largest_level_count_run():
    for n, levels in ((256, 1), (256, 7), (4096, 9)):
        a = ev.analyse(ev.Settings(n=n, levels=levels, method="SURE"))
        assert len(a.before) == levels and a.total == n - n // 2 ** levels


@pytest.mark.parametrize("signal", list(C.SIGNALS))
def test_figures_and_pdf_build_for_every_signal(signal):
    for cycle in C.CYCLE_MODES:
        a = ev.analyse(ev.Settings(n=512, signal=signal, levels=5, cycle=cycle, method="Hybrid"))
        for fig in (V.build_signals(a), V.build_error(a), V.build_coefficients(a), V.build_sorted(a), V.build_thresholds(a)):
            assert len(fig.data) >= 1
        assert build_pdf(a)[:4] == b"%PDF"


def test_the_sorted_figure_marks_the_kept_count_only_when_there_is_one():
    kept = ev.analyse(ev.Settings(n=512, levels=5))
    none = ev.analyse(ev.Settings(n=256, levels=4, method="fest", fixed=C.FIXED_MAX, rule="hart", snr=1.0))
    assert len(V.build_sorted(kept).layout.shapes) == 2 and len(V.build_sorted(none).layout.shapes) == 1
    assert V.coefficient_map(kept.before).shape == (5, 512)


def test_figures_for_the_experiments():
    assert len(V.build_methods(get("methods")).data) == 6 and len(V.build_curve(get("curve")).data) == 4 and len(V.build_cycle(get("cycle")).data) == 2 and len(V.build_nterm(get("nterm")).data) == 12
    assert len(V.build_sigma(get("noise")).data) == 1 and len(V.build_wrong_sigma(get("noise")).data) == 1 and V.level_labels(2) == ["Stufe 1", "Stufe 2"]
