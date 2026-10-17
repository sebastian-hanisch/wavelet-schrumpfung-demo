"""Fehlermaße und Kennzahlen: mittlerer quadratischer Fehler, Fehler gegen die Schwelle, Orakel, Näherung mit k Termen."""

import numpy as np
import pytest

import shr_resolution as R
import shr_shrink as S
import shr_wavelet as W
import signal_generator as G


def test_mse_by_hand_and_over_the_last_axis():
    assert float(R.mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])) == pytest.approx(4.0 / 3.0)
    assert R.mse(np.array([[1.0, 3.0], [0.0, 0.0]]), np.zeros((2, 2))).tolist() == [5.0, 0.0]


def test_threshold_errors_at_zero_equal_the_noise_and_at_infinity_the_smooth_error():
    h = W.daubechies(2)
    f = np.sin(np.linspace(0, 6, 64))
    y = f + G.SplitMix64(1).normal(64)
    errs = R.threshold_errors(y[None], f, h, 4, (0.0, 1e9))
    assert errs.shape == (1, 2) and errs[0, 0] == pytest.approx(float(R.mse(y, f)))
    d, a = W.dwt(y, h, 4)
    assert errs[0, 1] == pytest.approx(float(R.mse(W.idwt([np.zeros_like(v) for v in d], a, h), f)))
    hard = R.threshold_errors(y[None], f, h, 4, (0.0,), "hart")
    assert hard[0, 0] == pytest.approx(float(R.mse(y, f)))


def test_oracle_global_picks_the_smallest_error_per_row():
    h = W.daubechies(2)
    f = np.zeros(64)
    f[20:30] = 6.0
    y = f + G.SplitMix64(2).normal(3 * 64).reshape(3, 64)
    lams = np.linspace(0.0, 4.0, 9)
    best, lam = R.oracle_global(y, f, h, 3, lams)
    errs = R.threshold_errors(y, f, h, 3, lams)
    assert best.tolist() == pytest.approx(errs.min(axis=1).tolist()) and lam.tolist() == lams[errs.argmin(axis=1)].tolist() and np.all(best <= errs[:, 0] + 1e-12)


def test_wavelet_coefficients_and_n_term_error_follow_parseval():
    h = W.daubechies(4)
    x = np.random.default_rng(3).normal(size=128)
    flat = R.wavelet_coefficients(x, h, 4)
    assert len(flat) == 128 and (flat ** 2).sum() == pytest.approx((x ** 2).sum())
    ks = (1, 8, 32, 128)
    errs = R.n_term_error_wavelet(x, h, 4, ks)
    for k, e in zip(ks, errs):
        d, a = W.dwt(x, h, 4)
        kept = S.best_n_term(flat, k)
        parts = np.split(kept, np.cumsum([len(v) for v in d]))
        rec = W.idwt(parts[:-1], parts[-1], h)
        assert e == pytest.approx(np.linalg.norm(rec - x) / np.linalg.norm(x), abs=1e-9)
    assert errs == sorted(errs, reverse=True) and errs[-1] == pytest.approx(0.0, abs=1e-6)


def test_fourier_n_term_error_and_full_set():
    x = np.random.default_rng(4).normal(size=64)
    errs = R.n_term_error_fourier(x, (2, 16, 64))
    assert errs[-1] == pytest.approx(0.0, abs=1e-12) and errs[0] > errs[1] > 0
    t = np.arange(64)
    tone = np.cos(2 * np.pi * 5 * t / 64)
    assert R.n_term_error_fourier(tone, (2,))[0] == pytest.approx(0.0, abs=1e-12)          # ein Kosinus = ein konjugiertes Paar


def test_terms_for_energy_by_hand():
    assert R.terms_for_energy(np.array([3.0, 4.0, 0.0])) == 2 and R.terms_for_energy(np.array([4.0, 3.0]), 0.5) == 1
    assert R.terms_for_energy(np.array([1.0, 1.0, 1.0, 1.0]), 0.5) == 2 and R.terms_for_energy(np.array([1.0, 1.0, 1.0, 1.0])) == 4
    assert R.terms_for_energy(np.array([1.0 + 1j, 0.1])) == 1
