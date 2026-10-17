"""Daubechies-Filter und periodische DWT (auch für mehrere Reihen) gegen Formel, Tabelle und Rekonstruktion."""

import math

import numpy as np
import pytest

import shr_wavelet as W


def test_d4_equals_the_closed_form_and_haar_by_hand():
    ref = np.array([1 + math.sqrt(3), 3 + math.sqrt(3), 3 - math.sqrt(3), 1 - math.sqrt(3)]) / (4 * math.sqrt(2))
    assert W.daubechies(2) == pytest.approx(ref, abs=1e-14)
    assert W.daubechies(1).tolist() == pytest.approx([1 / math.sqrt(2)] * 2) and W.qmf(W.daubechies(1)).tolist() == pytest.approx([1 / math.sqrt(2), -1 / math.sqrt(2)])
    assert W.qmf([1.0, 2.0, 3.0, 4.0]).tolist() == [4.0, -3.0, 2.0, -1.0]


@pytest.mark.parametrize("p", [1, 2, 3, 4, 6, 8])
def test_filters_are_orthonormal_with_vanishing_moments(p):
    h = W.daubechies(p)
    g = W.qmf(h)
    assert len(h) == 2 * p and h.sum() == pytest.approx(math.sqrt(2), abs=1e-12) and (h ** 2).sum() == pytest.approx(1.0, abs=1e-12) and np.dot(h, g) == pytest.approx(0.0, abs=1e-12)
    for k in range(1, p):
        assert np.dot(h[2 * k:], h[:len(h) - 2 * k]) == pytest.approx(0.0, abs=1e-10)
    for m in range(p):
        assert abs(np.sum(g * np.arange(len(g)) ** m)) < 1e-6 * max(1.0, (2 * p) ** m)
    with pytest.raises(ValueError):
        W.daubechies(0)


def test_haar_mini_instance_by_hand(mini_x):
    h = W.daubechies(1)
    d, a = W.dwt(mini_x, h, 1)
    assert d[0].tolist() == pytest.approx([-1 / math.sqrt(2)] * 2) and a.tolist() == pytest.approx([3 / math.sqrt(2), 7 / math.sqrt(2)])
    d2, a2 = W.dwt(mini_x, h, 2)
    assert d2[1].tolist() == pytest.approx([-2.0]) and a2.tolist() == pytest.approx([5.0]) and W.idwt(d2, a2, h) == pytest.approx(mini_x)
    assert W.idwt([np.zeros(2)], a, h) == pytest.approx([1.5, 1.5, 3.5, 3.5])


def test_dwt_equals_the_explicit_sum_definition():
    h = W.daubechies(2)
    g = W.qmf(h)
    x = np.random.default_rng(1).normal(size=16)
    d, a = W.dwt(x, h, 2)
    c1 = np.array([sum(h[l] * x[(2 * k + l) % 16] for l in range(4)) for k in range(8)])
    d1 = np.array([sum(g[l] * x[(2 * k + l) % 16] for l in range(4)) for k in range(8)])
    d2 = np.array([sum(g[l] * c1[(2 * k + l) % 8] for l in range(4)) for k in range(4)])
    assert d[0] == pytest.approx(d1) and d[1] == pytest.approx(d2)


@pytest.mark.parametrize("p", [1, 2, 4, 8])
@pytest.mark.parametrize("n,J", [(64, 1), (256, 5), (512, 9)])
def test_perfect_reconstruction_and_energy(p, n, J):
    x = np.random.default_rng(2).normal(size=n)
    h = W.daubechies(p)
    d, a = W.dwt(x, h, J)
    assert W.idwt(d, a, h) == pytest.approx(x, abs=1e-12) and sum((v ** 2).sum() for v in d) + (a ** 2).sum() == pytest.approx((x ** 2).sum(), rel=1e-12)


def test_batches_equal_single_rows_and_keep_the_leading_shape():
    h = W.daubechies(4)
    x = np.random.default_rng(3).normal(size=(2, 3, 128))
    d, a = W.dwt(x, h, 4)
    assert [v.shape for v in d] == [(2, 3, 64), (2, 3, 32), (2, 3, 16), (2, 3, 8)] and a.shape == (2, 3, 8)
    single, sa = W.dwt(x[1, 2], h, 4)
    assert all(d[j][1, 2] == pytest.approx(single[j]) for j in range(4)) and a[1, 2] == pytest.approx(sa)
    rec = W.idwt(d, a, h)
    assert rec.shape == x.shape and rec == pytest.approx(x, abs=1e-12) and W.idwt(single, sa, h) == pytest.approx(x[1, 2], abs=1e-12)


def test_level_checks():
    h = W.daubechies(2)
    for n, J in ((100, 3), (64, 0), (64, -1), (8, 4)):
        with pytest.raises(ValueError):
            W.dwt(np.zeros(n), h, J)
    W.check_levels(64, 6)
    with pytest.raises(ValueError):
        W.check_levels(64, 7)
