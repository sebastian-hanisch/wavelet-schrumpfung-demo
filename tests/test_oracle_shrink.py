"""Schwellen, SURE, Hybrid, Rauschpegel, Cycle Spinning und beste n-Term-Näherung gegen Handrechnung und Simulation."""

import math

import numpy as np
import pytest

import shr_shrink as S
import shr_wavelet as W
import signal_generator as G


def test_soft_and_hard_by_hand():
    assert S.soft(3.0, 1.0) == 2.0 and S.soft(-3.0, 2.0) == -1.0 and S.soft(-1.0, 1.0) == 0.0 and S.soft(0.5, 1.0) == 0.0 and S.soft(1.0, 1.0) == 0.0
    assert S.hard(2.0, 2.0) == 0.0 and S.hard(2.0, 1.9) == 2.0 and S.hard(-2.0, 1.9) == -2.0
    assert S.soft(np.array([3.0, -1.0, 0.5]), 1.0).tolist() == [2.0, 0.0, 0.0] and S.hard(np.array([3.0, -1.0, 0.5]), 0.9).tolist() == [3.0, -1.0, 0.0]
    assert S.apply_rule(3.0, 1.0, "weich") == 2.0 and S.apply_rule(3.0, 1.0, "hart") == 3.0
    with pytest.raises(ValueError):
        S.apply_rule(3.0, 1.0, "mittel")


def test_universal_threshold_and_mad_by_hand():
    assert float(S.universal_threshold(1.0, 1024)) == pytest.approx(math.sqrt(2 * math.log(1024))) == pytest.approx(3.7233, abs=1e-4)
    assert float(S.universal_threshold(2.0, 256)) == pytest.approx(2 * math.sqrt(2 * math.log(256)))
    assert float(S.mad_sigma(np.array([1.0, -2.0, 3.0]))) == pytest.approx(2.0 / 0.6745)
    assert S.mad_sigma(np.array([[1.0, -2.0, 3.0], [4.0, 4.0, 4.0]])).tolist() == pytest.approx([2.0 / 0.6745, 4.0 / 0.6745])
    x = G.SplitMix64(1).normal(100 * 1024).reshape(100, 1024)
    assert S.mad_sigma(x).mean() == pytest.approx(1.0, abs=0.02)


def test_sure_by_hand(mini_d):
    assert [S.sure_risk(mini_d, lam) for lam in (0.0, 0.5, 1.0, 3.0)] == pytest.approx([4.0, 3.0, 3.25, 10.25])
    assert float(S.sure_threshold(mini_d, 5.0)) == 0.5
    assert float(S.sure_threshold(mini_d, 0.4)) == 0.0        # nur 0 ist erlaubt
    assert float(S.sure_threshold(np.array([5.0, 6.0, 7.0, 8.0]), 10.0)) == 0.0   # reines Signal: keine Schwelle verbessert SURE(0) = m


def test_sure_threshold_equals_the_direct_loop_and_handles_batches():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(4, 64))
    z[:, :6] += 5
    lam_max = math.sqrt(2 * math.log(64))
    batch = S.sure_threshold(z, lam_max)
    for row, b in zip(z, batch):
        cand = np.concatenate([[0.0], np.sort(np.abs(row))])
        cand = cand[cand <= lam_max]
        direct = cand[int(np.argmin([S.sure_risk(row, l) for l in cand]))]
        assert b == pytest.approx(direct)
    per_row = S.sure_threshold(z, np.array([0.5, 1.0, 2.0, 3.0]))
    assert per_row.shape == (4,) and per_row[0] <= 0.5 and per_row[1] <= 1.0 and per_row[2] <= 2.0


def test_sure_is_unbiased_for_the_true_loss():
    rng = G.SplitMix64(3)
    theta = np.zeros(200)
    theta[:20] = 5.0
    loss = {1.0: [], 2.0: []}
    est = {1.0: [], 2.0: []}
    for _ in range(3000):
        y = theta + rng.normal(200)
        for lam in loss:
            loss[lam].append(np.sum((S.soft(y, lam) - theta) ** 2))
            est[lam].append(S.sure_risk(y, lam))
    for lam in loss:
        assert np.mean(est[lam]) == pytest.approx(np.mean(loss[lam]), rel=0.01)


def test_level_threshold_branches():
    rng = np.random.default_rng(4)
    noise = rng.normal(size=(2, 256))
    sigma = np.array([1.0, 2.0])
    assert S.level_threshold(noise, sigma, "fest", 1024, 1.7).tolist() == [1.7, 1.7]
    assert S.level_threshold(noise, sigma, "universal", 1024).tolist() == pytest.approx([3.7233, 7.4466], abs=1e-3)
    sure = S.level_threshold(noise, sigma, "SURE", 1024)
    assert sure.shape == (2,) and np.all(sure <= sigma * math.sqrt(2 * math.log(256)) + 1e-12)
    signal = noise.copy()
    signal[:, :40] += 8.0
    # Hybrid: dünne Stufe (nur Rauschen) -> universal, dichte Stufe (viel Signal) -> SURE
    assert S.level_threshold(noise, sigma, "Hybrid", 1024).tolist() == pytest.approx(S.level_threshold(noise, sigma, "universal", 1024).tolist())
    assert S.level_threshold(signal, sigma, "Hybrid", 1024).tolist() == pytest.approx(S.level_threshold(signal, sigma, "SURE", 1024).tolist())
    with pytest.raises(ValueError):
        S.level_threshold(noise, sigma, "Bayes", 1024)


def test_hybrid_decision_boundary():
    m = 64
    crit = math.log2(m) ** 1.5 / math.sqrt(m)
    z = np.ones(m)
    low = z * math.sqrt(1 + crit - 1e-6)           # (sum z^2 - m)/m = crit - 1e-6: universal
    high = z * math.sqrt(1 + crit + 1e-6)          # knapp darüber: SURE
    uni = float(S.universal_threshold(1.0, 4096))
    assert float(S.level_threshold(low[None], np.array([1.0]), "Hybrid", 4096)[0]) == pytest.approx(uni)
    assert float(S.level_threshold(high[None], np.array([1.0]), "Hybrid", 4096)[0]) != pytest.approx(uni)


def test_shrink_with_threshold_zero_is_the_identity_and_infinite_keeps_the_smooth_part(mini_x):
    h = W.daubechies(2)
    x = np.random.default_rng(5).normal(size=64)
    assert S.shrink(x, h, 4, 1.0, "fest", "weich", 0.0)[0] == pytest.approx(x, abs=1e-12)
    assert S.shrink(x, h, 4, 1.0, "fest", "hart", 0.0)[0] == pytest.approx(x, abs=1e-12)
    d, a = W.dwt(x, h, 4)
    assert S.shrink(x, h, 4, 1.0, "fest", "weich", 1e9)[0] == pytest.approx(W.idwt([np.zeros_like(v) for v in d], a, h), abs=1e-12)
    est, thr = S.shrink(mini_x, W.daubechies(1), 1, 1.0, "fest", "weich", 0.8)
    assert est == pytest.approx([1.5, 1.5, 3.5, 3.5]) and float(thr[0]) == 0.8


def test_shrink_details_returns_thresholds_per_level_and_leaves_the_smooth_part():
    h = W.daubechies(2)
    x = np.random.default_rng(6).normal(size=128)
    d, a = W.dwt(x, h, 4)
    new, thr = S.shrink_details(d, np.asarray(1.0), "universal", "hart", 128)
    assert len(new) == len(thr) == 4 and all(float(t) == pytest.approx(math.sqrt(2 * math.log(128))) for t in thr)
    assert all(np.all((n == 0) | (np.abs(n) > float(t))) for n, t in zip(new, thr))
    est, _ = S.shrink(x, h, 4, 1.0, "universal", "hart")
    assert est == pytest.approx(W.idwt(new, a, h))


def test_cycle_spin_equals_the_loop_over_shifts_and_is_shift_invariant():
    h = W.daubechies(2)
    x = np.random.default_rng(7).normal(size=64) + 3 * (np.arange(64) > 20)
    J = 3
    loop = np.mean([np.roll(S.shrink(np.roll(x, k), h, J, 1.0, "fest", "hart", 1.0)[0], -k) for k in range(8)], axis=0)
    cs = S.cycle_spin(x, h, J, 1.0, "fest", "hart", 1.0)
    assert cs == pytest.approx(loop, abs=1e-12)
    assert S.cycle_spin(np.roll(x, 3), h, J, 1.0, "fest", "hart", 1.0) == pytest.approx(np.roll(cs, 3), abs=1e-12)
    plain = S.shrink(x, h, J, 1.0, "fest", "hart", 1.0)[0]
    assert np.abs(plain - np.roll(S.shrink(np.roll(x, 1), h, J, 1.0, "fest", "hart", 1.0)[0], -1)).max() > 0.01          # die DWT-Schwelle selbst ist nicht invariant
    assert S.cycle_spin(x, h, J, 1.0, "fest", "hart", 1.0, max_shifts=2) == pytest.approx(np.mean([np.roll(S.shrink(np.roll(x, k), h, J, 1.0, "fest", "hart", 1.0)[0], -k) for k in range(2)], axis=0), abs=1e-12)


def test_best_n_term_by_hand():
    c = np.array([1.0, -5.0, 3.0, 0.5, -4.0])
    assert S.best_n_term(c, 2).tolist() == [0.0, -5.0, 0.0, 0.0, -4.0] and S.best_n_term(c, 5).tolist() == c.tolist() and S.best_n_term(c, 0).tolist() == [0.0] * 5
    z = np.array([1 + 1j, 0.1, 3j])
    assert S.best_n_term(z, 1).tolist() == [0, 0, 3j]
    assert S.METHODS == ("universal", "SURE", "Hybrid", "fest") and S.RULES == ("weich", "hart")


def test_sure_threshold_allows_a_candidate_equal_to_the_upper_bound():
    z = np.full(4, 0.5)
    assert float(S.sure_threshold(z, 0.5)) == 0.5 and float(S.sure_threshold(z, 0.4999)) == 0.0
    assert S.sure_threshold(z[None, :], np.array([0.5])).tolist() == [0.5] and S.sure_threshold(z[None, :], np.array([0.4999])).tolist() == [0.0]
