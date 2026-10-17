"""Die vier Testsignale von Donoho und Johnstone: Handrechnung an den Rastermitten, Skalierung, Fehler."""

import math

import numpy as np
import pytest

import shr_signals as S


def test_blocks_by_hand():
    # n = 4: t = 0,125 / 0,375 / 0,625 / 0,875; Summe der Stufenhöhen mit t_j < t: 4 / 4 - 5 + 3 - 4 + 5 = 3 / 3 - 4,2 + 2,1 = 0,9 / 0,9 + 4,3 - 3,1 + 5,1 - 4,2 = 3,0
    assert S.blocks_raw(4).tolist() == pytest.approx([4.0, 3.0, 0.9, 3.0])


def test_heavisine_by_hand():
    # 4 sin(4 pi t) - sgn(t - 0,3) - sgn(0,72 - t) an t = 0,125 / 0,375 / 0,625 / 0,875
    t = (np.arange(4) + 0.5) / 4
    expected = 4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)
    assert S.heavisine_raw(4).tolist() == pytest.approx(expected.tolist()) and S.heavisine_raw(4).tolist() == pytest.approx([4.0, -6.0, 2.0, -4.0])


def test_bumps_and_doppler_by_hand():
    n = 1000
    t = (np.arange(n) + 0.5) / n
    b = S.bumps_raw(n)
    assert b.min() > 0 and b.argmax() in (np.argmin(np.abs(t - 0.25)), np.argmin(np.abs(t - 0.25)) + 1, np.argmin(np.abs(t - 0.25)) - 1, np.argmin(np.abs(t - 0.78)), np.argmin(np.abs(t - 0.13)))
    one = S.bumps_raw(2)
    assert one[0] == pytest.approx(sum(h * (1 + abs((0.25 - tj) / w)) ** -4 for h, tj, w in zip(S.BUMPS_H, S.KNOTS, S.BUMPS_W)))
    d = S.doppler_raw(4)
    for i, ti in enumerate((0.125, 0.375, 0.625, 0.875)):
        assert d[i] == pytest.approx(math.sqrt(ti * (1 - ti)) * math.sin(2 * math.pi * 1.05 / (ti + 0.05)))


def test_shape_is_scaled_to_unit_standard_deviation_and_signal_to_snr():
    for name in S.NAMES:
        f = S.shape(name, 1024)
        assert f.std() == pytest.approx(1.0) and S.signal(name, 1024, 7.0).std() == pytest.approx(7.0) and S.signal(name, 1024, 7.0) == pytest.approx(7.0 * f)
    assert S.shape("Blocks", 64).tolist() == pytest.approx((S.blocks_raw(64) / S.blocks_raw(64).std()).tolist())
    with pytest.raises(ValueError):
        S.shape("Sinus", 64)


def test_signals_have_the_expected_character():
    blocks = S.shape("Blocks", 1024)
    assert len(np.unique(np.round(blocks, 9))) == 11 and S.NAMES == ("Blocks", "Bumps", "HeaviSine", "Doppler")
    hs = S.heavisine_raw(1024)
    assert abs(hs[int(0.3 * 1024) - 3] - hs[int(0.3 * 1024) + 3]) > 1.5                                             # Sprung bei 0,3
    dop = S.doppler_raw(1024)
    assert np.abs(np.diff(np.sign(dop[:200]))).sum() > np.abs(np.diff(np.sign(dop[-200:]))).sum()               # schneller am Anfang
