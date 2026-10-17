"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

import math
from pathlib import Path

import numpy as np
import pytest

import shr_constants as C
import shr_evaluation as ev
import shr_shrink as S
import shr_signals as SG
import shr_wavelet as W
from sweeps_cache import get

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def preset(name, **over):
    p = C.PRESETS[name]
    base = dict(n=p["n"], signal=p["signal"], snr=p["snr"], p=p["p"], levels=p["levels"], method=p["method"], rule=p["rule"], fixed=p["fixed"], sigma_mode=p["sigma"], cycle=p["cycle"], seed=p["seed"])
    return ev.analyse(ev.Settings(**{**base, **over}))


def de(x, k=3):
    return f"{x:.{k}f}".replace(".", ",")


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_universal_soft_on_blocks_is_worse_than_the_noise():
    name = "Universale Schwelle auf Blocks: schlechter als das Rauschen"
    a = preset(name)
    assert a.mse_est > a.mse_noisy and de(a.universal / a.sigma, 1) == "3,7" and "3,7σ" in C.PRESET_HELP[name] and "Signal-Rausch-Verhältnis 7" in C.PRESET_HELP[name] and a.settings.snr == 7.0


def test_preset_sure_reduces_the_error_to_about_a_third():
    name = "SURE auf Blocks: der Fehler sinkt auf ein Drittel"
    a = preset(name)
    assert 0.30 < a.mse_est / a.mse_noisy < 0.36 and "etwa ein Drittel" in C.PRESET_HELP[name] and a.settings.method == "SURE"


def test_preset_cycle_spinning_halves_the_error():
    name = "Cycle Spinning: Gibbs-Artefakte verschwinden"
    on, off = preset(name), preset(name, cycle="aus")
    assert 0.4 < on.mse_est / off.mse_est < 0.55 and "halbiert den Fehler" in C.PRESET_HELP[name] and "64 Verschiebungen" in C.PRESET_HELP[name] and C.MAX_SHIFTS == 64 and 2 ** on.settings.levels == 64


def test_preset_heavisine_keeps_eight_of_1008_detail_coefficients():
    name = "HeaviSine: wenige Koeffizienten tragen alles"
    a = preset(name)
    assert (a.kept, a.total) == (8, 1008) and "nur 8 der 1008 Detailkoeffizienten" in C.PRESET_HELP[name]


def test_preset_low_snr_methods_are_far_better_than_no_threshold_and_sure_beats_universal():
    name = "Niedriger Rauschabstand: die Schwelle zählt"
    uni = preset(name)
    sure = preset(name, method="SURE")
    hard = preset(name, rule="hart")
    assert de(uni.mse_est, 2) == "0,16" and de(sure.mse_est, 2) == "0,10" and de(hard.mse_est, 2) == "0,11" and uni.mse_est < 0.2 * uni.mse_noisy
    assert "(0,16)" in C.PRESET_HELP[name] and "je etwa 0,10" in C.PRESET_HELP[name] and "Rauschen so stark wie das Signal" in C.PRESET_HELP[name] and uni.settings.snr == 1.0
    assert "Doppler bei Rauschabstand 1: 0,157 gegen 0,955 ohne Schwelle" in README and de(uni.mse_est) == "0,157" and de(uni.mse_noisy) == "0,955"


# --- Schwellenverfahren, Kurve ----------------------------------------------------------------------------------------------------------------


def test_method_numbers_in_the_readme():
    rows = {r["signal"]: r for r in get("methods")}
    lines = {"Blocks": "1,007 / 1,662 / 0,516 / 0,365 / 0,513 / 0,419", "Bumps": "1,008 / 2,011 / 0,534 / 0,395 / 0,457 / 0,450", "HeaviSine": "1,005 / 0,185 / 0,122 / 0,105 / 0,142 / 0,111",
             "Doppler": "0,998 / 0,757 / 0,239 / 0,229 / 0,285 / 0,260"}
    for name, line in lines.items():
        r = rows[name]
        assert " / ".join(de(r[k]) for k in ("noisy", "universal_soft", "universal_hard", "sure", "hybrid", "oracle")) == line and f"{name} {line}" in README
        assert r["sure"] < r["oracle"] and r["sure"] < r["universal_hard"]
    assert rows["Blocks"]["universal_soft"] > rows["Blocks"]["noisy"] and rows["Bumps"]["universal_soft"] > rows["Bumps"]["noisy"] and "verschlechtert Blocks und Bumps sogar" in README
    assert "schlägt auf allen vier Signalen das Orakel" in README and (C.EXP_N, C.EXP_SNR, C.EXP_P, C.EXP_J, C.EXP_METHOD_RUNS) == (1024, 7.0, 4, 6, 100) and "100 Läufe bei den Schwellenverfahren" in README


def test_curve_numbers_in_the_readme():
    rows = {r["signal"]: r for r in get("curve")}
    assert " / ".join(de(rows[k]["best_error"]) for k in SG.NAMES) == "0,414 / 0,443 / 0,111 / 0,268" and "mit den Fehlern 0,414 / 0,443 / 0,111 / 0,268" in README
    assert " / ".join(de(rows[k]["universal_error"]) for k in SG.NAMES) == "1,586 / 1,897 / 0,186 / 0,750" and "sind es 1,586 / 1,897 / 0,186 / 0,750" in README
    assert [rows[k]["best_lam"] for k in SG.NAMES] == [1.0, 1.0, 1.8, 1.4] and "bei 1,0 (Blocks), 1,0 (Bumps), 1,8 (HeaviSine) und 1,4 (Doppler)" in README and rows["Blocks"]["universal_lam"] == 3.8 and "universalen Schwelle 3,8" in README
    assert min(r["error_zero"] for r in rows.values()) > 0.99 and max(r["error_zero"] for r in rows.values()) < 1.006 and "das Rauschen (0,999 bis 1,005)" in README and C.EXP_CURVE_RUNS == 60
    assert "60 bei der Fehlerkurve" in README and de(rows["Blocks"]["error_zero"]) == "0,999" and de(rows["Doppler"]["error_zero"]) == "1,005"


# --- Cycle Spinning -----------------------------------------------------------------------------------------------------------------------------


def test_cycle_numbers_in_the_readme():
    rows = {(r["p"], r["signal"], r["rule"]): r for r in get("cycle")}

    def pair(p, sig, rule, a="plain", b="cycle", k=3):
        r = rows[(p, sig, rule)]
        return f"{de(r[a], k)} / {de(r[b], k)}"
    assert ", ".join(f"{sig} {pair(4, sig, 'hart')}" for sig in ("Blocks", "HeaviSine", "Doppler")) == "Blocks 0,521 / 0,231, HeaviSine 0,126 / 0,059, Doppler 0,249 / 0,119"
    assert "D8 hart: Blocks 0,521 / 0,231, HeaviSine 0,126 / 0,059, Doppler 0,249 / 0,119" in README
    assert ", ".join(pair(4, sig, "weich") for sig in ("Blocks", "HeaviSine", "Doppler")) == "1,676 / 1,340, 0,190 / 0,144, 0,755 / 0,547" and "D8 weich: 1,676 / 1,340, 0,190 / 0,144, 0,755 / 0,547" in README
    assert ", ".join(f"{sig} {pair(1, sig, 'hart')}" for sig in ("Blocks", "HeaviSine", "Doppler")) == "Blocks 0,099 / 0,052, HeaviSine 0,337 / 0,083, Doppler 0,964 / 0,277" and "Haar hart: Blocks 0,099 / 0,052, HeaviSine 0,337 / 0,083, Doppler 0,964 / 0,277" in README
    assert [pair(4, sig, "hart", "window_plain", "window_cycle", 2) for sig in ("Blocks", "HeaviSine", "Doppler")] == ["3,26 / 2,12", "2,66 / 2,21", "0,41 / 0,30"]
    assert "von 3,26 auf 2,12 (Blocks), 2,66 auf 2,21 (HeaviSine) und 0,41 auf 0,30 (Doppler)" in README and pair(1, "Blocks", "hart", "window_plain", "window_cycle", 2) == "0,89 / 0,87" and "(0,89 gegen 0,87)" in README
    assert all(r["cycle"] < r["plain"] for r in rows.values()) and (C.EXP_CYCLE_RUNS, C.EXP_CYCLE_WINDOW) == (40, 20) and "40 beim Cycle Spinning" in README and "± 20 Abtastwerten" in README


# --- Kompression ----------------------------------------------------------------------------------------------------------------------------------


def test_nterm_numbers_in_the_readme():
    rows = {r["signal"]: r for r in get("nterm")}
    b = rows["Blocks"]
    assert " / ".join(de(v) for v in b["haar"]) == "0,471 / 0,300 / 0,171 / 0,000 / 0,000" and "Haar 0,471 / 0,300 / 0,171 / 0,000 / 0,000" in README
    assert " / ".join(de(v) for v in b["d8"]) == "0,496 / 0,352 / 0,197 / 0,111 / 0,031" and "D8 0,496 / 0,352 / 0,197 / 0,111 / 0,031" in README
    assert " / ".join(de(v) for v in b["fourier"]) == "0,421 / 0,354 / 0,252 / 0,166 / 0,116" and "Fourier 0,421 / 0,354 / 0,252 / 0,166 / 0,116" in README and C.EXP_KS == (8, 16, 32, 64, 128)
    assert [(rows[k]["count_d8"], rows[k]["count_fourier"]) for k in SG.NAMES] == [(70, 168), (107, 410), (12, 9), (40, 175)] and "Blocks 70 / 168, Bumps 107 / 410, HeaviSine 12 / 9, Doppler 40 / 175" in README
    assert rows["HeaviSine"]["count_fourier"] < rows["HeaviSine"]["count_d8"] and "9 gegen 12" in APP and "Nur das glatte HeaviSine ist in Fourier dünner" in README


# --- Rauschpegel -----------------------------------------------------------------------------------------------------------------------------------


def test_noise_numbers_in_the_readme_and_the_app():
    res = get("noise")
    sig = {r["signal"]: r for r in res["sigma"]}
    line = ", ".join(f"{k} {de(sig[k]['mean'])} ± {de(sig[k]['sd'])}" for k in ("Blocks", "Bumps", "HeaviSine", "Doppler"))
    assert line == "Blocks 1,051 ± 0,060, Bumps 1,060 ± 0,056, HeaviSine 1,002 ± 0,050, Doppler 1,025 ± 0,050" and f"σ̂ bei {line}" in README
    assert round(100 * (sig["Blocks"]["mean"] - 1)) == 5 and round(100 * (sig["Bumps"]["mean"] - 1)) == 6 and "5 bis 6 % zu hoch" in APP
    wrong = {r["factor"]: r["error"] for r in res["wrong"]}
    assert " / ".join(de(wrong[f], 2) for f in (0.5, 0.75, 1.0, 1.5, 2.0)) == "0,58 / 1,01 / 1,53 / 2,73 / 3,94" and "0,58 / 1,01 / 1,53 / 2,73 / 3,94" in README
    zero = {r["n"]: r for r in res["zero"]}
    assert " / ".join(de(100 * zero[n]["share"], 1) for n in (256, 1024, 4096)) == "81,3 / 82,7 / 81,7" and " / ".join(de(100 * zero[n]["theory"], 1) for n in (256, 1024, 4096)) == "76,0 / 78,6 / 80,4"
    assert "81,3 / 82,7 / 81,7 % der Läufe (n = 256 / 1024 / 4096; Näherung 76,0 / 78,6 / 80,4 %)" in README and (C.EXP_NOISE_RUNS, C.EXP_ZERO_RUNS) == (200, 300) and "200 beim Rauschpegel" in README


def test_sure_is_unbiased_claim():
    rng = np.random.default_rng(11)
    theta = np.zeros(200)
    theta[:20] = 5.0
    loss, est = [], []
    for _ in range(1500):
        y = theta + rng.normal(size=200)
        loss.append(np.sum((S.soft(y, 1.0) - theta) ** 2))
        est.append(S.sure_risk(y, 1.0))
    assert abs(np.mean(est) / np.mean(loss) - 1) < 0.01 and "auf 1 % überein" in README and "20 großen Koeffizienten unter 200" in README


# --- Konstanten, Mini-Instanzen ----------------------------------------------------------------------------------------------------------------


def test_method_numbers_in_the_readme_match_the_constants():
    assert (min(C.N_CHOICES), max(C.N_CHOICES), C.SNR_MIN, C.SNR_MAX, C.EXP_SEED) == (256, 4096, 1.0, 20.0, 970000) and "n = 256 bis 4096" in README and "Signal-Rausch-Verhältnis 1 bis 20" in README and "Seed 970000" in README
    assert C.WAVELET_P == (1, 2, 3, 4, 6, 8) and "min(2^J, 64)" in README and C.MAD_DIVISOR == 0.6745 and "0,6745" in README
    assert ev.Settings().n == 1024 and ev.analyse(ev.Settings()).total == 1008


def test_hand_set_mini_instances_in_the_readme(mini_d=np.array([3.0, -1.0, 0.5, 2.0]), mini_x=np.array([1.0, 2.0, 3.0, 4.0])):
    assert [S.sure_risk(mini_d, lam) for lam in (0.0, 0.5, 1.0, 3.0)] == pytest.approx([4.0, 3.0, 3.25, 10.25]) and "SURE(0) = 4, SURE(0,5) = 3, SURE(1) = 3,25, SURE(3) = 10,25, Minimum bei 0,5" in README
    est, _ = S.shrink(mini_x, W.daubechies(1), 1, 1.0, "fest", "weich", 0.8)
    assert est == pytest.approx([1.5, 1.5, 3.5, 3.5]) and "(1,5; 1,5; 3,5; 3,5)" in README
    assert SG.blocks_raw(4).tolist() == pytest.approx([4.0, 3.0, 0.9, 3.0]) and SG.heavisine_raw(4).tolist() == pytest.approx([4.0, -6.0, 2.0, -4.0]) and "Blocks 4 / 3 / 0,9 / 3, HeaviSine 4 / −6 / 2 / −4" in README
    assert float(S.universal_threshold(1.0, 1024)) == pytest.approx(math.sqrt(2 * math.log(1024)))


def test_universal_threshold_is_three_point_seven_at_n_1024():
    assert de(float(S.universal_threshold(1.0, 1024)), 1) == "3,7" and "σ√(2 ln n)" in README and "σ√(2 ln n)" in APP
