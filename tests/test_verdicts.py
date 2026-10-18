"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import shr_evaluation as ev


def make_analysis(method="universal", rule="weich", p=4, cycle="aus", mse_noisy=0.955, mse_est=0.300, kept=272, total=1008, sigma=1.043, universal=3.884):
    z = np.zeros(1)
    s = ev.Settings(method=method, rule=rule, p=p, cycle=cycle)
    return ev.Analysis(s, z, z, z, sigma, [], [], [], kept, total, mse_noisy, mse_est, universal)


def test_main_verdict_better_and_worse_than_the_noise():
    kind, text = ev.verdict_main(make_analysis(method="SURE"))
    assert kind == "success"
    assert text == ("SURE je Stufe, weiche Schwelle, D8: Der Fehler sinkt auf 0,300 gegen 0,955 des verrauschten Signals (Faktor 3,2 besser); 272 von 1008 Detailkoeffizienten bleiben, Rauschpegel σ = 1,04.")
    kind, text = ev.verdict_main(make_analysis(mse_est=1.590, kept=88))
    assert kind == "warning"
    assert text == ("Universale Schwelle σ√(2 ln n), weiche Schwelle, D8: Der Fehler beträgt 1,590 gegen 0,955 des verrauschten Signals: Das Verfahren verschlechtert das Signal; 88 von 1008 Detailkoeffizienten bleiben, "
                    "Rauschpegel σ = 1,04. Die universale Schwelle 3,88 ist bei diesem Rauschabstand hoch; SURE, eine harte Schwelle oder eine feste kleinere Schwelle verzerren die großen Koeffizienten weniger.")
    assert ev.verdict_main(make_analysis(mse_est=0.954999))[0] == "success" and ev.verdict_main(make_analysis(mse_est=0.955))[0] == "warning"


def test_main_verdict_names_rule_wavelet_and_cycle_spinning():
    text = ev.verdict_main(make_analysis(method="Hybrid", rule="hart", p=1, cycle="an"))[1]
    assert text.startswith("Hybrid (SureShrink), harte Schwelle, Haar, Cycle Spinning: ") and "Feste Schwelle" in ev.verdict_main(make_analysis(method="fest"))[1]
    worse = ev.verdict_main(make_analysis(method="SURE", mse_est=2.0))[1]
    assert "Die universale Schwelle" not in worse and worse.endswith("σ = 1,04.")


def test_verdict_methods_text():
    rows = [{"signal": "Blocks", "noisy": 1.007, "universal_soft": 1.662, "universal_hard": 0.516, "sure": 0.365, "oracle": 0.419}]
    kind, text = ev.verdict_methods(rows)
    assert kind == "warning"
    assert text == ("Bei Blocks liegt der Fehler der weichen universalen Schwelle bei 1,66 gegen 1,01 des verrauschten Signals, SURE bei 0,36, die harte universale Schwelle bei 0,52; "
                    "SURE (0,36) schlägt sogar die beste globale Schwelle (0,42). **Bei Rauschabstand 7 ist die universale Schwelle zu hoch, und SURE, das je Stufe wählt, liegt unter dem Orakel mit einer gemeinsamen Schwelle.**")


def test_verdict_curve_text():
    rows = [{"signal": "Blocks", "best_lam": 1.0, "universal_lam": 3.8, "best_error": 0.414, "universal_error": 1.586, "error_zero": 0.999}, {"signal": "HeaviSine", "best_lam": 1.8, "universal_lam": 3.8, "best_error": 0.1,
                                                                                                                                                  "universal_error": 0.2, "error_zero": 1.0}]
    kind, text = ev.verdict_curve(rows)
    assert kind == "info"
    assert text == ("Die beste globale Schwelle liegt zwischen 1,0 und 1,8, die universale bei 3,8. Bei Blocks beträgt der Fehler 0,41 bei der besten und 1,59 bei der universalen Schwelle; ohne Schwelle (1,00) ist es das Rauschen. "
                    "**Der Fehler hat ein flaches Minimum, und die universale Schwelle liegt weit rechts davon.**")


def test_verdict_cycle_text():
    def row(p, signal, plain, cyc):
        return {"p": p, "signal": signal, "rule": "hart", "plain": plain, "cycle": cyc}
    rows = [row(4, "Blocks", 0.521, 0.231), row(4, "HeaviSine", 0.126, 0.059), row(4, "Doppler", 0.249, 0.119), row(1, "Blocks", 0.099, 0.052)]
    kind, text = ev.verdict_cycle(rows)
    assert kind == "info"
    assert text == ("Cycle Spinning senkt den Fehler der harten universalen Schwelle mit D8: Blocks 0,52 → 0,23, HeaviSine 0,13 → 0,06, Doppler 0,25 → 0,12. Bei Haar (Blocks) fällt er von 0,10 auf 0,05. "
                    "**Mitteln über die Verschiebungen halbiert bei harter Schwelle den Fehler (bei weicher Schwelle sinkt er weniger).**")


def test_verdict_nterm_text():
    rows = [{"signal": "Blocks", "count_d8": 70, "count_fourier": 168}, {"signal": "Bumps", "count_d8": 107, "count_fourier": 410}, {"signal": "HeaviSine", "count_d8": 12, "count_fourier": 9},
            {"signal": "Doppler", "count_d8": 40, "count_fourier": 175}]
    kind, text = ev.verdict_nterm(rows)
    assert kind == "info"
    assert text == ("Für 99 % der Energie braucht die beste Näherung bei Blocks 70 Koeffizienten in der D8-Basis gegen 168 in Fourier, bei Bumps 107 gegen 410, bei Doppler 40 gegen 175; nur HeaviSine braucht in Fourier "
                    "weniger (9 gegen 12). **Beim 99-%-Kriterium sind Wavelets bei Blocks, Bumps und Doppler die dünnere Basis, Fourier nur bei HeaviSine.**")


def test_verdict_noise_text():
    res = {"sigma": [{"signal": "Blocks", "mean": 1.051, "sd": 0.06}, {"signal": "HeaviSine", "mean": 1.002, "sd": 0.05}], "zero": [{"share": 0.813}, {"share": 0.827}, {"share": 0.817}],
           "wrong": [{"factor": 0.5, "error": 0.58}, {"factor": 1.0, "error": 1.53}]}
    kind, text = ev.verdict_noise(res)
    assert kind == "warning"
    assert text == ("Der Rauschpegel aus dem Median der Stufe 1 ist bei Blocks 1,051 ± 0,060 und bei HeaviSine 1,002 ± 0,050 (wahr 1): Bei Blocks ist er zu hoch, bei HeaviSine nicht. Bei reinem Rauschen nullt die universale Schwelle alle Details in "
                    "81 bis 83 % der Läufe. Mit einem halb so großen σ sinkt der Fehler bei Blocks von 1,53 auf 0,58. **Der Pegel und die Schwelle hängen zusammen; ein falscher Pegel verschiebt beide.**")
