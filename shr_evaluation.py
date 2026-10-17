"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import shr_constants as C
import shr_resolution as R
import shr_shrink as S
import shr_signals as SG
import shr_wavelet as W
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def wavelet_name(p: int) -> str:
    """Daubechies-Bezeichnung: Haar für p = 1, sonst D(2p)."""
    return "Haar" if p == 1 else f"D{2 * p}"


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    signal: str = C.DEFAULT_SIGNAL
    snr: float = C.DEFAULT_SNR
    p: int = C.DEFAULT_P
    levels: int = C.DEFAULT_LEVELS
    method: str = C.DEFAULT_METHOD
    rule: str = C.DEFAULT_RULE
    fixed: float = C.DEFAULT_FIXED
    sigma_mode: str = C.DEFAULT_SIGMA
    cycle: str = C.DEFAULT_CYCLE
    seed: int = C.DEFAULT_SEED

    @property
    def cycle_on(self) -> bool:
        return self.cycle == "an"

    @property
    def sigma_known(self) -> bool:
        return self.sigma_mode == "bekannt"


def build_measurement(s: Settings) -> tuple[np.ndarray, np.ndarray]:
    """(wahres Signal, verrauschtes Signal) mit Rauschen N(0, 1)."""
    f = SG.signal(s.signal, s.n, s.snr)
    return f, f + G.SplitMix64(s.seed).normal(s.n)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    truth: np.ndarray
    noisy: np.ndarray
    estimate: np.ndarray
    sigma: float                    # verwendeter Rauschpegel
    thresholds: list                # Schwelle je Stufe (ohne Verschiebung)
    before: list                    # Detailkoeffizienten je Stufe vor der Schwelle (ohne Verschiebung)
    after: list                     # nach der Schwelle
    kept: int                       # Zahl der Detailkoeffizienten ungleich null nach der Schwelle
    total: int                      # Zahl der Detailkoeffizienten
    mse_noisy: float
    mse_est: float
    universal: float                # universale Schwelle sigma sqrt(2 ln n)


def analyse(s: Settings) -> Analysis:
    f, y = build_measurement(s)
    h = W.daubechies(s.p)
    details, smooth = W.dwt(y, h, s.levels)
    sigma = 1.0 if s.sigma_known else float(S.mad_sigma(details[0]))
    new, thr = S.shrink_details(details, np.asarray(sigma), s.method, s.rule, s.n, s.fixed)
    if s.cycle_on:
        est = S.cycle_spin(y, h, s.levels, sigma, s.method, s.rule, s.fixed, C.MAX_SHIFTS)
    else:
        est = W.idwt(new, smooth, h)
    return Analysis(s, f, y, est, sigma, [float(t) for t in thr], details, new, int(sum(int(np.count_nonzero(d)) for d in new)), int(sum(len(d) for d in new)), float(R.mse(y, f)), float(R.mse(est, f)),
                    float(S.universal_threshold(sigma, s.n)))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def _noisy_rows(name: str, runs: int, n: int, snr: float, seed: int):
    f = SG.signal(name, n, snr)
    return f, f + G.SplitMix64(seed).normal(runs * n).reshape(runs, n)


def sweep_methods(runs: int = C.EXP_METHOD_RUNS, n: int = C.EXP_N, snr: float = C.EXP_SNR, p: int = C.EXP_P, levels: int = C.EXP_J, lams=C.EXP_LAMS) -> list[dict]:
    """Schwellwerte im Vergleich: Fehler gegen die Wahrheit je Verfahren (universal weich und hart, SURE, Hybrid je weich; beste globale Schwelle als Orakel je Lauf), σ aus der MAD der Stufe 1."""
    h = W.daubechies(p)
    rows = []
    for k, name in enumerate(SG.NAMES):
        f, y = _noisy_rows(name, runs, n, snr, C.EXP_SEED + 100 + k)
        sigma = S.mad_sigma(W.dwt(y, h, 1)[0][0])
        err = {}
        for key, method, rule in (("universal_soft", "universal", "weich"), ("universal_hard", "universal", "hart"), ("sure", "SURE", "weich"), ("hybrid", "Hybrid", "weich")):
            err[key] = float(np.mean(R.mse(S.shrink(y, h, levels, sigma, method, rule)[0], f)))
        best, lam = R.oracle_global(y, f, h, levels, lams)
        rows.append({"signal": name, "noisy": float(np.mean(R.mse(y, f))), **err, "oracle": float(np.mean(best)), "lam_oracle": float(np.mean(lam)), "sigma_mean": float(np.mean(sigma)),
                     "sigma_sd": float(np.std(sigma)), "lam_universal": float(S.universal_threshold(1.0, n)), "runs": runs})
    return rows


def sweep_curve(runs: int = C.EXP_CURVE_RUNS, n: int = C.EXP_N, snr: float = C.EXP_SNR, p: int = C.EXP_P, levels: int = C.EXP_J, lams=C.EXP_LAMS) -> list[dict]:
    """Fehler als Funktion der globalen weichen Schwelle (σ = 1 bekannt): Minimum und Fehler bei der universalen Schwelle (nächster Gitterpunkt)."""
    h = W.daubechies(p)
    lams = list(lams)
    rows = []
    for k, name in enumerate(SG.NAMES):
        f, y = _noisy_rows(name, runs, n, snr, C.EXP_SEED + 200 + k)
        errs = np.mean(R.threshold_errors(y, f, h, levels, lams), axis=0)
        best = int(np.argmin(errs))
        uni = int(np.argmin(np.abs(np.array(lams) - float(S.universal_threshold(1.0, n)))))
        rows.append({"signal": name, "lams": lams, "errors": errs.tolist(), "best_lam": lams[best], "best_error": float(errs[best]), "universal_lam": lams[uni], "universal_error": float(errs[uni]),
                     "error_zero": float(errs[0]), "error_max": float(errs[-1]), "runs": runs})
    return rows


def sweep_cycle(runs: int = C.EXP_CYCLE_RUNS, ps=C.EXP_CYCLE_P, signals=C.EXP_CYCLE_SIGNALS, n: int = C.EXP_N, snr: float = C.EXP_SNR, levels: int = C.EXP_J, window: int = C.EXP_CYCLE_WINDOW) -> list[dict]:
    """Cycle Spinning: Fehler der universalen Schwelle ohne und mit Mitteln über alle 2^J Verschiebungen (weich und hart) und größter Fehler im Fenster ± `window` um den Sprung."""
    rows = []
    for p in ps:
        h = W.daubechies(p)
        for k, name in enumerate(signals):
            f, y = _noisy_rows(name, runs, n, snr, C.EXP_SEED + 300 + 10 * p + k)
            sigma = S.mad_sigma(W.dwt(y, h, 1)[0][0])
            j0 = int(C.EXP_CYCLE_JUMPS[name] * n)
            win = slice(max(0, j0 - window), j0 + window)
            for rule in C.RULES:
                plain = S.shrink(y, h, levels, sigma, "universal", rule)[0]
                spun = np.array([S.cycle_spin(y[i], h, levels, sigma[i], "universal", rule, max_shifts=2 ** levels) for i in range(runs)])
                rows.append({"p": p, "signal": name, "rule": rule, "plain": float(np.mean(R.mse(plain, f))), "cycle": float(np.mean(R.mse(spun, f))),
                             "window_plain": float(np.mean(np.max(np.abs(plain - f)[:, win], axis=1))), "window_cycle": float(np.mean(np.max(np.abs(spun - f)[:, win], axis=1))), "runs": runs})
    return rows


def sweep_nterm(ks=C.EXP_KS, n: int = C.EXP_N, levels: int = C.EXP_J) -> list[dict]:
    """Beste k-Term-Näherung (relativer L2-Fehler, kein Rauschen): Haar, D8 und Fourier; Zahl der Koeffizienten für 99 % der Energie (D8 gegen Fourier)."""
    haar, d8 = W.daubechies(1), W.daubechies(4)
    rows = []
    for name in SG.NAMES:
        f = SG.shape(name, n)
        rows.append({"signal": name, "ks": list(ks), "haar": R.n_term_error_wavelet(f, haar, levels, ks), "d8": R.n_term_error_wavelet(f, d8, levels, ks), "fourier": R.n_term_error_fourier(f, ks),
                     "count_d8": R.terms_for_energy(R.wavelet_coefficients(f, d8, levels)), "count_fourier": R.terms_for_energy(np.fft.fft(f))})
    return rows


def sweep_noise(runs: int = C.EXP_NOISE_RUNS, n: int = C.EXP_N, snr: float = C.EXP_SNR, p: int = C.EXP_P, zero_ns=C.EXP_ZERO_NS, zero_runs: int = C.EXP_ZERO_RUNS, factors=C.EXP_SIGMA_FACTORS,
                factor_runs: int = C.EXP_SIGMA_RUNS, levels: int = C.EXP_J) -> dict:
    """Rauschpegel aus dem Median der Stufe 1 (je Signal), Anteil der Läufe, in denen die universale Schwelle bei reinem Rauschen alle Details nullt, und Fehler bei falschem σ (Blocks)."""
    h = W.daubechies(p)
    sigma = []
    for k, name in enumerate(SG.NAMES):
        _f, y = _noisy_rows(name, runs, n, snr, C.EXP_SEED + 400 + k)
        s_hat = S.mad_sigma(W.dwt(y, h, 1)[0][0])
        sigma.append({"signal": name, "mean": float(np.mean(s_hat)), "sd": float(np.std(s_hat))})
    zero = []
    h2 = W.daubechies(C.EXP_ZERO_P)
    for m in zero_ns:
        x = G.SplitMix64(C.EXP_SEED + 500 + m).normal(zero_runs * m).reshape(zero_runs, m)
        d, _a = W.dwt(x, h2, C.EXP_ZERO_J)
        top = np.max(np.abs(np.concatenate(d, axis=-1)), axis=-1)
        zero.append({"n": m, "share": float(np.mean(top <= S.universal_threshold(1.0, m))), "theory": 1.0 - 1.0 / math.sqrt(math.pi * math.log(m)), "runs": zero_runs})
    f, y = _noisy_rows("Blocks", factor_runs, n, snr, C.EXP_SEED + 600)
    wrong = []
    for fac in factors:
        est = S.shrink(y, h, levels, np.full(factor_runs, float(fac)), "universal", "weich")[0]
        wrong.append({"factor": fac, "error": float(np.mean(R.mse(est, f)))})
    return {"sigma": sigma, "zero": zero, "wrong": wrong, "runs": runs}


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    head = f"{C.METHOD_LABELS[s.method]}, {'weiche' if s.rule == 'weich' else 'harte'} Schwelle, {wavelet_name(s.p)}{', Cycle Spinning' if s.cycle_on else ''}: "
    if a.mse_est < a.mse_noisy:
        text = head + f"Der Fehler sinkt auf {de(a.mse_est, 3)} gegen {de(a.mse_noisy, 3)} des verrauschten Signals (Faktor {de(a.mse_noisy / a.mse_est, 1)} besser)"
        kind = "success"
    else:
        text = head + f"Der Fehler beträgt {de(a.mse_est, 3)} gegen {de(a.mse_noisy, 3)} des verrauschten Signals: Das Verfahren verschlechtert das Signal"
        kind = "warning"
    text += f"; {a.kept} von {a.total} Detailkoeffizienten bleiben, Rauschpegel σ = {de(a.sigma, 2)}."
    if s.method == "universal" and kind == "warning":
        text += f" Die universale Schwelle {de(a.universal, 2)} ist bei diesem Rauschabstand hoch; SURE, eine harte Schwelle oder eine feste kleinere Schwelle verzerren die großen Koeffizienten weniger."
    return kind, text


def verdict_methods(rows: list[dict]) -> tuple[str, str]:
    by = {r["signal"]: r for r in rows}
    b = by["Blocks"]
    return "warning", (f"Bei Blocks liegt der Fehler der weichen universalen Schwelle bei {de(b['universal_soft'], 2)} gegen {de(b['noisy'], 2)} des verrauschten Signals, SURE bei {de(b['sure'], 2)}, die harte "
                       f"universale Schwelle bei {de(b['universal_hard'], 2)}; SURE ({de(b['sure'], 2)}) schlägt sogar die beste globale Schwelle ({de(b['oracle'], 2)}). **Die universale Schwelle ist hier zu vorsichtig, "
                       "und ein Verfahren, das je Stufe wählt, gewinnt.**")


def verdict_curve(rows: list[dict]) -> tuple[str, str]:
    by = {r["signal"]: r for r in rows}
    lo, hi = min(r["best_lam"] for r in rows), max(r["best_lam"] for r in rows)
    b = by["Blocks"]
    return "info", (f"Die beste globale Schwelle liegt zwischen {de(lo, 1)} und {de(hi, 1)}, die universale bei {de(b['universal_lam'], 1)}. Bei Blocks beträgt der Fehler {de(b['best_error'], 2)} bei der besten und "
                    f"{de(b['universal_error'], 2)} bei der universalen Schwelle; ohne Schwelle ({de(b['error_zero'], 2)}) ist es das Rauschen. **Der Fehler hat ein flaches Minimum, und die universale Schwelle liegt weit rechts davon.**")


def verdict_cycle(rows: list[dict]) -> tuple[str, str]:
    def at(p, name, rule):
        return [r for r in rows if r["p"] == p and r["signal"] == name and r["rule"] == rule][0]
    parts = [f"{name} {de(at(4, name, 'hart')['plain'], 2)} → {de(at(4, name, 'hart')['cycle'], 2)}" for name in ("Blocks", "HeaviSine", "Doppler")]
    return "info", ("Cycle Spinning senkt den Fehler der harten universalen Schwelle mit D8: " + ", ".join(parts) + f". Bei Haar (Blocks) fällt er von {de(at(1, 'Blocks', 'hart')['plain'], 2)} auf "
                    f"{de(at(1, 'Blocks', 'hart')['cycle'], 2)}. **Mitteln über die Verschiebungen macht die Schwelle verschiebungsinvariant und halbiert den Fehler.**")


def verdict_nterm(rows: list[dict]) -> tuple[str, str]:
    by = {r["signal"]: r for r in rows}
    return "info", (f"Für 99 % der Energie braucht die beste Näherung bei Blocks {by['Blocks']['count_d8']} Koeffizienten in der D8-Basis gegen {by['Blocks']['count_fourier']} in Fourier, bei Bumps "
                    f"{by['Bumps']['count_d8']} gegen {by['Bumps']['count_fourier']}, bei Doppler {by['Doppler']['count_d8']} gegen {by['Doppler']['count_fourier']}; nur das glatte HeaviSine braucht in Fourier "
                    f"weniger ({by['HeaviSine']['count_fourier']} gegen {by['HeaviSine']['count_d8']}). **Wavelets sind die dünnere Basis für Sprünge und Spitzen, Fourier für glatte Schwingungen.**")


def verdict_noise(res: dict) -> tuple[str, str]:
    sig = {r["signal"]: r for r in res["sigma"]}
    z = res["zero"]
    wrong = {r["factor"]: r["error"] for r in res["wrong"]}
    return "warning", (f"Der Rauschpegel aus dem Median der Stufe 1 ist bei Blocks {de(sig['Blocks']['mean'], 3)} ± {de(sig['Blocks']['sd'], 3)} und bei HeaviSine {de(sig['HeaviSine']['mean'], 3)} ± "
                       f"{de(sig['HeaviSine']['sd'], 3)} (wahr 1): Sprünge verfälschen ihn nach oben. Bei reinem Rauschen nullt die universale Schwelle alle Details in {de(100 * z[0]['share'], 0)} bis "
                       f"{de(100 * max(r['share'] for r in z), 0)} % der Läufe. Mit einem halb so großen σ sinkt der Fehler bei Blocks von {de(wrong[1.0], 2)} auf {de(wrong[0.5], 2)}. "
                       "**Der Pegel und die Schwelle hängen zusammen; ein falscher Pegel verschiebt beide.**")
