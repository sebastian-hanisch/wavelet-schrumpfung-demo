"""Konstanten: Regler-Grenzen, Signale, feste Annahmen, Experimente, Farben und Presets (Kern: shr_wavelet.py, shr_signals.py, shr_shrink.py, shr_resolution.py)."""

N_CHOICES = (256, 512, 1024, 2048, 4096)
DEFAULT_N = 1024
SIGNALS = ("Blocks", "Bumps", "HeaviSine", "Doppler")
DEFAULT_SIGNAL = "Blocks"
SNR_MIN, SNR_MAX, SNR_STEP, DEFAULT_SNR = 1.0, 20.0, 0.5, 7.0                  # Standardabweichung des Signals bei Rauschen N(0, 1)
WAVELET_P = (1, 2, 3, 4, 6, 8)
DEFAULT_P = 4
LEVEL_MIN, DEFAULT_LEVELS, MAX_LEVELS = 1, 6, 9
METHODS = ("universal", "SURE", "Hybrid", "fest")
METHOD_LABELS = {"universal": "Universale Schwelle σ√(2 ln n)", "SURE": "SURE je Stufe", "Hybrid": "Hybrid (SureShrink)", "fest": "Feste Schwelle"}
DEFAULT_METHOD = "universal"
RULES = ("weich", "hart")
DEFAULT_RULE = "weich"
FIXED_MIN, FIXED_MAX, FIXED_STEP, DEFAULT_FIXED = 0.0, 6.0, 0.1, 1.0
SIGMA_MODES = ("geschätzt (MAD)", "bekannt")
DEFAULT_SIGMA = "geschätzt (MAD)"
CYCLE_MODES = ("aus", "an")
DEFAULT_CYCLE = "aus"
MAX_SHIFTS = 64                                                                # Cycle Spinning mittelt höchstens über so viele Verschiebungen
SEED_MAX, DEFAULT_SEED = 999999, 39
MAD_DIVISOR = 0.6745

# Experimente (feste Ströme)
EXP_SEED = 970000
EXP_N, EXP_SNR, EXP_P, EXP_J = 1024, 7.0, 4, 6
EXP_METHOD_RUNS = 100
EXP_LAMS = tuple(round(0.2 * i, 1) for i in range(26))                         # Schwellengitter 0 bis 5
EXP_CURVE_RUNS = 60
EXP_CYCLE_RUNS, EXP_CYCLE_P, EXP_CYCLE_SIGNALS = 40, (1, 4), ("Blocks", "HeaviSine", "Doppler")
EXP_CYCLE_WINDOW = 20
EXP_CYCLE_JUMPS = {"Blocks": 0.1, "HeaviSine": 0.3, "Doppler": 0.5}
EXP_KS = (8, 16, 32, 64, 128)
EXP_NOISE_RUNS, EXP_ZERO_NS, EXP_ZERO_RUNS, EXP_ZERO_P, EXP_ZERO_J = 200, (256, 1024, 4096), 300, 2, 4
EXP_SIGMA_FACTORS, EXP_SIGMA_RUNS = (0.5, 0.75, 1.0, 1.5, 2.0), 60

COLOR_TRUE = "#B03A2E"
COLOR_EST = "#4C9F70"
COLOR_NOISY = "rgba(120,120,120,0.7)"
COLOR_AUX = "#E8833A"
COLOR_BLUE = "rgba(76,120,168,0.95)"

PRESETS = {
    "Universale Schwelle auf Blocks: schlechter als das Rauschen": {"n": 1024, "signal": "Blocks", "snr": 7.0, "p": 4, "levels": 6, "method": "universal", "rule": "weich", "fixed": 1.0, "sigma": "geschätzt (MAD)",
                                                                   "cycle": "aus", "seed": DEFAULT_SEED},
    "SURE auf Blocks: der Fehler sinkt auf ein Drittel": {"n": 1024, "signal": "Blocks", "snr": 7.0, "p": 4, "levels": 6, "method": "SURE", "rule": "weich", "fixed": 1.0, "sigma": "geschätzt (MAD)", "cycle": "aus",
                                                         "seed": DEFAULT_SEED},
    "Cycle Spinning: Gibbs-Artefakte verschwinden": {"n": 1024, "signal": "HeaviSine", "snr": 7.0, "p": 4, "levels": 6, "method": "universal", "rule": "hart", "fixed": 1.0, "sigma": "geschätzt (MAD)", "cycle": "an",
                                                    "seed": DEFAULT_SEED},
    "HeaviSine: wenige Koeffizienten tragen alles": {"n": 1024, "signal": "HeaviSine", "snr": 7.0, "p": 4, "levels": 6, "method": "universal", "rule": "hart", "fixed": 1.0, "sigma": "geschätzt (MAD)", "cycle": "aus",
                                                    "seed": DEFAULT_SEED},
    "Niedriger Rauschabstand: die Schwelle zählt": {"n": 1024, "signal": "Doppler", "snr": 1.0, "p": 4, "levels": 6, "method": "universal", "rule": "weich", "fixed": 1.0, "sigma": "geschätzt (MAD)", "cycle": "aus",
                                              "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Universale Schwelle auf Blocks: schlechter als das Rauschen": "Blocks, Signal-Rausch-Verhältnis 7, D8: Die weiche universale Schwelle σ√(2 ln n) = 3,7σ verzerrt die vielen großen Koeffizienten; der Fehler liegt über dem des verrauschten Signals.",
    "SURE auf Blocks: der Fehler sinkt auf ein Drittel": "Dieselbe Messung mit SURE je Stufe: Die Schwelle wird je Stufe aus den Daten gewählt, der Fehler sinkt auf etwa ein Drittel des verrauschten.",
    "Cycle Spinning: Gibbs-Artefakte verschwinden": "HeaviSine mit harter universaler Schwelle und Cycle Spinning: Das Mitteln über 64 Verschiebungen halbiert den Fehler und glättet das Überschwingen an den Sprüngen; schalten Sie es aus, um den Unterschied zu sehen.",
    "HeaviSine: wenige Koeffizienten tragen alles": "HeaviSine ist glatt mit zwei Sprüngen: Nach der harten Schwelle bleiben nur 8 der 1008 Detailkoeffizienten übrig.",
    "Niedriger Rauschabstand: die Schwelle zählt": "Doppler bei Signal-Rausch-Verhältnis 1 (Rauschen so stark wie das Signal): Alle Verfahren sind weit besser als ohne Schwelle; die weiche universale Schwelle (0,16) liegt aber noch hinter SURE und der harten Schwelle (je etwa 0,10).",
}
