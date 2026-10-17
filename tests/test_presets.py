"""Presets: Vollständigkeit, gültige Werte, Standardwerte, Parameterschlüssel."""

import shr_constants as C
import shr_evaluation as ev
import shr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["signal"] in C.SIGNALS and p["p"] in C.WAVELET_P and p["method"] in C.METHODS and p["rule"] in C.RULES and p["sigma"] in C.SIGMA_MODES, name
        assert p["cycle"] in C.CYCLE_MODES and C.LEVEL_MIN <= p["levels"] <= min(C.MAX_LEVELS, p["n"].bit_length() - 2) and C.SNR_MIN <= p["snr"] <= C.SNR_MAX and C.FIXED_MIN <= p["fixed"] <= C.FIXED_MAX
        assert 0 <= p["seed"] <= C.SEED_MAX


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.signal in C.SIGNALS and s.p in C.WAVELET_P and s.method in C.METHODS and s.rule in C.RULES and s.sigma_mode in C.SIGMA_MODES and s.cycle in C.CYCLE_MODES
    assert C.LEVEL_MIN <= s.levels <= C.MAX_LEVELS and s.levels <= s.n.bit_length() - 2 and C.SNR_MIN <= s.snr <= C.SNR_MAX and set(C.METHOD_LABELS) == set(C.METHODS)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"n": s.n, "signal": s.signal, "snr": s.snr, "p": s.p, "levels": s.levels, "method": s.method, "rule": s.rule, "fixed": s.fixed, "sigma": s.sigma_mode, "cycle": s.cycle, "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_url_params_are_unique_and_not_password_like():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params)) and max(len(n) for n in C.PRESETS) <= 70
    assert not {"pw", "pass", "password", "pwd", "token", "key"} & set(params)


def test_bounds_and_the_constant_relations():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("snr_slider") == (C.SNR_MIN, C.SNR_MAX) and P.bounds("fixed_kept") == (C.FIXED_MIN, C.FIXED_MAX) and P.bounds("n_select") == (None, None)
    assert C.DEFAULT_P in C.WAVELET_P and C.DEFAULT_N in C.N_CHOICES and C.DEFAULT_METHOD in C.METHODS and all(n & (n - 1) == 0 for n in C.N_CHOICES)
