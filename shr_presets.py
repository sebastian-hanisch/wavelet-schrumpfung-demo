"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Die feste Schwelle erscheint nur beim Verfahren „fest“; ihr Wert steht unter `fixed_kept` (der Regler bekommt ihn als Startwert, die Adresszeile führt ihn als `lam`).
"""

import random
from dataclasses import dataclass

import streamlit as st

import shr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_select": SettingSpec("n", int, C.DEFAULT_N, choices=C.N_CHOICES),
    "signal_select": SettingSpec("s", str, C.DEFAULT_SIGNAL, choices=C.SIGNALS),
    "snr_slider": SettingSpec("snr", float, C.DEFAULT_SNR, C.SNR_MIN, C.SNR_MAX),
    "p_select": SettingSpec("w", int, C.DEFAULT_P, choices=C.WAVELET_P),
    "levels_select": SettingSpec("j", int, C.DEFAULT_LEVELS, choices=tuple(range(C.LEVEL_MIN, C.MAX_LEVELS + 1))),
    "method_select": SettingSpec("m", str, C.DEFAULT_METHOD, choices=C.METHODS),
    "rule_select": SettingSpec("r", str, C.DEFAULT_RULE, choices=C.RULES),
    "fixed_kept": SettingSpec("lam", float, C.DEFAULT_FIXED, C.FIXED_MIN, C.FIXED_MAX),
    "sigma_select": SettingSpec("sg", str, C.DEFAULT_SIGMA, choices=C.SIGMA_MODES),
    "cycle_select": SettingSpec("cs", str, C.DEFAULT_CYCLE, choices=C.CYCLE_MODES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_select", "signal": "signal_select", "snr": "snr_slider", "p": "p_select", "levels": "levels_select", "method": "method_select", "rule": "rule_select", "fixed": "fixed_kept",
               "sigma": "sigma_select", "cycle": "cycle_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        if key in preset:
            st.session_state[state_key] = preset[key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
