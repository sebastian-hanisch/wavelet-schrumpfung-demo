"""AppTest: Auswahlfelder, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import shr_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
FIXED = "Feste Schwelle λ"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


@pytest.mark.parametrize("signal", list(C.SIGNALS))
def test_every_signal_runs(signal):
    at = _run()
    at.selectbox(key="signal_select").set_value(signal).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_the_fixed_threshold_slider_exists_only_for_the_fixed_method_and_keeps_its_value():
    at = _run()
    assert FIXED not in _labels(at)
    at.radio(key="method_select").set_value("fest").run(timeout=TIMEOUT)
    assert not at.exception and FIXED in _labels(at)
    at2 = _run({"m": "fest", "lam": "3.5"})
    assert {x.label: x for x in at2.slider}[FIXED].value == 3.5
    at2.radio(key="method_select").set_value("SURE").run(timeout=TIMEOUT)
    at2.radio(key="method_select").set_value("fest").run(timeout=TIMEOUT)
    assert {x.label: x for x in at2.slider}[FIXED].value == 3.5


def test_the_level_choices_follow_the_record_length():
    at = _run({"n": "256", "j": "9"})
    assert at.select_slider(key="levels_select").value == 7
    at2 = _run({"n": "4096", "j": "9"})
    assert at2.select_slider(key="levels_select").value == 9
    at2.select_slider(key="n_select").set_value(256).run(timeout=TIMEOUT)
    assert not at2.exception and at2.select_slider(key="levels_select").value == 7


def test_changing_wavelet_method_rule_sigma_and_cycle_runs():
    at = _run()
    at.select_slider(key="p_select").set_value(1).run(timeout=TIMEOUT)
    at.radio(key="method_select").set_value("Hybrid").run(timeout=TIMEOUT)
    at.radio(key="rule_select").set_value("hart").run(timeout=TIMEOUT)
    at.radio(key="sigma_select").set_value("bekannt").run(timeout=TIMEOUT)
    at.radio(key="cycle_select").set_value("an").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.select_slider(key="p_select").value == 1 and at.radio(key="method_select").value == "Hybrid" and at.radio(key="cycle_select").value == "an"


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "2048", "s": "Doppler", "snr": "3.5", "w": "2", "j": "5", "m": "SURE", "r": "hart", "sg": "bekannt", "cs": "an", "seed": "42"})
    assert at.select_slider(key="n_select").value == 2048 and at.selectbox(key="signal_select").value == "Doppler" and at.select_slider(key="p_select").value == 2 and at.select_slider(key="levels_select").value == 5
    assert at.radio(key="method_select").value == "SURE" and at.radio(key="rule_select").value == "hart" and at.radio(key="sigma_select").value == "bekannt" and at.radio(key="cycle_select").value == "an"
    assert at.number_input(key="seed_input").value == 42 and {x.label: x for x in at.slider}["Signal-Rausch-Verhältnis"].value == 3.5
    at2 = _run({"n": "3000", "s": "unsinn", "snr": "99", "w": "5", "j": "99", "m": "Bayes", "r": "mittel", "sg": "x", "cs": "vielleicht"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.selectbox(key="signal_select").value == C.DEFAULT_SIGNAL and at2.select_slider(key="p_select").value == C.DEFAULT_P
    assert at2.select_slider(key="levels_select").value == C.DEFAULT_LEVELS and at2.radio(key="method_select").value == C.DEFAULT_METHOD and at2.radio(key="rule_select").value == C.DEFAULT_RULE
    assert at2.radio(key="sigma_select").value == C.DEFAULT_SIGMA and at2.radio(key="cycle_select").value == C.DEFAULT_CYCLE and {x.label: x for x in at2.slider}["Signal-Rausch-Verhältnis"].value == C.SNR_MAX


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Cycle Spinning: Gibbs-Artefakte verschwinden").click().run(timeout=TIMEOUT)
    assert not at.exception
    assert at.selectbox(key="signal_select").value == "HeaviSine" and at.radio(key="rule_select").value == "hart" and at.radio(key="cycle_select").value == "an"
    at.button(key="preset_Niedriger Rauschabstand: die Schwelle zählt").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="signal_select").value == "Doppler" and {x.label: x for x in at.slider}["Signal-Rausch-Verhältnis"].value == 1.0 and at.radio(key="cycle_select").value == "aus"


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
