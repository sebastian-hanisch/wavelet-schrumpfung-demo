"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_d():
    """Koeffizienten d = (3, -1, 0,5, 2) bei Rauschvarianz 1: SURE(lam) = m - 2 #{|d| <= lam} + sum min(|d|, lam)^2:
    SURE(0) = 4, SURE(0,5) = 4 - 2 + (0,25 + 0,25 + 0,25 + 0,25) = 3, SURE(1) = 4 - 4 + (1 + 1 + 0,25 + 1) = 3,25, SURE(3) = 4 - 8 + (9 + 1 + 0,25 + 4) = 10,25; Minimum bei lam = 0,5."""
    return np.array([3.0, -1.0, 0.5, 2.0])


@pytest.fixture
def mini_x():
    """Haar, x = (1, 2, 3, 4), eine Stufe: d = (-1/sqrt 2, -1/sqrt 2) = (-0,707, -0,707), a = (3/sqrt 2, 7/sqrt 2); weiche Schwelle 0,8 nullt beide Details (0,707 < 0,8),
    die Rücktransformation aus a allein ergibt (a0, a0, a1, a1)/sqrt 2 = (1,5, 1,5, 3,5, 3,5)."""
    return np.array([1.0, 2.0, 3.0, 4.0])
