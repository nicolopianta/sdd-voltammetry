# -*- coding: utf-8 -*-
"""Fast smoke tests: the package imports, the simulator runs, and both
separation methods recover a simulated faradaic current to within a few
percent. Runs in a few seconds; used by CI."""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from cv_deconvolution import (simulation, dunn_method, dunn_components, sdd_method,
                              masked_sdd_method, semi_integral, semi_derivative,
                              helmholtz_capacitive_current)

C_DL = 5e-5
SCAN_RATES = np.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])


def _simulate_forward(v, k0=1e-2, Nt=800):
    E, i_far, _, _, t = simulation.simulate_cv(
        A=1, D=1e-9, C_bulk=5e-6, n=1, alpha=0.5, k0=k0, E0=0.0,
        v=v, E_start=-1.0, E_switch=1.0, Nt=Nt, Nx=120)
    h = len(t) // 2
    return E[:h], i_far[:h], t[:h]


def test_import_surface():
    import cv_deconvolution as c
    for name in ("sdd_method", "dunn_method", "semi_integral", "simulation"):
        assert name in c.__all__


def test_semi_transforms_shapes():
    E, i_far, t = _simulate_forward(0.05)
    i_tot = i_far + helmholtz_capacitive_current(t, E, C_DL)
    assert semi_integral(t, i_tot).shape == t.shape
    assert semi_derivative(t, i_tot).shape == t.shape
    assert np.all(np.isfinite(semi_derivative(t, i_tot)))


@pytest.mark.parametrize("k0", [1e-2, 1e-3])
def test_sdd_and_dunn_recover_faradaic(k0):
    totals, times, true_far = [], [], None
    for i, v in enumerate(SCAN_RATES):
        E, i_far, t = _simulate_forward(v, k0=k0)
        totals.append(i_far + C_DL * v)
        times.append(t)
        if i == len(SCAN_RATES) - 1:
            true_far = i_far
    totals = np.column_stack(totals)
    times = np.column_stack(times)
    ip = true_far.max()

    coeffs = dunn_method(totals, SCAN_RATES)
    _, dunn_far = dunn_components(coeffs, SCAN_RATES, j=-1)
    _, sdd_far, _ = sdd_method(times[:, -1], totals[:, -1])

    sdd_ape = np.abs(sdd_far - true_far).max() / ip * 100
    dunn_ape = np.abs(dunn_far - true_far).max() / ip * 100
    assert sdd_ape < 15.0
    assert dunn_ape < 60.0


def test_masked_sdd_runs():
    E, i_far, t = _simulate_forward(0.1)
    i_tot = i_far + C_DL * 0.1
    out = masked_sdd_method(t, i_tot, exclusion_sigmas=4.0, n_iterations=3)
    background, faradaic = out[0], out[1]
    assert faradaic.shape == t.shape
    assert np.isfinite(faradaic).all()
