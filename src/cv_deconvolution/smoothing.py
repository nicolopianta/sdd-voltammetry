# -*- coding: utf-8 -*-
"""Savitzky-Golay smoothing helpers for extracting derivatives/semi-derivatives
from noisy experimental current-vs-potential data. The synthetic curves
produced by `simulation.py` are smooth by construction and don't need this;
real, measured CVs generally do, especially before any differentiation step
(np.gradient on raw noisy data amplifies noise into an unusable signal).
"""

import numpy as np
from scipy.signal import savgol_filter


def safe_savgol_window(n_points, window_length=None, polyorder=3):
    """Picks an odd Savitzky-Golay window length, > polyorder, that fits
    within `n_points`. Defaults to ~5% of the signal length."""
    if window_length is None:
        window_length = int(round(n_points * 0.05))
    window_length = max(window_length, polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > n_points:
        window_length = n_points if n_points % 2 else n_points - 1
    return window_length


def smooth_signal(y, window_length=None, polyorder=3):
    """Savitzky-Golay smoothing of a noisy measured signal `y`."""
    y = np.asarray(y, dtype=float)
    window_length = safe_savgol_window(len(y), window_length, polyorder)
    return savgol_filter(y, window_length, polyorder)


def smooth_derivative(y, dx, window_length=None, polyorder=3):
    """Savitzky-Golay estimate of dy/dx (a local polynomial fit evaluated
    through its derivative), used in place of np.gradient wherever `y` is
    noisy measured data rather than a clean simulated curve."""
    y = np.asarray(y, dtype=float)
    window_length = safe_savgol_window(len(y), window_length, polyorder)
    return savgol_filter(y, window_length, polyorder, deriv=1, delta=dx)
