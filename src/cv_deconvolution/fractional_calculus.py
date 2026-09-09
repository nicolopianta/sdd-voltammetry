# -*- coding: utf-8 -*-
"""Semi-differintegration and Riemann-Liouville fractional calculus, as used
by the Dunn/SDD deconvolution and the CPE capacitive-current model. Ported
from supporting_information_script_cpe.py, with an added optional
Savitzky-Golay smoothing step ahead of the derivative (`smooth_window`) for
use on noisy real data - the default (no smoothing) reproduces the original,
noise-free-only behaviour exactly.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gamma as gammafn

from .smoothing import smooth_derivative


def semi_integral(x, y):
    """Semi-integral (order 1/2) of y with respect to x."""
    dx = x[1] - x[0]
    kernel = 1 / np.sqrt(x)
    kernel[0] = 0
    return dx * fftconvolve(y, kernel, mode='full')[:len(x)] / np.sqrt(np.pi)


def semi_derivative(x, y, smooth_window=None, smooth_polyorder=3):
    """Semi-derivative (order 1/2) of y with respect to x. Pass
    `smooth_window` (a Savitzky-Golay window length, in samples) to estimate
    dy/dx from noisy data instead of a raw finite difference."""
    dx = x[1] - x[0]
    if smooth_window is not None:
        dy = smooth_derivative(y, dx, window_length=smooth_window, polyorder=smooth_polyorder)
    else:
        dy = np.gradient(y, x)
    return semi_integral(x, dy)


def fractional_integral(x, y, order):
    """Riemann-Liouville fractional integral of y wrt x, order in (0, 1).
    Generalizes `semi_integral` (order=0.5)."""
    dx = x[1] - x[0]
    kernel = x ** (order - 1)
    kernel[0] = 0
    return dx * fftconvolve(y, kernel, mode='full')[:len(x)] / gammafn(order)


def fractional_derivative(x, y, order, smooth_window=None, smooth_polyorder=3):
    """Riemann-Liouville fractional derivative of y wrt x, order in (0, 1).
    Generalizes `semi_derivative` (order=0.5); see its `smooth_window` note."""
    dx = x[1] - x[0]
    if smooth_window is not None:
        dy = smooth_derivative(y, dx, window_length=smooth_window, polyorder=smooth_polyorder)
    else:
        dy = np.gradient(y, x)
    return fractional_integral(x, dy, 1 - order)


def upsample_xy(x, y, n):
    """Upsamples two arrays by a factor n (linear interpolation)."""
    x_new = np.linspace(x[0], x[-1], len(x) * n)
    y_new = np.interp(x_new, x, y)
    return x_new, y_new
