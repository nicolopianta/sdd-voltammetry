# -*- coding: utf-8 -*-
"""Current models used by the deconvolution workflow: the fit models for
Dunn's method and the SDD peak, plus the Helmholtz and Constant-Phase-Element
(CPE) capacitive-current models used to generate/interpret the capacitive
branch. Ported unchanged from supporting_information_script_cpe.py - these
describe idealized current-vs-time relationships and don't themselves need a
"noisy data" adaptation (the noise lives in the measured E, i, t arrays
these are fit to or evaluated on).
"""

import numpy as np

from .fractional_calculus import fractional_derivative


def quad_no_intercept(x, a, b):
    """Fits current as a*x**2 + b*x, with x = sqrt(scan rate) (Dunn's method)."""
    return a * x**2 + b * x


def model_logistic_derivative(x, A, mu, sigma):
    """Derivative-of-a-logistic peak shape, used to fit the SDD semi-derivative peak."""
    z = (x - mu) / sigma
    z = np.clip(z, -50, 50)
    exp_term = np.exp(z)
    return A / sigma * exp_term / (1 + exp_term) ** 2


def helmholtz_capacitive_current(t, E, C):
    """Capacitive current through a simple, voltage-independent capacitance (I = C dE/dt)."""
    return C * np.gradient(E, t)


def cpe_capacitive_current(t, E, Y0, gamma, smooth_window=None, smooth_polyorder=3):
    """
    Capacitive current through a Constant Phase Element (CPE), following
    Charoen-amornkitt et al. (2020): I_CPE(t) = Y0 * D^gamma [E(t) - E(0)],
    the time-domain Riemann-Liouville fractional derivative equivalent of
    I_CPE(s) = Y0 * s**gamma * V(s). gamma=1 reduces to an ideal capacitor.

    Pass `smooth_window` if `E` is a noisy measured potential rather than a
    clean applied ramp (see fractional_calculus.fractional_derivative).
    """
    if gamma >= 1:
        return Y0 * np.gradient(E, t)
    return Y0 * fractional_derivative(t, E - E[0], gamma,
                                       smooth_window=smooth_window,
                                       smooth_polyorder=smooth_polyorder)
