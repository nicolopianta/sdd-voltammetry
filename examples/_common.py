# -*- coding: utf-8 -*-
"""Shared helpers for the simulated examples."""
import warnings
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("ignore")          # curve_fit covariance / RL-kernel noise

from cv_deconvolution.constants import F, R, T
from cv_deconvolution import simulation

# reference scan rate used to map the dimensionless kinetic number Lambda
# (Eq. 6 of the paper) onto a rate constant k0
V_REF = 5e-3          # V/s


def k0_from_lambda(lam, D, v_ref=V_REF):
    """Rate constant k0 that gives a dimensionless kinetic number Lambda,
    following the mapping used for the maps in the paper."""
    tau = (R * T) / (F * v_ref)
    return lam * np.sqrt(D) / np.sqrt(tau)


def apef(reconstructed, truth, i_peak):
    """Absolute percentage error of a reconstructed faradaic current,
    normalised by the peak current (Eq. 16 of the paper)."""
    return np.abs(reconstructed - truth) / abs(i_peak) * 100.0


def simulate_forward_lsv(v, D=1e-9, C_bulk=5e-6, n=1, alpha=0.5, k0=1e-2,
                         E0=0.0, E_start=-1.0, E_switch=1.0, Nt=2000, Nx=None, A=1.0):
    """One forward (linear-sweep) branch: returns E, i_faradaic, t.

    Nx defaults to a value that keeps the spatial grid fine enough to
    resolve a sharp diffusion peak (as in the paper's simulations)."""
    if Nx is None:
        Nx = int(np.sqrt(18 * Nt))
    E, i_far, _, _, t = simulation.simulate_cv(
        A=A, D=D, C_bulk=C_bulk, n=n, alpha=alpha, k0=k0, E0=E0,
        v=v, E_start=E_start, E_switch=E_switch, Nt=Nt, Nx=Nx)
    half = len(t) // 2
    return E[:half], i_far[:half], t[:half]
