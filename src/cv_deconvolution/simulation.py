# -*- coding: utf-8 -*-
"""Forward CV simulator (semi-infinite diffusion + Butler-Volmer kinetics),
plus a modified-Randles-circuit variant coupling it to a CPE capacitive
branch. Ported unchanged from supporting_information_script_cpe.py - these
generate clean, noise-free synthetic ground truth, useful for validating the
`deconvolution.py` methods (including their smoothing options) before
applying them to real data; they are not themselves something you run
against measured data.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma as gammafn
from scipy.linalg import solve_banded

from .constants import F, R, T


def potential_ramp(t, E_start, E_switch, scan_rate):
    """Potential ramp of a CV (works for rising or falling first branch)."""
    t_half = abs(E_start - E_switch) / scan_rate
    direction = 1.0 if E_switch > E_start else -1.0
    if t <= t_half:
        return E_start + direction * scan_rate * t
    else:
        return E_switch - direction * scan_rate * (t - t_half)


def butler_volmer_rates(E, E0, k0, alpha, n):
    """Reaction rates according to the Butler-Volmer equation."""
    exponent = n * F * (E - E0) / (R * T)
    exponent = np.clip(exponent, -700, 700)
    kf = k0 * np.exp(-alpha * exponent)
    kb = k0 * np.exp((1 - alpha) * exponent)
    return kf, kb


def crank_nicolson_dirichlet_banded(C_prev, r, C0_now, C0_prev, CL_now, CL_prev):
    """Concentration vector across x at one time step, via Crank-Nicolson."""
    N = len(C_prev)
    if N < 3:
        C_new = C_prev.copy()
        C_new[0] = C0_now
        C_new[-1] = CL_now
        return C_new

    m = N - 2
    ab = np.zeros((3, m))
    ab[0, 1:] = -r / 2
    ab[1, :] = 1 + r
    ab[2, :-1] = -r / 2

    rhs = (1 - r) * C_prev[1:-1] + (r / 2) * (C_prev[0:-2] + C_prev[2:])
    rhs[0] += (r / 2) * C0_now
    rhs[-1] += (r / 2) * CL_now

    interior = solve_banded((1, 1), ab, rhs)

    C_new = np.empty_like(C_prev)
    C_new[0] = C0_now
    C_new[-1] = CL_now
    C_new[1:-1] = interior
    return C_new


def electrode_flux(Cox_1, Cred_1, kf, kb, D, dx):
    """Flux of species at the electrode."""
    numerator = kf * Cox_1 - kb * Cred_1
    denominator = 1 + (kf * dx / D) + (kb * dx / D)
    J_ox = -numerator / denominator
    J_red = -J_ox
    return J_ox, J_red


def surface_concentration(C1, J, D, dx):
    """Concentration of a species at x=0."""
    return C1 + (J * dx / D)


def faradaic_current(J_ox, n, A):
    """Current from the flux of reactant (positive = net oxidation)."""
    return n * F * A * J_ox


def simulate_cv(
    A=1, D=1e-9, C_bulk=5e-6, n=1, alpha=0.5, k0=1e-3, E0=0,
    v=0.05, E_start=-1, E_switch=1, Nt=400, Nx=50
):
    """
    Simulates one CV cycle with semi-infinite diffusion, oxidizing a
    reduced species (Red -> Ox + n e-) starting at bulk concentration
    C_bulk. Returns E, current, Ox, Red, time.
    """
    t_total = 2 * abs(E_start - E_switch) / v
    dt = t_total / Nt
    dx = (6 * np.sqrt(D * t_total)) / Nx
    r = D * dt / dx**2

    time = np.linspace(0, t_total, Nt)
    E = np.zeros(Nt)
    E[0] = E_start
    current = np.zeros(Nt)

    Ox = np.zeros((Nt, Nx))
    Red = np.zeros((Nt, Nx))
    Ox[0, :] = 0.0
    Red[0, :] = C_bulk

    for j in range(1, Nt):
        E[j] = potential_ramp(time[j], E_start, E_switch, v)
        kf, kb = butler_volmer_rates(E[j], E0, k0, alpha, n)
        J_ox, J_red = electrode_flux(Ox[j-1, 1], Red[j-1, 1], kf, kb, D, dx)

        C0_O_now = surface_concentration(Ox[j-1, 1], J_ox, D, dx)
        C0_R_now = surface_concentration(Red[j-1, 1], J_red, D, dx)

        Ox[j, :] = crank_nicolson_dirichlet_banded(
            Ox[j-1, :], r, C0_O_now, Ox[j-1, 0], 0.0, 0.0)
        Red[j, :] = crank_nicolson_dirichlet_banded(
            Red[j-1, :], r, C0_R_now, Red[j-1, 0], C_bulk, C_bulk)

        Ox[j, -1] = 0.0
        Ox[j, 0] = C0_O_now

        current[j] = faradaic_current(J_ox, n, A)

    return E, current, Ox, Red, time


def simulate_cv_faradaic_cpe_randles(
    A=1, D=1e-9, C_bulk=5e-6, n=1, alpha=0.5, k0=1e-3, E0=0,
    v=0.05, E_start=-1, E_switch=1,
    Y0=1e-6, gamma=1.0, R_ohm=5.0, R_add=400.0,
    Nt=800, Nx=80
):
    """
    Simulates a CV in which a Faradaic branch (Butler-Volmer + semi-infinite
    diffusion) and a capacitive branch (a CPE Y0, gamma in series with
    R_add) are in parallel, in series with an internal resistance R_ohm -
    the modified Randles circuit of Charoen-amornkitt et al. (2020). Returns
    time, Vapp, Vact, Itotal, Ic, If.
    """
    t_total = 2 * abs(E_start - E_switch) / v
    dt = t_total / Nt
    dx = (6 * np.sqrt(D * t_total)) / Nx
    r = D * dt / dx**2
    time = np.linspace(0, t_total, Nt)

    Vapp = np.full(Nt, float(E_start))
    Vact = np.full(Nt, float(E_start))
    Vcpe = np.full(Nt, float(E_start))
    Itotal = np.zeros(Nt)
    Ic = np.zeros(Nt)
    If = np.zeros(Nt)

    Ox = np.zeros((Nt, Nx))
    Red = np.zeros((Nt, Nx))
    Ox[0, :] = 0.0
    Red[0, :] = C_bulk

    # Grunwald-Letnikov / L1 weights for the discretized fractional
    # derivative of order gamma: b_m = (m+1)^(1-gamma) - m^(1-gamma)
    p = 1 - gamma
    if p <= 0:
        b = np.zeros(Nt)
        b[0] = 1.0
    else:
        m = np.arange(Nt, dtype=float)
        b = (m + 1.0) ** p - m ** p
    alpha_c = Y0 / (dt ** gamma * gammafn(2 - gamma))

    I_bracket = 10.0  # generous current bracket (A) for the root solve

    for j in range(1, Nt):
        Vapp[j] = potential_ramp(time[j], E_start, E_switch, v)

        if j > 1:
            dVcpe_hist = Vcpe[1:j] - Vcpe[0:j - 1]
            weights = b[1:j][::-1]
            beta_hist = alpha_c * np.sum(weights * dVcpe_hist)
        else:
            beta_hist = 0.0

        def Ic_of_Vact(Vact_j):
            return (alpha_c * (Vact_j - Vcpe[j - 1]) + beta_hist) / (1 + alpha_c * R_add)

        def If_of_Vact(Vact_j):
            kf, kb = butler_volmer_rates(Vact_j, E0, k0, alpha, n)
            J_ox, _ = electrode_flux(Ox[j - 1, 1], Red[j - 1, 1], kf, kb, D, dx)
            return faradaic_current(J_ox, n, A)

        def residual(I_total_guess):
            Vact_j = Vapp[j] - I_total_guess * R_ohm
            return Ic_of_Vact(Vact_j) + If_of_Vact(Vact_j) - I_total_guess

        Itotal[j] = brentq(residual, -I_bracket, I_bracket)
        Vact[j] = Vapp[j] - Itotal[j] * R_ohm
        Ic[j] = Ic_of_Vact(Vact[j])
        If[j] = Itotal[j] - Ic[j]
        Vcpe[j] = Vact[j] - Ic[j] * R_add

        # diffusion update (semi-implicit), driven by V_act
        kf, kb = butler_volmer_rates(Vact[j], E0, k0, alpha, n)
        J_ox, J_red = electrode_flux(Ox[j - 1, 1], Red[j - 1, 1], kf, kb, D, dx)
        C0_O_now = surface_concentration(Ox[j - 1, 1], J_ox, D, dx)
        C0_R_now = surface_concentration(Red[j - 1, 1], J_red, D, dx)

        Ox[j, :] = crank_nicolson_dirichlet_banded(
            Ox[j - 1, :], r, C0_O_now, Ox[j - 1, 0], 0.0, 0.0)
        Red[j, :] = crank_nicolson_dirichlet_banded(
            Red[j - 1, :], r, C0_R_now, Red[j - 1, 0], C_bulk, C_bulk)
        Ox[j, -1] = 0.0
        Ox[j, 0] = C0_O_now

    return time, Vapp, Vact, Itotal, Ic, If
