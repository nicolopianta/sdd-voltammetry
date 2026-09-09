# -*- coding: utf-8 -*-
"""The Randles-Sevcik equation itself - the relationship this whole dataset
and package are named after, linking a cyclic voltammogram's Faradaic peak
current to the analyte's bulk concentration for a reversible,
diffusion-controlled redox couple. Also the semi-derivative-native
counterpart: the supremum (plateau) of the semi-*integrated* Faradaic
current, read directly off an SDD peak fit's own parameters rather than
from a reconstructed peak current.
"""

from .constants import F

RANDLES_SEVCIK_CONSTANT = 2.69e5  # C * mol^-1 * V^-1/2, at 298 K


def randles_sevcik_peak_current(concentration_mol_cm3, scan_rate_V_s, n=1, area_cm2=1.0,
                                 diffusion_coeff_cm2_s=1e-5):
    """
    Randles-Sevcik equation for a reversible, diffusion-controlled system
    at 298 K:

        i_p = 2.69e5 * n^(3/2) * A * D^(1/2) * v^(1/2) * C

    All arguments in the mixed cgs-style units electrochemistry
    conventionally uses this equation with: concentration in mol/cm^3, area
    in cm^2, diffusion coefficient in cm^2/s, scan rate in V/s. Returns the
    peak current in A.
    """
    return (
        RANDLES_SEVCIK_CONSTANT * n ** 1.5 * area_cm2
        * diffusion_coeff_cm2_s ** 0.5 * scan_rate_V_s ** 0.5 * concentration_mol_cm3
    )


def randles_sevcik_concentration(peak_current_A, scan_rate_V_s, n=1, area_cm2=1.0,
                                  diffusion_coeff_cm2_s=1e-5):
    """
    Inverts `randles_sevcik_peak_current` to estimate the bulk analyte
    concentration (mol/cm^3) from a measured Faradaic peak current - the
    quantity a Randles-Sevcik experiment is meant to recover. Only as good
    as its assumptions: a reversible, diffusion-controlled process, and the
    given electrode area/diffusion coefficient/electron count actually
    matching the real system. Treat the result as tentative, especially
    alongside evidence of a non-ideal peak shape.
    """
    return peak_current_A / (
        RANDLES_SEVCIK_CONSTANT * n ** 1.5 * area_cm2
        * diffusion_coeff_cm2_s ** 0.5 * scan_rate_V_s ** 0.5
    )


def randles_sevcik_diffusion_coefficient(peak_current_A, scan_rate_V_s, concentration_mol_cm3,
                                          n=1, area_cm2=1.0):
    """
    Inverts `randles_sevcik_peak_current` for the diffusion coefficient
    (cm^2/s), the other direction from `randles_sevcik_concentration`: given
    a *known* bulk concentration instead of a known diffusion coefficient,
    estimate D from the measured Faradaic peak current. Same caveats apply
    (reversible, diffusion-controlled process assumed) - treat the result as
    tentative.
    """
    denom = RANDLES_SEVCIK_CONSTANT * n ** 1.5 * area_cm2 * scan_rate_V_s ** 0.5 * concentration_mol_cm3
    return (peak_current_A / denom) ** 2


def semiderivative_fit_plateau(fit_result):
    """
    The supremum (plateau) `L` of the semi-*integrated* Faradaic current -
    a logistic ("sigmoidal") curve in time for a reversible system - implied
    by a fitted semi-derivative peak, read directly off the fit's own
    parameters rather than by reconstructing a peak current.

    `capacitive_models.model_logistic_derivative` is literally the
    derivative of that logistic curve, so its amplitude parameter `A`
    already *is* the sigmoid's plateau: L = A (which is also why
    `sdd_method`'s initial guess for `A` is the observed semi-integral's
    own max).

    Parameters
    ----------
    fit_result : lmfit.model.ModelResult
        From `sdd_method` / `sdd_fit` / `masked_sdd_method`.

    Returns
    -------
    float
        L, in the semi-derivative fit's own units (A * s^(1/2) when the fit
        was done with time, in seconds, as the independent variable - as
        every SDD-family function in this package does).
    """
    return fit_result.params["A"].value


def randles_sevcik_diffusion_coefficient_from_semiderivative(L, concentration_mol_cm3, n=1, area_cm2=1.0):
    """
    Estimates the diffusion coefficient directly from a semi-derivative
    fit's own plateau `L` (see `semiderivative_fit_plateau`), rather than
    from a reconstructed Randles-Sevcik peak current - the SDD-native
    counterpart to `randles_sevcik_diffusion_coefficient`.

    For a reversible system, Oldham's semi-integral electroanalysis gives
    the semi-integrated Faradaic current as a logistic curve in potential
    whose supremum is

        L = n * F * A * C_bulk * sqrt(D)

    Unlike a Randles-Sevcik peak current, `L` does not depend on scan rate
    at all - semi-differintegration is constructed specifically to remove
    that dependence - so this estimate needs no scan-rate term, and is
    comparable in principle across scan rates by construction (rather than
    only after separately correcting for v via i_p/sqrt(v)). Same caveats
    as `randles_sevcik_diffusion_coefficient` otherwise: only as good as
    the reversibility assumption and the given area/concentration/electron
    count.

    Returns D in cm^2/s.
    """
    denom = n * F * area_cm2 * concentration_mol_cm3
    return (L / denom) ** 2


def randles_sevcik_concentration_from_semiderivative(L, diffusion_coeff_cm2_s, n=1, area_cm2=1.0):
    """
    Estimates the bulk analyte concentration directly from a semi-derivative
    fit's own plateau `L` (see `semiderivative_fit_plateau`) and a *known*
    diffusion coefficient - the SDD-native counterpart to
    `randles_sevcik_concentration`, and the inverse of
    `randles_sevcik_diffusion_coefficient_from_semiderivative`.

    For a reversible system, Oldham's semi-integral electroanalysis gives
    the semi-integrated Faradaic current as a logistic curve whose supremum
    is

        L = n * F * A * C_bulk * sqrt(D)

    so C_bulk = L / (n * F * A * sqrt(D)). Like `L` itself, this estimate
    does not depend on scan rate. Same caveats as
    `randles_sevcik_concentration` otherwise: only as good as the
    reversibility assumption and the given area/diffusion coefficient/
    electron count.

    Returns C in mol/cm^3.
    """
    return L / (n * F * area_cm2 * diffusion_coeff_cm2_s ** 0.5)
