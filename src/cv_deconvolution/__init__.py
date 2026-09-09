# -*- coding: utf-8 -*-
"""
cv_deconvolution
=================

Separation of the faradaic and capacitive contributions to a voltammogram,
by the Semi-Derivative Decomposition (SDD) method and by the scan-rate-based
(Dunn) method it is benchmarked against.

- `simulation`: forward CV simulator (semi-infinite diffusion + Butler-Volmer
  kinetics; also a modified-Randles-circuit + CPE variant) - generates the
  noise-free synthetic ground truth used to validate the methods below.
- `fractional_calculus`: semi- and fractional integrals/derivatives, with an
  optional smoothing step ahead of any differentiation.
- `smoothing`: Savitzky-Golay smoothing / derivative helpers for noisy data.
- `alignment`: build aligned (potential, current, time) matrices from raw
  measured curves recorded at several scan rates; also interpolation and
  branch-splitting helpers.
- `capacitive_models`: the Dunn and SDD fit models, plus the Helmholtz and
  constant-phase-element (CPE) capacitive-current models.
- `background_trend` / `baseline`: fit or estimate a slowly-varying
  background (constant / linear / exponential / power-law trends; SNIP).
- `deconvolution`: Dunn's method and the SDD method themselves - including
  `masked_sdd_method`, which estimates the background iteratively - plus a
  combined runner and a plotting helper.
- `randles_sevcik`: the Randles-Sevcik equation and its semi-integral
  counterpart, for turning a reconstruction into a concentration / D.

Typical usage
-------------
    from cv_deconvolution import build_scan_rate_matrix, dunn_and_sdd_deconvolve, plot_deconvolution

    curves = {v_in_V_s: (E, i), ...}       # your measured curves
    E, totals, times, vs = build_scan_rate_matrix(curves, branch=0)
    result = dunn_and_sdd_deconvolve(totals, times, vs, j=-1,
                                     dunn_smooth_window=15, sdd_smooth_window=25)
    plot_deconvolution(E, totals, vs, -1, result["dunn_i_c"], result["dunn_i_f"],
                       result["sdd_background"], result["sdd_faradaic"])
"""

from .alignment import (
    build_scan_rate_matrix, align_branches_to_common_grid, reconstruct_time_axis,
    split_cv_branches, turning_point_index,
    interpolate_branch_onto, subtract_blank_branch,
    find_zero_crossing_index, trim_to_zero_crossing,
)
from .background_trend import (
    BACKGROUND_MODELS, background_constant, background_linear,
    background_exponential, background_power_law,
    fit_background_trend, compare_background_trends,
)
from .baseline import snip_background, snip_decompose
from .capacitive_models import (
    quad_no_intercept, model_logistic_derivative,
    helmholtz_capacitive_current, cpe_capacitive_current,
)
from .deconvolution import (
    dunn_method, dunn_components, sdd_method, sdd_fit, masked_sdd_method,
    semi_calculus_native, reconstruct_faradaic_current,
    dunn_and_sdd_deconvolve, plot_deconvolution,
)
from .fractional_calculus import (
    semi_integral, semi_derivative, fractional_integral, fractional_derivative, upsample_xy,
)
from .randles_sevcik import (
    RANDLES_SEVCIK_CONSTANT, randles_sevcik_peak_current, randles_sevcik_concentration,
    randles_sevcik_diffusion_coefficient, semiderivative_fit_plateau,
    randles_sevcik_diffusion_coefficient_from_semiderivative,
    randles_sevcik_concentration_from_semiderivative,
)
from .smoothing import safe_savgol_window, smooth_signal, smooth_derivative
from . import simulation

__all__ = [
    "build_scan_rate_matrix", "align_branches_to_common_grid", "reconstruct_time_axis",
    "split_cv_branches", "turning_point_index",
    "interpolate_branch_onto", "subtract_blank_branch",
    "find_zero_crossing_index", "trim_to_zero_crossing",
    "BACKGROUND_MODELS", "background_constant", "background_linear",
    "background_exponential", "background_power_law",
    "fit_background_trend", "compare_background_trends",
    "snip_background", "snip_decompose",
    "quad_no_intercept", "model_logistic_derivative",
    "helmholtz_capacitive_current", "cpe_capacitive_current",
    "dunn_method", "dunn_components", "sdd_method", "sdd_fit", "masked_sdd_method",
    "semi_calculus_native", "reconstruct_faradaic_current",
    "dunn_and_sdd_deconvolve", "plot_deconvolution",
    "semi_integral", "semi_derivative", "fractional_integral", "fractional_derivative", "upsample_xy",
    "RANDLES_SEVCIK_CONSTANT", "randles_sevcik_peak_current", "randles_sevcik_concentration",
    "randles_sevcik_diffusion_coefficient", "semiderivative_fit_plateau",
    "randles_sevcik_diffusion_coefficient_from_semiderivative",
    "randles_sevcik_concentration_from_semiderivative",
    "safe_savgol_window", "smooth_signal", "smooth_derivative",
    "simulation",
]
