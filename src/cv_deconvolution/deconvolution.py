# -*- coding: utf-8 -*-
"""Dunn's method and the SDD (Semi-Derivative Deconvolution) method for
separating the capacitive and Faradaic contributions to a CV, adapted from
supporting_information_script_cpe.py's `dunn_and_sdd_analysis` to work on
real, noisy, ground-truth-free data (see `alignment.build_scan_rate_matrix`
for how to get `totals`/`times`/`vs` from measured curves).
"""

import lmfit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .capacitive_models import quad_no_intercept, model_logistic_derivative
from .fractional_calculus import semi_integral, semi_derivative, upsample_xy
from .smoothing import smooth_signal


def reconstruct_faradaic_current(t, dy_faradaic, i_total, upsample_factor=1000):
    """
    Semi-integrates a semi-derivative-domain Faradaic component back into
    the current domain - the same reconstruction step `sdd_fit` and
    `masked_sdd_method` use internally, exposed separately so it
    can be applied to a Faradaic residual obtained some other way - e.g.
    the measured semi-derivative minus a *fitted background trend*
    (`background_trend.compare_background_trends`) instead of minus a
    fitted peak model.

    Parameters
    ----------
    t : np.ndarray
        Time, native grid.
    dy_faradaic : np.ndarray
        The semi-derivative-domain Faradaic component (native `t` grid) -
        e.g. `dy_measured - best_trend_fit_func(t)`.
    i_total : np.ndarray
        The real measured current for this segment, used for the
        background = i_total - faradaic step.

    Returns
    -------
    faradaic, background : np.ndarray
        The semi-integrated Faradaic current, and `i_total - faradaic`.
    """
    t = np.asarray(t, dtype=float)
    dy_faradaic = np.asarray(dy_faradaic, dtype=float)
    i_total = np.asarray(i_total, dtype=float)
    x_up, dyf_up = upsample_xy(t, dy_faradaic, upsample_factor)
    faradaic = semi_integral(x_up, dyf_up)[::upsample_factor]
    background = i_total - faradaic
    return faradaic, background


def dunn_method(totals, vs, smooth_window=None, smooth_polyorder=3):
    """Dunn's method: fits i = a*v + b*sqrt(v) independently at every
    potential point across the measured scan rates, separating the
    capacitive (a*v) and Faradaic (b*sqrt(v)) contributions.

    Parameters
    ----------
    totals : np.ndarray, shape (n_points, n_scan_rates)
        Measured total current, aligned across scan rates.
    vs : np.ndarray, shape (n_scan_rates,)
        Scan rates (V/s), matching the columns of `totals`.
    smooth_window : int, optional
        Savitzky-Golay window (samples) to smooth each scan-rate curve
        along the potential axis before fitting - recommended for noisy
        real data.

    Returns
    -------
    coeffs : np.ndarray, shape (n_points, 2)
        [a, b] fit coefficients at every potential point.
    """
    data = totals
    if smooth_window is not None:
        data = np.column_stack([
            smooth_signal(totals[:, col], window_length=smooth_window, polyorder=smooth_polyorder)
            for col in range(totals.shape[1])
        ])
    sqrt_v = np.sqrt(vs)
    coeffs = np.empty((data.shape[0], 2))
    for row in range(data.shape[0]):
        popt, _ = curve_fit(quad_no_intercept, sqrt_v, data[row])
        coeffs[row] = popt
    return coeffs


def dunn_components(coeffs, vs, j):
    """Capacitive (i_c) and Faradaic (i_f) current at scan rate `vs[j]`,
    from Dunn's method coefficients (see `dunn_method`)."""
    i_c = coeffs[:, 0] * vs[j]
    i_f = coeffs[:, 1] * np.sqrt(vs[j])
    return i_c, i_f


def semi_calculus_native(t, i, smooth_window=None, smooth_polyorder=3, upsample_factor=1000):
    """
    Computes the semi-derivative and semi-integral of (t, i) at native
    resolution (same length as t), via the same upsample -> smooth ->
    differentiate/integrate -> downsample pipeline `sdd_method` and
    `masked_sdd_method` use internally. Exposed separately so the
    convolution can be evaluated once over a **full CV cycle** and then
    sliced per branch afterward: semi-differintegration is a non-local
    ("memory") operator, so differentiating/integrating one branch in
    isolation - after resetting its own t=0 - discards whatever history
    came before it (e.g. the forward sweep, for the backward one), which
    the single-call convenience functions below implicitly assume doesn't
    matter. Feed the sliced (`dy`, `ys`) into `sdd_fit`
    to analyze just one branch correctly.

    Returns
    -------
    dy, ys : np.ndarray
        Semi-derivative and semi-integral of `i` wrt `t`, native `t` grid.
    """
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    i_for_calc = i
    if smooth_window is not None:
        i_for_calc = smooth_signal(i, window_length=smooth_window, polyorder=smooth_polyorder)
    x, y = upsample_xy(t, i_for_calc, upsample_factor)
    dy = semi_derivative(x, y)[::upsample_factor]
    ys = semi_integral(x, y)[::upsample_factor]
    return dy, ys


def sdd_fit(t, dy, ys, i_total, upsample_factor=1000):
    """`sdd_method`'s fit-and-reconstruct step, given an already-computed
    semi-derivative `dy` and semi-integral `ys` (native `t` grid, e.g. from
    `semi_calculus_native` run over a full cycle and then sliced to one
    branch) instead of computing them from `i_total` itself. `i_total` is
    still the real current for this segment - used for the final
    background = i_total - faradaic step.

    Parameters, returns: as `sdd_method`.
    """
    t = np.asarray(t, dtype=float)
    dy = np.asarray(dy, dtype=float)
    ys = np.asarray(ys, dtype=float)
    i_total = np.asarray(i_total, dtype=float)

    model = lmfit.Model(model_logistic_derivative)
    quarter = dy.size // 4
    params = model.make_params(
        A=ys.max(),
        mu=t[dy[quarter:].argmax() + quarter],
        sigma=ys[quarter:].max() / (4 * dy[quarter:].max()),
    )
    fit_result = model.fit(dy, params, x=t, method='leastsq')

    x_up, dyf_up = upsample_xy(t, fit_result.best_fit, upsample_factor)
    faradaic = semi_integral(x_up, dyf_up)[::upsample_factor]
    background = i_total - faradaic
    return background, faradaic, fit_result


def sdd_method(t, i_total, smooth_window=None, smooth_polyorder=3, upsample_factor=1000):
    """SDD (Semi-Derivative Deconvolution): fits the semi-derivative of the
    measured current with a logistic-derivative peak model, then
    reconstructs the Faradaic current by semi-integrating that fit; the
    difference from the measured current is the capacitive background.

    Parameters
    ----------
    t, i_total : np.ndarray
        Time and current of a single scan-rate curve (see
        `alignment.reconstruct_time_axis` to get `t` for real data).
    smooth_window : int, optional
        Savitzky-Golay window, in samples of the *original* (not upsampled)
        `t`/`i_total` - the curve is smoothed before the upsample+
        differentiate step below, since an unsmoothed np.gradient on noisy
        current turns the semi-derivative peak into noise. Strongly
        recommended for real data.

    Returns
    -------
    background : np.ndarray
        Estimated capacitive current (measured total minus the
        semi-integrated fit below - i.e. computed from the raw, unsmoothed
        `i_total`, so the reconstruction isn't itself over-smoothed).
    faradaic : np.ndarray
        Estimated (semi-integrated) Faradaic current.
    fit_result : lmfit.model.ModelResult
        The logistic-derivative fit to the semi-derivative peak.
    """
    t = np.asarray(t, dtype=float)
    i_total = np.asarray(i_total, dtype=float)
    dy, ys = semi_calculus_native(t, i_total, smooth_window, smooth_polyorder, upsample_factor)
    return sdd_fit(t, dy, ys, i_total, upsample_factor)


def masked_sdd_method(t, i_total, exclusion_sigmas=4.0, n_iterations=5,
                       smooth_window=None, smooth_polyorder=3, upsample_factor=1000):
    """
    A variant of `sdd_method` that iteratively separates the semi-derivative
    into a logistic-derivative peak and *everything else* (the background),
    instead of fitting the peak to the whole domain in one shot. A logistic-
    derivative peak is ~0 beyond a few `sigma` of its center by
    construction, so any real signal out there can't be peak - it gets
    folded into the background estimate, and the peak is refit on what's
    left. This avoids a failure mode plain `sdd_method` can hit: a broad,
    non-zero "shoulder" in the semi-derivative pulling a whole-domain fit
    into a peak that ends up too wide and too short.

    Algorithm, repeated up to `n_iterations` times:
      1. mask out points within `exclusion_sigmas` * sigma of the current
         peak center mu (where the peak model isn't negligible);
      2. rebuild the background by linearly interpolating across that
         excluded region from the surviving ("background-only") points;
      3. refit the logistic-derivative peak to (measured - background).

    Parameters
    ----------
    t, i_total : np.ndarray
        Time and current of a single scan-rate curve/branch.
    exclusion_sigmas : float
        Half-width of the region masked out around the peak, in units of
        the peak's own fitted sigma. Larger values trust the peak model
        further into its tails; smaller values are more conservative about
        what counts as "peak" but leave the background less informed right
        next to it.
    n_iterations : int
        Number of mask/rebuild-background/refit cycles.
    smooth_window : int, optional
        Savitzky-Golay window, in samples of the original `t`/`i_total` -
        see `sdd_method`.

    Returns
    -------
    background : np.ndarray
        Estimated capacitive current (measured total minus the Faradaic
        reconstruction below).
    faradaic : np.ndarray
        Estimated (semi-integrated) Faradaic current.
    fit_result : lmfit.model.ModelResult
        The final logistic-derivative fit, fit against the background-
        subtracted residual of its last iteration - so its own
        `.rsquared`/`.residual`/etc. describe that fit, not a fit against
        the raw measured semi-derivative. Use the separately-returned `dy`
        below for the measured curve (mirroring `sdd_method`'s
        `fit_result.data`, without the mismatch monkeypatching `.data`
        after the fact would create).
    dy_background : np.ndarray
        The final masked/interpolated background, in the semi-derivative
        domain (native `t` grid) - unlike `sdd_method`, this comes directly
        from data outside the peak rather than being a leftover residual.
    dy : np.ndarray
        The measured (total) semi-derivative, native `t` grid.
    """
    t = np.asarray(t, dtype=float)
    i_total = np.asarray(i_total, dtype=float)

    i_for_fit = i_total
    if smooth_window is not None:
        i_for_fit = smooth_signal(i_total, window_length=smooth_window, polyorder=smooth_polyorder)

    x, y = upsample_xy(t, i_for_fit, upsample_factor)
    dy = semi_derivative(x, y)[::upsample_factor]
    ys = semi_integral(x, y)[::upsample_factor]

    model = lmfit.Model(model_logistic_derivative)
    quarter = dy.size // 4
    # Initial guess only (no whole-domain fit): bootstrapping the first
    # mask from an unconstrained fit is unreliable, since that fit can
    # itself already be badly diverged (e.g. sigma wider than the entire
    # domain) whenever the whole-domain shape is a poor match to a single
    # peak - which is exactly the situation this method exists to recover
    # from. The raw peak-location/width heuristic below stays bounded to
    # the visible data even when that happens.
    mu = t[dy[quarter:].argmax() + quarter]
    sigma = abs(ys[quarter:].max() / (4 * dy[quarter:].max()))
    params = model.make_params(A=ys.max(), mu=mu, sigma=sigma)
    # Bounding sigma and A matters most for the unconstrained fallback fit
    # below (when masking collapses on the very first pass), and for
    # curves with no real Faradaic peak to find: without the bounds either
    # parameter can diverge to something many times larger than anything in
    # the data (a wide, thin, near-flat "peak" still integrates to something
    # small, so the reconstruction can look deceptively plausible even then).
    params["sigma"].min = 1e-12
    params["sigma"].max = t[-1] - t[0]
    A_bound = 10 * np.abs(dy).max() * params["sigma"].value * 4
    params["A"].min = -A_bound
    params["A"].max = A_bound

    fit_result = None
    dy_background = np.zeros_like(dy)
    for _ in range(n_iterations):
        mask = np.abs(t - mu) > exclusion_sigmas * sigma
        if mask.sum() < 3:
            break
        dy_background = np.interp(t, t[mask], dy[mask])
        residual_dy = dy - dy_background
        fit_result = model.fit(residual_dy, params, x=t, method="leastsq")
        params = fit_result.params
        mu = params["mu"].value
        sigma = abs(params["sigma"].value)

    if fit_result is None:
        # exclusion window covered the whole domain on the very first pass
        # (e.g. the initial guess itself was already this wide) - nothing
        # to mask against, so fall back to the plain whole-domain fit.
        fit_result = model.fit(dy, params, x=t, method="leastsq")
        dy_background = dy - fit_result.best_fit

    dy_faradaic = fit_result.best_fit
    _, dyf_up = upsample_xy(t, dy_faradaic, upsample_factor)
    faradaic = semi_integral(x, dyf_up)[::upsample_factor]
    background = i_total - faradaic

    return background, faradaic, fit_result, dy_background, dy


def dunn_and_sdd_deconvolve(totals, times, vs, j=-1,
                             dunn_smooth_window=None, sdd_smooth_window=None,
                             smooth_polyorder=3):
    """Runs Dunn's method (across all scan rates) and the SDD method (on
    scan rate `vs[j]`) on aligned CV data (see
    `alignment.build_scan_rate_matrix`).

    Returns
    -------
    dict with keys 'dunn_coeffs', 'dunn_i_c', 'dunn_i_f', 'sdd_background',
    'sdd_faradaic', 'sdd_fit'.
    """
    coeffs = dunn_method(totals, vs, smooth_window=dunn_smooth_window, smooth_polyorder=smooth_polyorder)
    dunn_i_c, dunn_i_f = dunn_components(coeffs, vs, j)

    sdd_background, sdd_faradaic, sdd_fit = sdd_method(
        times[:, j], totals[:, j],
        smooth_window=sdd_smooth_window, smooth_polyorder=smooth_polyorder,
    )

    return {
        "dunn_coeffs": coeffs, "dunn_i_c": dunn_i_c, "dunn_i_f": dunn_i_f,
        "sdd_background": sdd_background, "sdd_faradaic": sdd_faradaic, "sdd_fit": sdd_fit,
    }


def plot_deconvolution(E, totals, vs, j, dunn_i_c, dunn_i_f,
                        sdd_background, sdd_faradaic, true_capacitive=None, title=""):
    """Plots the Dunn and SDD decompositions side by side. Pass
    `true_capacitive` (e.g. from a synthetic-data validation run against
    `simulation.py`) to overlay a known-truth reference; real data has none,
    so it's optional."""
    total = totals[:, j]
    fig, ax = plt.subplots(2, sharex=True)
    rate_label = f"v = {1000 * vs[j]:.3g} mV/s"
    fig.suptitle(f"{title} — {rate_label}" if title else rate_label)

    ax[0].set_title("Dunn's method")
    ax[0].fill_between(E, 0, dunn_i_c * 1e6, color="steelblue", alpha=0.6, label="capacitive")
    ax[0].fill_between(E, dunn_i_c * 1e6, (dunn_i_c + dunn_i_f) * 1e6, color="orange", alpha=0.6, label="faradaic")
    ax[0].plot(E, total * 1e6, color="black", linewidth=2, label="total")
    if true_capacitive is not None:
        ax[0].plot(E, true_capacitive * 1e6, '--', color='r', label="true capacitive")
    ax[0].set_ylabel("Current (µA)")
    ax[0].legend(fontsize=8)
    ax[0].tick_params(labelbottom=False)

    ax[1].set_title("SDD method")
    ax[1].fill_between(E, 0, sdd_background * 1e6, color="steelblue", alpha=0.6, label="capacitive")
    ax[1].fill_between(E, sdd_background * 1e6, (sdd_background + sdd_faradaic) * 1e6, color="orange", alpha=0.6, label="faradaic")
    ax[1].plot(E, total * 1e6, color="black", linewidth=2, label="total")
    if true_capacitive is not None:
        ax[1].plot(E, true_capacitive * 1e6, '--', color='r', label="true capacitive")
    ax[1].set_xlabel("Potential (V)")
    ax[1].set_ylabel("Current (µA)")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    return fig, ax
