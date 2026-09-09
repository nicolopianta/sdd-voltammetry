# -*- coding: utf-8 -*-
"""Characterizing the *shape* of a semi-derivative's capacitive background,
instead of just reconstructing it via linear interpolation between
background-only points (as `masked_sdd_method` does
internally). Fits a small set of candidate trend models - constant, linear,
exponential decay, power-law - to those background-only points (the ones a
masked SDD fit already identified as outside its peak's exclusion window)
and ranks them by corrected Akaike Information Criterion (AICc), so the
comparison itself answers "does this look constant, drifting, decaying,
...?" empirically instead of assuming a shape up front. Whichever model
wins can then be evaluated over the whole domain for a smooth, parametric
background to subtract.
"""

import numpy as np
from scipy.optimize import curve_fit


def background_constant(t, c):
    """bg(t) = c."""
    return np.full_like(np.asarray(t, dtype=float), c)


def background_linear(t, a, b):
    """bg(t) = a + b*t - a flat background is b=0; a steadily
    charging/discharging one shows up as b != 0."""
    return a + b * np.asarray(t, dtype=float)


def background_exponential(t, a, tau, c):
    """bg(t) = a*exp(-t/tau) + c - a relaxing/decaying background settling
    toward a plateau `c` with time constant `tau`."""
    return a * np.exp(-np.asarray(t, dtype=float) / tau) + c


def background_power_law(t, a, gamma, c, t_offset=1e-6):
    """bg(t) = a*(t+t_offset)^(-gamma) + c - the shape a non-ideal (CPE-like)
    capacitive background follows (see `capacitive_models.cpe_capacitive_current`);
    `t_offset` avoids the singularity at t=0 without meaningfully changing
    the fit anywhere else."""
    return a * (np.asarray(t, dtype=float) + t_offset) ** (-gamma) + c


BACKGROUND_MODELS = {
    "constant": background_constant,
    "linear": background_linear,
    "exponential": background_exponential,
    "power_law": background_power_law,
}


def _initial_guess(name, t, y):
    span = t.max() - t.min()
    span = span if span > 0 else 1.0
    if name == "constant":
        return [float(np.mean(y))]
    if name == "linear":
        b0 = (y[-1] - y[0]) / span
        a0 = y[0] - b0 * t[0]
        return [float(a0), float(b0)]
    if name == "exponential":
        c0 = float(y[-1])
        a0 = float(y[0] - c0) or 1e-9
        return [a0, span / 2, c0]
    if name == "power_law":
        c0 = float(y[-1])
        a0 = float((y[0] - c0) * (t[0] + 1e-6)) or 1e-9
        return [a0, 1.0, c0]
    raise ValueError(f"unknown background model {name!r}")


def fit_background_trend(t, y, model_name, maxfev=20000):
    """
    Fits one named model from `BACKGROUND_MODELS` to (t, y) - typically the
    background-only points a masked SDD fit already identified (outside
    its peak's exclusion window).

    Returns
    -------
    params : np.ndarray
        The fitted parameters, in the order the model function takes them.
    fit_func : callable
        `fit_func(t_new)` evaluates the fitted model at any `t_new`.

    Raises
    ------
    RuntimeError
        If `scipy.optimize.curve_fit` fails to converge.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    func = BACKGROUND_MODELS[model_name]
    p0 = _initial_guess(model_name, t, y)
    popt, _ = curve_fit(func, t, y, p0=p0, maxfev=maxfev)

    def fit_func(t_new, _func=func, _popt=popt):
        return _func(np.asarray(t_new, dtype=float), *_popt)

    return popt, fit_func


def _aicc(y, y_fit, n_params):
    n = len(y)
    rss = max(float(np.sum((y - y_fit) ** 2)), 1e-300)
    aic = n * np.log(rss / n) + 2 * n_params
    denom = n - n_params - 1
    correction = (2 * n_params * (n_params + 1) / denom) if denom > 0 else np.inf
    return aic + correction


def compare_background_trends(t, y, model_names=None):
    """
    Fits every candidate background model (default: all of
    `BACKGROUND_MODELS`) to (t, y) and ranks them by corrected Akaike
    Information Criterion (AICc) - lower is better; AICc penalizes extra
    parameters, so a flat background won't lose to a power-law fit that's
    only marginally closer to noisy points just because it has more knobs.
    A model that fails to converge (e.g. fitting an exponential/power-law
    to a genuinely flat background is often ill-conditioned by
    construction) is silently skipped.

    Parameters
    ----------
    t, y : np.ndarray
        The background-only points to fit - e.g. `t[mask]`, `dy[mask]`
        from a converged `masked_sdd_method`
        (recompute `mask` from its final `mu` / `sigma`).

    Returns
    -------
    list of dict
        Sorted best-first (lowest AICc); each has keys 'model', 'params',
        'aicc', 'fit_func'.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    names = model_names if model_names is not None else list(BACKGROUND_MODELS.keys())

    results = []
    for name in names:
        try:
            popt, fit_func = fit_background_trend(t, y, name)
            y_fit = fit_func(t)
            if not np.all(np.isfinite(y_fit)):
                continue
            aicc = _aicc(y, y_fit, len(popt))
            if not np.isfinite(aicc):
                continue
            results.append({"model": name, "params": popt, "aicc": aicc, "fit_func": fit_func})
        except (RuntimeError, ValueError, TypeError):
            continue

    results.sort(key=lambda r: r["aicc"])
    return results
