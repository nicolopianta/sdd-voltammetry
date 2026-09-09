# -*- coding: utf-8 -*-
"""Non-parametric background/baseline estimation, as an alternative to the
parametric SDD peak fit in `deconvolution.py` - most useful when the
measured signal has real background curvature (e.g. a non-zero "shoulder"
away from the peak) that a single symmetric peak model can't represent,
which biases the SDD fit toward a peak that's too wide and too short (see
the SDD section's discussion).
"""

import numpy as np

from .smoothing import smooth_signal


def snip_background(y, iterations, decreasing=True):
    """
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
    background estimate, following Ryan, Clayton, Griffin, Sie & Cousens
    (1988). Repeatedly clips each point down to the average of its
    neighbors at growing distance m, so any feature narrower than roughly
    `iterations` samples gets flattened toward its surroundings while
    slower background variation survives untouched - no peak shape (or
    even a single peak) is assumed.

    Parameters
    ----------
    y : np.ndarray
        Input signal (e.g. current vs potential, or a semi-derivative).
    iterations : int
        Largest clipping half-width, in samples. Must stay below roughly
        the true peak's half-width in samples, or the peak itself starts
        getting clipped away as if it were background.
    decreasing : bool
        Sweep window half-widths m = iterations..1 (the standard, more
        stable SNIP order) if True, or m = 1..iterations if False.

    Returns
    -------
    np.ndarray
        Estimated background, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    z = y.copy()
    widths = range(iterations, 0, -1) if decreasing else range(1, iterations + 1)
    idx = np.arange(n)
    for m in widths:
        left = z[np.clip(idx - m, 0, n - 1)]
        right = z[np.clip(idx + m, 0, n - 1)]
        z = np.minimum(z, 0.5 * (left + right))
    return z


def snip_decompose(y, iterations, smooth_window=None, smooth_polyorder=3, decreasing=True):
    """Splits `y` into a SNIP background and a `y - background` remainder
    (the Faradaic-like component). Pass `smooth_window` to Savitzky-Golay
    smooth `y` first (see `smoothing.smooth_signal`) - SNIP's min-clipping
    otherwise tracks point-to-point measurement noise straight into the
    background estimate.

    Returns
    -------
    background, remainder : np.ndarray
    """
    y = np.asarray(y, dtype=float)
    y_for_snip = y if smooth_window is None else smooth_signal(
        y, window_length=smooth_window, polyorder=smooth_polyorder
    )
    background = snip_background(y_for_snip, iterations, decreasing=decreasing)
    remainder = y - background
    return background, remainder
