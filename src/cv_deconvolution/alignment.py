# -*- coding: utf-8 -*-
"""Turns raw experimental CV curves (E, i pairs at various scan rates, no
shared grid or timestamps guaranteed) into the aligned, common-potential-grid
matrices that `deconvolution.py` operates on. The original
supporting_information_script_cpe.py didn't need this step: its curves came
straight out of `simulation.simulate_cv`, already sharing one Nt/E grid and a
known time array for every scan rate.
"""

import numpy as np


def reconstruct_time_axis(E, scan_rate_V_s):
    """Reconstructs a physical time axis from a potential array and a
    (constant) scan rate: t[k] = cumulative sum of |dE|/v. Real .ocw-style
    exports store (E, i) but not timestamps, so this is required by any
    method (e.g. `sdd_method`) that differentiates with respect to time."""
    E = np.asarray(E, dtype=float)
    dt = np.abs(np.diff(E)) / scan_rate_V_s
    return np.concatenate(([0.0], np.cumsum(dt)))


def turning_point_index(E):
    """Index of a full CV cycle's turning point (the same rule
    `split_cv_branches` uses internally), exposed separately so other
    arrays computed over the *whole* cycle - e.g. a semi-derivative from
    `deconvolution.semi_calculus_native` - can be sliced at the identical
    point."""
    E = np.asarray(E, dtype=float)
    return int(np.argmax(E)) if E[1] > E[0] else int(np.argmin(E))


def split_cv_branches(E, i):
    """Splits one full CV cycle into its two monotonic-in-E branches (the
    outward sweep and the return sweep), at the turning point. Returns
    ((E_branch1, i_branch1), (E_branch2, i_branch2))."""
    E = np.asarray(E, dtype=float)
    i = np.asarray(i, dtype=float)
    turn = turning_point_index(E)
    return (E[:turn + 1], i[:turn + 1]), (E[turn:], i[turn:])


def find_zero_crossing_index(i):
    """Index of the first point where `i` reaches zero-or-positive - useful
    for discarding an initial negative-current segment (e.g. a memory
    effect carried over from the previous CV cycle) before analyzing a
    sweep. Returns 0 if `i` already starts non-negative, or if it never
    reaches zero-or-positive (nothing to trim either way)."""
    i = np.asarray(i, dtype=float)
    if i[0] >= 0:
        return 0
    positive_mask = i >= 0
    if not positive_mask.any():
        return 0
    return int(np.argmax(positive_mask))


def trim_to_zero_crossing(E, i):
    """Trims (E, i) to start at `find_zero_crossing_index(i)`. Returns
    (E_trimmed, i_trimmed, start_index)."""
    E = np.asarray(E, dtype=float)
    i = np.asarray(i, dtype=float)
    start = find_zero_crossing_index(i)
    return E[start:], i[start:], start


def interpolate_branch_onto(E_target, E_source, i_source):
    """Interpolates one CV branch (E_source, i_source), as returned by
    `split_cv_branches`, onto another branch's potential grid `E_target`."""
    E_source = np.asarray(E_source, dtype=float)
    i_source = np.asarray(i_source, dtype=float)
    order = np.argsort(E_source)
    return np.interp(E_target, E_source[order], i_source[order])


def subtract_blank_branch(E_branch, i_branch, E_blank_branch, i_blank_branch):
    """Background-corrects one CV branch by subtracting a blank/background
    scan's matching branch (e.g. a `Fondo` measurement), interpolated onto
    `E_branch`. Useful when a cycle starts away from zero current - often a
    leftover double-layer/background contribution from the previous cycle -
    since the blank scan carries the same background without the analyte's
    Faradaic signal."""
    return np.asarray(i_branch, dtype=float) - interpolate_branch_onto(
        E_branch, E_blank_branch, i_blank_branch
    )


def align_branches_to_common_grid(branches, n_points=None):
    """
    Aligns a dict of {key: (E_branch, i_branch)} arrays onto one common
    potential grid. This is `build_scan_rate_matrix`'s alignment step, but
    for branches that have *already* been extracted/processed (e.g.
    Fondo-corrected and/or zero-crossing-trimmed, so each may now start at
    a different potential and have a different length) - `build_scan_rate_matrix`
    can't be reused directly for that, since it expects a whole CV cycle
    per key and re-splits it into branches itself.

    Parameters
    ----------
    branches : dict[float, tuple[np.ndarray, np.ndarray]]
        Maps a key (e.g. scan rate in V/s) -> (E_branch, i_branch), each
        already a single monotonic-in-E branch.
    n_points : int, optional
        Points of the common grid; defaults to the shortest branch supplied.

    Returns
    -------
    E_common : np.ndarray, shape (n_points,)
    aligned : np.ndarray, shape (n_points, n_keys)
        Each branch's current, interpolated onto E_common; columns sorted
        by ascending key.
    keys : np.ndarray, shape (n_keys,)
        The sorted keys, matching the matrix columns.
    """
    keys = np.array(sorted(branches.keys()), dtype=float)
    sorted_branches = {}
    for k in keys:
        E_b = np.asarray(branches[k][0], dtype=float)
        i_b = np.asarray(branches[k][1], dtype=float)
        order = np.argsort(E_b)
        sorted_branches[k] = (E_b[order], i_b[order])

    e_lo = max(sorted_branches[k][0].min() for k in keys)
    e_hi = min(sorted_branches[k][0].max() for k in keys)
    if n_points is None:
        n_points = min(len(sorted_branches[k][0]) for k in keys)
    E_common = np.linspace(e_lo, e_hi, n_points)

    aligned = np.empty((n_points, len(keys)))
    for col, k in enumerate(keys):
        E_sorted, i_sorted = sorted_branches[k]
        aligned[:, col] = np.interp(E_common, E_sorted, i_sorted)
    return E_common, aligned, keys


def build_scan_rate_matrix(curves, branch=0, n_points=None):
    """
    Aligns CV curves recorded at different scan rates onto one common
    potential grid, ready for `dunn_method`/`sdd_method`.

    Parameters
    ----------
    curves : dict[float, tuple[np.ndarray, np.ndarray]] or dict[float, object]
        Maps scan rate (V/s) -> either an (E, i) tuple or an object exposing
        `.E`/`.i` attributes (e.g. the `CVCurve` from cv_loader.ipynb - just
        rebuild the dict with V/s keys:
        `{c.scan_rate_mV_s / 1000: c for c in ...}`), each holding one full
        CV cycle.
    branch : int
        0 = outward sweep, 1 = return sweep (see `split_cv_branches`).
    n_points : int, optional
        Points of the common grid; defaults to the shortest branch supplied.

    Returns
    -------
    E_common : np.ndarray, shape (n_points,)
    totals : np.ndarray, shape (n_points, n_scan_rates)
        Measured current, interpolated onto E_common; columns sorted by
        ascending scan rate.
    times : np.ndarray, shape (n_points, n_scan_rates)
        Reconstructed time axis (see `reconstruct_time_axis`), aligned to
        the same rows/columns as `totals`.
    vs : np.ndarray, shape (n_scan_rates,)
        Scan rates (V/s), sorted ascending, matching the matrix columns.
    """
    vs = np.array(sorted(curves.keys()), dtype=float)

    branches = {}
    for v in vs:
        c = curves[v]
        E, i = (c.E, c.i) if hasattr(c, "E") else c
        b0, b1 = split_cv_branches(np.asarray(E, dtype=float), np.asarray(i, dtype=float))
        branches[v] = b0 if branch == 0 else b1

    e_lo = max(branches[v][0].min() for v in vs)
    e_hi = min(branches[v][0].max() for v in vs)
    if n_points is None:
        n_points = min(len(branches[v][0]) for v in vs)
    E_common = np.linspace(e_lo, e_hi, n_points)

    totals = np.empty((n_points, len(vs)))
    times = np.empty((n_points, len(vs)))
    for col, v in enumerate(vs):
        E_b, i_b = branches[v]
        order = np.argsort(E_b)
        E_sorted, i_sorted = E_b[order], i_b[order]
        totals[:, col] = np.interp(E_common, E_sorted, i_sorted)
        t_b = reconstruct_time_axis(E_b, v)
        times[:, col] = np.interp(E_common, E_sorted, t_b[order])

    return E_common, totals, times, vs
