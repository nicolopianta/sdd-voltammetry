# -*- coding: utf-8 -*-
"""Example 2 - SDD vs Dunn on a constant (Helmholtz) capacitive background
(paper: Fig. 2).

For three kinetic regimes (Lambda = 20, 2, 0.2) we simulate the faradaic
voltammogram at several scan rates, add a constant capacitive current
C*v, then separate the two contributions with

    * Dunn's method   (fit i = a*v + b*sqrt(v) at every potential), and
    * the SDD method  (fit the logistic-derivative of the semi-integral),

and report the maximum APE of each reconstruction against the known
faradaic ground truth.

Runtime: ~30 s (a few seconds with --quick).
"""
import argparse
import numpy as np

from _common import k0_from_lambda, apef, simulate_forward_lsv
from cv_deconvolution import dunn_method, dunn_components, sdd_method

C_DL = 5e-5           # constant double-layer capacitance
D = 1e-9
SCAN_RATES = np.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])     # V/s (columns for Dunn)
LAMBDAS = [20.0, 2.0, 0.2]


def run(lam, Nt):
    k0 = k0_from_lambda(lam, D)
    totals, times, i_far_true = [], [], None
    for v in SCAN_RATES:
        E, i_far, t = simulate_forward_lsv(v, D=D, k0=k0, Nt=Nt)
        totals.append(i_far + C_DL * v)          # + constant capacitive current
        times.append(t)
        if v == SCAN_RATES[-1]:
            i_far_true = i_far
    totals = np.column_stack(totals)
    times = np.column_stack(times)
    i_peak = i_far_true.max()

    # Dunn (all scan rates) -> components at the fastest rate
    coeffs = dunn_method(totals, SCAN_RATES)
    _, dunn_far = dunn_components(coeffs, SCAN_RATES, j=-1)

    # SDD on the fastest rate
    _, sdd_far, _ = sdd_method(times[:, -1], totals[:, -1])

    return dict(lam=lam,
                dunn=apef(dunn_far, i_far_true, i_peak).max(),
                sdd=apef(sdd_far, i_far_true, i_peak).max(),
                E=E, true=i_far_true, dunn_far=dunn_far, sdd_far=sdd_far)


def main(show=True, quick=False):
    Nt = 800 if quick else 2000
    print(f"{'Lambda':>8} | {'SDD max APE':>12} | {'Dunn max APE':>13}")
    print("-" * 38)
    results = []
    for lam in LAMBDAS:
        r = run(lam, Nt)
        results.append(r)
        print(f"{r['lam']:>8.2g} | {r['sdd']:>11.1f}% | {r['dunn']:>12.1f}%")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(results), figsize=(4 * len(results), 3.2), sharey=True)
        for a, r in zip(np.atleast_1d(ax), results):
            a.plot(r["E"], r["true"] * 1e6, "k", label="true faradaic")
            a.plot(r["E"], r["sdd_far"] * 1e6, label="SDD")
            a.plot(r["E"], r["dunn_far"] * 1e6, label="Dunn")
            a.set_title(f"Lambda = {r['lam']:g}"); a.set_xlabel("E / V")
        np.atleast_1d(ax)[0].set_ylabel("faradaic current / uA")
        np.atleast_1d(ax)[0].legend(fontsize=8)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-show", dest="show", action="store_false")
    a = p.parse_args()
    main(show=a.show, quick=a.quick)
