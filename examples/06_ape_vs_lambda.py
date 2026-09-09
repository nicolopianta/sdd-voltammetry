# -*- coding: utf-8 -*-
"""Example 6 - maximum APE as a function of the kinetic number Lambda
(paper: Fig. 7).

Sweeps Lambda from the strongly mixed-control regime to the fully
reversible one, with a constant (Helmholtz) capacitive background, and
plots the maximum APE of the faradaic reconstruction for SDD and for
Dunn's method (reconstruction at 100 mV/s).  It reproduces the key
qualitative result of the paper: the SDD error stays almost flat, while
Dunn's error grows sharply as charge transfer stops being fast.

Runtime: ~1-2 min (a few seconds with --quick).
"""
import argparse
import numpy as np

from _common import apef, k0_from_lambda, simulate_forward_lsv
from cv_deconvolution import dunn_method, dunn_components, sdd_method

C_DL = 5e-5
D = 1e-9
SCAN_RATES = np.array([1e-4, 1e-3, 5e-3, 1e-2, 1e-1])   # V/s, as in the paper
RECON_INDEX = len(SCAN_RATES) - 1                       # 100 mV/s


def max_ape_at(lam, Nt):
    k0 = k0_from_lambda(lam, D)
    totals, times, true_far = [], [], None
    for k, v in enumerate(SCAN_RATES):
        E, i_far, t = simulate_forward_lsv(v, D=D, k0=k0, Nt=Nt)
        totals.append(i_far + C_DL * v)
        times.append(t)
        if k == RECON_INDEX:
            true_far = i_far
    totals = np.column_stack(totals)
    times = np.column_stack(times)
    ip = true_far.max()

    coeffs = dunn_method(totals, SCAN_RATES)
    _, dunn_far = dunn_components(coeffs, SCAN_RATES, j=RECON_INDEX)
    _, sdd_far, _ = sdd_method(times[:, RECON_INDEX], totals[:, RECON_INDEX])
    return apef(sdd_far, true_far, ip).max(), apef(dunn_far, true_far, ip).max()


def main(show=True, quick=False):
    n = 9 if quick else 21
    Nt = 900 if quick else 1800
    lams = np.logspace(-1, 3, n)

    sdd, dunn = [], []
    print(f"{'Lambda':>10} | {'SDD APE':>9} | {'Dunn APE':>9}")
    print("-" * 34)
    for lam in lams:
        s, d = max_ape_at(lam, Nt)
        sdd.append(s); dunn.append(d)
        print(f"{lam:>10.2g} | {s:>8.1f}% | {d:>8.1f}%")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.semilogx(lams, sdd, "o-", label="SDD")
        ax.semilogx(lams, dunn, "s-", label="Dunn's method")
        ax.set_xlabel(r"$\Lambda$"); ax.set_ylabel("maximum APE / %")
        ax.set_title("Reconstruction at 100 mV/s, Helmholtz background")
        ax.legend()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-show", dest="show", action="store_false")
    a = p.parse_args()
    main(show=a.show, quick=a.quick)
