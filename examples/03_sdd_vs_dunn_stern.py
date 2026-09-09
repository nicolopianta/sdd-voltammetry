# -*- coding: utf-8 -*-
"""Example 3 - SDD vs Dunn on a potential-dependent (Stern) capacitive
background (paper: Fig. 4 / Appendix A.2).

Same experiment as example 2, but the capacitive current is now
i_c(E) = C_Stern(E) * v, with C_Stern the series combination of a fixed
Helmholtz capacitance and a Gouy-Chapman capacitance that has a sharp
minimum at the potential of zero charge E_pzc.  We sweep E_pzc across the
window (-0.5, 0.2, 0.5 V) - the 0.2 V case puts the capacitance minimum
right under the faradaic peak, which is the hardest case for SDD.

Runtime: ~30 s (a few seconds with --quick).
"""
import argparse
import numpy as np

from _common import apef, simulate_forward_lsv, k0_from_lambda
from cv_deconvolution.constants import T
from cv_deconvolution import dunn_method, dunn_components, sdd_method

D = 1e-9
K0 = k0_from_lambda(2.0, 1e-9)   # mixed control (Lambda = 2), as in Fig. 4
SCAN_RATES = np.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
E_PZC_VALUES = [-0.5, 0.2, 0.5]


def stern_capacitance(E, C_helmholtz=5e-5, eps_r=35.0, c_salt_mol_L=0.1, E_pzc=0.2, n=1):
    """Series Helmholtz + Gouy-Chapman capacitance per unit area (F/cm^2),
    following Appendix A.2 of the paper (cosh form, minimum at E_pzc)."""
    eps0 = 8.85419e-12          # C^2 N^-1 m^-2
    kB = 1.380649e-23
    e = 1.60217663e-19
    n_inf = 1e3 * 6.022e23 * c_salt_mol_L                 # ions / m^3
    kappa = np.sqrt(2 * n**2 * e**2 * n_inf / (eps_r * eps0 * kB * T))
    C_gc = eps_r * eps0 * kappa * np.cosh(n * e * (E - E_pzc) / (2 * kB * T)) * 1e-4
    return C_helmholtz * C_gc / (C_helmholtz + C_gc)


def run(E_pzc, Nt):
    totals, times, i_far_true = [], [], None
    for v in SCAN_RATES:
        E, i_far, t = simulate_forward_lsv(v, D=D, k0=K0, Nt=Nt)
        i_cap = stern_capacitance(E, E_pzc=E_pzc) * v
        totals.append(i_far + i_cap)
        times.append(t)
        if v == SCAN_RATES[-1]:
            i_far_true = i_far
    totals = np.column_stack(totals)
    times = np.column_stack(times)
    i_peak = i_far_true.max()

    coeffs = dunn_method(totals, SCAN_RATES)
    _, dunn_far = dunn_components(coeffs, SCAN_RATES, j=-1)
    _, sdd_far, _ = sdd_method(times[:, -1], totals[:, -1])

    return dict(E_pzc=E_pzc, E=E, true=i_far_true,
                sdd=apef(sdd_far, i_far_true, i_peak).max(),
                dunn=apef(dunn_far, i_far_true, i_peak).max(),
                sdd_far=sdd_far, dunn_far=dunn_far)


def main(show=True, quick=False):
    Nt = 800 if quick else 2000
    print(f"{'E_pzc / V':>10} | {'SDD max APE':>12} | {'Dunn max APE':>13}")
    print("-" * 40)
    results = []
    for e in E_PZC_VALUES:
        r = run(e, Nt)
        results.append(r)
        print(f"{r['E_pzc']:>10.2f} | {r['sdd']:>11.1f}% | {r['dunn']:>12.1f}%")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(results), figsize=(4 * len(results), 3.2), sharey=True)
        for a, r in zip(np.atleast_1d(ax), results):
            a.plot(r["E"], r["true"] * 1e6, "k", label="true faradaic")
            a.plot(r["E"], r["sdd_far"] * 1e6, label="SDD")
            a.plot(r["E"], r["dunn_far"] * 1e6, label="Dunn")
            a.set_title(f"E_pzc = {r['E_pzc']:g} V"); a.set_xlabel("E / V")
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
