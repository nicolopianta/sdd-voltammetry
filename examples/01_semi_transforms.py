# -*- coding: utf-8 -*-
"""Example 1 - the semi-integral and semi-derivative of a voltammogram
(paper: Section 2 / Fig. S2).

Simulate one reversible linear-sweep voltammogram, add a constant Helmholtz
capacitive current of comparable magnitude, and show:
    (a) the total current vs time,
    (b) its semi-integral m(t)   -> a sigmoid, plateau = L,
    (c) its semi-derivative      -> a logistic-derivative peak.

Runtime: a few seconds.
"""
import argparse
import numpy as np

np.seterr(divide="ignore", invalid="ignore")   # harmless 1/sqrt(0) in the RL kernel

from cv_deconvolution import (simulation, semi_integral, semi_derivative,
                              helmholtz_capacitive_current)

C_DL = 5e-5          # constant double-layer capacitance (F, per unit area A=1)


def main(show=True, quick=False):
    Nt = 800 if quick else 2000
    E, i_far, _, _, t = simulation.simulate_cv(
        A=1, D=1e-9, C_bulk=5e-6, n=1, alpha=0.5, k0=1e-2, E0=0.0,
        v=0.05, E_start=-1.0, E_switch=1.0, Nt=Nt)

    half = len(t) // 2                       # forward (linear-sweep) branch
    E, i_far, t = E[:half], i_far[:half], t[:half]

    i_cap = helmholtz_capacitive_current(t, E, C_DL)      # ~ constant = C*v
    i_tot = i_far + i_cap

    m = semi_integral(t, i_tot)
    dm = semi_derivative(t, i_tot)

    print(f"peak faradaic current      : {i_far.max() * 1e6:8.2f} uA")
    print(f"capacitive current (C*v)   : {i_cap.mean() * 1e6:8.2f} uA")
    print(f"semi-integral plateau L    : {m[-1] * 1e6:8.2f} uA s^1/2")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
        ax[0].plot(t, i_tot * 1e6, "k"); ax[0].plot(t, i_cap * 1e6, "0.6", ls="--")
        ax[0].set_ylabel("total current / uA")
        ax[1].plot(t, m * 1e6); ax[1].set_ylabel("semi-integral m(t)")
        ax[2].plot(t, dm * 1e6); ax[2].set_ylabel("semi-derivative")
        ax[2].set_xlabel("time / s")
        fig.suptitle("Example 1: current, semi-integral, semi-derivative")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="coarser time grid")
    p.add_argument("--no-show", dest="show", action="store_false", help="don't open a figure")
    a = p.parse_args()
    main(show=a.show, quick=a.quick)
