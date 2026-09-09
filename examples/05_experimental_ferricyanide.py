# -*- coding: utf-8 -*-
"""Example 5 - SDD vs Dunn on real data: ferri-/ferrocyanide at a
glassy-carbon electrode (paper: Section 3.4 / Fig. 6 / Fig. S8).

Loads the measured voltammograms in data/experimental/, aligns them across
scan rate, and separates the faradaic and capacitive currents of the
forward (oxidative) scan with Dunn's method and with SDD (iterative
background estimation).  It then estimates the bulk analyte concentration
from each reconstruction via the Randles-Sevcik relation and compares with
the nominal 3 mM.

Runtime: ~20 s.
"""
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from _ivium import load_curves
from cv_deconvolution import (build_scan_rate_matrix, dunn_method, dunn_components,
                              masked_sdd_method, semiderivative_fit_plateau,
                              randles_sevcik_concentration,
                              randles_sevcik_concentration_from_semiderivative)

AREA_CM2 = 0.071
D_CM2_S = 6.7e-6
N_E = 1
C_NOMINAL_mM = 3.0
SMOOTH = 15                     # Savitzky-Golay window for the noisy real data


def main(show=True, quick=False):
    analyte = load_curves("analyte", pattern="*3agg_05-03*.ocw", rate_range=(0.02, 0.5))
    blank = load_curves("blank")
    print("analyte scan rates (mV/s):", sorted(round(v * 1000) for v in analyte))

    E, totals, times, vs = build_scan_rate_matrix(analyte, branch=0)   # forward branch
    j = -1                                                             # fastest scan rate
    v = vs[j]

    # --- Dunn ---
    coeffs = dunn_method(totals, vs, smooth_window=SMOOTH)
    dunn_i_c, dunn_i_f = dunn_components(coeffs, vs, j)
    dunn_ip = dunn_i_f.max()
    c_dunn = randles_sevcik_concentration(dunn_ip, v, n=N_E, area_cm2=AREA_CM2,
                                          diffusion_coeff_cm2_s=D_CM2_S) * 1e6  # mol/cm3 -> mM

    # --- SDD (iterative background estimation) ---
    sdd_bg, sdd_far, sdd_fit, *_ = masked_sdd_method(times[:, j], totals[:, j],
                                                     exclusion_sigmas=4.0, n_iterations=5,
                                                     smooth_window=SMOOTH)
    L = semiderivative_fit_plateau(sdd_fit)
    c_sdd = randles_sevcik_concentration_from_semiderivative(
        L, D_CM2_S, n=N_E, area_cm2=AREA_CM2) * 1e6

    print(f"\nscan rate analysed        : {v * 1000:.0f} mV/s")
    print(f"nominal concentration     : {C_NOMINAL_mM:.1f} mM")
    print(f"Dunn  (Randles-Sevcik i_p): {c_dunn:.2f} mM")
    print(f"SDD   (semi-integral L)   : {c_sdd:.2f} mM   <- closer to nominal")

    if show:
        import matplotlib.pyplot as plt
        blank_v = min(blank, key=lambda x: abs(x - v))
        Eb, ib = blank[blank_v]
        from cv_deconvolution import split_cv_branches, interpolate_branch_onto
        (Ebf, Ibf), _ = split_cv_branches(Eb, ib)
        bl = interpolate_branch_onto(E, Ebf, Ibf)

        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        for a, (name, i_c, i_f) in zip(ax, [("SDD", sdd_bg, sdd_far),
                                            ("Dunn", dunn_i_c, dunn_i_f)]):
            a.fill_between(E, 0, i_c / AREA_CM2 * 1e3, color="orange", alpha=.7)
            a.fill_between(E, i_c / AREA_CM2 * 1e3, (i_c + i_f) / AREA_CM2 * 1e3,
                           color="slateblue", alpha=.7)
            a.plot(E, totals[:, j] / AREA_CM2 * 1e3, "k", lw=1.3, label="measured total")
            a.plot(E, bl / AREA_CM2 * 1e3, "r--", lw=1, label="measured blank")
            a.set_ylabel("i / mA cm$^{-2}$"); a.set_title(name); a.legend(fontsize=8)
        ax[1].set_xlabel("E vs. Hg$_2$Cl$_2$ / V")
        fig.suptitle(f"GC / K3Fe(CN)6, forward scan, {v*1000:.0f} mV/s")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-show", dest="show", action="store_false")
    a = p.parse_args()
    main(show=a.show, quick=a.quick)
