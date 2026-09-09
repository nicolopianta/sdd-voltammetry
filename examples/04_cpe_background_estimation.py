# -*- coding: utf-8 -*-
"""Example 4 - SDD with background estimation for a memory-carrying (CPE)
capacitive current (paper: Section 3.3 / Fig. 5 / Appendix A.4).

A constant-phase element is *not* annihilated by the differentiation step:
its semi-derivative is a slowly growing power law ~ t^(1/2 - phi), which
biases a single whole-domain logistic fit.  This example implements the
iterative background-estimation algorithm of Appendix A.4:

    1. preliminary logistic-derivative fit  -> peak centre t0, width sigma
    2. mask |t - t0| <= 4 sigma
    3. fit the CPE semi-derivative shape (Q, phi) to the remaining points
    4. subtract that background from the semi-derivative
    5. refit the peak; repeat 2-5

and compares it with the plain (whole-domain) SDD fit.

Runtime: ~20 s (a few seconds with --quick).
"""
import argparse
import numpy as np
import lmfit
from scipy.optimize import curve_fit

from _common import apef, simulate_forward_lsv
from cv_deconvolution import (cpe_capacitive_current, semi_integral, semi_derivative,
                              model_logistic_derivative, sdd_method, upsample_xy)

try:
    TRAPZ = np.trapezoid
except AttributeError:
    TRAPZ = np.trapz

D = 1e-9
K0 = 1e-3
V_DISPLAY = 0.5                 # V/s
C_REF = 5e-5                    # capacitor whose cumulative charge we match
PHIS = [0.75, 0.85, 0.93]


def sdd_cpe_background(t, E, i_total, excl_sigmas=4.0, n_iter=5, upf=1000):
    """SDD with the CPE background-estimation step of Appendix A.4.
    Returns (faradaic, capacitive, (Q, phi))."""
    x = np.linspace(t[0], t[-1], len(t) * upf)
    y = np.interp(x, t, i_total)
    dm = semi_derivative(x, y)[::upf]
    ys = semi_integral(x, y)[::upf]

    model = lmfit.Model(model_logistic_derivative)
    q = dm.size // 4
    t0 = t[dm[q:].argmax() + q]
    sigma = abs(ys[q:].max() / (4 * dm[q:].max()))
    params = model.make_params(A=ys.max(), mu=t0, sigma=sigma)
    params["sigma"].min = 1e-9
    params["sigma"].max = t[-1] - t[0]

    def cpe_semideriv(tt, Q, phi):
        ic = cpe_capacitive_current(t, E, Q, phi)
        xu, yu = upsample_xy(t, ic, 50)
        return np.interp(tt, t, semi_derivative(xu, yu)[::50])

    fit, q_phi = None, (np.nan, np.nan)
    for _ in range(n_iter):
        mask = np.abs(t - t0) > excl_sigmas * sigma
        if mask.sum() < 6:
            break
        probe = cpe_semideriv(t[mask], 1.0, 0.7)
        scale = float(np.mean(np.abs(probe))) or 1.0
        Q_guess = max(float(np.mean(np.abs(dm[mask]))) / scale, 1e-12)
        try:
            (Q, phi), _ = curve_fit(cpe_semideriv, t[mask], dm[mask],
                                    p0=[Q_guess, 0.8], bounds=([0.0, 0.05], [np.inf, 1.0]),
                                    maxfev=20000)
            dm_bg = cpe_semideriv(t, Q, phi)
            q_phi = (Q, phi)
        except Exception:
            dm_bg = np.interp(t, t[mask], dm[mask])
        fit = model.fit(dm - dm_bg, params, x=t, method="leastsq")
        params = fit.params
        t0 = params["mu"].value
        sigma = abs(params["sigma"].value)

    if fit is None:
        fit = model.fit(dm, params, x=t, method="leastsq")

    xu, dyf = upsample_xy(t, fit.best_fit, upf)
    far = semi_integral(xu, dyf)[::upf]
    return far, i_total - far, q_phi


def run(phi, Nt):
    E, i_far, t = simulate_forward_lsv(V_DISPLAY, D=D, k0=K0, Nt=Nt)
    i_peak = i_far.max()

    i_cpe_unit = cpe_capacitive_current(t, E, Y0=1.0, gamma=phi)
    q_ref = TRAPZ(np.abs(C_REF * np.gradient(E, t)), t)
    Y0 = q_ref / TRAPZ(np.abs(i_cpe_unit), t)
    i_cpe = Y0 * i_cpe_unit
    i_tot = i_far + i_cpe

    _, plain_far, _ = sdd_method(t, i_tot)
    bg_far, _, (Q_fit, phi_fit) = sdd_cpe_background(t, E, i_tot)

    return dict(phi=phi, E=E, true=i_far,
                plain=apef(plain_far, i_far, i_peak).max(),
                bg=apef(bg_far, i_far, i_peak).max(),
                phi_fit=phi_fit, plain_far=plain_far, bg_far=bg_far,
                i_cpe_over_peak=np.abs(i_cpe).max() / i_peak)


def main(show=True, quick=False):
    Nt = 1200 if quick else 2400
    print(f"{'phi':>5} | {'plain SDD APE':>14} | {'SDD+background APE':>19} | {'phi recovered':>14}")
    print("-" * 62)
    results = []
    for phi in PHIS:
        r = run(phi, Nt)
        results.append(r)
        print(f"{r['phi']:>5.2f} | {r['plain']:>13.1f}% | {r['bg']:>18.1f}% | {r['phi_fit']:>14.3f}")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(results), figsize=(4 * len(results), 3.2), sharey=True)
        for a, r in zip(np.atleast_1d(ax), results):
            a.plot(r["E"], r["true"] * 1e6, "k", label="true faradaic")
            a.plot(r["E"], r["bg_far"] * 1e6, label="SDD + background")
            a.plot(r["E"], r["plain_far"] * 1e6, label="plain SDD")
            a.set_title(f"phi = {r['phi']}"); a.set_xlabel("E / V")
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
