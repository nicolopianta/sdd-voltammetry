# Semi-Derivative Decomposition of voltammetry

[![CI](https://github.com/<your-github-username>/sdd-voltammetry/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-github-username>/sdd-voltammetry/actions)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Code and data for

> N. Pianta, M. Spotti, F. La Mantia, G. Di Liberto, R. Ruffo,
> *Separation of Faradaic and Capacitive Currents in Voltammetry through
> Semi-Derivative Decomposition* (2026). DOI: `<article DOI>`

**Semi-Derivative Decomposition (SDD)** separates the faradaic and capacitive
contributions to a linear-sweep voltammogram by fitting the derivative of the
current's semi-integral with a logistic-derivative peak, then reconstructing
the faradaic current by semi-integrating that fit. This repository contains
the `cv_deconvolution` Python package that implements SDD and the
scan-rate-based (Dunn) method it is benchmarked against, together with
runnable example scripts that reproduce every figure of the paper on
simulated data, and the experimental ferri-/ferrocyanide dataset.

## Install

```bash
git clone https://github.com/<your-github-username>/sdd-voltammetry.git
cd sdd-voltammetry
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .            # loose deps; or  pip install -r requirements.txt  for exact versions
```

Python >= 3.10; the only dependencies are numpy, scipy, lmfit and matplotlib.

## Run the examples

```bash
python examples/01_semi_transforms.py            # current, semi-integral, semi-derivative     (Fig. S2)
python examples/02_sdd_vs_dunn_helmholtz.py      # SDD vs Dunn, constant capacitance           (Fig. 2)
python examples/03_sdd_vs_dunn_stern.py          # SDD vs Dunn, potential-dependent capacitance (Fig. 4)
python examples/04_cpe_background_estimation.py  # SDD with background estimation for a CPE     (Fig. 5, Appendix A.4)
python examples/05_experimental_ferricyanide.py  # SDD vs Dunn on measured data + Randles-Sevcik (Fig. 6, S8)
python examples/06_ape_vs_lambda.py              # maximum APE vs kinetic number Lambda         (Fig. 7)
```

Every script prints its numeric result and (unless `--no-show`) opens a
figure. The simulated examples accept `--quick` for a coarser, faster run.
See [`examples/README.md`](examples/README.md) for expected runtimes and
outputs.

## Minimal usage

```python
import numpy as np
from cv_deconvolution import simulation, sdd_method, helmholtz_capacitive_current

# one simulated linear-sweep voltammogram + a constant capacitive current
E, i_faradaic, _, _, t = simulation.simulate_cv(k0=1e-2, v=0.05, Nt=2000)
half = len(t) // 2
E, i_faradaic, t = E[:half], i_faradaic[:half], t[:half]
i_total = i_faradaic + helmholtz_capacitive_current(t, E, C=5e-5)

capacitive, faradaic, fit = sdd_method(t, i_total)     # <- the SDD reconstruction
```

For measured data recorded at several scan rates:

```python
from cv_deconvolution import build_scan_rate_matrix, dunn_method, masked_sdd_method

curves = {v_in_V_per_s: (E, i), ...}                   # your measured CV cycles
E, totals, times, vs = build_scan_rate_matrix(curves, branch=0)     # forward branch
dunn_coeffs = dunn_method(totals, vs, smooth_window=15)
cap, far, fit, *_ = masked_sdd_method(times[:, -1], totals[:, -1], smooth_window=15)
```

## Repository layout

```
src/cv_deconvolution/   the package: SDD + Dunn, fractional calculus, forward CV simulator
examples/               one script per figure of the paper (see examples/README.md)
data/experimental/      the measured GC / K3Fe(CN)6 voltammograms (see data/experimental/README.md)
tests/                  fast smoke test used by CI
```

## Data availability / how to cite

Please cite **both** the article and the archived software release (see
`CITATION.cff`; GitHub shows a "Cite this repository" button). The tagged
releases of this repository are archived on Zenodo with a permanent DOI.

## Licence

Code: MIT (`LICENSE`). Experimental data under `data/`: CC-BY-4.0
(`data/experimental/README.md`).
