# Examples

One script per figure of the paper. Each is self-contained, prints its
numeric result, and opens a figure unless run with `--no-show`. The
simulated ones take `--quick` for a coarse, fast run.

| script | paper | what it shows | runtime (full / --quick) |
|---|---|---|---|
| `01_semi_transforms.py` | Fig. S2 | the current, its semi-integral (a sigmoid) and its semi-derivative (a peak) for one simulated voltammogram + constant capacitance | ~3 s / ~1 s |
| `02_sdd_vs_dunn_helmholtz.py` | Fig. 2 | SDD vs Dunn on a constant (Helmholtz) capacitive background, for three kinetic regimes (Lambda = 20, 2, 0.2) | ~30 s / ~3 s |
| `03_sdd_vs_dunn_stern.py` | Fig. 4 | SDD vs Dunn when the capacitance depends on potential (Stern model), with the minimum swept across the window (E_pzc = -0.5, 0.2, 0.5 V) | ~30 s / ~3 s |
| `04_cpe_background_estimation.py` | Fig. 5, Appendix A.4 | the iterative background-estimation step for a memory-carrying (constant-phase-element) capacitive current; also recovers the CPE exponent phi | ~20 s / ~4 s |
| `05_experimental_ferricyanide.py` | Fig. 6, Fig. S8 | SDD vs Dunn on the measured GC / K3Fe(CN)6 data, plus a bulk-concentration estimate from each method via Randles-Sevcik | ~20 s |
| `06_ape_vs_lambda.py` | Fig. 7 | maximum APE vs the kinetic number Lambda: SDD stays flat, Dunn degrades sharply under mixed control | ~2 min / ~6 s |

### Expected output (approximate)

The exact numbers depend on the simulation grid; with the default settings:

* **02** - SDD max APE stays a few percent across all three Lambda; Dunn grows
  from ~4 % (Lambda = 20) to ~50 % (Lambda = 0.2).
* **03** - SDD max APE ~5 % away from the peak, up to ~12 % when the
  capacitance minimum sits under the faradaic peak (E_pzc = 0.2 V); Dunn is
  degraded by the peak shift of the mixed-control faradaic process.
* **04** - plain SDD fails badly (APE > 100 %); SDD with background estimation
  reaches ~3 % APE and recovers phi to within ~1 %.
* **05** - Dunn gives c ~ 2 mM, SDD gives c closer to the nominal 3 mM.
* **06** - SDD max APE ~3-6 % for every Lambda; Dunn ~60 % at Lambda = 0.1,
  falling below 1 % once charge transfer is fast.

### Shared helpers (not examples)

* `_common.py` - simulation wrapper, the Lambda -> k0 mapping, the APE metric.
* `_ivium.py` - reader for the Ivium `.ocw` / `.icw` files under `data/`.

### The 2-D APE map

The full 2-D map over the (Lambda, Pi) plane of Fig. 3 / Figs. S4-S7 is
computationally heavy (a CV simulation at several scan rates plus two fits
per grid point, over a 40x40 or 50x50 grid). It is not included here as an
example; `06_ape_vs_lambda.py` is the 1-D slice of it.
