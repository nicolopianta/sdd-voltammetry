# Experimental dataset

Cyclic voltammograms of the ferri-/ferrocyanide couple at a glassy-carbon
electrode, used for the experimental validation in the paper (Section 3.4,
Fig. 6, Fig. S8).

## Measurement

| | |
|---|---|
| Analyte | 3 mM potassium hexacyanoferrate(III), K3Fe(CN)6 |
| Supporting electrolyte | 0.1 M KCl (aqueous) |
| Working electrode | glassy-carbon disc, geometric area 0.071 cm2 |
| Reference | saturated calomel electrode (Hg2Cl2) |
| Potential window | -0.3 V to +0.5 V vs Hg2Cl2 |
| Scan rates | 10, 20, 50, 100, 200, 500 mV/s |
| Instrument | Ivium potentiostat |

`analyte/` holds the voltammograms of the analyte solution (`3agg` = third
successive addition of the stock solution; `05-03` = the +0.5 / -0.3 V
window). `blank/` holds the matching supporting-electrolyte-only
voltammograms at each scan rate. The blank is **not** subtracted in the
analysis; it is shown only as a reference for the reconstructed capacitive
current.

## File format (Ivium)

Each measurement is a triplet:

* `*.ocw` - the curve: 2 header lines, then whitespace-separated
  `E / V   i / A` pairs (one full CV cycle).
* `*.icw` - the parameter list; line index 65 (0-based) is the scan rate in
  V/s.
* `*.ici` - the plot-settings file (not used).

`examples/_ivium.py` reads these:

```python
from _ivium import load_curves
analyte = load_curves("analyte", pattern="*3agg_05-03*.ocw", rate_range=(0.02, 0.5))
# -> {0.02: (E, i), 0.05: (E, i), 0.1: (E, i), 0.2: (E, i), 0.5: (E, i)}
```

## Licence

This dataset is released under the Creative Commons Attribution 4.0
International licence (CC-BY-4.0): https://creativecommons.org/licenses/by/4.0/
If you use it, please cite the paper (see the top-level `CITATION.cff`).
