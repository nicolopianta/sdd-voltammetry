# -*- coding: utf-8 -*-
"""Minimal reader for the Ivium .ocw / .icw text files under data/experimental/.

.ocw  -> two-column table  "E / V   i / A"  (2 header lines, then the curve)
.icw  -> plain-text parameter list; line index 65 is the scan rate in V/s
"""
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "experimental"


def _lines(path):
    return Path(path).read_text(encoding="latin-1").splitlines()


def scan_rate_V_s(icw_path):
    """Scan rate (V/s) recorded in an Ivium .icw parameter file."""
    return float(_lines(icw_path)[65].strip())


def read_curve(ocw_path):
    """Return (E, i) arrays for one CV cycle from an Ivium .ocw file."""
    E, i = [], []
    for ln in _lines(ocw_path)[2:]:
        parts = ln.split()
        if len(parts) == 2:
            E.append(float(parts[0]))
            i.append(float(parts[1]))
    return np.array(E), np.array(i)


def load_curves(subdir, pattern="*.ocw", rate_range=(0.0, np.inf)):
    """Load every .ocw in data/experimental/<subdir> into a
    {scan_rate_V_s: (E, i)} dict, keeping one curve per (rounded) rate and
    restricting to `rate_range` (V/s)."""
    folder = DATA / subdir
    out = {}
    for ocw in sorted(folder.glob(pattern)):
        icw = ocw.with_suffix(".icw")
        if not icw.exists():
            continue
        v = round(scan_rate_V_s(icw), 4)
        if not (rate_range[0] <= v <= rate_range[1]) or v in out:
            continue
        out[v] = read_curve(ocw)
    return out
