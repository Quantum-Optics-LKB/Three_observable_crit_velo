"""Uncertainty on the critical Mach numbers, propagated from measurement limits.

For one scan and one observable the standard uncertainty on beta_c combines, in
quadrature, three independent contributions:

* **sampling**: beta_c is only known to within the gap between the last point
  below and the first point above the criterion (rectangular distribution,
  half-width = gap / 2);
* **criterion**: spread of beta_c when the threshold definition is varied within
  reasonable bounds (rectangular distribution, half-width = half the full range);
* **calibration**: beta = v0 / c_s, so a relative error ``cs_rel_error`` on the
  sound velocity moves beta_c proportionally (systematic, common to the three
  observables of a scan).

Rectangular half-widths are divided by sqrt(3) to give standard uncertainties.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Uncertainty:
    sampling: float
    criterion: float
    calibration: float

    @property
    def total(self) -> float:
        return float(np.sqrt(self.sampling**2 + self.criterion**2 + self.calibration**2))


def beta_c_uncertainty(nominal, variants, cs_rel_error: float = 0.0) -> Uncertainty:
    """Combine the three contributions.

    Args:
        nominal: ``thresholds.Onset`` obtained with the reference criterion.
        variants: critical Mach numbers obtained with the other criteria
            (NaN entries are ignored).
        cs_rel_error: relative uncertainty on the sound velocity c_s.
    """
    gap = nominal.first - nominal.last_before
    sampling = 0.0 if np.isnan(gap) else abs(gap) / 2 / np.sqrt(3)
    v = np.array([nominal.first, *variants], dtype=float)
    v = v[np.isfinite(v)]
    criterion = 0.0 if len(v) < 2 else (v.max() - v.min()) / 2 / np.sqrt(3)
    return Uncertainty(sampling, criterion, abs(nominal.first) * cs_rel_error)
