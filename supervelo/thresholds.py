"""Critical-velocity onset criteria.

All three observables (vortex count, drag force, impurity velocity) are reduced
to the same question: at which Mach number beta does a signal first leave its
sub-critical behaviour? ``find_onset`` answers it for any signal, and the
helpers below build the signal-specific baseline and threshold.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter


@dataclass
class Onset:
    """Critical Mach number found for one scan.

    ``first`` is the first beta at which the criterion is met (``consecutive``
    successive points), ``last_before`` is the point just before it. The paper
    quotes the latter for the force ("last data point preceding ...").
    """

    first: float
    last_before: float

    @property
    def found(self) -> bool:
        return not np.isnan(self.first)


def sort_by_beta(beta, *signals):
    """Sort all signals by increasing beta and drop non-finite points."""
    beta = np.asarray(beta, dtype=float)
    keep = np.isfinite(beta)
    for s in signals:
        keep &= np.isfinite(s)
    order = np.argsort(beta[keep])
    return (beta[keep][order],) + tuple(np.asarray(s, float)[keep][order] for s in signals)


def find_onset(beta, excess, threshold, consecutive=3, beta_min=0.0) -> Onset:
    """First beta where ``excess > threshold`` for ``consecutive`` points in a row.

    Args:
        beta: increasing Mach numbers.
        excess: signal minus its baseline.
        threshold: scalar or array, same units as ``excess``.
        consecutive: number of successive points required (rejects isolated spikes).
        beta_min: ignore points below this Mach number.
    """
    above = excess > threshold
    for i in range(len(beta) - consecutive + 1):
        if beta[i] >= beta_min and above[i : i + consecutive].all():
            return Onset(beta[i], beta[i - 1] if i > 0 else np.nan)
    return Onset(np.nan, np.nan)


def robust_line(x, y, n_iter=5):
    """Straight-line fit with iterative 3-sigma (MAD) clipping. Returns (slope, offset)."""
    keep = np.ones_like(x, dtype=bool)
    for _ in range(n_iter):
        p = np.polyfit(x[keep], y[keep], 1)
        res = y - np.polyval(p, x)
        sigma = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) < 3 * max(sigma, 1e-3)
    return p


def velocity_onset(
    beta, delta_v, threshold=0.04, lag=(0.03, 0.25), consecutive=3, beta_min=0.1
) -> Onset:
    """Onset of the impurity upstream motion (Fig. 3).

    The signal is the deviation ``delta_v - beta`` of Delta v_i / c_s from the
    superfluid line (the black line of Fig. 3), smoothed with a 3-point median
    filter. The onset is the first beta where the signal exceeds its *local*
    sub-critical level by ``threshold`` (in units of c_s) for ``consecutive``
    successive points. The local level is the median of the signal over
    ``[beta - lag[1], beta - lag[0]]``, i.e. over the points just before.

    Comparing with the preceding points instead of a fixed baseline detects the
    first departure from the previous behaviour, whether it is a step or the start
    of a ramp, and is not fooled by the slow drift of the baseline or by a larger
    response at higher beta.
    """
    beta, delta_v = sort_by_beta(beta, delta_v)
    raw = delta_v - beta
    dev = median_filter(raw, size=3, mode="nearest")
    excess = np.full_like(beta, np.nan)
    for i, b in enumerate(beta):
        before = (beta >= max(b - lag[1], 0.03)) & (beta <= b - lag[0])
        if before.sum() >= 3:
            excess[i] = dev[i] - np.median(raw[before])
    return find_onset(beta, excess, threshold, consecutive, beta_min)


def force_onset(
    beta, force, noise_window=0.2, fraction=0.1, n_sigma=None, consecutive=3
) -> Onset:
    """Onset of the hydrodynamic force along the flow (Fig. 2c).

    The baseline is the mean force for ``beta < noise_window`` (superfluid regime,
    zero force). The threshold is either ``fraction`` of the rise
    (smoothed maximum minus baseline), or ``n_sigma`` times the standard
    deviation of the noise if ``n_sigma`` is given.
    """
    beta, force = sort_by_beta(beta, force)
    quiet = beta < noise_window
    base, noise = force[quiet].mean(), force[quiet].std()
    smooth = median_filter(force, size=5, mode="nearest")
    rise = np.abs(smooth - base).max()
    thr = n_sigma * noise if n_sigma is not None else fraction * rise
    sign = np.sign(smooth[np.argmax(np.abs(smooth - base))] - base)
    return find_onset(beta, sign * (force - base), thr, consecutive, beta_min=noise_window)


def vortex_onset(beta, pairs, consecutive=2, beta_min=0.05) -> Onset:
    """Onset of vortex nucleation (Fig. 2a): first emission of a vortex-antivortex pair.

    ``pairs`` is the number of pairs per frame. A pair must be seen in
    ``consecutive`` successive frames above ``beta_min`` to count; isolated
    detections are phase noise.
    """
    beta, pairs = sort_by_beta(beta, pairs)
    return find_onset(beta, pairs, 0.5, consecutive, beta_min)
