"""Impurity radius sigma / xi from the reconstructed potential (Fig. 1d).

Clean version of ``compute_potential_2`` in ``myrann_processing.py``. The potential
map is obtained from the Bernoulli equation (Eq. 3 of the paper),

    V = rho_ref - rho + (|sqrt(rho_ref) v_ref|^2 - |sqrt(rho) v|^2) / 2,

and a Gaussian exp(-2 x^2 / w0^2) is fitted along the axis perpendicular to the flow
through the minimum of the unwrapped phase. The 1/e^2 radius w0, in units of xi, is
the quoted sigma / xi.

Faithfulness to the legacy code, kept on purpose so that Fig. 1(d) is reproduced:
``|sqrt(rho) v|^2 = rho v^2`` is used for the kinetic term, xi is the constant
``XI_LEGACY`` and the frame ``FRAME`` of the scan is analysed.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage, optimize
from skimage.restoration import unwrap_phase

from . import params

XI_LEGACY = 48e-6  # m, healing length used to express lengths in the legacy code
FRAME = 3  # frame of the scan analysed (fluid nearly at rest)
ROI = 200  # half-size of the region of interest (px)
SMOOTH = 1  # px, Gaussian smoothing of the fields


def _roi(img):
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    return img[cy - ROI : cy + ROI, cx - ROI : cx + ROI]


def _prepare(field, xi_cs):
    """Normalised density, and rho * v^2 map, of one smoothed field (ROI only)."""
    from . import velocity  # slow import (numba/pyfftw)

    field = ndimage.gaussian_filter(field, SMOOTH)
    amp = np.abs(field) ** 2
    amp = amp / amp.max()
    phase = np.angle(field)
    amp_roi, phase_roi = _roi(amp), _roi(phase)
    v = np.sqrt(amp_roi) * velocity.velocity(phase_roi, dx=params.D_REAL / xi_cs)
    return amp_roi, phase_roi, v[0] ** 2 + v[1] ** 2


def impurity_radius(scan_dir, frame: int = FRAME, xi: float = XI_LEGACY, return_profile=False):
    """sigma / xi of one scan: 1/e^2 radius of the Gaussian fitted to the potential."""
    scan_dir = Path(scan_dir)
    ff = np.load(scan_dir / "field_fluid.npy", mmap_mode="r")
    fr = np.load(scan_dir / "field_fluid_ref.npy", mmap_mode="r")
    cs = np.load(scan_dir / "cs_quench.npy")
    xi_cs = 1 / (params.K0 * cs)

    amp, phase, kin = _prepare(np.asarray(ff[frame]), xi_cs)
    amp_ref, phase_ref, kin_ref = _prepare(np.asarray(fr[frame]), xi_cs)
    potential = amp_ref - amp + kin_ref / 2 - kin / 2

    unwrapped = unwrap_phase(phase)
    unwrapped = unwrapped + abs(unwrapped.min())
    _, x_min = np.unravel_index(np.argmin(unwrapped), unwrapped.shape)
    profile = potential[:, x_min]  # cut perpendicular to the flow, through the impurity

    n = profile.size
    x = np.linspace(-n // 2, n // 2, n) * params.D_REAL / xi
    outer = np.ones(n, dtype=bool)
    outer[n // 2 - 50 : n // 2 + 50] = False
    background = profile[outer].mean()
    fit_mask = np.ones(n, dtype=bool)
    fit_mask[n // 2 - 10 : n // 2 + 10] = False

    def gauss(xx, a0, x0, w0):
        return a0 * np.exp(-2 * (xx - x0) ** 2 / w0**2) + background

    popt, _ = optimize.curve_fit(gauss, x[fit_mask], profile[fit_mask], p0=[0.8, 0, 10])
    sigma = abs(popt[2])
    if return_profile:
        return sigma, x, profile, gauss(x, *popt)
    return sigma
