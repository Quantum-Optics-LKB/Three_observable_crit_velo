"""The three observables of the paper, computed from one scan folder.

A scan folder (``Data_defect/<MMDDhhmm>_scan_k_defect``) must contain
``kx_fluid.npy``, ``kx_defect.npy``, ``cs_quench.npy``, ``field_fluid.npy``,
``field_fluid_ref.npy`` and ``field_defect.npy``.

Axis convention: the code's x axis is the flow axis, which is the y axis of the
paper. ``F_flow`` is the force along the flow (F_y in the paper) and
``F_trans`` the transverse one (F_x in the paper).

Expensive results are cached in ``cache/<scan>/``. The cache is keyed by the scan
name, unlike the legacy ``load_cached_array``, so scans can never be mixed up.
"""

from pathlib import Path

import numpy as np

from . import params

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache"


def _cache(scan_dir: Path, name: str) -> Path:
    path = CACHE_ROOT / Path(scan_dir).name
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def _mask_bad(scan_dir, values):
    """Copy of per-frame ``values`` with the known bad frames of the scan set to NaN.

    ``thresholds.sort_by_beta`` drops non-finite points, so masked frames disappear
    from every threshold and plot as soon as beta (or the signal) is NaN there.
    """
    values = np.array(values, dtype=float)
    bad = params.BAD_FRAMES.get(Path(scan_dir).name, [])
    values[bad] = np.nan
    return values


def mach_number(scan_dir) -> np.ndarray:
    """Mach number beta = 2 pi kx_fluid / (k0 c_s) of every frame (positive-oriented).

    Bad frames (``params.BAD_FRAMES``) are NaN.
    """
    scan_dir = Path(scan_dir)
    kx = np.load(scan_dir / "kx_fluid.npy")
    cs = np.load(scan_dir / "cs_quench.npy")
    if kx[-1] < 0:
        kx = -kx
    return _mask_bad(scan_dir, 2 * np.pi * kx / (params.K0 * cs))


def defect_velocity(scan_dir, clean: bool = False):
    """Impurity velocity relative to the flow, Delta v_i / c_s = beta - 2 pi kx_defect / (ki c_s).

    The defect tilt is referenced to its first frame, as in ``defect_velocity_2``.
    Returns (beta, delta_v), both in the original frame order.
    """
    scan_dir = Path(scan_dir)
    kx_fluid = np.load(scan_dir / "kx_fluid.npy")
    kx_def = np.load(scan_dir / "kx_defect.npy")
    kx_def = kx_def - kx_def[0]
    cs = np.load(scan_dir / "cs_quench.npy")
    beta = 2 * np.pi * kx_fluid / (params.K0 * cs)
    delta_v = beta - 2 * np.pi * kx_def / (params.KI * cs)
    if clean:
        # Display filter of ``defect_velocity_2``: drop frames where the tilt of the
        # impurity or of the fluid jumps by more than 3x the median step.
        d_beta = np.abs(np.diff(beta, append=0))
        d_kx = np.abs(np.diff(kx_def, append=0))
        glitch = (d_kx >= 3 * np.median(d_kx)) | (d_beta >= 3 * np.median(d_beta))
        beta, delta_v = np.where(glitch, np.nan, beta), np.where(glitch, np.nan, delta_v)
    return _mask_bad(scan_dir, beta), _mask_bad(scan_dir, delta_v)


def _aligned_defect(frame, shift):
    """Flip the impurity image (mirror imaging) and shift it onto the fluid image."""
    return np.roll(np.flip(frame, axis=0)[:, ::-1], shift, axis=(0, 1))


def _shift():
    sx = params.BARY_FLUID_PX[0] - params.BARY_DEFECT_PX[0]
    sy = params.BARY_FLUID_PX[1] - params.BARY_DEFECT_PX[1]
    return sy, sx


def force_density(scan_dir, frame: int) -> np.ndarray:
    """Surface force density rho(x,y) d rho_i / d(flow axis) of one frame (Fig. 2d)."""
    scan_dir = Path(scan_dir)
    ff = np.load(scan_dir / "field_fluid.npy", mmap_mode="r")
    fr = np.load(scan_dir / "field_fluid_ref.npy", mmap_mode="r")
    fd = np.load(scan_dir / "field_defect.npy", mmap_mode="r")
    fluid = ff[frame] * np.exp(-1j * np.angle(fr[frame]))
    rho_f = np.abs(fluid) ** 2
    rho_d = np.abs(_aligned_defect(fd[frame], _shift())) ** 2
    d_dy, d_dx = np.gradient(rho_d, params.D_DEF)
    return -d_dx * rho_f


def drag_force(scan_dir, use_cache: bool = True):
    """Hydrodynamic force on the impurity for every frame (Eq. 4 of the paper).

    Returns (F_flow, F_trans). Same operation as ``drag_force_crit`` of
    ``myrann_processing.py``, done one frame at a time to limit memory.
    """
    scan_dir = Path(scan_dir)
    cache = _cache(scan_dir, "drag_force.npz")
    if use_cache and cache.exists():
        z = np.load(cache)
        return _mask_bad(scan_dir, z["F_flow"]), _mask_bad(scan_dir, z["F_trans"])
    ff = np.load(scan_dir / "field_fluid.npy", mmap_mode="r")
    fr = np.load(scan_dir / "field_fluid_ref.npy", mmap_mode="r")
    fd = np.load(scan_dir / "field_defect.npy", mmap_mode="r")
    shift = _shift()
    f_flow = np.empty(ff.shape[0])
    f_trans = np.empty(ff.shape[0])
    for i in range(ff.shape[0]):
        rho_f = np.abs(ff[i] * np.exp(-1j * np.angle(fr[i]))) ** 2
        rho_d = np.abs(_aligned_defect(fd[i], shift)) ** 2
        d_dy, d_dx = np.gradient(rho_d, params.D_DEF)
        f_flow[i] = -np.sum(d_dx * rho_f)
        f_trans[i] = -np.sum(d_dy * rho_f)
    np.savez(cache, F_flow=f_flow, F_trans=f_trans)  # cached unmasked
    return _mask_bad(scan_dir, f_flow), _mask_bad(scan_dir, f_trans)


def _vortex_counts(scan_dir, use_cache: bool = True, rho_min=None) -> np.ndarray:
    """Counts of (+1, -1) vortices near the impurity for every frame, shape (2, n_frames).

    Circulation on a plaquette of radius ``VORTEX_PLAQUETTE_RADIUS`` (see
    ``velocity.vortex_detection``), counted in a window around the impurity, as
    in ``scan_k_tur_setup.vortices``.
    """
    from . import velocity  # imported lazily (slow: numba/pyfftw)

    scan_dir = Path(scan_dir)
    from scipy.ndimage import gaussian_filter

    rho_min = params.VORTEX_RHO_MIN if rho_min is None else rho_min
    cache = _cache(scan_dir, f"vortex_counts_rho{rho_min}.npy")
    if use_cache and cache.exists():
        return np.load(cache)
    ff = np.load(scan_dir / "field_fluid.npy", mmap_mode="r")
    counts = np.zeros((2, ff.shape[0]))
    for i in range(ff.shape[0]):
        field = np.asarray(ff[i])
        phi = np.angle(field)
        vort = velocity.vortex_detection(phi, r=params.VORTEX_PLAQUETTE_RADIUS)
        x = (vort[:, 0] - phi.shape[-1] / 2) * 2 * params.D_REAL
        y = (vort[:, 1] - phi.shape[-2] / 2) * 2 * params.D_REAL
        w = params.VORTEX_WINDOW
        inside = (np.abs(x) < w) & (np.abs(y) < w)
        rho = gaussian_filter(np.abs(field) ** 2, params.VORTEX_RHO_SMOOTH)
        half = int(params.VORTEX_WINDOW / (2 * params.D_REAL))
        cy, cx = phi.shape[0] // 2, phi.shape[1] // 2
        rho0 = np.percentile(rho[cy - half : cy + half, cx - half : cx + half], 90)
        iy = np.clip(vort[:, 1].astype(int), 0, phi.shape[0] - 1)
        ix = np.clip(vort[:, 0].astype(int), 0, phi.shape[1] - 1)
        inside &= rho[iy, ix] > rho_min * rho0
        counts[0, i] = np.sum(inside & (vort[:, 2] > 0))
        counts[1, i] = np.sum(inside & (vort[:, 2] < 0))
    np.save(cache, counts)
    return counts


def vortex_number(scan_dir, use_cache: bool = True, rho_min=None) -> np.ndarray:
    """Total number of vortices near the impurity in every frame (Fig. 2a)."""
    return _mask_bad(scan_dir, _vortex_counts(scan_dir, use_cache, rho_min).sum(axis=0))


def vortex_pairs(scan_dir, use_cache: bool = True, rho_min=None) -> np.ndarray:
    """Number of vortex-antivortex pairs, min(N+, N-), in every frame.

    Phase noise produces isolated detections of a single charge; requiring both
    charges rejects most of them.
    """
    return _mask_bad(scan_dir, _vortex_counts(scan_dir, use_cache, rho_min).min(axis=0))


def impurity_center(scan_dir, frame: int = 0, smooth: float = 6.0):
    """Pixel position (row, col) of the impurity in the fluid image of one frame.

    Taken as the maximum of the (aligned) impurity-beam intensity.
    """
    from scipy.ndimage import gaussian_filter

    fd = np.load(Path(scan_dir) / "field_defect.npy", mmap_mode="r")
    rho = gaussian_filter(np.abs(_aligned_defect(np.asarray(fd[frame]), _shift())) ** 2, smooth)
    return np.unravel_index(np.argmax(rho), rho.shape)
