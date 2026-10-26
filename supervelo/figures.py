"""Reproduction of Figs. 2, 3 and 4 of the paper from the analysis results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import observables, params
from .thresholds import sort_by_beta

COLORS = {"vortex": "tab:blue", "force": "tab:orange", "velocity": "tab:green"}
MARKERS = {"vortex": "^", "force": "s", "velocity": "o"}


def _crop(img, center=None, half=60):
    cy, cx = center if center is not None else (img.shape[0] // 2, img.shape[1] // 2)
    return img[cy - half : cy + half, cx - half : cx + half]


def _extent(half=60):
    um = half * params.D_REAL * 1e6
    return (-um, um, -um, um)


def fig2(scan_dir, beta_c, out, beta_map=0.4):
    """Fig. 2: vortex number, amplitude maps, force and force density at one sigma/xi.

    ``beta_c`` is a dict with the ``vortex`` and ``force`` onsets (floats).
    """
    scan_dir = Path(scan_dir)
    beta = observables.mach_number(scan_dir)
    n_vort = observables.vortex_number(scan_dir)
    f_flow, f_trans = observables.drag_force(scan_dir)
    # Remove the sub-critical offset (alignment residual), orient the rise as positive
    quiet = (beta < 0.2) & (beta > 0)
    f_flow = f_flow - f_flow[quiet].mean()
    f_trans = f_trans - f_trans[quiet].mean()
    sign = np.sign(f_flow[np.nanargmax(np.abs(f_flow))])
    ff = np.load(scan_dir / "field_fluid.npy", mmap_mode="r")
    frames = [int(np.nanargmin(np.abs(beta - b))) for b in (0.0, beta_map)]
    center = observables.impurity_center(scan_dir, frames[0])
    dens = [_crop(observables.force_density(scan_dir, i), center) for i in frames]
    dens_max = max(np.abs(d).max() for d in dens)

    fig = plt.figure(figsize=(8, 5), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    b_s, n_s = sort_by_beta(beta, n_vort)
    ax_a.plot(b_s, n_s, "k")
    ax_a.axvline(beta_c["vortex"], color="k", ls="--")
    ax_a.set_ylabel("Number of vortices")
    ax_a.set_ylim(0, 12)  # isolated frames with hundreds of detections are camera glitches
    b_s, ff_s, ft_s = sort_by_beta(beta, sign * f_flow, sign * f_trans)
    ax_c.plot(b_s, ff_s, color="limegreen", label=r"$F_y$ (along flow)")
    ax_c.plot(b_s, ft_s, color="navy", label=r"$F_x$ (transverse)")
    ax_c.axvline(beta_c["force"], color="limegreen", ls="--")
    ax_c.set_xlabel(r"$\beta$")
    ax_c.set_ylabel(r"$F_{x,y}$ (a.u.)")
    ax_c.legend(fontsize=8)
    for k, i in enumerate(frames):
        rho = np.abs(np.asarray(ff[i])) ** 2
        crop = _crop(rho, center)
        ax_b = fig.add_subplot(gs[0, 1 + k])
        ax_b.imshow(np.sqrt(crop / np.percentile(crop, 99)), cmap="gray", vmin=0, vmax=1,
                    extent=_extent())
        ax_b.set_title(rf"$\beta\simeq{beta[i]:.2f}$", fontsize=9)
        ax_d = fig.add_subplot(gs[1, 1 + k])
        ax_d.imshow(sign * dens[k] / dens_max, cmap="viridis", vmin=0, vmax=1, extent=_extent())
        ax_d.set_xlabel(r"$x$ ($\mu$m)")
        if k == 0:
            ax_b.set_ylabel(r"$y$ ($\mu$m)")
            ax_d.set_ylabel(r"$y$ ($\mu$m)")
    fig.suptitle(f"Fig. 2 reproduction: {scan_dir.name}", fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _soft(color, alpha=0.5):
    """Light fill colour of the markers used in the paper figures."""
    from matplotlib.colors import to_rgba

    return to_rgba(color, alpha)


def fig3(data_root, scans, labels, beta_c, out):
    """Fig. 3 in the style of the paper: Delta v_i / c_s versus beta.

    ``beta_c`` maps scan name to the velocity onset (dashed vertical line).
    """
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, ax = plt.subplots(figsize=(3.6, 2.9), layout="constrained")
    handles = []
    for scan, label, color in zip(scans, labels, colors):
        beta, dv = observables.defect_velocity(Path(data_root) / scan, clean=True)
        beta, dv = sort_by_beta(beta, dv)
        ax.plot(beta, dv, "-", color=color, lw=1.2)
        h, = ax.plot(beta, dv, "o", ms=6, mfc=_soft(color), mec=color, mew=1.2, label=label)
        handles.append(h)
        ax.axvline(beta_c[scan], color=color, ls="--", lw=1.5, zorder=0)
    xs = np.array([0.0, 0.72])
    ax.plot(xs, xs, "k", lw=2, zorder=1)
    ax.text(0.745, 0.72, r"$\beta$", va="center")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\Delta v_\mathrm{i}/c_\mathrm{s}$")
    ax.set_xlim(-0.04, 0.78)
    ax.set_ylim(-0.04, 1.16)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    leg = ax.legend(handles=handles, title=r"$\sigma/\xi$", loc="lower right", frameon=False,
                    handletextpad=0.3)
    leg._legend_box.align = "center"
    # underline the legend title, as in the paper
    fig.canvas.draw()
    box = leg.get_title().get_window_extent().transformed(ax.transAxes.inverted())
    ax.plot([box.x0, box.x1], [box.y0 - 0.012] * 2, transform=ax.transAxes, color="k", lw=0.8)
    _save(fig, out)
    plt.close(fig)


def fig4(table, err, sigma, out, extra=None):
    """Fig. 4 in the style of the paper: beta_c of the three observables versus sigma / xi.

    Args:
        table: dict observable -> array of beta_c (one per scan).
        err: dict observable -> array of standard uncertainties on beta_c, from
            ``uncertainty.beta_c_uncertainty`` (sampling, threshold definition and
            sound-velocity calibration added in quadrature).
        sigma: sigma / xi of each scan.
        extra: optional dict observable -> (sigma, beta_c, error) for the run of Fig. 2,
            drawn with the same symbols but without connecting line.

    The dashed line is the average of all the points of the six scans.
    """
    labels = {"vortex": "Vortex count", "force": "Hydrodynamic force",
              "velocity": "Impurity velocity"}
    fig, ax = plt.subplots(figsize=(3.7, 2.5), layout="constrained")
    top = 0.0
    for key in ("vortex", "force", "velocity"):
        y, e = np.asarray(table[key], float), np.asarray(err[key], float)
        ax.errorbar(sigma, y, yerr=e, color=COLORS[key], marker=MARKERS[key], ms=4.5, mew=1,
                    mfc=_soft(COLORS[key]), mec=COLORS[key], capsize=2, lw=1.2, elinewidth=1,
                    label=labels[key])
        top = max(top, np.nanmax(y + e))
    if extra is not None:
        for key, (s, b, e) in extra.items():
            ax.errorbar(s, b, yerr=e, color=COLORS[key], marker=MARKERS[key], ms=4.5, mew=1,
                        mfc=_soft(COLORS[key]), mec=COLORS[key], capsize=2, elinewidth=1, ls="")
            top = max(top, b + e)
    mean = float(np.nanmean([table[k] for k in table]))
    ax.axhline(mean, color="k", ls="--", lw=1.5, zorder=0)
    ax.text(0.04, mean - 0.035, rf"$\langle\beta_\mathrm{{c}}\rangle$ = {mean:.2f}",
            transform=ax.get_yaxis_transform(), va="top")
    ax.set_xlabel(r"$\sigma/\xi$")
    ax.set_ylabel(r"$\beta_\mathrm{c}$")
    ax.set_xlim(4.05, 6.6)
    ax.set_xticks([4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_ylim(0.1, max(0.8, top + 0.04))
    ax.legend(loc="upper right", fontsize=7)
    _save(fig, out)
    plt.close(fig)


def _save(fig, out):
    """Save a figure as PNG (preview) and as vector PDF (manuscript) next to each other."""
    out = Path(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
