"""Montage of the phase maps with the detected vortices, to check false positives.

    python scripts/inspect_vortices.py --scan 08291816_scan_k_defect --beta-range 0 0.35

Red/blue dots are +1/-1 vortices from ``velocity.vortex_detection``; the white square
is the counting window used for the vortex number.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supervelo import observables, params, velocity  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "Data_defect")
    p.add_argument("--scan", default=params.SCANS_FIG4[-1])
    p.add_argument("--beta-range", type=float, nargs=2, default=(0.0, 0.35))
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--radius", type=int, default=params.VORTEX_PLAQUETTE_RADIUS)
    p.add_argument("--half", type=int, default=150, help="half-size of the displayed crop (px)")
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "vortex_check.png")
    args = p.parse_args()

    scan_dir = args.data_root / args.scan
    beta = observables.mach_number(scan_dir)
    ff = np.load(scan_dir / "field_fluid.npy", mmap_mode="r")
    sel = np.flatnonzero((beta >= args.beta_range[0]) & (beta <= args.beta_range[1]))
    sel = sel[np.argsort(beta[sel])]
    sel = sel[np.unique(np.linspace(0, len(sel) - 1, args.n_frames).astype(int))]

    ncol = 4
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), layout="constrained")
    w_px = params.VORTEX_WINDOW / (2 * params.D_REAL)
    for ax, i in zip(np.ravel(axs), sel):
        phi = np.angle(np.asarray(ff[i]))
        vort = velocity.vortex_detection(phi, r=args.radius)
        cy, cx = phi.shape[0] // 2, phi.shape[1] // 2
        h = args.half
        ax.imshow(phi[cy - h : cy + h, cx - h : cx + h], cmap="twilight_shifted",
                  extent=(-h, h, h, -h))
        for charge, color in ((1, "red"), (-1, "blue")):
            m = vort[:, 2] == charge
            ax.scatter(vort[m, 0] - cx, vort[m, 1] - cy, s=14, c=color, edgecolors="w", lw=0.4)
        ax.add_patch(plt.Rectangle((-w_px, -w_px), 2 * w_px, 2 * w_px, fill=False, ec="w", lw=0.8))
        ax.set_xlim(-h, h)
        ax.set_ylim(h, -h)
        inside = (np.abs(vort[:, 0] - cx) < w_px) & (np.abs(vort[:, 1] - cy) < w_px)
        ax.set_title(f"frame {i}, beta={beta[i]:.2f}, in window: N+={int((inside & (vort[:, 2] > 0)).sum())}, "
                     f"N-={int((inside & (vort[:, 2] < 0)).sum())}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in np.ravel(axs)[len(sel):]:
        ax.axis("off")
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
