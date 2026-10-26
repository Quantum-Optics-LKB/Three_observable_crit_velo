"""Compute the critical Mach numbers of the three observables and redraw Figs. 2-4.

Usage (from the repository root):

    python scripts/run_analysis.py --data-root Data_defect

Outputs ``results/critical_betas.csv`` and ``figures/fig{2,3,4}.png``.
Nothing in the data folder is modified; heavy intermediate results go to ``cache/``.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supervelo import figures, observables, params, thresholds, uncertainty  # noqa: E402

OBSERVABLES = ("vortex", "force", "velocity")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "Data_defect")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--figures", type=Path, default=ROOT / "figures")
    p.add_argument("--velocity-threshold", type=float, default=0.04,
                   help="excess of Delta v_i / c_s - beta over its local sub-critical level "
                        "defining the onset (units of c_s)")
    p.add_argument("--force-fraction", type=float, default=0.1,
                   help="fraction of the force rise defining the onset")
    p.add_argument("--cs-rel-error", type=float, default=0.05,
                   help="ASSUMED relative uncertainty on the sound velocity c_s "
                        "(calibration error on beta); to be replaced by the measured value")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args()


def scan_onsets(scan_dir, args):
    """Critical Mach numbers of one scan, with the onsets of the varied criteria.

    Returns two dicts keyed by observable: the nominal ``Onset`` and the list of
    ``beta_c`` obtained with other reasonable criteria (used for the uncertainty).
    """
    beta, dv = observables.defect_velocity(scan_dir)
    f_flow, _ = observables.drag_force(scan_dir)
    beta_m = observables.mach_number(scan_dir)

    vel = thresholds.velocity_onset(beta, dv, threshold=args.velocity_threshold)
    vel_var = [thresholds.velocity_onset(beta, dv, threshold=f * args.velocity_threshold).first
               for f in (0.75, 1.5)]

    force = thresholds.force_onset(beta_m, f_flow, fraction=args.force_fraction)
    force_var = [thresholds.force_onset(beta_m, f_flow, fraction=f).first
                 for f in (0.05, 0.15)]

    vort = thresholds.vortex_onset(beta_m, observables.vortex_pairs(scan_dir))
    vort_var = [thresholds.vortex_onset(beta_m, observables.vortex_pairs(scan_dir),
                                        consecutive=c).first for c in (1, 3)]
    vort_var += [thresholds.vortex_onset(beta_m, observables.vortex_pairs(scan_dir, rho_min=r)).first
                 for r in (0.05, 0.2)]

    onsets = {"vortex": vort, "force": force, "velocity": vel}
    variants = {"vortex": vort_var, "force": force_var, "velocity": vel_var}
    return onsets, variants


def scan_uncertainties(onsets, variants, args):
    return {k: uncertainty.beta_c_uncertainty(onsets[k], variants[k], args.cs_rel_error)
            for k in onsets}


def row(scan, power, s, onsets, unc):
    out = {"scan": scan, "power_mW": power, "sigma_over_xi": f"{s:.3f}"}
    for k in OBSERVABLES:
        out[f"{k}_first"] = f"{onsets[k].first:.3f}"
        out[f"{k}_last_before"] = f"{onsets[k].last_before:.3f}"
        out[f"{k}_err"] = f"{unc[k].total:.3f}"
        out[f"{k}_err_sampling"] = f"{unc[k].sampling:.3f}"
        out[f"{k}_err_criterion"] = f"{unc[k].criterion:.3f}"
        out[f"{k}_err_calibration"] = f"{unc[k].calibration:.3f}"
    return out


def main():
    args = parse_args()
    args.results.mkdir(exist_ok=True)
    args.figures.mkdir(exist_ok=True)
    sigma = np.loadtxt(ROOT / "data" / "pot_size.txt")

    rows = []
    table = {k: [] for k in OBSERVABLES}
    err = {k: [] for k in OBSERVABLES}
    velocity_bc = {}
    for scan, power, s in zip(params.SCANS_FIG4, params.POWERS_MW, sigma):
        print(f"[{scan}] {power} mW, sigma/xi = {s:.2f}", flush=True)
        on, var = scan_onsets(args.data_root / scan, args)
        unc = scan_uncertainties(on, var, args)
        rows.append(row(scan, power, s, on, unc))
        for k in OBSERVABLES:
            table[k].append(on[k].first)
            err[k].append(unc[k].total)
        velocity_bc[scan] = on["velocity"].first
        print("   ", {k: f"{on[k].first:.3f} +- {unc[k].total:.3f}" for k in OBSERVABLES})

    with open(args.results / "critical_betas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Fig. 2 data set
    fig2_dir = args.data_root / params.SCAN_FIG2
    on2, var2 = scan_onsets(fig2_dir, args)
    unc2 = scan_uncertainties(on2, var2, args)
    print(f"[{params.SCAN_FIG2}] (Fig. 2)",
          {k: f"{on2[k].first:.3f} +- {unc2[k].total:.3f}" for k in OBSERVABLES})
    with open(args.results / "critical_betas_fig2_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row(params.SCAN_FIG2, "", np.nan, on2, unc2)))
        w.writeheader()
        w.writerow(row(params.SCAN_FIG2, "", np.nan, on2, unc2))

    if args.no_figures:
        return
    figures.fig2(fig2_dir, {k: v.first for k, v in on2.items()}, args.figures / "fig2.png")
    labels = [f"{sigma[params.SCANS_FIG4.index(s)]:.1f}" for s in params.SCANS_FIG3]
    figures.fig3(args.data_root, params.SCANS_FIG3, labels, velocity_bc, args.figures / "fig3.png")
    figures.fig4({k: np.array(v) for k, v in table.items()},
                 {k: np.array(v) for k, v in err.items()}, sigma, args.figures / "fig4.png",
                 extra={k: (params.SIGMA_FIG2, on2[k].first, unc2[k].total) for k in OBSERVABLES})


if __name__ == "__main__":
    main()
