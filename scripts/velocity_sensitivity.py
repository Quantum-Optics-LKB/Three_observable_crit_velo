"""Sensitivity of the impurity-velocity onset (Fig. 3) to the threshold.

    python scripts/velocity_sensitivity.py --data-root Data_defect
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supervelo import observables, params, thresholds  # noqa: E402

THRESHOLDS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "Data_defect")
    args = p.parse_args()

    print("scan".ljust(26), *[f"{t:7.2f}" for t in THRESHOLDS])
    for scan in params.SCANS_FIG4 + [params.SCAN_FIG2]:
        beta, dv = observables.defect_velocity(args.data_root / scan)
        vals = [thresholds.velocity_onset(beta, dv, threshold=t).first for t in THRESHOLDS]
        print(scan.ljust(26), *[f"{v:7.3f}" for v in vals])


if __name__ == "__main__":
    main()
