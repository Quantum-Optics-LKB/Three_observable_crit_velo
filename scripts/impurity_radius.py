"""Impurity radius sigma / xi of the scans (Fig. 1d), recomputed and compared with pot_size.txt.

    python scripts/impurity_radius.py --data-root Data_defect
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supervelo import params, potential  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "Data_defect")
    p.add_argument("--scans", nargs="*", default=params.SCANS_FIG4 + [params.SCAN_FIG2])
    p.add_argument("--frames", type=int, nargs="*", default=[potential.FRAME],
                   help="frames analysed (the first is the nominal one)")
    args = p.parse_args()

    reference = dict(zip(params.SCANS_FIG4, np.loadtxt(ROOT / "data" / "pot_size.txt")))
    print("scan".ljust(26), "sigma/xi".rjust(9), "pot_size.txt".rjust(13),
          "other frames (min-max)".rjust(24))
    for scan in args.scans:
        radii = [potential.impurity_radius(args.data_root / scan, frame=f) for f in args.frames]
        ref = reference.get(scan)
        spread = f"{min(radii):.2f}-{max(radii):.2f}" if len(radii) > 1 else ""
        print(scan.ljust(26), f"{radii[0]:9.3f}",
              (f"{ref:13.3f}" if ref is not None else "-".rjust(13)), spread.rjust(24))


if __name__ == "__main__":
    main()
