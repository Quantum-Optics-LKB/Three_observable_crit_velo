"""Experimental constants and scan lists shared by the analysis.

Every number that used to be hard-coded in ``myrann_processing.py`` lives here.
"""

import numpy as np

# Optics
K0 = 2 * np.pi / 780e-9  # fluid beam wavenumber (rad/m)
KI = 2 * np.pi / 795e-9  # impurity beam wavenumber (rad/m)

# Pixel sizes (m)
D_REAL = 3.45e-6 * 2  # fluid camera, 2x binned
D_DEF = 1.1e-6  # impurity camera

# Defect-to-fluid alignment (pixels), identical for all scans in the original code.
# The defect image is shifted by (BARY_FLUID - BARY_DEFECT) along each axis.
BARY_FLUID_PX = (614, 515)  # (x, y)
BARY_DEFECT_PX = (606, 523)  # (x, y)

# Scans of Fig. 1(d), 3 and 4, in order of increasing impurity power.
SCANS_FIG4 = [
    "08291714_scan_k_defect",
    "08291746_scan_k_defect",
    "08291752_scan_k_defect",
    "08291758_scan_k_defect",
    "08291809_scan_k_defect",
    "08291816_scan_k_defect",
]
POWERS_MW = [72, 77, 82, 88, 100, 104]

# Scan of Fig. 2 (a different data set from Fig. 4; to be confirmed by the authors).
SCAN_FIG2 = "08301817_scan_k_defect"

# Scans shown in Fig. 3 (indices into SCANS_FIG4: sigma/xi = 4.3, 5.8, 6.4).
SCANS_FIG3 = [SCANS_FIG4[0], SCANS_FIG4[3], SCANS_FIG4[5]]

# sigma / xi of the Fig. 2 run, from ``scripts/impurity_radius.py`` (same method as Fig. 1d)
SIGMA_FIG2 = 5.63

# Camera glitches, identified by eye (see ``bad_points.py``). These frames are set to
# NaN by ``observables`` and therefore dropped from every analysis. Only the
# Fig. 2 scan has a list; no bad frames are recorded for the six Fig. 4 scans.
BAD_FRAMES = {
    SCAN_FIG2: [38, 56, 64, 70, 95],
}

# Vortex detection (see ``scan_k_tur_setup.vortices``)
VORTEX_PLAQUETTE_RADIUS = 5  # px
VORTEX_WINDOW = 1.5e-3  # m, half-width of the counting window around the impurity
# Detections inside the impurity core (phase undefined there) are rejected: the density
# smoothed over VORTEX_RHO_SMOOTH pixels must exceed VORTEX_RHO_MIN times the background.
# Smoothing is wide enough to keep the impurity hole but not the small vortex cores.
VORTEX_RHO_MIN = 0.1
VORTEX_RHO_SMOOTH = 8  # px
