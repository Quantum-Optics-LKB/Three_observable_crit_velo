# Critical velocity of a paraxial superfluid of light past a mobile impurity

Analysis code accompanying

> T. Aladjidi, M. Baker-Rasooli, T. D. Ferreira, A. Bramati, M. Albert, Q. Glorieux, and P.-É. Larré,
> *Three-observable measurement of the critical velocity in a paraxial superfluid of light flowing past a mobile impurity*
> (SciPost Physics, submitted).

The critical Mach number `beta_c` of the superfluid-to-dissipative transition is extracted from three observables
measured simultaneously on the same runs:

1. the number of vortex-antivortex pairs nucleated in the wake of the impurity,
2. the hydrodynamic force exerted on the impurity,
3. the displacement velocity of the impurity relative to the flow,

for six impurity radii `sigma/xi` between 4.3 and 6.4, plus one run from a different data set (Fig. 2 of the paper).
The code recomputes `beta_c` with the same explicit criterion for all runs, propagates the uncertainties, and redraws Figs. 3 and 4.

## Repository layout

| Path | Content |
|---|---|
| `supervelo/` | Analysis package (see below) |
| `scripts/` | Command-line entry points |
| `data/pot_size.txt` | Impurity radii `sigma/xi` of the six runs (Fig. 1d) |
| `results/` | Tables written by `scripts/run_analysis.py` (`beta_c` and uncertainties) |
| `legacy/` | Original analysis scripts, kept for reference only (lab-specific paths, not runnable as is) |
| `Data_defect/` | Experimental data, **not included** (about 24 GB, see below) |

Modules of `supervelo/`:

| Module | Role |
|---|---|
| `params.py` | Constants, scan lists, bad camera frames, alignment of the impurity image |
| `observables.py` | Mach number, impurity velocity, hydrodynamic force, vortex detection and counting |
| `thresholds.py` | Onset criteria for the three observables |
| `uncertainty.py` | Propagation of the uncertainty on `beta_c` |
| `potential.py` | Impurity radius from the reconstructed potential (Bernoulli equation) |
| `figures.py` | Figures 2, 3 and 4 |
| `velocity.py` | Phase unwrapping, velocity field and vortex detection routines |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

`numba` and `pyfftw` are needed by `velocity.py` (vortex detection); `cupy` is optional (GPU).

## Data

The raw and processed data are not versioned (about 24 GB). Place them in `Data_defect/`, one folder per run:

```
Data_defect/<MMDDhhmm>_scan_k_defect/
    kx_fluid.npy, kx_defect.npy       tilt of the fluid and impurity beams for each frame
    cs_quench.npy                     sound velocity
    field_fluid.npy, field_fluid_ref.npy, field_defect.npy    reconstructed complex fields
```

Runs used: `08291714`, `08291746`, `08291752`, `08291758`, `08291809`, `08291816` (Figs. 1d, 3, 4; impurity power 72, 77, 82, 88, 100, 104 mW)
and `08301817` (Fig. 2, a different data set). The data are available from the authors on request.

## Usage

Run from the repository root.

```
python scripts/run_analysis.py --data-root Data_defect          # beta_c of the three observables, Figs. 2-4
python scripts/impurity_radius.py --data-root Data_defect       # sigma/xi of each run (checks Fig. 1d)
python scripts/velocity_sensitivity.py --data-root Data_defect  # dependence of the velocity onset on its threshold
python scripts/inspect_vortices.py --scan 08291816_scan_k_defect --beta-range 0 0.35   # visual check of vortex detections
```

Outputs:

- `results/critical_betas.csv`: `beta_c` for each run and observable, with the three components of its uncertainty;
- `results/critical_betas_fig2_run.csv`: same for the run of Fig. 2;
- `figures/fig2.png`, `figures/fig3.{png,pdf}`, `figures/fig4.{png,pdf}`.

The drag force and the vortex counts are slow to compute; they are cached in `cache/<run>/` (not versioned).

## Criteria for the critical Mach number

The same three-successive-points rule is used for all observables (`supervelo/thresholds.py`):

| Observable | `beta_c` is the first beta at which... |
|---|---|
| Vortex count | a vortex-antivortex pair is detected in two successive images (`beta >= 0.05`), outside the impurity core (smoothed density below 10 % of the background) |
| Hydrodynamic force | the force along the flow exceeds its sub-critical level (mean for `beta < 0.2`) by 10 % of its total rise, for three successive points |
| Impurity velocity | `Delta v_i / c_s - beta` (3-point median filter) exceeds its local sub-critical level (median over `beta - 0.25 ... beta - 0.03`) by `0.04 c_s`, for three successive points |

Camera glitches listed in `supervelo/params.py` (`BAD_FRAMES`) are ignored.
