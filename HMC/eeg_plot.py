"""
Three-panel figure of HMC EEG phase estimate:

(A) Uncalibrated phase error distribution (polar)
(B) Calibrated phase error distribution (polar)
(C) Mean phase error (uncal & cal) vs. IAF CV per recording,
    with paired points and regression lines.

Expected inputs
---------------
1) NPZ file containing arrays (degrees):
   - phase_err_unc_deg_all
   - phase_err_cal_deg_all

2) CSV with per-recording phase-error means (degrees):
   - required: file, mean_unc_deg, mean_cal_deg
   - optional: a weight column (e.g., n_windows, n_samples, ...)

3) CSV with per-segment IAF estimates used to compute CV per recording:
   - required: file, segment_index, had_alpha, paf_hz
"""

import numpy as np
import pandas as pd

from utils import (
    make_figure,
)

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

NPZ_PATH = "phase_error_all.npz"
PHASE_CSV = "phase_error_per_file.csv"
IAF_CSV = "iaf_per_segment.csv"


# --------------------------------------------------------------------
# IAF helpers
# --------------------------------------------------------------------
def compute_iaf_cv_per_file(iaf_csv: str):
    """Compute IAF CV per recording from iaf_per_segment.csv."""
    df = pd.read_csv(iaf_csv)
    # Keep only valid segments with alpha and finite PAF
    mask_valid = (
        (df["segment_index"] >= 0)
        & (df["had_alpha"] == 1)
        & np.isfinite(df["paf_hz"])
    )
    df_valid = df.loc[mask_valid].copy()

    if df_valid.empty:
        raise RuntimeError("No valid segments with alpha found in iaf_per_segment.csv")

    grouped = df_valid.groupby("file")["paf_hz"]
    stats = grouped.agg(
        mean_iaf_hz="mean",
        std_iaf_hz="std",
        n_segments="count",
    )
    stats["cv_iaf"] = stats["std_iaf_hz"] / stats["mean_iaf_hz"]

    return stats


# --------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------
def plot_phase_error(
    phase_err_unc_deg_all,
    phase_err_cal_deg_all,
    phase_csv_path=PHASE_CSV,
    iaf_csv_path=IAF_CSV,
    save_base="phase_error_HMC",
):
    """
    Three-panel figure:
      (A) Uncalibrated phase-error distribution (polar)
      (B) Calibrated phase-error distribution (polar)
      (C) Mean phase error (uncalibrated & calibrated) vs IAF CV per recording,
          with paired points and regression lines, styled like the phase panels.
    """
    # ------------------------------------------------------------
    # Data for panels A & B (all samples)
    # ------------------------------------------------------------
    phase_err_unc_deg_all = np.asarray(phase_err_unc_deg_all)
    phase_err_cal_deg_all = np.asarray(phase_err_cal_deg_all)

    phase_err_unc_rad = np.radians(phase_err_unc_deg_all)
    phase_err_cal_rad = np.radians(phase_err_cal_deg_all)

    # For the significance annotation between panels A & B
    phase_df_for_perm = pd.read_csv(phase_csv_path)

    # per-file circular mean directions (in radians)
    if "mean_unc_deg" not in phase_df_for_perm.columns or "mean_cal_deg" not in phase_df_for_perm.columns:
        raise KeyError(
            "phase_error_per_file.csv must contain columns 'mean_unc_deg' and 'mean_cal_deg' "
            "to run the paired circular permutation test."
        )
    trial_mu_unc = np.radians(phase_df_for_perm["mean_unc_deg"].to_numpy(dtype=float))
    trial_mu_cal = np.radians(phase_df_for_perm["mean_cal_deg"].to_numpy(dtype=float))

    # weights: try a few common column names; fall back to equal weights
    weight_candidates = [
        "n_windows",
        "n_win",
        "n_samples",
        "n_segments",
        "n",
        "count",
    ]
    w_col = next((c for c in weight_candidates if c in phase_df_for_perm.columns), None)
    if w_col is None:
        trial_nwin = np.ones_like(trial_mu_unc, dtype=float)
    else:
        trial_nwin = phase_df_for_perm[w_col].to_numpy(dtype=float)


    # ------------------------------------------------------------
    # Data for panel C: per-file mean errors vs IAF CV
    # ------------------------------------------------------------
    phase_df = pd.read_csv(phase_csv_path)

    iaf_stats = compute_iaf_cv_per_file(iaf_csv_path)
    # Merge to CV
    merged = phase_df.merge(
        iaf_stats[["cv_iaf", "mean_iaf_hz", "std_iaf_hz", "n_segments"]],
        on="file",
        how="inner",
    )

    if merged.empty:
        raise RuntimeError(
            "No overlapping recordings between phase_error_per_file.csv "
            "and iaf_per_segment.csv after merging on 'file'."
        )

    x_cv = merged["cv_iaf"].values
    y_unc = merged["mean_unc_deg"].values
    y_cal = merged["mean_cal_deg"].values


    # Plot
    make_figure(
        err_unc_rad=phase_err_unc_rad,
        err_cal_rad=phase_err_cal_rad,
        trial_freq_cv=x_cv,
        trial_abs_unc_rad=np.radians(y_unc),
        trial_abs_cal_rad=np.radians(y_cal),
        trial_mu_unc=trial_mu_unc,
        trial_mu_cal=trial_mu_cal,
        trial_nwin=trial_nwin,
        save_base=save_base,
        n_perm=int(1e5),
        perm_seed=0,
        panel_c_xlabel="IAF coefficient of variation (CV)",
        panel_c_title=r"$\mathbf{(C)}$ Phase error vs. IAF variability",
    )


def main(npz_path=NPZ_PATH, phase_csv_path=PHASE_CSV, iaf_csv_path=IAF_CSV):
    data = np.load(npz_path)
    if "phase_err_unc_deg_all" not in data.files or "phase_err_cal_deg_all" not in data.files:
        raise KeyError(
            "NPZ file must contain 'phase_err_unc_deg_all' and 'phase_err_cal_deg_all'."
        )

    phase_err_unc_deg_all = data["phase_err_unc_deg_all"]
    phase_err_cal_deg_all = data["phase_err_cal_deg_all"]

    print(f"Loaded {phase_err_unc_deg_all.size} uncalibrated samples "
          f"and {phase_err_cal_deg_all.size} calibrated samples from {npz_path}")

    plot_phase_error(
        phase_err_unc_deg_all,
        phase_err_cal_deg_all,
        phase_csv_path=phase_csv_path,
        iaf_csv_path=iaf_csv_path,
    )

if __name__ == "__main__":
    main()
