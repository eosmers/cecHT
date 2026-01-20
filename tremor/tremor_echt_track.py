"""
Tremor ecHT plot w/ tracked f0:
(A) Uncalibrated phase error distribution (polar)
(B) Calibrated phase error distribution (polar)
(C) Mean |phase error| vs tremor-frequency CV per trial (paired points + regressions)

Also:
- Significance A vs B shown as a bracket/bar like plot.py
- p-value from paired circular permutation (within-trial label swap)
"""

import sys
from pathlib import Path
import importlib.util

import numpy as np
from scipy.stats import circmean, circstd
from scipy.signal import hilbert, butter, sosfiltfilt

from tremor_echt import collect_trials, compute_endpoint_errors

from utils import (
    _wrap_phase,
    make_figure
)


# ---------------------------------------------------------------------
# Import ecHT implementation
# ---------------------------------------------------------------------
file_path = Path(__file__).resolve().parent.parent / "phase_track.py"
spec = importlib.util.spec_from_file_location("ECHT", file_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
ECHT = module.ECHT


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _estimate_f0(seg: np.ndarray, fs: float, f_min: float, f_max: float):
    """Estimate dominant frequency in [f_min, f_max] using rFFT + parabolic peak."""
    x = np.asarray(seg, float).ravel()
    if x.size < 8:
        return np.nan

    # Window to reduce leakage
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)

    band = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(band):
        return np.nan

    mag = np.abs(X[band])
    if mag.size < 3:
        return np.nan

    k0 = int(np.argmax(mag))
    freqs_b = freqs[band]

    # Parabolic interpolation around the peak (in magnitude domain)
    if 0 < k0 < (mag.size - 1):
        y1, y2, y3 = mag[k0 - 1], mag[k0], mag[k0 + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom  # in bins
            bin_hz = freqs_b[1] - freqs_b[0]
            return float(freqs_b[k0] + delta * bin_hz)

    return float(freqs_b[k0])


# ---------------------------------------------------------------------
# Per-trial processing
# ---------------------------------------------------------------------
def process_one_trial(
    x,
    phi_ds,
    fs,
    f0,
    N=256,
    filt_order=2,
    freq_win_len=2048,
    freq_stride=2048,
):
    """
    Returns:
      err_unc, err_cal             : per-window endpoint errors (paired)
      freq_cv                      : trial tremor frequency CV (FFT-based)
      trial_abs_unc, trial_abs_cal : per-trial mean |error| (rad)
      trial_mu_unc, trial_mu_cal   : per-trial circular mean error (rad)
      n_windows                    : number of windows
    """
    L = len(x)
    if L < N:
        return (np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    l_freq = max(0.1, f0 - f0 / 2.0)
    h_freq = min(0.5 * fs - 0.1, f0 + f0 / 2.0)
    if not (0.0 < l_freq < h_freq < 0.5 * fs):
        return (np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    sos = butter(filt_order, [l_freq, h_freq], fs=fs, btype="bandpass", output="sos")
    x_filt = sosfiltfilt(sos, x)

    z_offline = hilbert(x_filt)
    phi_offline = np.angle(z_offline)

    L = min(len(x_filt), len(phi_offline), len(phi_ds))
    x_filt = x_filt[:L]
    phi_offline = phi_offline[:L]

    first_seg = x_filt[:N]

    echt_unc = ECHT(
        l_freq=l_freq, h_freq=h_freq, sfreq=fs, filt_order=filt_order,
        calibrate=False, f0=None,
        bandpass_tracking=False, bandpass_update_mode="threshold"
    )
    echt_unc.fit(first_seg)

    echt_cal = ECHT(
        l_freq=l_freq, h_freq=h_freq, sfreq=fs, filt_order=filt_order,
        calibrate=True, f0=f0,
        bandpass_tracking=True, bandpass_update_mode="threshold"
    )
    echt_cal.fit(first_seg)

    out_len = L - (N - 1)
    if out_len <= 0:
        return (np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    err_unc = np.empty(out_len, dtype=float)
    err_cal = np.empty(out_len, dtype=float)

    f0_track = float(f0)
    alpha = 0.1
    last_f0_hat = np.nan

    k = 0
    for end_idx in range(N - 1, L):
        # ecHT window
        start_idx = end_idx - N + 1
        seg_echt = x_filt[start_idx:end_idx + 1]  # length N

        # --- f0 tracking buffer ---
        # Update only every freq_stride samples to reduce compute
        if ((end_idx - (N - 1)) % freq_stride) == 0:
            start_f0 = max(0, end_idx - freq_win_len + 1)
            seg_f0 = x_filt[start_f0:end_idx + 1]  # up to freq_win_len

            # Track in a band around the current estimate
            f_min = max(0.1, f0_track - 0.5 * f0_track)
            f_max = min(0.5 * fs - 0.1, f0_track + 0.5 * f0_track)

            last_f0_hat = _estimate_f0(seg_f0, fs, f_min, f_max)

        if np.isfinite(last_f0_hat):
            f0_track = (1 - alpha)*f0_track + alpha * float(last_f0_hat)

        zu = np.squeeze(np.asarray(echt_unc.transform(seg_echt, f0=f0_track)))
        zc = np.squeeze(np.asarray(echt_cal.transform(seg_echt, f0=f0_track)))

        phi_unc_end = float(np.asarray(np.angle(zu[-1])).item())
        phi_cal_end = float(np.asarray(np.angle(zc[-1])))
        phi_true_end = float(np.asarray(phi_offline[end_idx]))

        err_unc[k] = _wrap_phase(phi_unc_end - phi_true_end)
        err_cal[k] = _wrap_phase(phi_cal_end - phi_true_end)
        k += 1

    trial_abs_unc = float(np.mean((err_unc)))
    trial_abs_cal = float(np.mean((err_cal)))
    trial_mu_unc = float(circmean(err_unc, high=np.pi, low=-np.pi))
    trial_mu_cal = float(circmean(err_cal, high=np.pi, low=-np.pi))
    n_windows = int(err_unc.size)

    # Tremor frequency CV (dominant FFT frequency per window)
    freqs_win = []
    if L >= freq_win_len:
        for start in range(0, L - freq_win_len + 1, freq_stride):
            seg_f = x_filt[start:start + freq_win_len]
            Xf = np.fft.rfft(seg_f)
            freqs = np.fft.rfftfreq(len(seg_f), d=1.0 / fs)
            mask = (freqs >= 0.5) & (freqs <= 20.0)
            if not np.any(mask):
                continue
            mag = np.abs(Xf[mask])
            if mag.size == 0:
                continue
            freqs_win.append(float(freqs[mask][np.argmax(mag)]))

    if len(freqs_win) >= 2:
        freqs_win = np.asarray(freqs_win, dtype=float)
        mean_f = float(np.mean(freqs_win))
        std_f = float(np.std(freqs_win, ddof=1))
        freq_cv = (std_f / mean_f) if mean_f != 0.0 else np.nan
    else:
        freq_cv = np.nan

    return (err_unc, err_cal, freq_cv, trial_abs_unc, trial_abs_cal, trial_mu_unc, trial_mu_cal, n_windows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        mat_paths = sys.argv[1:]
    else:
        mat_paths = [
            "data_sbj_original_2016.mat",
            # "data_sbj_repeated_2019.mat",
            # "data_sbj_new_2019.mat",
        ]

    print("Using the following .mat files (non-existing ones will be skipped):")
    for p in mat_paths:
        print("  -", p)

    trials = collect_trials(mat_paths)
    print(f"Total trials collected from all selected files: {len(trials)}")

    if len(trials) == 0:
        print("No trials found. Check that the .mat files exist and have the expected structure.")
        return

    filt_order = 2
    N = 128

    (
        err_unc, err_cal,
        trial_freq_cv,
        trial_abs_unc, trial_abs_cal,
        trial_mu_unc, trial_mu_cal,
        trial_nwin,
    ) = compute_endpoint_errors(
        trials,
        process_fn=process_one_trial,
        filt_order=filt_order,
        N=N,
        n_jobs=-1,
        freq_win_len=2048,
        freq_stride=2048,
    )

    if err_unc.size == 0 or err_cal.size == 0:
        print("No windows produced any errors (check N and trial lengths).")
        return

    mean_unc = _wrap_phase(circmean(err_unc, high=np.pi, low=-np.pi))
    std_unc = circstd(err_unc)
    mean_cal = _wrap_phase(circmean(err_cal, high=np.pi, low=-np.pi))
    std_cal = circstd(err_cal)

    print("\nSummary of ecHT vs offline Hilbert reference")
    print(f"Mean uncalibrated error (deg): {np.degrees(mean_unc):.2f} ± {np.degrees(std_unc):.2f}")
    print(f"Mean calibrated error   (deg): {np.degrees(mean_cal):.2f} ± {np.degrees(std_cal):.2f}")

    make_figure(
        err_unc_rad=err_unc,
        err_cal_rad=err_cal,
        trial_freq_cv=trial_freq_cv,
        trial_abs_unc_rad=trial_abs_unc,
        trial_abs_cal_rad=trial_abs_cal,
        trial_mu_unc=trial_mu_unc,
        trial_mu_cal=trial_mu_cal,
        trial_nwin=trial_nwin,
        save_base="tremor_track",
        n_perm=int(1e5),
        perm_seed=0,
        panel_c_xlabel="Tremor frequency CV",
        panel_c_title=r"$\mathbf{(C)}$ Phase error vs. tremor variability",
    )


if __name__ == "__main__":
    main()
