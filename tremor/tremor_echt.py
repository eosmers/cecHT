"""
Tremor ecHT plot:
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
from scipy.io import loadmat
from scipy.stats import circmean, circstd
from scipy.signal import hilbert, butter, sosfiltfilt

from joblib import Parallel, delayed

from utils import (
    _wrap_phase,
    make_figure
)

# ---------------------------------------------------------------------
# Import ecHT implementation
# ---------------------------------------------------------------------
file_path = Path(__file__).resolve().parent.parent / "phase.py"
spec = importlib.util.spec_from_file_location("ECHT", file_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
ECHT = module.ECHT


# ---------------------------------------------------------------------
# Data loading (compatible with data_sbj_original_2016.mat)
# ---------------------------------------------------------------------
def iter_trials(mat_path):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data_all_res = mat["data_all_res"]

    for subj in np.ravel(data_all_res):
        for cond in np.ravel(subj):
            if cond is None:
                continue
            for trial in np.ravel(cond):
                if trial is None:
                    continue
                if not hasattr(trial, "ADC_in_raw") or trial.ADC_in_raw is None:
                    continue

                x = np.asarray(trial.ADC_in_raw, dtype=float).ravel()
                if x.size < 10:
                    continue

                dt_mean = float(np.asarray(trial.dt_mean).squeeze())
                fs = 1 / dt_mean
                f0 = float(np.asarray(trial.cal_freq).squeeze())

                phi_ds = np.asarray(trial.ADC_in_phase, dtype=float).ravel()

                L = min(len(x), len(phi_ds))
                yield x[:L], phi_ds[:L], fs, f0


def collect_trials(mat_paths):
    all_trials = []
    for mat_path in mat_paths:
        mat_path = Path(mat_path)
        if not mat_path.exists():
            print(f"[WARNING] File not found, skipping: {mat_path}")
            continue
        this_trials = list(iter_trials(mat_path))
        print(f"Collected {len(this_trials)} trials from {mat_path.name}")
        all_trials.extend(this_trials)
    return all_trials


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

    l_freq = max(0.1, f0 - f0 / 2)
    h_freq = min(0.5 * fs - 0.1, f0 + f0 / 2)
    if not (0 < l_freq < h_freq < 0.5 * fs):
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
        calibrate=False, f0=None
    )
    echt_unc.fit(first_seg)

    echt_cal = ECHT(
        l_freq=l_freq, h_freq=h_freq, sfreq=fs, filt_order=filt_order,
        calibrate=True, f0=f0,
    )
    echt_cal.fit(first_seg)

    out_len = L - (N - 1)
    if out_len <= 0:
        return (np.array([]), np.array([]), np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    err_unc = np.empty(out_len, dtype=float)
    err_cal = np.empty(out_len, dtype=float)

    k = 0
    for end_idx in range(N - 1, L):
        start_idx = end_idx - N + 1
        seg = x_filt[start_idx:end_idx + 1]

        zu = np.squeeze(np.asarray(echt_unc.transform(seg)))
        zc = np.squeeze(np.asarray(echt_cal.transform(seg)))

        phi_unc_end = np.angle(zu[-1])
        phi_cal_end = np.angle(zc[-1])
        phi_true_end = phi_offline[end_idx]

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
            freqs = np.fft.rfftfreq(len(seg_f), d=1 / fs)
            mask = (freqs >= 0.5) & (freqs <= 20)
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
        freq_cv = (std_f / mean_f) if mean_f != 0 else np.nan
    else:
        freq_cv = np.nan

    return (err_unc, err_cal, freq_cv, trial_abs_unc, trial_abs_cal, trial_mu_unc, trial_mu_cal, n_windows)


def compute_endpoint_errors(
    trials,
    process_fn=process_one_trial,
    filt_order=2,
    N=256,
    n_jobs=-1,
    freq_win_len=2048,
    freq_stride=2048,
):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_fn)(
            x, phi_ds, fs, f0, N, filt_order, freq_win_len, freq_stride
        )
        for (x, phi_ds, fs, f0) in trials
    )

    err_unc_all = []
    err_cal_all = []

    trial_freq_cv = []
    trial_abs_unc = []
    trial_abs_cal = []
    trial_mu_unc = []
    trial_mu_cal = []
    trial_nwin = []

    n_windows_total = 0

    for eu, ec, cv, absu, absc, muu, muc, nwin in results:
        if eu.size and ec.size:
            if eu.size != ec.size:
                m = min(eu.size, ec.size)
                eu = eu[:m]
                ec = ec[:m]
                nwin = m
            err_unc_all.append(eu)
            err_cal_all.append(ec)
            n_windows_total += int(eu.size)

        trial_freq_cv.append(cv)
        trial_abs_unc.append(absu)
        trial_abs_cal.append(absc)
        trial_mu_unc.append(muu)
        trial_mu_cal.append(muc)
        trial_nwin.append(nwin)

    err_unc_all = np.concatenate(err_unc_all) if err_unc_all else np.array([], dtype=float)
    err_cal_all = np.concatenate(err_cal_all) if err_cal_all else np.array([], dtype=float)

    trial_freq_cv = np.asarray(trial_freq_cv, dtype=float)
    trial_abs_unc = np.asarray(trial_abs_unc, dtype=float)
    trial_abs_cal = np.asarray(trial_abs_cal, dtype=float)
    trial_mu_unc = np.asarray(trial_mu_unc, dtype=float)
    trial_mu_cal = np.asarray(trial_mu_cal, dtype=float)
    trial_nwin = np.asarray(trial_nwin, dtype=int)

    print(f"compute_endpoint_errors: {n_windows_total} ecHT windows total")

    return (
        err_unc_all, err_cal_all,
        trial_freq_cv,
        trial_abs_unc, trial_abs_cal,
        trial_mu_unc, trial_mu_cal,
        trial_nwin
    )


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
        save_base="tremor_echt",
        n_perm=int(1e5),
        perm_seed=0,
        panel_c_xlabel="Tremor frequency CV",
        panel_c_title=r"$\mathbf{(C)}$ Phase error vs. tremor variability",
    )


if __name__ == "__main__":
    main()
