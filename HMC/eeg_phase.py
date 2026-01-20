"""Compute EEG phase-estimation errors for ecHT.

This script evaluates the online phase output of the Endpoint-Corrected Hilbert
Transform (ecHT / ECHT) against an acausal reference (band-pass + Hilbert).

It supports two modes:

1) Offline IAF
   - Crop each recording from start until the first sleep-stage change
   - Estimate the individual alpha frequency (IAF) once on an initial segment
   - Run ecHT online on the full cropped segment

2) Online IAF
   - Split the cropped segment into non-overlapping windows
   - Estimate IAF in each window from the previous window
   - Use that IAF to run ecHT online in the current window

Results are saved as:
  - a CSV with per-file circular statistics
  - an NPZ with concatenated phase-error samples (deg)

Notes
-----
- Sleep-stage change time is determined from a companion
  *_sleepscoring.edf file that contains annotations.
- The output phase error is defined as the circular difference
  angle(z_echt * conj(z_hilbert)) in degrees.
"""

from joblib import Parallel, delayed
from pathlib import Path

import argparse
import csv

import numpy as np

import mne
from philistine.mne import savgol_iaf
from scipy.signal import hilbert, butter, filtfilt

from utils import _circ_stats

# ---------------------------------------------------------------------
# Import ECHT from phase.py
# ---------------------------------------------------------------------
import importlib.util

_file = Path(__file__).resolve().parent.parent / "phase.py"
_spec = importlib.util.spec_from_file_location("ECHT", _file)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ECHT = _module.ECHT

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
EDF_DIR = Path("/home/eike/PycharmProjects/hori/HMC/original")
EXCLUDE_SUBJECTS = {"SN026", "SN032", "SN049"}
CSV_PATH = "phase_error_per_file.csv"
NPZ_PATH = "phase_error_all.npz"
CHANNEL_NAME = "EEG O2-M1"
N_JOBS = -1

mne.set_log_level("ERROR")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _is_sleep_stage(description, ignore_prefixes):
    """Heuristic: treat any annotation not starting with ignore_prefixes as a stage."""
    return not description.startswith(ignore_prefixes)


def get_first_stage_change_end(scoring_path):
    """
    Load sleep scoring annotations and return time (in s) of first
    change in sleep stage from the start of the recording.

    Assumes scoring EDF is annotations-only (no signals).
    """
    ann = mne.read_annotations(scoring_path)
    if len(ann) == 0:
        raise RuntimeError("No annotations found in sleep scoring file.")

    # Sort annotations by onset
    order = np.argsort(ann.onset)
    onset = np.array(ann.onset)[order]
    desc = np.array(ann.description)[order]

    # ignore purely technical annotations
    ignore_prefixes = ("Recording", "Lights", "Marker")

    # Find first sleep-stage annotation
    first_idx = None
    for i, d in enumerate(desc):
        if _is_sleep_stage(d, ignore_prefixes):
            first_idx = i
            break
    if first_idx is None:
        raise RuntimeError("No sleep-stage annotations found in scoring file.")
    first_desc = desc[first_idx]

    # Find first different sleep stage after the first one
    change_idx = None
    for i in range(first_idx + 1, len(desc)):
        if _is_sleep_stage(desc[i], ignore_prefixes) and desc[i] != first_desc:
            change_idx = i
            break

    if change_idx is None:
        # No change -> use end of last annotation
        end_time = onset[-1] + ann.duration[order][-1]
    else:
        # First time the stage changes
        end_time = onset[change_idx]

    return float(end_time)


def _fail(edf_path, reason, had_alpha = False):
    return {
        "edf_path": str(edf_path),
        "ok": False,
        "had_alpha": bool(had_alpha),
        "reason": reason,
        "phase_err_unc": None,
        "phase_err_cal": None,
    }


def _load_cropped_channel(
    edf_path, channel_name,
):
    """
    Load EDF, pick a single channel, and crop from start until first sleep-stage change.
    Returns: (signal_1d, sfreq, info_for_rawarray)
    """
    scoring_path = edf_path.with_name(edf_path.stem + "_sleepscoring.edf")
    if not scoring_path.exists():
        raise FileNotFoundError(f"Scoring file not found: {scoring_path}")

    t_segment_end = get_first_stage_change_end(str(scoring_path))

    raw = mne.io.read_raw_edf(str(edf_path), preload=True)
    raw.pick([channel_name])
    fs = float(raw.info["sfreq"])

    # Ensure we don't exceed actual recording duration
    t_segment_end = min(t_segment_end, raw.times[-1])
    if t_segment_end <= 0:
        raise RuntimeError("Segment end <= 0 s (no usable data).")

    raw_seg = raw.copy().crop(tmin=0, tmax=t_segment_end)
    x = raw_seg.get_data().squeeze()  # (n_times,)
    info = raw_seg.info.copy()
    return x, fs, info


def _estimate_paf_from_data(
    data_1d, info,
):
    """
    Estimate Peak Alpha Frequency (PAF) from 1D data using savgol_iaf.
    Returns float PAF (Hz) or None if unavailable/invalid.
    """
    raw_tmp = mne.io.RawArray(data_1d[np.newaxis, :], info)
    try:
        iaf = savgol_iaf(raw_tmp)
    except Exception:
        return None

    paf = getattr(iaf, "PeakAlphaFrequency", None)
    if paf is None:
        return None

    paf = float(paf)
    if (not np.isfinite(paf)) or paf <= 0:
        return None
    return paf


def _params_from_f0(fs, f0):
    """
    Given f0, compute:
      - win_len (samples): 2 * int(fs / f0)
      - l_freq/h_freq: +/- 0.25*f0 (since bw_rel=0.5*f0)
    """
    win_len = 2 * int(fs / f0)

    bw_rel = f0 * 0.5
    l_freq = f0 - bw_rel / 2
    h_freq = f0 + bw_rel / 2
    if l_freq <= 0:
        l_freq = 0.1
    return win_len, l_freq, h_freq


def _run_echt_vs_hilbert(
    data_1d,
    fs,
    filt_order,
    f0,
    l_freq,
    h_freq,
    win_len,
):
    """
    Run ecHT "online" on data_1d (uncalibrated + calibrated) and compare to
    band-pass + Hilbert reference. Returns phase error arrays in degrees:
      (phase_err_unc_deg, phase_err_cal_deg)
    """
    n = data_1d.size
    if win_len < 3 or win_len >= n:
        raise ValueError(f"invalid window length (win_len={win_len}, N={n})")

    echt_uncal = ECHT(
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=fs,
        n_fft=None,
        filt_order=filt_order,
        calibrate=False,
    )
    echt_cal = ECHT(
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=fs,
        n_fft=None,
        filt_order=filt_order,
        f0=f0,
        calibrate=True,
    )

    # Initialize ecHT with first window
    first_win = data_1d[:win_len]
    echt_uncal.fit(first_win)
    echt_cal.fit(first_win)

    # Reference analytic signal via band-pass + Hilbert
    b, a = butter(filt_order, [l_freq, h_freq], fs=fs, btype="band")
    xh = filtfilt(b, a, data_1d)
    analytic_hilbert = hilbert(xh)

    analytic_uncal = np.zeros(n, dtype=np.complex128)
    analytic_cal = np.zeros(n, dtype=np.complex128)

    # Online ecHT over the segment
    start_idx = win_len - 1
    for i in range(start_idx, n):
        w = data_1d[i - win_len + 1 : i + 1]
        analytic_uncal[i] = echt_uncal.transform(w)[-1, 0]
        analytic_cal[i] = echt_cal.transform(w)[-1, 0]

    z_hilbert = analytic_hilbert[start_idx:]
    z_unc = analytic_uncal[start_idx:]
    z_cal = analytic_cal[start_idx:]

    phase_err_unc = np.degrees(np.angle(z_unc * np.conj(z_hilbert)))
    phase_err_cal = np.degrees(np.angle(z_cal * np.conj(z_hilbert)))
    return phase_err_unc, phase_err_cal


def compute_phase_errors_for_file(
    edf_path,
    channel_name = CHANNEL_NAME,
    online_iaf = False,
    online_iaf_window = 60,
    offline_iaf_window = 0,
):
    """
    Returns a dict with:
      - ok: bool
      - had_alpha: bool
      - reason: str
      - phase_err_unc / phase_err_cal: np.ndarray (deg) or None
    """
    mne.set_log_level("ERROR")
    edf_path = Path(edf_path)

    try:
        x, fs, info = _load_cropped_channel(edf_path, channel_name)
        n_total = x.size
        if n_total < 10:
            return _fail(edf_path, f"error: segment too short (N={n_total})")

        filt_order = 1

        # ============================================================
        # Branch 1: ONLINE IAF (window-wise)
        # ============================================================
        if online_iaf:
            window_samples = int(round(fs * online_iaf_window))
            if window_samples <= 0:
                return _fail(edf_path, "error: invalid online_iaf_window (<= 0 s)")

            n_windows = n_total // window_samples
            if n_windows < 2:
                return _fail(
                    edf_path,
                    "error: not enough full windows (< 2) before stage change",
                )

            phase_err_unc_list: list[np.ndarray] = []
            phase_err_cal_list: list[np.ndarray] = []
            had_any_iaf = False

            for win_idx in range(1, n_windows):
                prev = x[(win_idx - 1) * window_samples : win_idx * window_samples]
                curr = x[win_idx * window_samples : (win_idx + 1) * window_samples]
                if curr.size == 0:
                    break

                paf = _estimate_paf_from_data(prev, info)
                if paf is None:
                    continue
                had_any_iaf = True

                win_len, l_freq, h_freq = _params_from_f0(fs, paf)
                if win_len < 3 or win_len >= curr.size:
                    continue

                try:
                    pe_unc, pe_cal = _run_echt_vs_hilbert(
                        curr, fs, filt_order, paf, l_freq, h_freq, win_len
                    )
                except Exception:
                    continue

                phase_err_unc_list.append(pe_unc)
                phase_err_cal_list.append(pe_cal)

            if (not had_any_iaf) or (len(phase_err_unc_list) == 0):
                return _fail(edf_path, "No alpha: no valid window-wise IAF", had_alpha=False)

            phase_err_unc_all = np.concatenate(phase_err_unc_list)
            phase_err_cal_all = np.concatenate(phase_err_cal_list)

        # ============================================================
        # Branch 2: SINGLE IAF on full segment (offline behaviour)
        # ============================================================
        else:
            if offline_iaf_window is not None and offline_iaf_window > 0:
                n_iaf = int(min(n_total, round(fs * offline_iaf_window)))
                if n_iaf < 10:
                    return _fail(
                        edf_path,
                        f"error: offline IAF window too short (N={n_iaf})",
                    )
                iaf_data = x[:n_iaf]
            else:
                iaf_data = x

            # Keep the "no_alpha:" wording so your aggregation logic stays the same
            try:
                paf = _estimate_paf_from_data(iaf_data, info)
            except Exception as e:
                return _fail(edf_path, f"no_alpha: {e}", had_alpha=False)

            if paf is None:
                return _fail(edf_path, "no_alpha: no valid PAF", had_alpha=False)

            win_len, l_freq, h_freq = _params_from_f0(fs, paf)
            if win_len < 3 or win_len >= n_total:
                return _fail(
                    edf_path,
                    f"error: invalid window length (win_len={win_len}, N={n_total})",
                    had_alpha=False,
                )

            try:
                phase_err_unc_all, phase_err_cal_all = _run_echt_vs_hilbert(
                    x, fs, filt_order, paf, l_freq, h_freq, win_len
                )
            except Exception as e:
                return _fail(edf_path, f"error: {e}", had_alpha=True)

        return {
            "edf_path": str(edf_path),
            "ok": True,
            "had_alpha": True,
            "reason": "",
            "phase_err_unc": phase_err_unc_all,
            "phase_err_cal": phase_err_cal_all,
        }

    except Exception as e:
        return _fail(edf_path, f"error: {e}", had_alpha=False)


def main(
    online_iaf = False,
    online_iaf_window = 60,
    offline_iaf_window = 300,
    edf_dir = EDF_DIR,
    csv_path = CSV_PATH,
    npz_path = NPZ_PATH,
):
    # ----------------------------------------------------------------
    # 1) Collect EDF files (exclude *_sleepscoring.edf)
    # ----------------------------------------------------------------
    edf_dir = Path(edf_dir)
    edf_files = sorted(
        p
        for p in edf_dir.glob("*.edf")
        if not p.name.endswith("_sleepscoring.edf")
        and not any(subj in p.name for subj in EXCLUDE_SUBJECTS)
    )

    if not edf_files:
        print(f"No EDF files found in {edf_dir}")
        return

    print(f"Found {len(edf_files)} EDF files in {edf_dir}")
    print(f"Online IAF adaptation: {'ON' if online_iaf else 'OFF'}")
    if online_iaf:
        print(f"  Online IAF window: {online_iaf_window:.2f} s")
    else:
        if offline_iaf_window is not None and offline_iaf_window > 0:
            print(
                f"  Offline IAF estimation window: {offline_iaf_window:.2f} s (from start)"
            )
        else:
            print("  Offline IAF estimation window: full available segment")

    # ----------------------------------------------------------------
    # 2) Run in parallel
    # ----------------------------------------------------------------
    n_jobs = N_JOBS

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
    )(
        delayed(compute_phase_errors_for_file)(
            str(p),
            CHANNEL_NAME,
            online_iaf,
            online_iaf_window,
            offline_iaf_window,
        )
        for p in edf_files
    )

    for res in results:
        print(f"Finished: {Path(res['edf_path']).name} -> {res['reason'] or 'OK'}")

    # ----------------------------------------------------------------
    # 3) Aggregate phase errors across all valid files
    # ----------------------------------------------------------------
    all_phase_err_unc = []
    all_phase_err_cal = []
    no_alpha_files = []
    error_files = []
    per_file_stats = []

    for res in results:
        if not res["ok"]:
            if "no_alpha" in res["reason"]:
                no_alpha_files.append((res["edf_path"], res["reason"]))
            else:
                error_files.append((res["edf_path"], res["reason"]))
            continue

        all_phase_err_unc.append(res["phase_err_unc"])
        all_phase_err_cal.append(res["phase_err_cal"])

        # Per-file statistics
        pe_unc_deg = np.asarray(res["phase_err_unc"])
        pe_cal_deg = np.asarray(res["phase_err_cal"])

        pe_unc_rad = np.radians(pe_unc_deg)
        pe_cal_rad = np.radians(pe_cal_deg)

        mean_unc_rad, std_unc_rad, plv_unc, pli_unc = _circ_stats(pe_unc_rad)
        mean_cal_rad, std_cal_rad, plv_cal, pli_cal = _circ_stats(pe_cal_rad)

        per_file_stats.append(
            {
                "file": Path(res["edf_path"]).name,
                "n_samples": pe_unc_deg.size,
                "mean_unc_deg": np.degrees(mean_unc_rad),
                "std_unc_deg": np.degrees(std_unc_rad),
                "plv_unc": plv_unc,
                "mean_cal_deg": np.degrees(mean_cal_rad),
                "std_cal_deg": np.degrees(std_cal_rad),
                "plv_cal": plv_cal,
            }
        )

    print("\n=== Summary ===")
    print(f"Total EDF files found:         {len(edf_files)}")
    print(f"Files with usable alpha/phase: {len(all_phase_err_unc)}")
    print(f"Files without alpha (ignored): {len(no_alpha_files)}")
    print(f"Files with other errors:       {len(error_files)}")

    if no_alpha_files:
        print("\nFiles without identifiable alpha (savgol_iaf):")
        for path, reason in no_alpha_files:
            print(f"  - {Path(path).name}: {reason}")

    if error_files:
        print("\nFiles that failed for other reasons:")
        for path, reason in error_files:
            print(f"  - {Path(path).name}: {reason}")

    if not all_phase_err_unc:
        print("\nNo valid phase error data to save. Exiting.")
        return

    # Concatenate all phase errors (degrees)
    phase_err_unc_deg_all = np.concatenate(all_phase_err_unc)
    phase_err_cal_deg_all = np.concatenate(all_phase_err_cal)

    # ----------------------------------------------------------------
    # 3b) Save per-file stats to CSV
    # ----------------------------------------------------------------
    if per_file_stats:
        fieldnames = [
            "file",
            "n_samples",
            "mean_unc_deg",
            "std_unc_deg",
            "plv_unc",
            "mean_cal_deg",
            "std_cal_deg",
            "plv_cal",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_file_stats)

        print(f"\nPer-file phase statistics written to: {csv_path}")
    else:
        print("\nNo per-file statistics to write (no valid alpha/phase data).")

    # ----------------------------------------------------------------
    # 3c) Save aggregated phase errors to NPZ
    # ----------------------------------------------------------------
    np.savez(
        npz_path,
        phase_err_unc_deg_all=phase_err_unc_deg_all,
        phase_err_cal_deg_all=phase_err_cal_deg_all,
    )
    print(f"Aggregated phase errors saved to: {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute phase errors for EEG EDFs (with optional online IAF)."
    )
    parser.add_argument(
        "--online-iaf",
        action="store_true",
        default=False,
        help="Use minute-wise online IAF adaptation (as in *_online_IAF.py).",
    )
    parser.add_argument(
        "--online-iaf-window",
        type=float,
        default=60,
        help="Window length in seconds for online IAF estimation/adaptation (default: 60).",
    )
    parser.add_argument(
        "--offline-iaf-window",
        type=float,
        default=300,
        help=(
            "Window length in seconds for offline (single) IAF estimation. "
            "If <= 0, use the full segment up to the first stage change. "
            "Default: 300 s (5 minutes)."
        ),
    )
    parser.add_argument(
        "--edf-dir",
        type=str,
        default=str(EDF_DIR),
        help="Directory with EDF files.",
    )

    args = parser.parse_args()

    main(
        online_iaf=args.online_iaf,
        online_iaf_window=args.online_iaf_window,
        offline_iaf_window=args.offline_iaf_window,
    )
