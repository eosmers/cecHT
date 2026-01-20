"""
ECHT Latency Metrics
====================

Measure both algorithmic latency (phase-based effective delay) and
computational latency (wall-clock runtime) of the Endpoint-Corrected Hilbert
Transform (ECHT).

Computational latency
    Runtime is measured as wall-clock time per processed block, and also
    normalised as time per sample per channel.

Runtime comparisons include SciPy's hilbert as a baseline, using the same
FFT length as ECHT for a fair comparison.
"""

import numpy as np
import time
from scipy.signal import hilbert

# ---------------------------------------------------------------------
# Import ECHT from phase.py
# ---------------------------------------------------------------------
from pathlib import Path
import importlib.util
_file = Path(__file__).resolve().parent.parent / "phase.py"
_spec = importlib.util.spec_from_file_location("ECHT", _file)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ECHT = _module.ECHT


def wrap_phase(phi):
    """
    Wrap phase angle(s) to [-pi, pi].

    Parameters
    ----------
    phi : float or ndarray
        Input phase angle(s).

    Returns
    -------
    float or ndarray
        Wrapped angle(s).
    """
    return (phi + np.pi) % (2 * np.pi) - np.pi

# ---------------------------------------------------------------------
# Computational latency
# ---------------------------------------------------------------------
def benchmark_runtime(
    sfreq,
    l_freq,
    h_freq,
    N_values,
    C_values,
    filt_order=1,
    calibrate=True,
    n_runs=50,
    random_state=0,
):
    """
    Benchmark wall-clock time of ECHT vs SciPy Hilbert.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz).
    l_freq, h_freq : float
        Band-pass edges (Hz).
    N_values : sequence of int
        Block sizes (samples) to test.
    C_values : sequence of int
        Number of channels to test.
    filt_order : int
        IIR filter order.
    filter_type : str
        Filter type passed to ECHT (e.g., 'butter').
    calibrate : bool
        Whether to use endpoint calibration.
    n_runs : int
        Repetitions per configuration for averaging.
    random_state : int
        Seed for reproducible random data.

    Returns
    -------
    results : list of RuntimeResult
        Runtime metrics for each (N, C) configuration.
    """
    rng = np.random.RandomState(random_state)
    f0 = (l_freq + h_freq) / 2
    results = []

    for C in C_values:
        for N in N_values:
            X = rng.randn(N, C)

            # ECHT instance
            echt = ECHT(
                l_freq=l_freq,
                h_freq=h_freq,
                sfreq=sfreq,
                n_fft=None,
                filt_order=filt_order,
                calibrate=calibrate,
                f0=f0,
            )
            echt.fit(X)

            # Warm up
            for _ in range(3):
                _ = echt.transform(X)

            # Time ECHT
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = echt.transform(X)
            t1 = time.perf_counter()
            T_block_echt = (t1 - t0) / n_runs

            # Baseline SciPy Hilbert: use same FFT length as ECHT for fairness
            n_fft = echt.n_fft

            def hilbert_baseline(X_in):
                Y = hilbert(X_in, N=n_fft, axis=0)
                return Y[: X_in.shape[0], :]

            # Warm up
            for _ in range(3):
                _ = hilbert_baseline(X)

            # Time baseline
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = hilbert_baseline(X)
            t1 = time.perf_counter()
            T_block_hilb = (t1 - t0) / n_runs

            per_sample_per_chan_echt = T_block_echt / (N * C)
            ratio = T_block_echt / T_block_hilb if T_block_hilb > 0 else np.nan

            results.append(
                dict(
                    N=int(N),
                    C=int(C),
                    T_block_echt=T_block_echt,
                    T_block_hilb=T_block_hilb,
                    per_sample_per_chan_echt=per_sample_per_chan_echt,
                    ratio_echt_to_hilb=ratio,
                )
            )

    return results

def main():
    # ----------------- User-adjustable parameters -----------------
    sfreq = 1000      # Sampling frequency in Hz
    f0 = 12           # Central frequency for the band (Hz)
    bandwidth = 6     # Total bandwidth (Hz)
    l_freq = f0 - bandwidth / 2
    h_freq = f0 + bandwidth / 2
    filt_order = 1
    calibrate = True

    # N and C grids for runtime benchmark
    N_values = [256, 512, 1024, 2048, 4096, 8192, 16384]
    C_values = [1]
    n_runs = 10000
    # --------------------------------------------------------------

    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Band-pass: [{l_freq:.2f}, {h_freq:.2f}] Hz (order={filt_order})")
    print(f"Calibration: {calibrate} at f0={f0:.2f} Hz")

    # Runtime benchmark
    results = benchmark_runtime(
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        N_values=N_values,
        C_values=C_values,
        filt_order=filt_order,
        calibrate=calibrate,
        n_runs=n_runs,
    )

    print("\n=== Computational runtime (wall-clock) ===")
    print("N\tC\tT_echt_us\tT_hilb_us\tus_per_sample_chan\tE/H_ratio")
    for r in results:
        print(
            f"{r['N']:d}\t{r['C']:d}\t"
            f"{r['T_block_echt'] * 1e6: .3f}\t"
            f"{r['T_block_hilb'] * 1e6: .3f}\t"
            f"{r['per_sample_per_chan_echt'] * 1e6: .3f}\t"
            f"{r['ratio_echt_to_hilb']: .3f}"
        )

    ns_per = np.array([r["per_sample_per_chan_echt"] for r in results]) * 1e9

    print("\nSummary over all N:")
    print("  median ns per sample per channel (ECHT): "
          f"{np.median(ns_per):.3f} ns")
    print("  mean ns per sample per channel (ECHT): "
          f"{np.mean(ns_per):.3f} ns")
    print("  stdev ns per sample per channel (ECHT): "
          f"{np.std(ns_per):.3f} ns")
    print("  max ns per sample per channel (ECHT): "
          f"{np.max(ns_per):.3f} ns")
    print("  min ns per sample per channel (ECHT): "
          f"{np.min(ns_per):.3f} ns")


if __name__ == "__main__":
    main()
