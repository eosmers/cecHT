"""
ecHT harmonic experiments: cosine-based simulations.

Panels:

  1) Bandwidth sweep
  2) Filter order sweep
  3) Frequency drift
  4) SNR sweep
  5) Filter-type comparison (Butterworth, Bessel, Cheby I/II, Elliptic)
  6) Window length sweep (wide range, non-integer cycles).

For all sweep-type plots we show:
  - mean absolute phase error (deg)
  - ±1 std of |error| as a shaded band around the mean

Uncalibrated ecHT is displayed in orange and calibrated ecHT (c-ecHT) in blue.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from concurrent.futures import ProcessPoolExecutor

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

# ---------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------
F0 = 10                # central frequency in Hz
SFREQ = 256            # sampling frequency in Hz (for most experiments)
BW_DEFAULT = F0 * 0.3  # default bandwidth (Hz)

N_PHASE_SAMPLES = 90            # Samples of initial phase
N_NOISE_TRIALS = int(1e4)       # Monte-Carlo trials for noise
N_CYCLES = 2.1
SMOOTH_WINDOW_CYCLES = 0
N_WINDOW_STEPS = 100

N_WORKERS = min(9, max(1, (os.cpu_count() or 2) - 1))

MASTER_RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------
blue = "#4E79A7"
orange = "#F28E2B"
red = "#E15759"
green = "#59A14F"
yellow = "#EDC948"
purple = "#B07AA1"
teal = "#76B7B2"
gray = "#BAB0AC"
global_alpha = 0.6

COL_UNCAL = orange
COL_CAL = blue

def set_mpl_style():
    """Global Matplotlib style"""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.1,
        "figure.dpi": 300,
        "text.usetex": True,
    })


COMMON_BBOX = dict(
    facecolor="white",
    edgecolor=gray,
    boxstyle="round,pad=0.2",
    alpha=0.9,
)


def add_panel_label(ax, label, text):
    """
    Panel label in the same style as intro diagram, e.g. '(A) Window length'.
    """
    ax.text(
        0.02,
        1.02,
        rf"\textbf{{({label})}} {text}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        # bbox=COMMON_BBOX,
    )

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def wrap_phase(phi):
    """Wrap phase to [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def angle_diff(phi1, phi2):
    """Smallest signed difference phi1 - phi2 in [-pi, pi]."""
    return wrap_phase(phi1 - phi2)


def generate_cosine_window(n_cycles, f0, sfreq, phi0=0.0):
    """
    Generate x[n] = cos(2π f0 t + phi0) with a given number of cycles.

    Returns
    -------
    t : ndarray
        Time vector (s).
    x : ndarray
        Cosine samples.
    true_phase_end : float
        True underlying phase at the last sample (rad).
    """
    duration = n_cycles / f0
    n_samples = int(np.round(duration * sfreq))
    t = np.arange(n_samples) / sfreq
    phase = 2 * np.pi * f0 * t + phi0
    x = np.cos(phase)
    true_phase_end = wrap_phase(phase[-1])
    return t, x, true_phase_end


def endpoint_phase_echt(
    x,
    f0,
    sfreq,
    bw=None,
    order=1,
    filter_type="butter",
    calibrate=False,
):
    """Endpoint phase using ECHT (with optional calibration)."""
    if bw is None:
        bw = BW_DEFAULT
    l_freq = f0 - bw / 2.0
    h_freq = f0 + bw / 2.0
    echt = ECHT(
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=sfreq,
        filt_order=order,
        filter_type=filter_type,
        calibrate=calibrate,
        f0=(f0 if calibrate else None),
        # mmse=(True if calibrate else False),
        # mmse_sigma_w2=(1e-4 if calibrate else None),
        # mmse_n_grid=512,
        # mmse_amp_profile=(True if calibrate else None),
    )
    z = echt.fit_transform(x).ravel()
    return np.angle(z[-1])


# ---------------------------------------------------------------------
# Window length sweep
# ---------------------------------------------------------------------
def compute_window_length_sweep():
    """
    Window-length sweep summarized as violin plots over ±0.5 cycles.

    For each central cycle count c in {1,2,3,4} we draw many non-integer
    window lengths n_cycles in [c-0.5, c+0.5], generate signals, and
    aggregate the resulting phase errors into one distribution per band.
    """
    cycles = np.array([1, 2, 3, 4])
    unc = []
    cal = []

    window_span = 0.5
    n_window_steps = N_WINDOW_STEPS if "N_WINDOW_STEPS" in globals() else 25

    for c in cycles:
        # Sample window lengths uniformly in [c-0.5, c+0.5]
        lo = max(c - window_span, 0.2)  # avoid ridiculously short windows
        hi = c + window_span
        n_cycles_vals = np.linspace(lo, hi, n_window_steps)

        errs_unc_band = []
        errs_cal_band = []

        for n_cyc in n_cycles_vals:
            for i in range(N_PHASE_SAMPLES):
                phi0 = i / N_PHASE_SAMPLES * 2 * np.pi

                # Non-integer number of cycles here
                _, x, true_phase_end = generate_cosine_window(
                    n_cyc, F0, SFREQ, phi0
                )

                phi_unc = endpoint_phase_echt(
                    x, F0, SFREQ, calibrate=False
                )
                phi_cal = endpoint_phase_echt(
                    x, F0, SFREQ, calibrate=True
                )

                err_unc = np.degrees(
                    (angle_diff(phi_unc, true_phase_end))
                )
                err_cal = np.degrees(
                    (angle_diff(phi_cal, true_phase_end))
                )

                errs_unc_band.append(err_unc)
                errs_cal_band.append(err_cal)

        # One big distribution per cycle band
        unc.append(errs_unc_band)
        cal.append(errs_cal_band)

    return {
        "cycles": cycles,
        "unc": unc,
        "cal": cal,
    }


def plot_window_length_sweep(ax, r):
    cycles = r["cycles"]
    unc = [np.abs(u) for u in r["unc"]]
    cal = [np.abs(ca) for ca in r["cal"]]

    positions = np.arange(len(cycles))

    # pad each distribution with 0° and 180°
    unc = [np.concatenate([u, [0, 180]]) for u in unc]
    cal = [np.concatenate([ca, [0, 180]]) for ca in cal]

    v_unc = ax.violinplot(
        unc,
        positions=positions,
        widths=0.8,
        points=1000,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    v_cal = ax.violinplot(
        cal,
        positions=positions,
        widths=0.8,
        points=1000,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    # half-violin
    for i, pos in enumerate(positions):
        for body, side, color in [
            (v_unc["bodies"][i], "left", COL_UNCAL),
            (v_cal["bodies"][i], "right", COL_CAL),
        ]:
            path = body.get_paths()[0]
            verts = path.vertices
            xs = verts[:, 0]

            if side == "left":
                verts[:, 0] = np.minimum(xs, pos)
            else:
                verts[:, 0] = np.maximum(xs, pos)

            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(global_alpha)


    # median of |error| to display in violins
    mean_unc = [np.median(u) for u in unc]
    mean_cal = [np.median(ca) for ca in cal]

    ax.scatter(positions - 0.1, mean_unc, marker=">", s=20,
               color=COL_UNCAL, zorder=3, label="uncalibrated",
               edgecolor="black", linewidth=0.75)
    ax.scatter(positions + 0.1, mean_cal, marker="<", s=20,
               color=COL_CAL, zorder=3, label="calibrated",
               edgecolor="black", linewidth=0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels([rf"{c}\,$\pm$\,0.5" for c in cycles])
    ax.set_xlabel(r"cycles of $1/f_0$")
    ax.set_ylim(0, 45)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels([r"$10^\circ$", r"$20^\circ$", r"$30^\circ$", r"$40^\circ$"])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)



# ---------------------------------------------------------------------
# Bandwidth sweep
# ---------------------------------------------------------------------
def compute_bandwidth_sweep():
    bw_factors = np.linspace(0.1, 1, 100)
    nyq = SFREQ / 2.0

    valid_f = []
    mean_unc = []
    std_unc = []
    max_unc = []
    mean_cal = []
    std_cal = []
    max_cal = []

    n_cycles = N_CYCLES

    for f in bw_factors:
        bw = f * F0
        l_freq = F0 - bw / 2.0
        h_freq = F0 + bw / 2.0
        if l_freq <= 0.0 or h_freq >= nyq:
            continue

        errs_unc = []
        errs_cal = []
        for i in range(N_PHASE_SAMPLES):
            phi0 = i/N_PHASE_SAMPLES * 2*np.pi
            _, x, true_phase_end = generate_cosine_window(
                n_cycles, F0, SFREQ, phi0
            )
            phi_unc = endpoint_phase_echt(
                x, F0, SFREQ, bw=bw, order=2, calibrate=False
            )
            phi_cal = endpoint_phase_echt(
                x, F0, SFREQ, bw=bw, order=2, calibrate=True
            )
            err_unc = np.degrees(np.abs(angle_diff(phi_unc, true_phase_end)))
            err_cal = np.degrees(np.abs(angle_diff(phi_cal, true_phase_end)))
            errs_unc.append(err_unc)
            errs_cal.append(err_cal)

        errs_unc = np.asarray(errs_unc)
        errs_cal = np.asarray(errs_cal)

        valid_f.append(f)
        mean_unc.append(float(errs_unc.mean()))
        std_unc.append(float(errs_unc.std(ddof=0)))
        max_unc.append(float(errs_unc.max()))
        mean_cal.append(float(errs_cal.mean()))
        std_cal.append(float(errs_cal.std(ddof=0)))
        max_cal.append(float(errs_cal.max()))

    return {
        "bw_factors": np.array(valid_f),
        "mean_unc": np.array(mean_unc),
        "std_unc": np.array(std_unc),
        "max_unc": np.array(max_unc),
        "mean_cal": np.array(mean_cal),
        "std_cal": np.array(std_cal),
        "max_cal": np.array(max_cal),
    }


def plot_bandwidth_sweep(ax, r):
    f = r["bw_factors"]
    mean_unc, std_unc, max_unc = r["mean_unc"], r["std_unc"], r["max_unc"]
    mean_cal, std_cal, max_cal = r["mean_cal"], r["std_cal"], r["max_cal"]

    line_mu_unc, = ax.plot(
        f, mean_unc, "-", label="uncalibrated", color=COL_UNCAL
    )
    line_mu_cal, = ax.plot(
        f, mean_cal, "--", label="calibrated", color=COL_CAL
    )

    lower_unc = np.clip(mean_unc - std_unc, 0, None)
    upper_unc = mean_unc + std_unc
    ax.fill_between(
        f, lower_unc, upper_unc,
        color=line_mu_unc.get_color(), alpha=0.18,
    )

    lower_cal = np.clip(mean_cal - std_cal, 0, None)
    upper_cal = mean_cal + std_cal
    ax.fill_between(
        f, lower_cal, upper_cal,
        color=line_mu_cal.get_color(), alpha=0.18,
    )

    ax.set_xlabel(r"Bandwidth / $f_0$")
    ax.set_ylabel(r"$|$Phase error$|$ $[^\circ]$")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)


# ---------------------------------------------------------------------
# Filter order sweep
# ---------------------------------------------------------------------
def compute_order_sweep():
    orders = np.array([1, 2, 3, 4, 5])

    mean_unc = []
    std_unc = []
    max_unc = []
    mean_cal = []
    std_cal = []
    max_cal = []

    n_cycles = N_CYCLES

    for order in orders:
        errs_unc = []
        errs_cal = []
        for i in range(N_PHASE_SAMPLES):
            phi0 = i/N_PHASE_SAMPLES *2*np.pi
            _, x, true_phase_end = generate_cosine_window(
                n_cycles, F0, SFREQ, phi0
            )
            phi_unc = endpoint_phase_echt(
                x, F0, SFREQ, bw=BW_DEFAULT, order=order, calibrate=False
            )
            phi_cal = endpoint_phase_echt(
                x, F0, SFREQ, bw=BW_DEFAULT, order=order, calibrate=True
            )
            err_unc = np.degrees(np.abs(angle_diff(phi_unc, true_phase_end)))
            err_cal = np.degrees(np.abs(angle_diff(phi_cal, true_phase_end)))
            errs_unc.append(err_unc)
            errs_cal.append(err_cal)

        errs_unc = np.asarray(errs_unc)
        errs_cal = np.asarray(errs_cal)

        mean_unc.append(float(errs_unc.mean()))
        std_unc.append(float(errs_unc.std(ddof=0)))
        max_unc.append(float(errs_unc.max()))
        mean_cal.append(float(errs_cal.mean()))
        std_cal.append(float(errs_cal.std(ddof=0)))
        max_cal.append(float(errs_cal.max()))

    return {
        "orders": orders,
        "mean_unc": np.array(mean_unc),
        "std_unc": np.array(std_unc),
        "max_unc": np.array(max_unc),
        "mean_cal": np.array(mean_cal),
        "std_cal": np.array(std_cal),
        "max_cal": np.array(max_cal),
    }


def plot_order_sweep(ax, r):
    orders = r["orders"]
    mean_unc, std_unc, max_unc = r["mean_unc"], r["std_unc"], r["max_unc"]
    mean_cal, std_cal, max_cal = r["mean_cal"], r["std_cal"], r["max_cal"]

    line_mu_unc, = ax.plot(
        orders, mean_unc, "-", label="uncalibrated", color=COL_UNCAL
    )
    line_mu_cal, = ax.plot(
        orders, mean_cal, "--", label="calibrated", color=COL_CAL
    )

    lower_unc = np.clip(mean_unc - std_unc, 0, None)
    upper_unc = mean_unc + std_unc
    ax.fill_between(
        orders, lower_unc, upper_unc,
        color=line_mu_unc.get_color(), alpha=0.18,
    )

    lower_cal = np.clip(mean_cal - std_cal, 0, None)
    upper_cal = mean_cal + std_cal
    ax.fill_between(
        orders, lower_cal, upper_cal,
        color=line_mu_cal.get_color(), alpha=0.18,
    )

    ax.set_xlabel("Filter order (Butterworth)")
    ax.set_xticks(orders)
    ax.set_xticklabels(2 * orders)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)



# ---------------------------------------------------------------------
# SNR sweep
# ---------------------------------------------------------------------
def compute_snr_sweep(seed):
    rng = np.random.default_rng(seed)
    snr_db_list = np.linspace(-10, 20, 1000)

    n_cycles = N_CYCLES
    _, x_clean, true_phase_end = generate_cosine_window(
        n_cycles, F0, SFREQ, phi0=0.0
    )
    signal_power = np.mean(x_clean ** 2)

    mean_unc = []
    std_unc = []
    mean_cal = []
    std_cal = []

    for snr_db in snr_db_list:
        snr_lin = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_lin
        noise_std = np.sqrt(noise_power)

        errs_unc = []
        errs_cal = []
        for _ in range(N_NOISE_TRIALS):
            noise = rng.normal(0.0, noise_std, size=x_clean.shape)
            x_noisy = x_clean + noise
            phi_unc = endpoint_phase_echt(
                x_noisy, F0, SFREQ, bw=BW_DEFAULT, order=2, calibrate=False
            )
            phi_cal = endpoint_phase_echt(
                x_noisy, F0, SFREQ, bw=BW_DEFAULT, order=2, calibrate=True
            )
            err_unc = np.degrees(np.abs(angle_diff(phi_unc, true_phase_end)))
            err_cal = np.degrees(np.abs(angle_diff(phi_cal, true_phase_end)))
            errs_unc.append(err_unc)
            errs_cal.append(err_cal)

        errs_unc = np.asarray(errs_unc)
        errs_cal = np.asarray(errs_cal)

        mean_unc.append(float(errs_unc.mean()))
        std_unc.append(float(errs_unc.std(ddof=0)))
        mean_cal.append(float(errs_cal.mean()))
        std_cal.append(float(errs_cal.std(ddof=0)))

    return {
        "snr_db": snr_db_list,
        "mean_unc": np.array(mean_unc),
        "std_unc": np.array(std_unc),
        "mean_cal": np.array(mean_cal),
        "std_cal": np.array(std_cal),
    }


def plot_snr_sweep(ax, r):
    snr_db = r["snr_db"]
    mean_unc, std_unc = r["mean_unc"], r["std_unc"]
    mean_cal, std_cal = r["mean_cal"], r["std_cal"]

    line_mu_unc, = ax.semilogy(
        snr_db, mean_unc, "-", label="uncalibrated", color=COL_UNCAL
    )
    line_mu_cal, = ax.semilogy(
        snr_db, mean_cal, "--", label="calibrated", color=COL_CAL
    )

    eps = 1e-6
    lower_unc = np.clip(mean_unc - std_unc, eps, None)
    upper_unc = np.clip(mean_unc + std_unc, eps, None)
    ax.fill_between(
        snr_db, lower_unc, upper_unc,
        color=line_mu_unc.get_color(), alpha=0.18,
    )

    lower_cal = np.clip(mean_cal - std_cal, eps, None)
    upper_cal = np.clip(mean_cal + std_cal, eps, None)
    ax.fill_between(
        snr_db, lower_cal, upper_cal,
        color=line_mu_cal.get_color(), alpha=0.18,
    )

    ax.set_xlabel(r"Input SNR (dB)")
    ax.set_ylabel(r"$|$phase error$|$ $[^\circ]$")
    ax.set_ylim(0.8 * min(mean_cal), 1.2 * max(mean_unc))
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)


# ---------------------------------------------------------------------
# Chirp robustness
# ---------------------------------------------------------------------
def generate_detuned_cosine(n_cycles, f0, sfreq, delta_frac=0.0, phi0=0.0):
    duration = n_cycles / f0
    n_samples = int(round(duration * sfreq))
    t = np.arange(n_samples) / sfreq

    f_sig = f0 * (1.0 + delta_frac)
    phase = 2 * np.pi * f_sig * t + phi0
    x = np.cos(phase)
    true_phase_end = wrap_phase(phase[-1])
    return t, x, true_phase_end

def compute_chirp_robustness():
    delta_fracs = np.linspace(-0.1, 0.1, 100)
    n_cycles = N_CYCLES

    mean_unc = []; std_unc = []
    mean_cal = []; std_cal = []

    for df in delta_fracs:
        errs_unc = []
        errs_cal = []
        for i in range(N_PHASE_SAMPLES):
            phi0 = i / N_PHASE_SAMPLES * 2 * np.pi
            _, x, true_phase_end = generate_detuned_cosine(
                n_cycles, F0, SFREQ, delta_frac=df, phi0=phi0
            )
            phi_unc = endpoint_phase_echt(x, F0, SFREQ, bw=BW_DEFAULT,
                                          order=2, calibrate=False)
            phi_cal = endpoint_phase_echt(x, F0, SFREQ, bw=BW_DEFAULT,
                                          order=2, calibrate=True)

            err_unc = np.degrees(np.abs(angle_diff(phi_unc, true_phase_end)))
            err_cal = np.degrees(np.abs(angle_diff(phi_cal, true_phase_end)))
            errs_unc.append(err_unc)
            errs_cal.append(err_cal)

        errs_unc = np.asarray(errs_unc)
        errs_cal = np.asarray(errs_cal)

        mean_unc.append(float(errs_unc.mean()))
        std_unc.append(float(errs_unc.std(ddof=0)))
        mean_cal.append(float(errs_cal.mean()))
        std_cal.append(float(errs_cal.std(ddof=0)))

    return {
        "delta_fracs": np.array(delta_fracs),
        "mean_unc": np.array(mean_unc),
        "std_unc": np.array(std_unc),
        "mean_cal": np.array(mean_cal),
        "std_cal": np.array(std_cal),
    }


def plot_chirp_robustness(ax, r):
    df = r["delta_fracs"]
    mean_unc, std_unc = r["mean_unc"], r["std_unc"]
    mean_cal, std_cal = r["mean_cal"], r["std_cal"]

    line_mu_unc, = ax.plot(
        df, mean_unc, "-", label="uncalibrated", color=COL_UNCAL
    )
    line_mu_cal, = ax.plot(
        df, mean_cal, "--", label="calibrated", color=COL_CAL
    )

    lower_unc = np.clip(mean_unc - std_unc, 0, None)
    upper_unc = mean_unc + std_unc
    ax.fill_between(
        df, lower_unc, upper_unc,
        color=line_mu_unc.get_color(), alpha=0.18,
    )

    lower_cal = np.clip(mean_cal - std_cal, 0, None)
    upper_cal = mean_cal + std_cal
    ax.fill_between(
        df, lower_cal, upper_cal,
        color=line_mu_cal.get_color(), alpha=0.18,
    )

    ax.set_xlabel(r"$\Delta f/f_0$")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5, alpha=0.7)



# ---------------------------------------------------------------------
# Filter-type comparison
# ---------------------------------------------------------------------
def compute_filter_type():
    n_cycles = N_CYCLES
    order = 1
    bw = BW_DEFAULT

    filter_types = ["butter", "bessel", "cheby1", "cheby2", "ellip"]
    labels = ["Butter", "Bessel", "Cheby I", "Cheby II", "Cauer"]

    mean_unc = []
    std_unc = []
    mean_cal = []
    std_cal = []

    for ftype in filter_types:
        errs_rad_unc = []
        errs_rad_cal = []

        for i in range(N_PHASE_SAMPLES):
            phi0 = i / N_PHASE_SAMPLES * 2 * np.pi
            _, x, true_phase_end = generate_cosine_window(
                n_cycles, F0, SFREQ, phi0
            )

            # ECHT with given filter type, uncalibrated
            phi_unc = endpoint_phase_echt(
                x,
                F0,
                SFREQ,
                bw=bw,
                order=order,
                filter_type=ftype,
                calibrate=False,
            )
            # ECHT with given filter type, calibrated
            phi_cal = endpoint_phase_echt(
                x,
                F0,
                SFREQ,
                bw=bw,
                order=order,
                filter_type=ftype,
                calibrate=True,
            )

            err_unc = angle_diff(phi_unc, true_phase_end)
            err_cal = angle_diff(phi_cal, true_phase_end)
            errs_rad_unc.append(err_unc)
            errs_rad_cal.append(err_cal)

        errs_rad_unc = np.asarray(errs_rad_unc)
        errs_rad_cal = np.asarray(errs_rad_cal)

        errs_deg_unc = np.degrees(np.abs(errs_rad_unc))
        errs_deg_cal = np.degrees(np.abs(errs_rad_cal))

        mean_unc.append(float(errs_deg_unc.mean()))
        std_unc.append(float(errs_deg_unc.std(ddof=0)))
        mean_cal.append(float(errs_deg_cal.mean()))
        std_cal.append(float(errs_deg_cal.std(ddof=0)))

    return {
        "labels": labels,
        "mean_unc": np.array(mean_unc),
        "std_unc": np.array(std_unc),
        "mean_cal": np.array(mean_cal),
        "std_cal": np.array(std_cal),
    }


def plot_filter_type(ax, r):
    labels = np.array(r["labels"])
    mean_unc, std_unc = r["mean_unc"], r["std_unc"]
    mean_cal, std_cal = r["mean_cal"], r["std_cal"]

    # Console summary
    print("Filter-type comparison (ECHT, order=1, BW = f0/2):")
    print("{:<10} {:>12} {:>12} {:>12} {:>12}".format(
        "Filter", "mean|err| unc", "std unc",
        "mean|err| cal", "std cal"
    ))
    for lab, mu_u, cs_u, mu_c, cs_c in zip(
        labels, mean_unc, std_unc, mean_cal, std_cal
    ):
        print("{:<10} {:12.3f} {:12.3f} {:12.3f} {:12.3f}".format(
            lab, mu_u, cs_u, mu_c, cs_c
        ))
    print()

    # Identify Cheby II index
    outlier_name = "Cheby II"
    idx_out = np.where(labels == outlier_name)[0][0] if outlier_name in labels else None

    # y-limit from non–Cheby-II bars
    if idx_out is not None:
        mask_other = np.ones_like(labels, dtype=bool)
        mask_other[idx_out] = False
        y_max_others = float(
            max(
                (mean_unc[mask_other] + std_unc[mask_other]).max(),
                (mean_cal[mask_other] + std_cal[mask_other]).max(),
            )
        )
        y_lim = 1.05 * y_max_others
    else:
        y_lim = 1.05 * float(max(mean_unc.max(), mean_cal.max()))

    x = np.arange(len(labels))
    width = 0.35

    face_unc = mcolors.to_rgba(COL_UNCAL, alpha=global_alpha)
    face_cal = mcolors.to_rgba(COL_CAL, alpha=global_alpha)

    bars_unc = ax.bar(
        x - width / 2,
        mean_unc,
        width,
        yerr=std_unc,
        capsize=2,
        label="uncalibrated",
        color=face_unc,
        edgecolor="black",
        linewidth=1,
    )
    bars_cal = ax.bar(
        x + width / 2,
        mean_cal,
        width,
        yerr=std_cal,
        capsize=2,
        label="calibrated",
        color=face_cal,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, y_lim)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    # Annotate Cheby II uncalibrated height on its clipped bar
    if idx_out is not None:
        bar_out = bars_unc[idx_out]
        x_out = bar_out.get_x() + 1.1*bar_out.get_width() / 2.0
        val_out = mean_unc[idx_out]

        ax.text(
            x_out,
            y_lim * 0.95,
            f"{val_out:.1f}°",
            ha="center",
            va="top",
            fontsize=7,
            rotation=90,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(output_prefix="ecHT_simulations"):
    set_mpl_style()

    # Prepare seeds for the six independent computations
    seed = MASTER_RNG.integers(0, 2**32 - 1, size=1)

    # Computations in parallel
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [
            ex.submit(compute_window_length_sweep),  # 0
            ex.submit(compute_bandwidth_sweep),  # 1
            ex.submit(compute_order_sweep),  # 2
            ex.submit(compute_snr_sweep, seed),  # 3
            ex.submit(compute_chirp_robustness),  # 4
            ex.submit(compute_filter_type),  # 5
        ]
        results = [f.result() for f in futures]

        (res_window,
         res_bandwidth,
         res_order,
         res_snr,
         res_chirp,
         res_filter) = results

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))
    ax = axes.ravel()

    plot_bandwidth_sweep(ax[0], res_bandwidth)  # Bandwidth
    plot_order_sweep(ax[1], res_order)  # Filter order
    plot_chirp_robustness(ax[2], res_chirp)  # Frequency drift
    plot_snr_sweep(ax[3], res_snr)  # Noise robustness
    plot_filter_type(ax[4], res_filter)  # Filter type
    plot_window_length_sweep(ax[5], res_window)  # Window length

    # Panel labels in intro style
    add_panel_label(ax[0], "A", "Bandwidth")
    add_panel_label(ax[1], "B", "Filter order")
    add_panel_label(ax[2], "C", "Frequency drift")
    add_panel_label(ax[3], "D", "Noise robustness")
    add_panel_label(ax[4], "E", "Filter type")
    add_panel_label(ax[5], "F", "Window length")


    # Shared legend (uncalibrated, calibrated, ±1 SD)
    line_unc = mlines.Line2D([], [], color=COL_UNCAL, linestyle="-", label="uncalibrated")
    line_cal = mlines.Line2D([], [], color=COL_CAL, linestyle="--", label="calibrated")
    std_patch = mpatches.Patch(
        facecolor=gray,
        alpha=0.18,
        edgecolor="k",
        linewidth=0.5,
        label=r"$\pm 1$ SD",
    )

    legend = fig.legend(
        handles=[line_unc, line_cal, std_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        frameon=True,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_edgecolor("0.5")

    fig.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])

    png_name = f"{output_prefix}.png"
    pdf_name = f"{output_prefix}.pdf"
    fig.savefig(png_name, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_name, bbox_inches="tight")


if __name__ == "__main__":
    main()
