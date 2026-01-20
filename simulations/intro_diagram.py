"""
ecHT vs HT Infographic
======================

Create a compact multi-panel infographic comparing the standard Hilbert
Transform (HT) analytic signal to the Endpoint-Corrected Hilbert Transform
(ecHT) analytic signal.

Outputs
-------
Two files are written:
- ecHT_vs_HT_infographic.png
- ecHT_vs_HT_infographic.pdf

Notes
-----
- The figure is rendered using Matplotlib and SciPy.
- LaTeX rendering is optional. If use_tex=True and LaTeX is not installed,
  Matplotlib may raise an error.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle, ConnectionPatch

from scipy.fft import fft, ifft, fftshift, next_fast_len
from scipy.signal import butter, freqz

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


def analytic_mask(n_fft: int) -> np.ndarray:
    """
    Standard Hilbert analytic-signal multiplier (no band-pass).

    Parameters
    ----------
    n_fft : int
        FFT length.

    Returns
    -------
    h : ndarray, shape (n_fft,)
        Real-valued frequency-domain multiplier implementing the analytic
        signal construction (DC=1, positive frequencies doubled, negative=0;
        Nyquist=1 for even n_fft).
    """
    h = np.zeros(n_fft, dtype=float)
    h[0] = 1.0
    if n_fft % 2 == 0:
        # even length: DC + positive + Nyquist
        h[1:n_fft // 2] = 2.0
        h[n_fft // 2] = 1.0
    else:
        # odd length: DC + positive
        h[1:(n_fft + 1) // 2] = 2.0
    return h

def make_echt_vs_ht_infographic(
    sfreq: float = 1000,
    f0: float = 5.23,        # central frequency of interest
    n_cycles: float = 1.87,  # number of cycles in the window
    extra_cycles: float = 0.2,
    filt_order: int = 2,     # Butterworth band-pass order
    pad_factor: int = 64,
    save_base: str = "ecHT_vs_HT_infographic",
    use_tex: bool = True,
):
    """
    Generate a 6-panel ecHT vs HT infographic with a Butterworth band-pass.

    Panels
    ------
    (1) Long signal with the analysis window at the end.
    (2) DFT magnitude spectrum.
    (3) Analytic spectrum using standard HT multiplier.
    (4) ecHT spectrum (analytic + band-pass) plus band-pass |H(f)|.
    (5) Time-domain analytic signals (real parts).
    (6) Phase estimates near the window end + zoom panel.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz).
    f0 : float
        Central frequency of the cosine (Hz).
    n_cycles : float
        Number of cycles in the analysis window.
    extra_cycles : float
        Extra cycles shown before the analysis window for context in (A).
    filt_order : int
        Butterworth band-pass order.
    pad_factor : int
        Zero-padding factor controlling FFT length via ``next_fast_len``.
    save_base : str
        Output filename stem (without extension).
    use_tex : bool
        If True, use LaTeX rendering in Matplotlib.

    Returns
    -------
    None
        Writes ``{save_base}.png`` and ``{save_base}.pdf``.
    """
    # --------------------------------------------------------------
    # Signal: finite window extracted from a longer cosine
    # --------------------------------------------------------------
    n_samples = int(round(n_cycles * sfreq / f0))
    t_win = np.arange(n_samples) / sfreq
    phi0 = 0
    x_win = np.cos(2 * np.pi * f0 * t_win + phi0)

    extra_samples = int(round(extra_cycles * sfreq / f0))
    t_pre = np.arange(-extra_samples, 0) / sfreq
    t_long = np.concatenate([t_pre, t_win])
    x_long = np.cos(2 * np.pi * f0 * t_long + phi0)

    window_start = 0.0
    window_end = t_win[-1]

    # --------------------------------------------------------------
    # ecHT setup (we only use its analytic kernel h_)
    # --------------------------------------------------------------
    BW = f0 * 0.8
    l_freq = f0 - BW / 2.0
    h_freq = f0 + BW / 2.0

    n_fft = next_fast_len(pad_factor * n_samples)

    echt = ECHT(
        l_freq=l_freq,
        h_freq=h_freq,
        sfreq=sfreq,
        n_fft=n_fft,
        filt_order=filt_order,
        calibrate=False,
        f0=None,
    )
    echt.fit(x_win)

    n_fft = echt.n_fft
    h_echt = echt.h_  # analytic (Hilbert) multiplier

    # --------------------------------------------------------------
    # Frequency-domain objects
    # --------------------------------------------------------------
    X_fft = fft(x_win, n_fft)
    freqs = np.fft.fftfreq(n_fft, d=1.0 / sfreq)
    freqs_shift = fftshift(freqs)
    X_fft_shift = fftshift(X_fft)

    # Standard HT analytic spectrum
    h_ht = analytic_mask(n_fft)
    X_fft_ht = X_fft * h_ht
    X_fft_ht_shift = fftshift(X_fft_ht)

    # --------------------------------------------------------------
    # Real Butterworth band-pass on the actual FFT grid
    # --------------------------------------------------------------
    Wn = [l_freq / (sfreq / 2), h_freq / (sfreq / 2)]
    b, a = butter(filt_order, Wn, btype="band")

    # Evaluate H(e^{jω}) at FFT grid frequencies:
    # freqs are in Hz; digital rad/sample is ω = 2π f / sfreq.
    w = 2 * np.pi * freqs / sfreq
    w_mod = np.mod(w, 2 * np.pi)  # bring into [0, 2π)
    _, H_fftgrid = freqz(b, a, worN=w_mod)  # complex response on FFT grid

    H_bp = H_fftgrid                 # length n_fft, same order as X_fft
    H_bp_shift = fftshift(H_bp)      # centered for plotting
    H_bp_amp = np.abs(H_bp_shift)
    if H_bp_amp.max() > 0:
        H_bp_amp /= H_bp_amp.max()   # normalize for plotting

    # ecHT spectrum: analytic kernel + Butterworth band-pass
    X_fft_analytic = X_fft * h_echt
    X_fft_echt = X_fft_analytic * H_bp
    X_fft_echt_shift = fftshift(X_fft_echt)

    # --------------------------------------------------------------
    # Time-domain analytic signals
    # --------------------------------------------------------------
    z_ht_full = ifft(X_fft_ht)
    z_echt_full = ifft(X_fft_echt)

    z_ht = z_ht_full[:n_samples]
    z_echt = z_echt_full[:n_samples]

    # True analytic phase of the pure cosine
    phase_true_rad = 2 * np.pi * f0 * t_win + phi0

    phase_true_deg = np.rad2deg(np.angle(np.exp(1j * phase_true_rad)))
    phase_ht_deg = np.rad2deg(np.angle(z_ht))
    phase_echt_deg = np.rad2deg(np.angle(z_echt))

    # Index range for last ~2 cycles
    samples_per_cycle = int(round(sfreq / f0))
    n_show = min(n_samples, 2 * samples_per_cycle)
    idx_end = np.arange(n_samples - n_show, n_samples)

    # --------------------------------------------------------------
    # Figure with 6 main panels + legend + zoom
    # --------------------------------------------------------------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "figure.dpi": 300,
        "text.usetex": use_tex,
    })

    blue = "#4E79A7"
    orange = "#F28E2B"
    red = "#E15759"
    green = "#59A14F"
    yellow = "#EDC948"
    purple = "#B07AA1"
    teal = "#76B7B2"
    gray = "#BAB0AC"

    col_sig = blue
    col_true = blue
    col_ht = yellow
    col_echt = orange
    col_bp = gray

    # 2 × 4 layout with narrower 4th column
    fig, axes = plt.subplots(
        2, 4, figsize=(7, 4.5),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.5]}
    )
    ax1 = axes[0, 0]        # (A)
    ax2 = axes[0, 1]        # (B)
    ax3 = axes[0, 2]        # (C)
    ax_legend = axes[0, 3]  # legend (top-right special)
    ax4 = axes[1, 0]        # (D)
    ax5 = axes[1, 1]        # (E)
    ax6 = axes[1, 2]        # (F)
    ax_zoom = axes[1, 3]    # zoom (bottom-right special)

    # Common vertical position (in axes coordinates) for labels under each plot
    y_under = -0.02

    # (A) Long signal + faint analysis window at the end
    ax1.plot(t_long, x_long, lw=1, color=gray, label="Underlying signal")
    ax1.axvspan(window_start, window_end, alpha=0.15, color=gray)

    mask_win = (t_long >= window_start) & (t_long <= window_end)
    ax1.plot(
        t_long[mask_win],
        x_long[mask_win],
        lw=1.5,
        color=col_sig,
        label="Analysis window",
    )

    transA = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.text(
        window_start,
        y_under,
        r"$n=0$",
        transform=transA,
        ha="left",
        va="top",
        fontsize=10,
        color="k",
        clip_on=False,
    )
    ax1.text(
        window_end,
        y_under,
        r"$N-1$",
        transform=transA,
        ha="right",
        va="top",
        fontsize=10,
        color="k",
        clip_on=False,
    )

    # (B) DFT spectrum
    ax2.plot(freqs_shift, np.abs(X_fft_shift), lw=1.5, color=col_sig)
    ax2.set_xlim(-3 * f0, 3 * f0)

    transB2 = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
    ax2.text(-f0, y_under, r"$-f_0$", transform=transB2,
             ha="center", va="top", fontsize=10, clip_on=False)
    ax2.text(0, y_under, r"$0$", transform=transB2,
             ha="center", va="top", fontsize=10, clip_on=False)
    ax2.text(f0, y_under, r"$+f_0$", transform=transB2,
             ha="center", va="top", fontsize=10, clip_on=False)

    # (C) Analytic spectrum (Hilbert)
    ax3.plot(freqs_shift, np.abs(X_fft_ht_shift), lw=1.5, color=col_sig)
    ax3.set_xlim(-3 * f0, 3 * f0)

    transB3 = mtransforms.blended_transform_factory(ax3.transData, ax3.transAxes)
    ax3.text(-f0, y_under, r"$-f_0$", transform=transB3,
             ha="center", va="top", fontsize=10, clip_on=False)
    ax3.text(0, y_under, r"$0$", transform=transB3,
             ha="center", va="top", fontsize=10, clip_on=False)
    ax3.text(f0, y_under, r"$+f_0$", transform=transB3,
             ha="center", va="top", fontsize=10, clip_on=False)

    # Legend in dedicated axis
    legend_elements = [
        Line2D([0], [0], color=col_true, lw=1.5, ls="-",
               label="True signal"),
        Line2D([0], [0], color=col_ht, lw=1.5, ls="--",
               label="HT"),
        Line2D([0], [0], color=col_echt, lw=1.5, ls="-.",
               label="ecHT"),
    ]
    ax_legend.set_axis_off()
    ax_legend.legend(
        handles=legend_elements,
        loc="center",
        fontsize=8,
        frameon=True,
        handlelength=2.5,
    )

    # (D) ecHT vs HT spectra + Butterworth band-pass amplitude response
    ax4.plot(
        freqs_shift,
        np.abs(X_fft_echt_shift),
        lw=1.5,
        ls="-.",
        label="ecHT (analytic + band-pass)",
        color=col_echt,
    )
    ax4.set_xlim(-3 * f0, 3 * f0)

    ax4b = ax4.twinx()
    ax4b.plot(
        freqs_shift,
        H_bp_amp,
        lw=1.5,
        color=col_bp,
        alpha=0.9,
        label="Band-pass |H(f)| (norm., Butterworth)",
    )

    transB4 = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
    ax4.text(-f0, y_under, r"$-f_0$", transform=transB4,
             ha="center", va="top", fontsize=10, clip_on=False)
    ax4.text(0, y_under, r"$0$", transform=transB4,
             ha="center", va="top", fontsize=10, clip_on=False)
    ax4.text(f0, y_under, r"$+f_0$", transform=transB4,
             ha="center", va="top", fontsize=10, clip_on=False)

    # (E) Time-domain over full window + true analytic signal
    ax5.plot(t_long, x_long, lw=0.7, color=gray, alpha=0)

    ax5.plot(t_win, x_win, lw=1.5, ls="-",
             label="True analytic (real part)", color=col_true)
    ax5.plot(t_win, np.real(z_ht), lw=1.5, ls="--",
             label="HT (real part of analytic)", color=col_ht)
    ax5.plot(t_win, np.real(z_echt), lw=1.5, ls="-.",
             label="ecHT (real part of analytic)", color=col_echt)

    transE = mtransforms.blended_transform_factory(ax5.transData, ax5.transAxes)
    ax5.text(window_start, y_under, r"$n=0$",
             transform=transE, ha="left", va="top",
             fontsize=10, color="k", clip_on=False)
    ax5.text(window_end, y_under, r"$N-1$",
             transform=transE, ha="right", va="top",
             fontsize=10, color="k", clip_on=False)

    # (F) Wrapped phase (deg) near window end, with dots at last samples
    ax6.plot(t_long, x_long, lw=0.7, color=gray, alpha=0)
    ax6.plot(
        t_win[idx_end],
        phase_true_deg[idx_end],
        lw=1.5,
        ls="-",
        label="True signal",
        color=col_true,
    )
    line_ht_p, = ax6.plot(
        t_win[idx_end],
        phase_ht_deg[idx_end],
        lw=1.5,
        ls="--",
        label="HT",
        color=col_ht,
    )
    line_echt_p, = ax6.plot(
        t_win[idx_end],
        phase_echt_deg[idx_end],
        lw=1.5,
        ls="-.",
        label="ecHT",
        color=col_echt,
    )

    ax6.plot(
        t_win[-1],
        phase_ht_deg[-1],
        ".",
        color=line_ht_p.get_color(),
        markersize=6,
    )
    ax6.plot(
        t_win[-1],
        phase_echt_deg[-1],
        ".",
        color=line_echt_p.get_color(),
        markersize=6,
    )

    transF = mtransforms.blended_transform_factory(ax6.transData, ax6.transAxes)
    ax6.text(
        window_start,
        y_under,
        r"$n=0$",
        transform=transF,
        ha="left",
        va="top",
        fontsize=10,
        color="k",
        clip_on=False,
    )
    ax6.text(
        window_end,
        y_under,
        r"$N-1$",
        transform=transF,
        ha="right",
        va="top",
        fontsize=10,
        color="k",
        clip_on=False,
    )

    # --- Zoom region indices and limits (used for both rectangle and zoom panel) ---
    zoom_cycles = 0.1
    zoom_samples = int(round(zoom_cycles * samples_per_cycle))
    zoom_samples = max(2, zoom_samples)
    zoom_idx = np.arange(n_samples - zoom_samples, n_samples)

    # Determine y-range of zoomed region
    y_min = np.min([
        phase_true_deg[zoom_idx],
        phase_ht_deg[zoom_idx],
        phase_echt_deg[zoom_idx],
    ])
    y_max = np.max([
        phase_true_deg[zoom_idx],
        phase_ht_deg[zoom_idx],
        phase_echt_deg[zoom_idx],
    ])

    pad = 0.05 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    x0 = t_win[zoom_idx[0]]
    x1 = t_win[zoom_idx[-1]]
    width = x1 - x0
    y0 = y_min
    height = y_max - y_min

    scale = 0.20
    dx = scale * width / 2
    dy = scale * height / 2

    # rectangle on the original phase panel (F) – gray
    rect = Rectangle(
        (x0 - dx, y0 - dy),
        width + 2 * dx,
        height + 2 * dy,
        fill=False,
        edgecolor=gray,
        linewidth=0.8,
        linestyle="-",
        zorder=0.5,
    )
    ax6.add_patch(rect)

    # Zoom panel in the dedicated 4th column
    ax_zoom.plot(
        t_win[zoom_idx],
        phase_true_deg[zoom_idx],
        lw=1.5,
        ls="-",
        color=col_true,
    )
    ax_zoom.plot(
        t_win[zoom_idx],
        phase_ht_deg[zoom_idx],
        lw=1.2,
        ls="--",
        color=col_ht,
    )
    ax_zoom.plot(
        t_win[zoom_idx],
        phase_echt_deg[zoom_idx],
        lw=1.2,
        ls="-.",
        color=col_echt,
    )

    # Highlight last sample in zoom as well
    ax_zoom.plot(
        t_win[-1],
        phase_ht_deg[-1],
        ".",
        color=col_ht,
        markersize=5,
    )
    ax_zoom.plot(
        t_win[-1],
        phase_echt_deg[-1],
        ".",
        color=col_echt,
        markersize=5,
    )

    # Match data limits of the rectangle (preserves data-space of zoom)
    ax_zoom.set_xlim(x0 - dx, x1 + dx)
    ax_zoom.set_ylim(y0 - dy, y0 - dy + height + 2 * dy)

    # Remove ticks & labels in zoom panel but KEEP spines (box around zoom)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_xticklabels([])
    ax_zoom.set_yticklabels([])

    # Make zoom box (spines) gray
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor(gray)
        spine.set_linewidth(0.8)

    # Lines indicating the zoom:
    # from right edge of rectangle in ax6 to left edge of zoom-box in ax_zoom
    rect_right_x = x1 + dx
    rect_bottom_y = y0 - dy
    rect_top_y = y0 - dy + height + 2 * dy

    # bottom connection: rect bottom-right -> zoom bottom-left corner of axes box
    con1 = ConnectionPatch(
        xyA=(rect_right_x, rect_bottom_y),
        coordsA=ax6.transData,
        xyB=(0.0, 0.0),
        coordsB=ax_zoom.transAxes,
        color=gray,
        linewidth=0.8,
        linestyle="-",
    )
    # top connection: rect top-right -> zoom top-left corner of axes box
    con2 = ConnectionPatch(
        xyA=(rect_right_x, rect_top_y),
        coordsA=ax6.transData,
        xyB=(0.0, 1.0),
        coordsB=ax_zoom.transAxes,
        color=gray,
        linewidth=0.8,
        linestyle="-",
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

    # --------------------------------------------------------------
    # Panel labels
    # --------------------------------------------------------------
    text_ha_align = "left"
    textx = 0.02
    texty = 1.07
    ax1.text(textx, texty, r"\textbf{(A)} $x(n)$", transform=ax1.transAxes,
             va="top", ha=text_ha_align)#, bbox=common_bbox)
    ax2.text(textx, texty, r"\textbf{(B)} DFT: $X(k)$", transform=ax2.transAxes,
             va="top", ha=text_ha_align)#, bbox=common_bbox)
    ax3.text(textx, texty, r"\textbf{(C)} Analytic: $X^+(k)$", transform=ax3.transAxes,
             va="top", ha=text_ha_align)#, bbox=common_bbox)
    ax4.text(textx, texty, r"\textbf{(D)} Bandpass: $X^+(k)H(k)$", transform=ax4.transAxes,
             va="top", ha=text_ha_align)#, bbox=common_bbox)
    ax5.text(textx, texty, r"\textbf{(E)} $\mathrm{Re}\ \hat{z}(n)$", transform=ax5.transAxes,
             va="top", ha=text_ha_align)#, bbox=common_bbox)
    ax6.text(textx, texty, r"\textbf{(F)} $\hat{\theta}(n) = \arg \hat{z}(n)$", transform=ax6.transAxes,
             va="top", ha=text_ha_align)#, bbox=common_bbox)

    for ax in [ax1, ax2, ax3, ax4, ax4b, ax5, ax6, ax_legend]:
        # Remove ticks/labels and hide spines for a clean look.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)


    fig.tight_layout()
    fig.savefig(f"{save_base}.png", dpi=300)
    fig.savefig(f"{save_base}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_echt_vs_ht_infographic()
