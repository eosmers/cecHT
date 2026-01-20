import numpy as np
import matplotlib as mpl
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
    "text.usetex": True,
})

# Color palette
BLUE   = "#4E79A7"
ORANGE = "#F28E2B"
RED    = "#E15759"
GREEN  = "#59A14F"
YELLOW = "#EDC948"
PURPLE = "#B07AA1"
TEAL   = "#76B7B2"

COL_HT   = ORANGE
COL_ECHT = BLUE

COMMON_BBOX = dict(
    facecolor="white",
    edgecolor="0.8",
    boxstyle="round,pad=0.2",
    alpha=0.85,
)

# ---------------------------------------------------------------------
# Circular helpers
# ---------------------------------------------------------------------
def _wrap_phase(phi):
    """Wrap phase to [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _wmean_angle(mu, w):
    """Weighted circular mean (radians)."""
    v = np.sum(w * np.exp(1j * mu))
    return float(np.angle(v))


def _circ_stats(phi_rad):
    """
    Returns:
        mu, sd, plv, pli
    """
    phi = np.asarray(phi_rad, float)
    if phi.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    mu = float(circmean(phi, high=np.pi, low=-np.pi))
    sd = float(circstd(phi))
    plv = float(np.abs(np.mean(np.exp(1j * phi))))
    pli = float(np.abs(np.mean(np.sign(np.sin(phi)))))

    return mu, sd, plv, pli


def _p_value_text(p, decimals=4):
    """Pretty LaTeX-formatted p-value."""
    if not np.isfinite(p):
        return "p = n/a"

    threshold = 10 ** (-decimals)
    if p < threshold:
        n = int(np.ceil(-np.log10(p)))
        return rf"$p < 10^{{-{n}}}$"
    return rf"$p = {p:.{decimals}f}$"

# ---------------------------------------------------------------------
# Permutation test (paired, circular)
# ---------------------------------------------------------------------
def circular_permutation_test_paired(
    mu_a,
    mu_b,
    weights,
    n_perm=5000,
    seed=0,
):
    """
    Paired circular permutation test using within-recording label swaps.
    Returns (T_obs_deg, p_value).
    """
    mu_a = np.asarray(mu_a, float)
    mu_b = np.asarray(mu_b, float)
    w = np.asarray(weights, float)

    mask = np.isfinite(mu_a) & np.isfinite(mu_b) & np.isfinite(w) & (w > 0)
    mu_a, mu_b, w = mu_a[mask], mu_b[mask], w[mask]

    if mu_a.size < 2:
        return np.nan, np.nan

    def T(mua, mub):
        da = _wmean_angle(mua, w)
        db = _wmean_angle(mub, w)
        return abs(_wrap_phase(da - db))

    T_obs = T(mu_a, mu_b)

    rng = np.random.default_rng(seed)
    swaps = rng.random((n_perm, mu_a.size)) < 0.5

    exceed = 0
    for s in swaps:
        mua = np.where(s, mu_b, mu_a)
        mub = np.where(s, mu_a, mu_b)
        if T(mua, mub) >= T_obs - 1e-15:
            exceed += 1

    p = (exceed + 1) / (n_perm + 1)
    return float(np.degrees(T_obs)), float(p)

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _style_polar_axis(ax, r_max, radial_ticks):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=[""] * 8)

    ax.set_ylim(0, r_max)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([rf"{t:.0f}\%" for t in radial_ticks])

    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_rlabel_position(90)

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("center")

    r0 = r_max * 0.8
    ax.text(np.deg2rad(45),  r0, r"$+45^\circ$", ha="center", va="center")
    ax.text(np.deg2rad(-45), r0, r"$-45^\circ$", ha="center", va="center")


def _style_cart_axis(ax):
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.2")

def make_figure(
    err_unc_rad,
    err_cal_rad,
    trial_freq_cv,
    trial_abs_unc_rad,
    trial_abs_cal_rad,
    trial_mu_unc,
    trial_mu_cal,
    trial_nwin,
    save_base="tremor_echt_summary",
    n_perm=5000,
    perm_seed=0,
    panel_c_xlabel="Tremor frequency CV",
    panel_c_title=r"$\mathbf{(C)}$ Phase error vs. tremor variability",
):
    # --- stats ---
    mu_u, sd_u, plv_u, pli_u = _circ_stats(err_unc_rad)
    mu_c, sd_c, plv_c, pli_c = _circ_stats(err_cal_rad)

    # --- paired circular permutation test ---
    _, p_perm = circular_permutation_test_paired(
        mu_a=trial_mu_unc,
        mu_b=trial_mu_cal,
        weights=trial_nwin,
        n_perm=n_perm,
        seed=perm_seed,
    )
    p_txt = _p_value_text(p_perm)

    # --- histogram prep ---
    bin_width_deg = 10
    bin_width_rad = np.radians(bin_width_deg)
    bins = np.arange(-180, 181, bin_width_deg)
    centers = (np.radians(bins[:-1]) + np.radians(bins[1:])) / 2

    err_unc_deg = np.degrees(_wrap_phase(err_unc_rad))
    err_cal_deg = np.degrees(_wrap_phase(err_cal_rad))

    pu = np.histogram(err_unc_deg, bins=bins)[0] / max(1, err_unc_deg.size) * 100
    pc = np.histogram(err_cal_deg, bins=bins)[0] / max(1, err_cal_deg.size) * 100

    r_max = max(pu.max(), pc.max()) * 1.05
    r_max = max(r_max, 5)
    r_ticks = [t for t in [10, 20, 30] if t < r_max]

    # --- layout ---
    fig = plt.figure(figsize=(7.0, 2.4))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 1.2],
        height_ratios=[1.0],
        wspace=0.3,
    )

    axA = fig.add_subplot(gs[0, 0], projection="polar")
    axB = fig.add_subplot(gs[0, 1], projection="polar")
    axC = fig.add_subplot(gs[0, 2])

    # --- Panel A ---
    axA.bar(centers, pu, width=bin_width_rad, color=COL_HT, edgecolor="0", linewidth=0.75)
    axA.plot([mu_u, mu_u], [0, r_max], color="0", linewidth=2)
    _style_polar_axis(axA, r_max, r_ticks)
    axA.text(
        0.05, 0.45,
        rf"${np.degrees(mu_u):.1f}^\circ \pm {np.degrees(sd_u):.1f}^\circ$" "\n"
        rf"PLV$\uparrow$: {plv_u:.3f}" "\n"
        rf"PLI$\downarrow$: {pli_u:.3f}",
        transform=axA.transAxes, bbox=COMMON_BBOX, va="top"
    )

    # --- Panel B ---
    axB.bar(centers, pc, width=bin_width_rad, color=COL_ECHT, edgecolor="0", linewidth=0.75)
    axB.plot([mu_c, mu_c], [0, r_max], color="0", linewidth=2)
    _style_polar_axis(axB, r_max, r_ticks)
    axB.text(
        0.05, 0.45,
        rf"${np.degrees(mu_c):.1f}^\circ \pm {np.degrees(sd_c):.1f}^\circ$" "\n"
        rf"PLV$\uparrow$: {plv_c:.3f}" "\n"
        rf"PLI$\downarrow$: {pli_c:.3f}",
        transform=axB.transAxes, bbox=COMMON_BBOX, va="top"
    )

    # --- Panel C (NO legend, NO slope text) ---
    mask = np.isfinite(trial_freq_cv) & np.isfinite(trial_abs_unc_rad) & np.isfinite(trial_abs_cal_rad)
    x = trial_freq_cv[mask]
    y_unc = np.degrees(trial_abs_unc_rad[mask])
    y_cal = np.degrees(trial_abs_cal_rad[mask])

    for xi, yu, yc in zip(x, y_unc, y_cal):
        axC.plot([xi, xi], [yu, yc], color="0.7", linewidth=0.8, zorder=1)

    axC.scatter(x, y_unc, s=16, marker="X", facecolor=COL_HT, edgecolor="0", linewidth=0.5, zorder=3)
    axC.scatter(x, y_cal, s=18, marker="o", facecolor=COL_ECHT, edgecolor="0", linewidth=0.5, zorder=2)

    if x.size > 2:
        xs = np.sort(x)
        line_unc = axC.plot(
            xs,
            np.polyval(np.polyfit(x, y_unc, 1), xs),
            color=COL_HT,
            linewidth=1.2,
            zorder=0,
        )[0]
        line_cal = axC.plot(
            xs,
            np.polyval(np.polyfit(x, y_cal, 1), xs),
            color=COL_ECHT,
            linewidth=1.2,
            zorder=0,
        )[0]

        outline = [pe.Stroke(linewidth=2.2, foreground="black"), pe.Normal()]
        line_unc.set_path_effects(outline)
        line_cal.set_path_effects(outline)




    _style_cart_axis(axC)
    axC.set_xlabel(panel_c_xlabel)
    axC.set_ylabel(r"Mean phase error $[^\circ]$")
    axC.yaxis.set_label_coords(-0.18, 0.5)

    fig.tight_layout()

    # --- force panel C to match polar panels' height (as in plot.py) ---
    bboxA = axA.get_position()
    bboxB = axB.get_position()

    top = max(bboxA.y1, bboxB.y1) + 0.06
    bottom = max(bboxA.y0, bboxB.y0)
    height = top - bottom

    bboxC = axC.get_position()
    axC.set_position([bboxC.x0, bottom, bboxC.width, height])

    # --- headings (aligned) ---
    bbA, bbB, bbC = axA.get_position(), axB.get_position(), axC.get_position()
    y_head = 0.96
    fig.text(bbA.x0, y_head, r"$\mathbf{(A)}$ Uncalibrated phase error", ha="left", va="top")
    fig.text(bbB.x0, y_head, r"$\mathbf{(B)}$ Calibrated phase error", ha="left", va="top")
    fig.text(bbC.x0 - 0.02, y_head, panel_c_title, ha="left", va="top")

    bbA, bbB, bbC = axA.get_position(), axB.get_position(), axC.get_position()
    x1 = bbA.x0 + bbA.width / 2
    x2 = bbB.x0 + bbB.width / 2
    y = max(bbA.y1, bbB.y1) + 0.01
    h = 0.05

    fig.add_artist(Line2D(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        transform=fig.transFigure,
        color="black",
        linewidth=0.9,
    ))

    fig.text(
        (x1 + x2) / 2,
        y - h + 0.05,
        rf"({p_txt}, circular permutation)",
        ha="center",
        va="bottom",
        fontsize=7,
        transform=fig.transFigure,
    )

    fig.savefig(f"{save_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{save_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)