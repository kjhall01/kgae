import kgae
import xarray as xr
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.stats import gaussian_kde

# -----------------------------
# Load + interleave helpers
# -----------------------------
l = kgae.open_latents("xval", experiment="kgvae_cv1940-2014")
l2 = xr.concat([l.isel(time=slice(i, None, 5)).drop("time") for i in range(5)], "fold")

modes = [1, 2, 3, 4, 5]
ks = 15
dx = 5
kernel_size_pre = 7
ticks = [40, 20, 12, 7, 5, 2, 1]

# -----------------------------
# PSD-based sorting (DESCENDING frequency)
# -----------------------------
def psd_based_mode_order_desc(
    psd_fm: np.ndarray,
    freqs: np.ndarray | None = None,
    mode_axis: int = 1,
    peak_weight: float = 1.0,
    centroid_weight: float = 0.5,
    bandwidth_weight: float = 0.25,
    peak_to_total_weight: float = 0.25,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Returns indices that sort modes from HIGH to LOW frequency (descending),
    using a composite PSD-based score.
    psd_fm is (freq, mode).
    freqs is (freq,).
    """
    psd = np.asarray(psd_fm)
    if psd.ndim != 2:
        raise ValueError(f"psd_fm must be 2D (freq, mode); got shape {psd.shape}")
    if mode_axis != 1:
        psd = np.moveaxis(psd, mode_axis, 1)

    nfreq, nmode = psd.shape
    if freqs is None:
        f = np.arange(nfreq, dtype=float)
    else:
        f = np.asarray(freqs, dtype=float)
        if f.shape != (nfreq,):
            raise ValueError(f"freqs must have shape ({nfreq},); got {f.shape}")

    psd = np.maximum(psd, 0.0)
    P = psd / (np.sum(psd, axis=0, keepdims=True) + eps)  # (freq, mode)

    f_centroid = np.sum(f[:, None] * P, axis=0)
    f2_centroid = np.sum((f[:, None] ** 2) * P, axis=0)
    bandwidth = np.sqrt(np.maximum(f2_centroid - f_centroid**2, 0.0))

    peak_idx = np.argmax(psd, axis=0)
    f_peak = f[peak_idx]

    peak_power = psd[peak_idx, np.arange(nmode)]
    total_power = np.sum(psd, axis=0) + eps
    peak_to_total = peak_power / total_power

    score = (
        peak_weight * f_peak
        + centroid_weight * f_centroid
        + bandwidth_weight * bandwidth
        + peak_to_total_weight * peak_to_total
    )

    # DESCENDING frequency score (high -> low)
    return np.argsort(score)[::-1]


def simple_peakfreq_order_desc(psd_fm: torch.Tensor, freqs_fm: torch.Tensor, eps=1e-12) -> np.ndarray:
    """
    Simple sort: order modes by frequency of maximum power, descending.
    psd_fm: (freq, mode)
    freqs_fm: (freq, mode) or (freq,)
    """
    if freqs_fm.ndim == 2:
        f = freqs_fm[:, 0]
    else:
        f = freqs_fm
    peak_idx = torch.argmax(psd_fm, dim=0)     # (mode,)
    f_peak = f[peak_idx].clamp_min(eps)        # (mode,)
    order = torch.argsort(f_peak, descending=True).cpu().numpy()
    return order


def peak_period_years_from_psd(psd_fm: torch.Tensor, freqs_fm: torch.Tensor, eps=1e-12) -> np.ndarray:
    """
    psd_fm: (freq, mode)
    freqs_fm: (freq, mode) or (freq,)
    Returns per-mode peak period in years, shape (mode,).
    """
    if freqs_fm.ndim == 2:
        f = freqs_fm[:, 0]
    else:
        f = freqs_fm
    peak_idx = torch.argmax(psd_fm, dim=0)  # (mode,)
    f_peak = f[peak_idx].clamp_min(eps)     # (mode,)
    period_years = (1.0 / (f_peak * 12.0)).cpu().numpy()
    return period_years


# -----------------------------
# Plotting utilities
# -----------------------------
def plot_psd_stack(ax, psd, freqs, title):
    for i in range(psd.shape[1]):
        ax.plot(1 / freqs[:, i].numpy() / 12, psd[:, i].numpy(), alpha=0.25, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Normalized power")
    ax.set_xscale("log", base=2)
    ax.set_xticks([t for t in ticks], labels=ticks)


def _kde_on_log2_period(periods, x_grid):
    """
    Fit KDE in log2 space for stability on log-scale axes.
    periods: array of periods in years (>0)
    x_grid: grid in years (>0)
    Returns density on x_grid (in year-space), properly transformed.
    """
    periods = np.asarray(periods)
    periods = periods[np.isfinite(periods) & (periods > 0)]
    if periods.size < 5:
        return None

    y = np.log2(periods)
    kde = gaussian_kde(y)

    y_grid = np.log2(x_grid)
    dens_y = kde(y_grid)

    # change-of-variables: y = log2(x) => dy/dx = 1 / (x ln 2)
    dens_x = dens_y * (1.0 / (x_grid * np.log(2.0)))
    return dens_x


def overlay_step_hist_and_kde(ax, datasets, labels, colors, bins=18, xlim=(1, 40)):
    """
    Overlay multiple step histograms (density=True) + KDEs on one axis.
    datasets: list of 1D arrays of periods (years)
    """
    # Log-spaced bins are more sensible on a log axis
    bin_edges = np.logspace(np.log2(xlim[0]), np.log2(xlim[1]), bins + 1, base=2)
    x_grid = np.logspace(np.log2(xlim[0]), np.log2(xlim[1]), 300, base=2)

    for data, lab, col in zip(datasets, labels, colors):
        d = np.asarray(data)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size == 0:
            continue

        ax.hist(d, bins=bin_edges, density=True, histtype="step", linewidth=1.8, label=lab, color=col)

        dens = _kde_on_log2_period(d, x_grid)
        if dens is not None:
            ax.plot(x_grid, dens, linewidth=2.0, color=col, alpha=0.9)

    ax.set_xscale("log", base=2)
    ax.set_xticks([t for t in ticks], labels=ticks)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylabel("Density")


# -----------------------------
# PRE: plot PSDs per original mode
# -----------------------------
fig, ax = plt.subplots(ncols=1, nrows=len(modes), figsize=(9, 3.0 * len(modes)), sharex=True)
fig.suptitle("Pre-sorting: PSD stacks")

for r, mode in enumerate(modes):
    all_latents = l2.sel(mode=mode).stack(sample=("seed", "fold"))
    all_latents = all_latents.transpose("time", "sample")
    torch_latents = torch.from_numpy(all_latents.values).float()

    psd, freqs = kgae.compute_smoothed_power_spectra(torch_latents, kernel_size=kernel_size_pre, dx=dx)
    plot_psd_stack(ax[r], psd, freqs, title=f"Original mode={mode}")

ax[-1].set_xlabel("Period (years)")
plt.tight_layout()
plt.show()


# -----------------------------
# BUILD POST: two re-bucketings (composite vs simple), both DESC frequency
# -----------------------------
new_modes_composite = {1: [], 2: [], 3: [], 4: [], 5: []}
new_modes_simple = {1: [], 2: [], 3: [], 4: [], 5: []}

peak_periods_pre = {k: [] for k in modes}
peak_periods_post_composite = {k: [] for k in modes}
peak_periods_post_simple = {k: [] for k in modes}

n_seeds = int(l.sizes["seed"])
n_folds = 5

for seed in range(n_seeds):
    for fold in range(n_folds):
        data = l.isel(time=slice(fold, None, 5), seed=seed).transpose("time", "mode")  # (time, mode)
        torch_latents = torch.from_numpy(data.values).float()

        psd, freqs = kgae.compute_smoothed_power_spectra(torch_latents, kernel_size=ks, dx=dx)

        # per-original-mode peak periods (for "pre")
        pre_peaks = peak_period_years_from_psd(psd, freqs)  # length=mode
        for k in range(5):
            peak_periods_pre[k + 1].append(float(pre_peaks[k]))

        # composite ordering (DESC freq)
        order_comp = psd_based_mode_order_desc(psd.numpy(), freqs[:, 0].numpy(), mode_axis=1)

        # simple ordering (DESC freq)
        order_simple = simple_peakfreq_order_desc(psd, freqs)

        # append sorted latent series into buckets
        for newmode in range(5):
            new_modes_composite[newmode + 1].append(torch_latents[:, order_comp[newmode]].reshape(-1, 1))
            new_modes_simple[newmode + 1].append(torch_latents[:, order_simple[newmode]].reshape(-1, 1))

        # store post peak periods for each newmode (based on sorted indices)
        post_peaks_comp = pre_peaks[order_comp]
        post_peaks_simple = pre_peaks[order_simple]
        for newmode in range(5):
            peak_periods_post_composite[newmode + 1].append(float(post_peaks_comp[newmode]))
            peak_periods_post_simple[newmode + 1].append(float(post_peaks_simple[newmode]))

new_modes_composite = {k: torch.hstack(new_modes_composite[k]) for k in new_modes_composite.keys()}
new_modes_simple = {k: torch.hstack(new_modes_simple[k]) for k in new_modes_simple.keys()}


# -----------------------------
# POST: plot PSD stacks for composite + simple (optional but useful)
# -----------------------------
fig, ax = plt.subplots(ncols=2, nrows=len(modes), figsize=(14, 3.0 * len(modes)), sharex=True)
fig.suptitle("Post-sorting PSD stacks (left=composite, right=simple peak-frequency)")

for r, mode in enumerate(modes):
    psd_c, freqs_c = kgae.compute_smoothed_power_spectra(new_modes_composite[mode], kernel_size=ks, dx=dx)
    psd_s, freqs_s = kgae.compute_smoothed_power_spectra(new_modes_simple[mode], kernel_size=ks, dx=dx)

    plot_psd_stack(ax[r, 0], psd_c, freqs_c, title=f"New mode={mode} (composite DESC)")
    plot_psd_stack(ax[r, 1], psd_s, freqs_s, title=f"New mode={mode} (simple DESC)")

ax[-1, 0].set_xlabel("Period (years)")
ax[-1, 1].set_xlabel("Period (years)")
plt.tight_layout()
plt.show()


# -----------------------------
# FINAL COMPARISON: overlay PRE vs composite POST vs simple POST on same panel
# - step hist (no fill)
# - different outline colors
# - KDE curves for each
# -----------------------------
fig, ax = plt.subplots(ncols=1, nrows=len(modes), figsize=(10, 2.8 * len(modes)), sharex=True)
fig.suptitle("Peak-period distributions (overlay): pre vs composite vs simple (DESC sorting)")

# Choose outline colors (user asked different outlines)
colors = ["black", "tab:blue", "tab:orange"]
labels = ["Pre (original)", "Post (composite)", "Post (simple peak sort)"]

for r, mode in enumerate(modes):
    pre = np.array(peak_periods_pre[mode], dtype=float)
    post_c = np.array(peak_periods_post_composite[mode], dtype=float)
    post_s = np.array(peak_periods_post_simple[mode], dtype=float)

    overlay_step_hist_and_kde(
        ax[r],
        datasets=[pre, post_c, post_s],
        labels=labels,
        colors=colors,
        bins=20,
        xlim=(1, 40),
    )
    ax[r].set_title(f"Mode {mode}: peak-period (years)")
    ax[r].legend(loc="upper right", frameon=False)

ax[-1].set_xlabel("Period (years)")
plt.tight_layout()
plt.show()
