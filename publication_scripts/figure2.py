import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import kgae
from pathlib import Path
import torch
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import pandas as pd 
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

# -----------------------
# configuration
# -----------------------
experiment = "../final_scripts/large-ensemble-2-1940-2014"

experiment = Path(experiment)
n_ensemble = 100
modes_to_examine = [4, 3, 2]
titles = ['Decadal', 'Interannual', 'Quasibiennial']

# -----------------------
# utilities
# -----------------------

# -----------------------
# load data
# -----------------------
xval_latents = kgae.open_latents('xval', experiment=experiment, n_ensemble=n_ensemble)
test_latents = kgae.open_latents('test', experiment=experiment, n_ensemble=n_ensemble)
test_start_time = pd.to_datetime(test_latents.time.values[0]).to_pydatetime()

full_encodings = (
    xr.concat([xval_latents, test_latents], 'time')
      .isel(mode=modes_to_examine)
      .assign_coords({'mode': titles})
)
t_end = pd.to_datetime(full_encodings.time.values[-1]).to_pydatetime()


# -----------------------
# figure + layout
# -----------------------
fig = plt.figure(figsize=(8, 10))

outer = gridspec.GridSpec(
    nrows=3, ncols=1,
    height_ratios=[1.1, 2.2, 1.2],
    hspace=0.35
)

# ============================================================
# TOP: KDEs + histograms (1 × 3)
# ============================================================
gs_kde = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[0], wspace=0.25
)

kde_axs = [fig.add_subplot(gs_kde[0, 0])]
for i in range(1, 3):
    kde_axs.append(fig.add_subplot(gs_kde[0, i], sharey=kde_axs[0]))
kde_letters = ['a)', 'b)', 'c)']

# ---- shared limits across all modes ----
all_data = np.concatenate([
    full_encodings.sel(mode=m).values.reshape(-1)
    for m in titles
])

xmin, xmax = np.percentile(all_data, [0.5, 99.5])
pad = 0.15 * (xmax - xmin)
xlim = (xmin - pad, xmax + pad)

xgrid = np.linspace(*xlim, 600)
for i in range(3):
    # pull xval/test samples for this mode
    xval_data = xval_latents.isel(mode=modes_to_examine[i]).values.reshape(-1)
    test_data = test_latents.isel(mode=modes_to_examine[i]).values.reshape(-1)

    # xval histogram only (density-normalized)
    kde_axs[i].hist(
        xval_data,
        bins=40,
        density=True,
        color='gray',
        alpha=0.30,
        edgecolor='none'
    )

    # KDEs (same bandwidth so the comparison is fair)
    kde_xval = gaussian_kde(xval_data, bw_method=0.2)
    kde_test = gaussian_kde(test_data, bw_method=0.2)

    kde_axs[i].plot(xgrid, kde_xval(xgrid), color='k', linewidth=1.5, linestyle='-')
    kde_axs[i].plot(xgrid, kde_test(xgrid), color='k', linewidth=1.0, linestyle='--')

    kde_axs[i].axvline(0, color='k', linestyle='--', linewidth=0.5)

    kde_axs[i].set_xlim(xlim)
    if i > 0:
        kde_axs[i].tick_params(axis='y', which='both', left=False, labelleft=False)
        kde_axs[i].spines['left'].set_visible(False)
    else:
        kde_axs[i].tick_params(axis='y', which='both', left=True, labelleft=True)
        kde_axs[i].set_ylabel('Density')
    kde_axs[i].set_title(kde_letters[i], loc='left', fontweight='bold')
    kde_axs[i].set_xlabel("Latent Posterior Mean ($\\mu$)")

    kde_axs[i].spines['right'].set_color('none')
    kde_axs[i].spines['top'].set_color('none')

kde_axs[2].plot([], [], 'k-', label='xval')
kde_axs[2].plot([], [], 'k--', label='test')
kde_axs[2].legend(
    frameon=False,
    fontsize=7,
    loc='best'
)

# ============================================================
# MIDDLE: Time series (3 × 1)
# ============================================================
gs_ts = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=outer[1], hspace=0.15
)

timeseries_axs = [fig.add_subplot(gs_ts[0, 0])]
for i in range(1, 3):
    timeseries_axs.append(fig.add_subplot(gs_ts[i, 0], sharex=timeseries_axs[0]))

ts_letters = ['d)', 'e)', 'f)']

for i in range(3):
    mean_ts = full_encodings.sel(mode=titles[i]).mean('seed')
    lo = full_encodings.sel(mode=titles[i]).quantile(0.025, 'seed')
    hi = full_encodings.sel(mode=titles[i]).quantile(0.975, 'seed')

    timeseries_axs[i].plot(full_encodings.time, mean_ts, color='k', linewidth=1)
    timeseries_axs[i].fill_between(
        full_encodings.time, lo, hi,
        color='gray', alpha=0.3
    )

    timeseries_axs[i].axhline(0, color='k', linestyle='--', linewidth=0.5)
    timeseries_axs[i].set_ylabel(
        ts_letters[i],
        rotation=0,
        fontweight='bold',
        labelpad=15
    )
    timeseries_axs[i].set_yticks([-3, 0, 3])

    timeseries_axs[i].spines['right'].set_color('none')
    timeseries_axs[i].spines['top'].set_color('none')

    if i < 2:
        timeseries_axs[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        timeseries_axs[i].spines['bottom'].set_visible(False)
    else:
        timeseries_axs[i].tick_params(axis='x', which='both', bottom=True, labelbottom=True)

    # ---- test period shading ----
    timeseries_axs[i].set_facecolor('none')

# ---- ONE continuous test-period shading across the whole TS panel ----
bbox = mtransforms.Bbox.union([ax.get_position() for ax in timeseries_axs])

ax0 = timeseries_axs[0]
y_mid = 0.5 * sum(ax0.get_ylim())

# convert python datetimes -> matplotlib date floats
x0 = mdates.date2num(test_start_time)
x1 = mdates.date2num(t_end)

to_fig = fig.transFigure.inverted()

x0_fig = to_fig.transform(ax0.transData.transform((x0, y_mid)))[0]
x1_fig = to_fig.transform(ax0.transData.transform((x1, y_mid)))[0]

rect = mpatches.Rectangle(
    (x0_fig, bbox.y0),
    x1_fig - x0_fig,
    bbox.height,
    transform=fig.transFigure,
    color='k',
    alpha=0.06,
    linewidth=0,
    zorder=-10,      # <-- PUT IT FAR IN THE BACK
    clip_on=False
)
fig.patches.append(rect)


# ============================================================
# BOTTOM: Power spectra
# ============================================================
powerspectra_ax = fig.add_subplot(outer[2])

psd_list = []
for seed in full_encodings.seed.values:
    coeffs, freqs = kgae.compute_smoothed_power_spectra(
        torch.tensor(full_encodings.sel(seed=seed).values, dtype=torch.float32),
        dx=1,
        kernel_size=7
    )
    psd_list.append(coeffs.detach().numpy())

psd = np.stack(psd_list, axis=0)
frequencies = freqs[:, 0].detach().numpy()
psd_styles = ['-', '--', ':']
psd_lw = [1.5, 1.5, 1.5]
labels = ['Decadal', 'Interannual', 'Quasibiennial']
for i in range(3):
    mean_psd = psd[:, :, i].mean(axis=0)
    lo = np.percentile(psd[:, :, i], 2.5, axis=0)
    hi = np.percentile(psd[:, :, i], 97.5, axis=0)

    powerspectra_ax.plot(
        frequencies,
        mean_psd,
        linestyle=psd_styles[i],
        linewidth=psd_lw[i],
        color='k',
        label=labels[i]
    )
    powerspectra_ax.fill_between(frequencies, lo, hi, alpha=0.25, color='k')

ticks = [40, 20, 12, 7, 5, 2, 1]
powerspectra_ax.set_xscale('log', base=2)
powerspectra_ax.set_xticks([1 / (t * 12) for t in ticks], labels=ticks)

powerspectra_ax.set_ylabel(r"Density $\left(\frac{E_K^2}{\sum E_K^2}\right)$")
powerspectra_ax.set_xlabel('Period (years)')
powerspectra_ax.set_title('g)', loc='left', fontweight='bold')
powerspectra_ax.legend(
    loc='upper right',
    fontsize=8,
    frameon=False
)

powerspectra_ax.grid()

powerspectra_ax.spines['right'].set_color('none')
powerspectra_ax.spines['top'].set_color('none')
plt.savefig('figures/figure2_timeseries.pdf')
plt.show()

