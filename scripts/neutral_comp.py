import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import kgae
import cartopy.crs as ccrs
from pathlib import Path
from matplotlib.gridspec import GridSpec

# -----------------------------
# Settings
# -----------------------------
experiment = Path("kgvae_cv1940-2014")
n_ensemble = 30
data_set = "xval"

thresh = 0.1                      # frac-of-ensemble sig threshold for alpha/beta rows
cols = [5,4,3]                   # (Decadal, Interannual, Quasibiennial)
col_titles = ["Decadal (i=3)", "Interannual (i=2)", "Quasibiennial (i=1)"]

# composite selection + scaling knobs
k_sigma = 1.0                      # target-mode event threshold: z_m > +kσ, z_m < -kσ
other_sigma = 1.0                  # restrict *other modes in cols* to +/- other_sigma*sigma
scale_to = "none"                  # "none" | "1sigma" | "2sigma"  (scales using realized Δz)
eps = 1e-12

# bootstrap + plotting knobs
n_boot = 500
boot_alpha = 0.05                  # two-sided
cmap = "RdBu_r"
n_fill_levels = 21
n_contours = 11

proj = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree(central_longitude=180)

# -----------------------------
# Load ensemble means
# -----------------------------
alphas = 0
betas = 0
alpha_sigs = 0
beta_sigs = 0
zs = []

for seed in range(n_ensemble):
    alpha = xr.open_dataset(experiment / f"seed{seed}" / f"{data_set}_alpha.nc")["__xarray_dataarray_variable__"]
    alpha_sig = xr.open_dataset(
        experiment / f"seed{seed}" / f'{data_set}_alpha_sig{"s" if data_set != "xval" else ""}.nc'
    )["__xarray_dataarray_variable__"]

    beta = xr.open_dataset(experiment / f"seed{seed}" / f"{data_set}_beta.nc")["__xarray_dataarray_variable__"]
    beta_sig = xr.open_dataset(
        experiment / f"seed{seed}" / f'{data_set}_beta_sig{"s" if data_set != "xval" else ""}.nc'
    )["__xarray_dataarray_variable__"]

    z = xr.open_dataset(experiment / f"seed{seed}" / f"{data_set}_latent.nc")["__xarray_dataarray_variable__"]
    zs.append(z)

    alphas += alpha
    betas += beta
    alpha_sigs += alpha_sig
    beta_sigs += beta_sig

alphas = alphas / n_ensemble
betas = betas / n_ensemble
alpha_sigs = alpha_sigs / n_ensemble
beta_sigs = beta_sigs / n_ensemble

# Latents -> (time, seed, mode)
latents = xr.concat([z.expand_dims(seed=[i]) if "seed" not in z.dims else z
                     for i, z in enumerate(zs)], dim="seed")
latents = latents.transpose("time", "seed", "mode")

sst = xr.open_dataset(os.path.expanduser("~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc")).sst
end_year = int(str(experiment)[-4:])
sst = sst.sel(time=slice(None, pd.Timestamp(end_year, 12, 31)))
sst = sst.rename({k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in sst.dims})

ssta, fit, gwm, p = kgae.global_detrend(sst, deg=2)
ssta, mc = kgae.remove_climo(ssta)
ssta = ssta.transpose("time", "lat", "lon")

# --- Neutral-regime bootstrapped SSTA composites:  -1σ < z < 1σ  ---
# Assumes the prerequisite arrays are already available:
#   ssta: xarray.DataArray with dims ('time','lat','lon')
#   latents: xarray.DataArray with dims ('time','seed','mode')   (we'll mean over seed)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# -----------------------------
# knobs
# -----------------------------
mode_idx = 3          # <-- pick latent mode (0-based)
k_sigma = 1.0         # neutral band: -k_sigma < z < +k_sigma   (in *standardized* units)
n_boot = 500
boot_alpha = 0.05     # two-sided 95% CI
seed = 42

proj = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree(central_longitude=180)

# -----------------------------
# helpers
# -----------------------------
#def standardize(da_time):
#    mu = da_time.mean("time")
#    sd = da_time.std("time")
#    return (da_time - mu) / (sd + 1e-12)

def bootstrap_composite_mean(ssta, mask_time, n_boot=500, alpha=0.05, seed=0):
    """
    ssta: (time, lat, lon) DataArray
    mask_time: boolean DataArray over time (True = in regime)
    Returns: comp_mean, comp_lo, comp_hi (lat, lon)
    """
    # indices of selected times
    t_idx = np.where(mask_time.values)[0]
    N = t_idx.size
    if N < 5:
        raise ValueError(f"Too few samples in neutral regime: N={N}")

    rng = np.random.default_rng(seed)

    # bootstrap means (n_boot, lat, lon)
    boots = []
    for b in range(n_boot):
        draw = rng.choice(t_idx, size=N, replace=True)
        boots.append(ssta.isel(time=draw).mean("time"))
    boots = xr.concat(boots, dim="boot").assign_coords(boot=np.arange(n_boot))

    comp_mean = ssta.isel(time=t_idx).mean("time")
    comp_lo = boots.quantile(alpha / 2, dim="boot")
    comp_hi = boots.quantile(1 - alpha / 2, dim="boot")
    return comp_mean, comp_lo, comp_hi

def plot_map(da2d, title, ax, vlim=None):
    # infer lon/lat coords
    lon = da2d["lon"]
    lat = da2d["lat"]
    if vlim is None:
        vmax = float(np.nanmax(np.abs(da2d.values)))
        vlim = (-vmax, vmax)

    pcm = ax.pcolormesh(lon, lat, da2d, transform=data_crs,
                        vmin=vlim[0], vmax=vlim[1], shading="auto", cmap="RdBu_r")
    ax.coastlines(linewidth=0.6)
    ax.set_title(title)
    return pcm

# -----------------------------
# prep z (mean over seeds), align time, build neutral mask
# -----------------------------
z = latents.mean("seed").isel(mode=mode_idx)           # (time,)
z = z.sel(time=ssta.time)                              # align to ssta timeline (safe even if identical)
#z = standardize(z)                                     # unit std
low = z.quantile(0.33, 'time')
high = z.quantile(0.67, 'time')
mask = (z > low) & (z < high)

print(f"Neutral samples (-{k_sigma}σ < z < {k_sigma}σ): {int(mask.sum().values)} / {z.sizes['time']}")

# -----------------------------
# compute bootstrap composite
# -----------------------------
comp, lo, hi = bootstrap_composite_mean(ssta, mask, n_boot=n_boot, alpha=boot_alpha, seed=seed)
ci_half = 0.5 * (hi - lo)

# -----------------------------
# plot
# -----------------------------
# common symmetric limits for mean & CI (separate for readability)
vmax_mean = float(np.nanmax(np.abs(comp.values)))
vlim_mean = (-vmax_mean, vmax_mean)

vmax_ci = float(np.nanmax(ci_half.values))
vlim_ci = (0.0, vmax_ci)

fig = plt.figure(figsize=(12, 4), constrained_layout=True)
gs = fig.add_gridspec(1, 2)

ax0 = fig.add_subplot(gs[0, 0], projection=proj)
pcm0 = plot_map(comp, f"SSTA composite mean (mode={mode_idx}, |z|<{k_sigma}σ)", ax0, vlim=vlim_mean)
cb0 = plt.colorbar(pcm0, ax=ax0, shrink=0.85, pad=0.02)
cb0.set_label("K")

ax1 = fig.add_subplot(gs[0, 1], projection=proj)
lon = ci_half["lon"]; lat = ci_half["lat"]
pcm1 = ax1.pcolormesh(lon, lat, ci_half, transform=data_crs,
                      vmin=vlim_ci[0], vmax=vlim_ci[1], shading="auto", cmap="viridis")
ax1.coastlines(linewidth=0.6)
ax1.set_title(f"Bootstrap 95% CI half-width (n_boot={n_boot})")
cb1 = plt.colorbar(pcm1, ax=ax1, shrink=0.85, pad=0.02)
cb1.set_label("K")

plt.show()
