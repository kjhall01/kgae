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

#alphas = kgae.smooth_spatial(alphas, sigma_latlon=1.5)
#betas  = kgae.smooth_spatial(betas,  sigma_latlon=1.5)

# Latents -> (time, seed, mode)
latents = xr.concat([z.expand_dims(seed=[i]) if "seed" not in z.dims else z
                     for i, z in enumerate(zs)], dim="seed")
latents = latents.transpose("time", "seed", "mode")

import numpy as np
import xarray as xr

def sel_mode_1based(da, dim, m):
    if dim in da.coords and np.any(da[dim].values == m):
        return da.sel({dim: m})
    return da.isel({dim: m - 1})


# ------------------------------------------------------------
# CONDITIONAL BACKGROUND + CONDITIONAL ENSO AMPLITUDE CHECKS
# (paste after you already have: ssta (time,lat,lon), latents, cols,
#  alphas/betas loaded, and sel_mode_1based defined)
# ------------------------------------------------------------
import numpy as np
import xarray as xr

# --- pick which entry of cols is dec/int/qb in your convention ---
m_dec, m_int, m_qb = cols[0], cols[1], cols[2]   # e.g. [5,4,3]

# ensemble-mean latents
zbar = latents.mean("seed").transpose("time", "mode")

# patterns
alpha_int = sel_mode_1based(alphas, "tendency_mode", m_int).transpose("lat", "lon")


sst = xr.open_dataset(os.path.expanduser("~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc")).sst
end_year = int(str(experiment)[-4:])
sst = sst.sel(time=slice(None, pd.Timestamp(end_year, 12, 31)))
sst = sst.rename({k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in sst.dims})

ssta, fit, gwm, p = kgae.global_detrend(sst, deg=2)
ssta, mc = kgae.remove_climo(ssta)
ssta = ssta.transpose("time", "lat", "lon")

# ------------------------------------------------------------
# 3-panel disentangling plot:
# (1) pos z_dec - neg z_dec      (signed contrast; mostly linear state + some nonlinear)
# (2) |z_dec| high - |z_dec| low (amplitude/curvature-like contrast)
# (3) alpha_dec                 (pure linear decadal sensitivity pattern)
#
# Assumes you already have:
#   - ssta: DataArray (time, lat, lon)
#   - latents: DataArray (time, seed, mode)
#   - alphas: DataArray (tendency_mode, lat, lon)
#   - cols, sel_mode_1based, proj/data_crs, cmap
# ------------------------------------------------------------
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# --- configure which mode is decadal in your cols convention ---
m_dec = cols[0]      # e.g. cols=[5,4,3] so dec=5
k = 1.0              # threshold in sigma units for "high"
other_sigma = 1.0    # gate other modes in cols to within +/- other_sigma*sigma; set None to disable
smooth_sigma = None  # e.g. 1.5 if you want kgae.smooth_spatial; otherwise None

# ensemble-mean latent + select modes
zbar = latents.mean("seed").transpose("time", "mode")
z_dec = sel_mode_1based(zbar, "mode", m_dec)

# optionally gate only other modes that ARE in cols (not modes outside cols)
if other_sigma is None:
    gate = xr.ones_like(z_dec, dtype=bool)
else:
    gate = xr.ones_like(z_dec, dtype=bool)
    for m_other in cols:
        if m_other == m_dec:
            continue
        z_other = sel_mode_1based(zbar, "mode", m_other)
        sig_other = float(z_other.std("time").values)
        gate = gate & (np.abs(z_other) <= other_sigma * sig_other)

# align to common time axis
z_dec, gate, ssta_al = xr.align(z_dec, gate, ssta.transpose("time", "lat", "lon"), join="inner")

# thresholds
sig_dec = float(z_dec.std("time").values)
thr = k * sig_dec

pos = (z_dec >  thr) & gate
neg = (z_dec < -thr) & gate
abs_hi = (np.abs(z_dec) >  thr) & gate
abs_lo = (np.abs(z_dec) <= thr) & gate

# composites
mean_pos = ssta_al.where(pos, drop=True).mean("time")
mean_neg = ssta_al.where(neg, drop=True).mean("time")
diff_posneg = mean_pos - mean_neg

mean_abs_hi = ssta_al.where(abs_hi, drop=True).mean("time")
mean_abs_lo = ssta_al.where(abs_lo, drop=True).mean("time")
diff_abshi = mean_abs_hi - mean_abs_lo

# alpha_dec pattern
alpha_dec = sel_mode_1based(alphas, "tendency_mode", m_dec).transpose("lat", "lon")

# (optional) smoothing if you want it consistent with other figures
if smooth_sigma is not None:
    import kgae
    diff_posneg = kgae.smooth_spatial(diff_posneg, sigma_latlon=smooth_sigma)
    diff_abshi  = kgae.smooth_spatial(diff_abshi,  sigma_latlon=smooth_sigma)
    alpha_dec   = kgae.smooth_spatial(alpha_dec,   sigma_latlon=smooth_sigma)

# shared levels (makes comparison easier)
fields_for_scale = [diff_posneg, diff_abshi, alpha_dec]
vmax = max(float(np.nanmax(np.abs(f.values))) for f in fields_for_scale)
if not np.isfinite(vmax) or vmax == 0:
    vmax = 1.0

# plot
proj = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree(central_longitude=180)

fig, axes = plt.subplots(
    1, 3, figsize=(15.5, 4.2),
    subplot_kw={"projection": proj},
    constrained_layout=True
)

titles = [
    f"(1) mean(SSTa | z_dec>+{k}σ) − mean(SSTa | z_dec<−{k}σ)\n"
    f"n+={int(pos.sum())}, n-={int(neg.sum())}, gate(other cols ±{other_sigma}σ)",
    f"(2) mean(SSTa | |z_dec|>{k}σ) − mean(SSTa | |z_dec|≤{k}σ)\n"
    f"n_hi={int(abs_hi.sum())}, n_lo={int(abs_lo.sum())}, gate(other cols ±{other_sigma}σ)",
    f"(3) α_dec (linear decadal sensitivity pattern)"
]

for ax, fld, ttl in zip(axes, [diff_posneg, diff_abshi, alpha_dec], titles):
    im = ax.pcolormesh(
        fld.lon, fld.lat, fld,
        cmap=cmap, shading="auto",
        vmin=-vmax, vmax=vmax,
        transform=data_crs
    )
    ax.coastlines(linewidth=0.6)
    ax.set_title(ttl, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

plt.show()

# optional: quick pattern correlations vs alpha_int if you have alpha_int in scope
# print("corr(diff_posneg, alpha_int) =", pattcorr(diff_posneg, alpha_int))
# print("corr(diff_abshi , alpha_int) =", pattcorr(diff_abshi , alpha_int))
# print("corr(alpha_dec  , alpha_int) =", pattcorr(alpha_dec  , alpha_int))
