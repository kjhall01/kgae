import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter

# ============================
# SETTINGS (match plot_alphas_final.py vibe)
# ============================
experiment_root = Path("kgvae_cv2-1940-2014")  # <-- CHANGE: folder containing seed_* dirs
seed_glob = "seed*"

# plotting
nlev_r2 = 21
nlev_gap = 21
r2_levels = np.linspace(-0.5, 1.0, nlev_r2)         # R^2 can be negative out-of-sample
gap_levels = np.linspace(-0.5, 0.5, nlev_gap)       # will auto-rescale per mode below if desired

cmap_r2 = "viridis"
cmap_gap = "RdBu_r"

# stippling density (smaller = denser)
stipple_stride = 3

# Require this fraction of seeds to agree on significance for stippling
sig_seed_frac = 0.5   # 0.5 = majority vote; 0.8 is stricter; 1.0 is unanimous

# Modes to plot
tendency_modes = [5, 4, 3]
tendency_mode_labels = {
    5: "Decadal Mode",
    4: "Interannual Mode",
    3: "Quasibiennial Mode",
}

# filenames produced by training script
FN_R2_CV   = "xval_tendencyreg_r2_median.nc"
FN_R2_TEST = "test_tendencyreg_r2_median.nc"
FN_GAP_MED = "tendencyreg_gen_gap_median.nc"
FN_GAP_SIG = "tendencyreg_gen_gap_sig.nc"   # <-- please write this correctly

sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.mean('time') 
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))
mask = xr.ones_like(sst).where(~np.isnan(sst))  # 1 over ocean, nan over land; used to mask out land in gap plots

# ============================
# Helpers
# ============================
def _open_singlevar(fp: Path) -> xr.DataArray:
    ds = xr.open_dataset(fp)
    var = list(ds.data_vars)[0]
    return ds[var]

def _ensure_mode_dim(da: xr.DataArray) -> xr.DataArray:
    # Support either "tendency_mode" or "mode"
    if "mode" in da.dims and "tendency_mode" not in da.dims:
        da = da.rename({"mode": "tendency_mode"})
    if "tendency_mode" not in da.dims:
        raise ValueError(f"Expected a mode dim; got dims {da.dims}")
    return da

def _load_across_seeds(root: Path):
    seed_dirs = sorted([p for p in root.glob(seed_glob) if p.is_dir()])
    if not seed_dirs:
        raise RuntimeError(f"No seed dirs found under {root} with glob '{seed_glob}'")

    r2cv_list, r2te_list, gap_list, sig_list = [], [], [], []

    for sdir in seed_dirs:
        r2cv = _ensure_mode_dim(_open_singlevar(sdir / FN_R2_CV)).expand_dims(seed=[sdir.name])
        r2te = _ensure_mode_dim(_open_singlevar(sdir / FN_R2_TEST)).expand_dims(seed=[sdir.name])
        gap  = _ensure_mode_dim(_open_singlevar(sdir / FN_GAP_MED)).expand_dims(seed=[sdir.name])

        sig_fp = sdir / FN_GAP_SIG
        if sig_fp.exists():
            sig = _ensure_mode_dim(_open_singlevar(sig_fp)).expand_dims(seed=[sdir.name])
            sig_list.append(sig)

        r2cv_list.append(r2cv)
        r2te_list.append(r2te)
        gap_list.append(gap)

    r2cv_all = xr.concat(r2cv_list, dim="seed")
    r2te_all = xr.concat(r2te_list, dim="seed")
    gap_all  = xr.concat(gap_list,  dim="seed")

    sig_all = xr.concat(sig_list, dim="seed") if sig_list else None
    return r2cv_all, r2te_all, gap_all, sig_all

def _seed_median(da: xr.DataArray) -> xr.DataArray:
    return da.quantile(0.5, dim="seed", skipna=True).drop_vars("quantile", errors="ignore")

def _seed_sigmask(sig_all: xr.DataArray, frac=0.5) -> xr.DataArray:
    # sig_all: (seed, tendency_mode, lat, lon) boolean
    p = sig_all.astype(float).mean("seed")
    return p >= frac

def _stipple(ax, mask2d: xr.DataArray, stride=3):
    lat = mask2d["lat"].values
    lon = mask2d["lon"].values
    M = mask2d.values.astype(bool)
    jj, ii = np.where(M)
    if jj.size == 0:
        return
    # downsample points
    sel = (np.arange(jj.size) % stride) == 0
    ax.scatter(
        lon[ii[sel]], lat[jj[sel]],
        s=2, marker=".", linewidths=0, alpha=0.7,
        transform=ccrs.PlateCarree(central_longitude=180)
    )

def _add_map_features(ax):
    ax.coastlines(linewidth=0.7)

def _plot_panel(ax, field, levels, cmap, title, fmt=None, vmin=None, vmax=None):
    im = ax.pcolormesh(
        field["lon"], field["lat"], field,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(central_longitude=180)
    )
    _add_map_features(ax)
    ax.set_title(title)
    if fmt is not None:
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.95)
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
    else:
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.95)
    return im


# ============================
# Main
# ============================
r2cv_all, r2te_all, gap_all, sig_all = _load_across_seeds(experiment_root)

# Take seed-median for shading.
r2cv_med = _seed_median(r2cv_all)
r2te_med = _seed_median(r2te_all)
gap_med  = _seed_median(gap_all)

gap_sig = None
if sig_all is not None:
    gap_sig = _seed_sigmask(sig_all, frac=sig_seed_frac)

# sanity print (catch R^2>>1 immediately)
print("R2_cv median min/max:", float(r2cv_med.min()), float(r2cv_med.max()))
print("R2_test median min/max:", float(r2te_med.min()), float(r2te_med.max()))
print("Gap median min/max:", float(gap_med.min()), float(gap_med.max()))

# Plot one figure per mode (3 panels)
outdir = experiment_root / "figs_r2"
outdir.mkdir(exist_ok=True)

for mi in tendency_modes:
    A = r2cv_med.sel(tendency_mode=mi) *mask
    B = r2te_med.sel(tendency_mode=mi) *mask
    C = gap_med.sel(tendency_mode=mi) *mask

    # Optional: autoscale gap levels per mode to avoid “all white” maps
    g_lim = float(np.nanpercentile(np.abs(C.values), 98))
    g_lim = max(g_lim, 0.05)
    gap_levels_mode = np.linspace(-g_lim, g_lim, nlev_gap)

    fig = plt.figure(figsize=(14, 4.5))
    proj = ccrs.PlateCarree(central_longitude=180)

    ax1 = plt.subplot(1, 3, 1, projection=proj)
    ax2 = plt.subplot(1, 3, 2, projection=proj)
    ax3 = plt.subplot(1, 3, 3, projection=proj)

    label = tendency_mode_labels.get(int(mi), f"Mode {mi}")
    S = gap_sig.sel(tendency_mode=mi)

    _plot_panel(ax1, A, r2_levels, cmap_r2, vmin=-0.1, vmax=0.1, title=f"{label}\nCV median $R^2$", fmt="%.2f")
    _plot_panel(ax2, B, r2_levels, cmap_r2, vmin=-0.1, vmax=0.1, title="Test median $R^2$", fmt="%.2f")
    _plot_panel(ax3, C.where(~S), gap_levels_mode,  cmap=cmap_gap,vmin=-0.25, vmax=0.25, title="Median generalization gap\n$\\Delta R^2 = R^2_{test}-R^2_{cv}$", fmt="%.2f")

    #if gap_sig is not None:
    #    _stipple(ax3, S, stride=stipple_stride)

    plt.tight_layout()
    plt.show() 

    #fout = outdir / f"r2_threepanel_mode_{int(mi)}.png"
    #plt.savefig(fout, dpi=200, bbox_inches="tight")
    #plt.close(fig)

print("Saved figures to:", outdir)
