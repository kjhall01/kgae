import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import kgae
from sklearn.decomposition import PCA

# ---------------------------
# Regression + preprocessing
# ---------------------------

def regress_out(y, x, dim="time"):
    """Remove linear regression of y on x (gridpoint-wise if y has lat/lon)."""
    y, x = xr.align(y, x, join="inner")
    x_anom = x - x.mean(dim)
    y_anom = y - y.mean(dim)
    b = xr.cov(y_anom, x_anom, dim=dim) / x_anom.var(dim)
    return y - b * x  # keeps mean of y

def standardize_ts(x, dim="time", eps=1e-12):
    x = x - x.mean(dim)
    s = x.std(dim)
    s = xr.where(s < eps, 1.0, s)
    return x / s

def regression_slope_map(grid, ts, dim="time", standardize_index=False):
    """
    grid: DataArray with dims (time, lat, lon)
    ts:   DataArray with dim (time,)
    Returns slope map b(lat,lon) for grid ~ b*ts + a
    """
    grid, ts = xr.align(grid, ts, join="inner")

    if standardize_index:
        ts_use = standardize_ts(ts, dim=dim)
    else:
        ts_use = ts - ts.mean(dim)

    grid_anom = grid - grid.mean(dim)

    var = ts_use.var(dim)
    # avoid divide-by-zero
    var = xr.where(var == 0, np.nan, var)

    b = xr.cov(grid_anom, ts_use, dim=dim) / var
    return b

# ---------------------------
# Block bootstrap
# ---------------------------

BLOCK_LEN_MONTHS = 12

def moving_block_bootstrap_indices(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    """Moving-block bootstrap indices for length-n."""
    if block_len <= 1:
        return rng.integers(0, n, size=n)

    if n <= block_len:
        start = int(rng.integers(0, n))
        return (np.arange(n) + start) % n

    n_starts = n - block_len + 1
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n_starts, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)
    return idx[:n]

def bootstrap_slope_patterns(
    rng: np.random.Generator,
    grid: xr.DataArray,  # (time, lat, lon)
    ts: xr.DataArray,    # (time,)
    n_boot=200,
    block_len=12,
    standardize_index=False,
    dim="time",
):
    """Return bootstrapped slope maps: (boot, lat, lon)."""
    grid, ts = xr.align(grid, ts, join="inner")
    n = ts.sizes[dim]
    out = []
    for _ in range(n_boot):
        idx = moving_block_bootstrap_indices(rng, n, block_len)
        pat = regression_slope_map(
            grid.isel({dim: idx}),
            ts.isel({dim: idx}),
            dim=dim,
            standardize_index=standardize_index,
        )
        out.append(pat)
    return xr.concat(out, dim="boot")

def significance_mask_from_ci(boot_patterns: xr.DataArray, alpha=0.05):
    """Significant if CI excludes 0 at each gridpoint."""
    lo = boot_patterns.quantile(alpha / 2, dim="boot")
    hi = boot_patterns.quantile(1 - alpha / 2, dim="boot")
    sig = (lo > 0) | (hi < 0)
    return sig, lo, hi

# ---------------------------
# Plotting
# ---------------------------

def make_levels_from_data(data, n_levels=21, vlim=None):
    """Symmetric levels around 0."""
    if vlim is None:
        vmax = float(np.nanmax(np.abs(data)))
    else:
        vmax = float(vlim)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return np.linspace(-1.2, 1.2, n_levels)

def plot_panel(ax, pat, sig, title, levels, cmap="RdBu_r", linewidth=0.9, central_longitude=180):
    data_crs = ccrs.PlateCarree(central_longitude=central_longitude)
    # filled ONLY where significant
    pat_sig = pat#.where(sig)
    cf = ax.contourf(
        pat["lon"], pat["lat"], pat_sig,
        levels=levels, 
        cmap=cmap, 
        extend="both",
        transform=data_crs
    )

    # contours everywhere: dashed negative, solid positive
    neg_levels = [l for l in levels if l < 0]
    pos_levels = [l for l in levels if l > 0]
    if neg_levels:
        ax.contour(
            pat["lon"], pat["lat"], pat,
            levels=neg_levels, colors="k",
            linestyles="dashed", linewidths=linewidth,
            transform=data_crs
        )
    if pos_levels:
        ax.contour(
            pat["lon"], pat["lat"], pat,
            levels=pos_levels, colors="k",
            linestyles="solid", linewidths=linewidth,
            transform=data_crs
        )

    ax.coastlines()
    ax.set_title(title)
    return cf

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    time_start, time_end = "1940-01-01", "2014-12-31"
    rng = np.random.default_rng(42)

    # knobs
    n_boot = 10
    block_len = 12
    alpha_sig = 0.5
    n_levels = 21
    vlim = None                 # set e.g. 0.6 if you want fixed scaling
    standardize_index = True    # True => pattern is °C per 1σ(index)
    central_longitude = 180

    # preprocess sst
    sst = xr.open_dataset(
        "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"
    ).sst.sel(time=slice(time_start, time_end))
    sst = sst.rename({"latitude": "lat", "longitude": "lon"})

    ssta, fit, gwm, p = kgae.global_detrend(sst, deg=2)
    ssta, monthly_clim = kgae.remove_climo(ssta)

    # indices
    oni = ssta.sel(lat=slice(-5, 5), lon=slice(10, 60)).mean(["lat", "lon"])
    oni.name = "ONI"


    pmm = xr.open_dataset("pmm.nc").value
    pmm, _ = xr.align(pmm, oni, join="inner")
    pmm.name = "PMM"

    nino4 = ssta.sel(lat=slice(-5, 5), lon=slice(-20, 30)).mean(["lat", "lon"])
    nino3 = ssta.sel(lat=slice(-5, 5), lon=slice(30, 90)).mean(["lat", "lon"])
    nino12 = ssta.sel(lat=slice(-10, 0), lon=slice(90, 100)).mean(["lat", "lon"])

    # simple Takahashi-style combos (standardize at end by standardize_index=True in regression)
    C_index = 1.7 * nino4 - 0.1 * nino12
    C_index.name = "C-Index"
    E_index = nino12 - 0.5 * nino4
    E_index.name = "E-Index"

    # PDO (EOF1 of North Pacific SSTa)
    np_sst = ssta.sel(lat=slice(21, None)).stack(point=("lat", "lon")).dropna("point", how="any")
    pca = PCA(n_components=1)
    pca.fit(np_sst.values)
    pdo_index = pca.transform(np_sst.values).squeeze() * -1
    pdo_index = xr.DataArray(pdo_index, name="PDO", dims=("time",), coords={"time": np_sst.time})

    # IPO / TPI
    #t1 = ssta.sel(lat=slice(25, 45), lon=slice(-40, 35)).mean(["lat", "lon"])
    #t2 = ssta.sel(lat=slice(-10, 10), lon=slice(-10, 90)).mean(["lat", "lon"])
    #t3 = ssta.sel(lat=slice(-50, -15), lon=slice(-30, 20)).mean(["lat", "lon"])
    #tpi_index = t2 - (t1 + t3) / 2
    #tpi_index.name = "TPI"

    npgo_index = xr.open_dataset('npgo.nc').value 
    npgo_index, _ = xr.align(npgo_index, oni, join="inner")
    npgo_index.name = "NPGO"

    cold_tongue_index = ssta.sel(lat=slice(-6, 6), lon=slice(0, 90)).mean(["lat", "lon"])
    cold_tongue_index.name = "CTI"

    # residual fields for “remove ENSO/CTI”
    ssta_no_cti = regress_out(ssta, cold_tongue_index)
    ssta_no_oni = regress_out(ssta, oni)

    indices = [npgo_index, oni, E_index, C_index, pmm, pdo_index]
    fields  = [ssta, ssta, ssta, ssta, ssta_no_cti, ssta]  # adjust as desired

    # first pass slopes for shared color scale
    slope_maps = []
    for idx, grd in zip(indices, fields):
        grd_a, idx_a = xr.align(grd, idx, join="inner")
        slope_maps.append(regression_slope_map(grd_a, idx_a, standardize_index=standardize_index))
    slopes_da = xr.concat(slope_maps, dim="index").assign_coords(index=[p.name for p in indices])
    
    levels = make_levels_from_data(slopes_da.values, n_levels=n_levels, vlim=vlim)

    # compute slopes + significance
    slopes = []
    sigs = []
    for idx, grd in zip(indices, fields):
        print("Index:", idx.name)
        grd_a, idx_a = xr.align(grd, idx, join="inner")

        pat = regression_slope_map(grd_a, idx_a, standardize_index=standardize_index)
      #  boot = bootstrap_slope_patterns(
      #      rng, grd_a, idx_a,
      #      n_boot=n_boot, block_len=block_len,
      #      standardize_index=standardize_index
      #  )
       # sig, lo, hi = significance_mask_from_ci(boot, alpha=alpha_sig)

        slopes.append(pat)
       # sigs.append(sig)

    names = [p.name for p in indices]
    slopes_da = xr.concat(slopes, dim="mode").assign_coords(mode=names)
    slopes_da.name = "pattern"
    slopes_da.to_netcdf("climate_mode_patterns.nc")
    indices = [idx.rename("mode") for idx in indices]
    indices_da = xr.concat([indices.drop('number') if hasattr(indices, 'number') else indices for indices in indices], dim="mode").assign_coords(mode=names)
    indices_da.name = "magnitude"
    indices_da.to_netcdf("climate_mode_indices.nc")
    
    #import sys; sys.exit() 

#    sigs_da   = xr.concat(sigs,   dim="index").assign_coords(index=[p.name for p in indices])

    # plot
    n_panels = len(indices)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(10, 4.2 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_longitude)},
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    mappable = None
    for i, name in enumerate(slopes_da["mode"].values):
        ax = axes[i]
        mappable = plot_panel(
            ax,
            slopes_da.sel(mode=name),
            #sigs_da.sel(index=name),
            None,
            title=f"{name}",
            levels=levels,
            cmap="RdBu_r",
            linewidth=0.9,
            central_longitude=central_longitude,
        )

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    if mappable is not None:
        units = "°C / 1σ(index)" if standardize_index else "°C / index-unit"
        cbar = fig.colorbar(mappable, ax=axes[:n_panels], orientation="horizontal", pad=0.04, fraction=0.06)
        cbar.set_label(f"Regression slope ({units})")
    plt.savefig('climate_modes.png', dpi=300)
    plt.show()
