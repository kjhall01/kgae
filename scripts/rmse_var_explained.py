import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs 
from pathlib import Path 

experiment = Path('kgvae_cv1940-2014')
n_ensemble = 30
modes_to_examine = [0,1,2,3,4]
mode_labels = ["HF1", "HF2", "Quasibiennial (i=1)", "Interannual (i=2)", "Decadal (i=3)", ]

def open_variance(data_set, n_ensemble=30):
    latents1 = []
    for seed in range(n_ensemble):
        latents = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_reconstruction_variance.nc')
        latents = latents['__xarray_dataarray_variable__']
        latents1.append(latents)
    return xr.concat(latents1, 'seed')

def open_mse(data_set, n_ensemble=30):
    latents1 = []
    for seed in range(n_ensemble):
        latents = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_mse.nc')
        latents = latents['__xarray_dataarray_variable__']
        latents1.append(latents)
    return xr.concat(latents1, 'seed')


def open_reference(data_set, n_ensemble=30):
    latents = xr.open_dataset(experiment / f'{data_set}_references.nc')
    latents = latents['__xarray_dataarray_variable__']
    return latents.var('time')

def plot_mean_with_spread(
    xval_var,
    test_var,
    titles=("Cross-validation", "Test"),
    cmap="magma",
    n_std_levels=3,
    figsize=(12, 4),
    vmin=None, 
    vmax=None
):
    """
    Side-by-side plot of mean fields with std-dev contours.

    Parameters
    ----------
    xval_var, test_var : xarray.DataArray
        Dimensions must include (lat, lon, seed)
    titles : tuple
        Titles for the two panels
    cmap : str
        Colormap for the mean field
    n_std_levels : int
        Number of contour levels for std-dev
    figsize : tuple
        Figure size
    """

    # Compute statistics
    xval_mean = xval_var.mean("seed")
    xval_std  = xval_var.std("seed")

    test_mean = test_var.mean("seed")
    test_std  = test_var.std("seed")

    # Shared color limits for means
    vmin = vmin if vmin is not None else min(xval_mean.min(), test_mean.min()).item()
    vmax = vmax if vmax is not None else max(xval_mean.max(), test_mean.max()).item()

    proj = ccrs.PlateCarree(central_longitude=180)
    data_crs = ccrs.PlateCarree(central_longitude=180)

    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        subplot_kw=dict(projection=proj),
        constrained_layout=True
    )

    for ax, mean, std, title in zip(
        axes,
        [xval_mean, test_mean],
        [xval_std, test_std],
        titles
    ):
        # Mean field
        pcm = ax.pcolormesh(
            mean.lon,
            mean.lat,
            mean,
            transform=data_crs,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto"
        )

        # Std-dev contours
        levels = np.linspace(0, std.max().item(), n_std_levels + 1)[1:]
        cs = ax.contour(
            std.lon,
            std.lat,
            std,
            levels=levels,
            colors="k",
            linewidths=0.8,
            transform=data_crs
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

        ax.coastlines(linewidth=1.0)
        ax.set_title(title, fontsize=12)

    # Shared colorbar
    cbar = fig.colorbar(
        pcm, ax=axes, orientation="vertical",
        fraction=0.045, pad=0.03
    )
    cbar.set_label("Mean value")

    plt.show()


if __name__ == "__main__":
    xval_var = open_variance('xval', n_ensemble=n_ensemble)
    ref_xval_var = open_reference('xval', n_ensemble=n_ensemble)
    test_var = open_variance('test', n_ensemble=n_ensemble)
    ref_test_var = open_reference('test', n_ensemble=n_ensemble)

    xval_var = xval_var / ref_xval_var 
    test_var = test_var / ref_test_var

    plot_mean_with_spread(
        xval_var,
        ( 
            test_var
            .rename({'seed': 'mem'})
            .stack(seed=('mem', 'fold'))
        ),
        cmap='viridis'
    )

    xval_mse = open_mse('xval', n_ensemble=n_ensemble)
    test_mse = open_mse('test', n_ensemble=n_ensemble)

    plot_mean_with_spread(
        xval_mse,
        ( 
            test_mse
            .rename({'seed': 'mem'})
            .stack(seed=('mem', 'fold'))
        ),
        cmap='viridis',
        vmin=0,
        vmax=2
    )
    



