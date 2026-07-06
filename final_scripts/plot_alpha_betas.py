import xarray as xr 
import kgae 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs 
import numpy as np 
from pathlib import Path 
from matplotlib.ticker import FormatStrFormatter

def moving_block_bootstrap_indices(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    """
    Moving-block bootstrap indices for a length-n time series.

    Draw contiguous blocks of length block_len with replacement, concatenate until length n,
    then truncate to n.

    This preserves autocorrelation structure up to ~block_len lags.
    """
    if block_len <= 1:
        # degenerate -> IID bootstrap
        return rng.integers(0, n, size=n)

    if n <= block_len:
        # if record shorter than one block, just sample cyclic shifts of the full record
        start = int(rng.integers(0, n))
        return (np.arange(n) + start) % n

    n_starts = n - block_len + 1
    n_blocks = int(np.ceil(n / block_len))

    starts = rng.integers(0, n_starts, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)
    return idx[:n]

def bootstrap_significance(
        da, 
        ensemble_size=None, 
        n_resamples = 1000,
        dim='seed', 
        block_length=1,
        rng_seed=-1,
        ci = 0.95
    ):

    rng = np.random.default_rng(rng_seed)
    ensemble_size = da.sizes[dim] if ensemble_size is None else ensemble_size

    ests = []
    for _ in range(n_resamples):
        print(_)
        idx = moving_block_bootstrap_indices(rng, ensemble_size, block_len=block_length)
        ests.append( da.isel(**{dim: idx}).mean(dim) )


    ests = xr.concat(ests, 'boot')
    pct_lt_0 = (ests < 0).mean('boot').where(np.isfinite(ests.mean('boot')))
    return pct_lt_0
   # lo, hi = ests.quantile((1 - ci)/2, 'boot'), ests.quantile(ci + (1-ci)/2, 'boot')
   # mask = (lo > 0) | (hi < 0)
   # return mask


if __name__ == "__main__":
    experiment = './large-ensemble-3-1940-2014'
    n_ensemble = 3
    n_bootstraps = 100
    ci = 0.9
    overwrite = True

    modes_to_examine = [5, 4, 3, 2, 1]
    mode_labels = ["Decadal Mode", "Interannual Mode", "Quasibiennial Mode", "HF2", "HF1"]
    alpha_label = r"Global Sensitivity $(\alpha$)"
    beta_labels = [ r"Decadal Dependence $(\beta)$", r"Interannual Dependence $(\beta)$", r"Quasibiennial Dependence $(\beta)$", "HF2", "HF1" ]

    # do bootstrapping for alpha if necessary
    alpha_full, beta_full = kgae.open_alphabetas('xval', experiment=experiment, n_ensemble=n_ensemble)
    alpha_full_mean, beta_full_mean = alpha_full.mean('seed').mean('fold'), beta_full.mean('seed').mean('fold')
    alpha_full = alpha_full.mean('fold')
    beta_full = beta_full.mean('fold')
    print(alpha_full_mean)
    
    alpha_pct_lt_zero_at_ci_cache_path = f"alpha_full_pct-lt-zero_bootstrap={n_bootstraps}_ensemble={n_ensemble}.nc"
    if not Path(alpha_pct_lt_zero_at_ci_cache_path).is_file() or overwrite:
        alpha_pct_lt_zero_at_ci = bootstrap_significance(
            alpha_full, 
            ensemble_size = n_ensemble,
            n_resamples = n_bootstraps,
            block_length = 1,
            rng_seed = 0,
            ci = ci,
            dim='seed'
        )
        alpha_pct_lt_zero_at_ci.to_netcdf(alpha_pct_lt_zero_at_ci_cache_path)
    else: 
        alpha_pct_lt_zero_at_ci = xr.open_dataset(alpha_pct_lt_zero_at_ci_cache_path)
        alpha_pct_lt_zero_at_ci = alpha_pct_lt_zero_at_ci["__xarray_dataarray_variable__"]

    alpha_full_mean_masked = alpha_full_mean.where((alpha_pct_lt_zero_at_ci > ci + (1-ci)/2) | (alpha_pct_lt_zero_at_ci < (1-ci)/2))

    # do bootstrapping for beta if necessary
    beta_pct_lt_zero_at_ci_cache_path = f"beta_full_pct-lt-zero_bootstrap={n_bootstraps}_ensemble={n_ensemble}.nc"
    if not Path(beta_pct_lt_zero_at_ci_cache_path).is_file() or overwrite:
        beta_pct_lt_zero_at_ci = bootstrap_significance(
            beta_full, 
            ensemble_size = n_ensemble,
            n_resamples = n_bootstraps,
            block_length = 1,
            rng_seed = 0,
            ci = ci,
            dim='seed'
        )
        beta_pct_lt_zero_at_ci.to_netcdf(beta_pct_lt_zero_at_ci_cache_path)
    else: 
        beta_pct_lt_zero_at_ci = xr.open_dataset(beta_pct_lt_zero_at_ci_cache_path)
        beta_pct_lt_zero_at_ci = beta_pct_lt_zero_at_ci["__xarray_dataarray_variable__"]
    beta_full_mean_masked = beta_full_mean.where((beta_pct_lt_zero_at_ci > ci + (1-ci)/2) | (beta_pct_lt_zero_at_ci < (1-ci)/2))


    # now plot them 
    n_modes = len(modes_to_examine)
    n_rows = n_modes + 1
    n_cols = n_modes 
    pidx = 0 

    panel_letters = [f"{chr(ord('a') + i)})" for i in range(n_rows * n_cols)]

    fig, axes = plt.subplots(
        nrows = n_rows,
        ncols = n_cols, 
        subplot_kw = {
            'projection': ccrs.PlateCarree(central_longitude=180)
        },
        figsize=(4.3 * n_cols, 2.6 * n_rows),
        constrained_layout = False
    )
    axes = np.atleast_2d(axes)

    alpha_levels = np.linspace(-0.35, 0.35, 21)
    alpha_levels_contours = np.linspace(-0.35, 0.35, 7)

    beta_levels = np.linspace(-0.01, 0.01, 21)
    beta_levels_contours = np.linspace(-0.01, 0.01, 7)

    # add alpha left-axis label (alpha state idnenpendent yada yada)
    axes[0, 0].annotate(
        alpha_label,
        xy=(-0.07, 0.5), xycoords="axes fraction",
        rotation=90, ha="center", va="center",
        fontsize=10
    )

    mappable_alpha, mappable_beta = None, None
    for ci, tendency_mode in enumerate(modes_to_examine):
        ax = axes[0, ci]
        ax.text(
            0.0, 1.07, panel_letters[pidx],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
        )
        pidx += 1

        # shading 
        field_filled = alpha_full_mean_masked.sel(tendency_mode=tendency_mode)
        mappable_alpha = field_filled.plot.contourf(
            ax=ax, transform=ccrs.PlateCarree(central_longitude=180), levels=alpha_levels,
            add_colorbar=False, extend="both"
        )
        ax.set_title("")
        ax.set_title(mode_labels[ci])
        
        # negative contours
        field = alpha_full_mean.sel(tendency_mode = tendency_mode)
        ax.contour(field["lon"], field["lat"], field.values,
            levels=[lv for lv in alpha_levels_contours if lv < 0], 
            colors="k", linestyles="dashed",
            linewidths=0.7, 
            transform=ccrs.PlateCarree(central_longitude=180)
        )

        # positive contours
        ax.contour(field["lon"], field["lat"], field.values,
            levels=[lv for lv in alpha_levels_contours if lv >= 0], 
            colors="k", linestyles="solid",
            linewidths=0.7, 
            transform=ccrs.PlateCarree(central_longitude=180)
        )
        ax.coastlines()

        for ri, predictor_mode in enumerate(modes_to_examine, start=1):
            ax = axes[ri, ci]

            # add pannel label (letter)
            ax.text(
                0.0, 1.07, panel_letters[pidx],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
            )
            pidx += 1

            # shading 
            field_filled = beta_full_mean_masked.sel(tendency_mode=tendency_mode, predictor_mode=predictor_mode)
            mappable_beta = field_filled.plot.contourf(
                ax=ax, transform=ccrs.PlateCarree(central_longitude=180), levels=beta_levels,
                add_colorbar=False, extend="both"
            )
            ax.set_title("")
            ax.coastlines()

            
            # negative contours
            field = beta_full_mean.sel(tendency_mode = tendency_mode, predictor_mode=predictor_mode)
            ax.contour(field["lon"], field["lat"], field.values,
                levels=[lv for lv in beta_levels_contours if lv < 0], 
                colors="k", linestyles="dashed",
                linewidths=0.7, 
                transform=ccrs.PlateCarree(central_longitude=180)
            )

            # positive contours
            ax.contour(field["lon"], field["lat"], field.values,
                levels=[lv for lv in beta_levels_contours if lv >= 0], 
                colors="k", linestyles="solid",
                linewidths=0.7, 
                transform=ccrs.PlateCarree(central_longitude=180)
            )

            # add beta label to each of the first column's left sides:
            if ci == 0:
                ax.annotate(
                    beta_labels[ri-1],
                    xy=(-0.07, 0.5), xycoords="axes fraction",
                    rotation=90, ha="center", va="center",
                    fontsize=10
                )
    
    plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.06, right=0.88, top=0.93, bottom=0.10)


    # colorbars
    # alpha colorbar next to top row
    if mappable_alpha is not None:
        row_boxes = [axes[0, j].get_position() for j in range(n_cols)]
        y0 = min(b.y0 for b in row_boxes)
        y1 = max(b.y1 for b in row_boxes)
        x1 = max(b.x1 for b in row_boxes)
        cax = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
        cb = fig.colorbar(mappable_alpha, cax=cax, orientation="vertical")
        cb.set_label(r"sensitivity $(K$ $\sigma_i^{-1})$", fontsize=10)
        cb.formatter = FormatStrFormatter('%.3f')
        cb.update_ticks()

    # composite colorbar next to bottom row
    if mappable_beta is not None:
        beta_boxes = [axes[r, c].get_position() for r in range(1, n_rows ) for c in range(n_cols)]
        y0 = min(b.y0 for b in beta_boxes)
        y1 = max(b.y1 for b in beta_boxes)
        x1 = max(b.x1 for b in beta_boxes)
        cax_beta = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
        cb = fig.colorbar(mappable_beta, cax=cax_beta, orientation="vertical")
        cb.set_label(r"$\beta$ $(K$ $\sigma_i^{-1}$ $\sigma_j^{-1})$", fontsize=10)
        cb.formatter = FormatStrFormatter('%.3f')
        cb.update_ticks()

    #plt.savefig("alphas_betas_final-test.png", dpi=300)
    plt.show()
