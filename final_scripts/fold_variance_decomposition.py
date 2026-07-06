import xarray as xr 
import kgae 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs 
import numpy as np 
from pathlib import Path 
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde


def plot_fold_boot_distributions(
    da: xr.DataArray,
    ax=None,
    boot_dim: str = "boot",
    fold_dim: str = "fold",
    bins: int | str = "fd",     # "fd" = Freedman–Diaconis, or int
    kde_points: int = 400,
    kde_bw: float | str = "scott",  # "scott" | "silverman" | float
    hist_alpha: float = 0.25,
    kde_lw: float = 2.0,
    clip_to_data: bool = True,
):
    """
    Overlay, on one axis, per-fold bootstrap distributions along `boot_dim`:
      - histogram (density=True)
      - KDE curve

    Parameters
    ----------
    da : xr.DataArray
        Must have dims (boot_dim, fold_dim) or (fold_dim, boot_dim).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if boot_dim not in da.dims or fold_dim not in da.dims:
        raise ValueError(f"da must have dims including {boot_dim!r} and {fold_dim!r}. Got {da.dims}.")

    # Ensure ordering: (fold, boot)
    x = da.transpose(fold_dim, boot_dim)

    # Build a common x-grid for KDEs so curves are comparable
    all_vals = x.values.reshape(-1)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError("No finite values found in da.")

    xmin, xmax = np.min(all_vals), np.max(all_vals)
    pad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    xgrid = np.linspace(xmin - pad, xmax + pad, kde_points)

    folds = x[fold_dim].values
    for f in folds:
        vals = x.sel({fold_dim: f}).values
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue

        # Histogram (density)
        ax.hist(vals, bins=bins, density=True, alpha=hist_alpha, label=f"fold {f}")

        # KDE
        kde = gaussian_kde(vals, bw_method=kde_bw)
        y = kde(xgrid)
        if clip_to_data:
            y = np.where((xgrid >= vals.min()) & (xgrid <= vals.max()), y, np.nan)
        ax.plot(xgrid, y, lw=kde_lw)

    ax.set_xlabel(str(da.name) if da.name is not None else "value")
    ax.set_ylabel("density")
    ax.set_title("Bootstrap distributions by fold")
    ax.legend(ncols=2, fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0,1)

    return fig, ax


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

def bootstrap_fold_diff_var_ratio(
        da_fold, 
        da_full,
        ensemble_size=None, 
        n_resamples = 1000,
        dim='seed', 
        block_length=1,
        rng_seed=-1,
    ):

    rng = np.random.default_rng(rng_seed)
    ensemble_size = da_fold.sizes[dim] if ensemble_size is None else ensemble_size
    
    ests, cors = [], []
    for _ in range(n_resamples):
        print(_)
        idx = rng.choice(ensemble_size, size=ensemble_size, replace=True)
        fold_mean = da_fold.isel(**{dim: idx}).mean(dim) 
        full_mean = da_full.isel(**{dim: idx}).mean(dim)
        diff_mag = np.sqrt(((fold_mean - full_mean)**2).sum(['lat', 'lon']))
        full_mag = np.sqrt((full_mean**2).sum(['lat', 'lon']))
        cor = xr.corr(fold_mean, full_mean, dim=['lat', 'lon'])
        cors.append(cor)
        ests.append(diff_mag / full_mag)

    return xr.concat(ests, 'boot'), xr.concat(cors, 'boot')



if __name__ == "__main__":
    experiment = 'large-ensemble-2-1940-2014'
    n_ensemble = 100
    n_bootstraps = 1000
    overwrite = False

    modes_to_examine = [5, 4, 3]
    mode_labels = ["Decadal Mode", "Interannual Mode", "Quasibiennial Mode"]
    alpha_label = r"State-Independent Sensitivity $(\alpha$)"
    beta_labels = [ r"Decadal \nState-Dependence $(\beta)$", r"Interannual\nState-Dependence $(\beta)$", r"Quasibiennial\nState-Dependent $(\beta)$" ]

    # calculate fold-difference variance and optimizer-noise variance
    alpha_full, beta_full = kgae.open_alphabetas('full',n_ensemble=n_ensemble, experiment=experiment)
    alpha_cv, beta_cv = kgae.open_alphabetas('xval', n_ensemble=n_ensemble, experiment=experiment)
    
    fold_diff_var_boot_alpha, fold_diff_patcor_boot_alpha = bootstrap_fold_diff_var_ratio(
        alpha_cv.sel(tendency_mode=modes_to_examine), 
        alpha_full.sel(tendency_mode=modes_to_examine),
        ensemble_size = n_ensemble,
        n_resamples = n_bootstraps,
        block_length = 1,
        rng_seed = 0,
        dim='seed'
    )

    fold_diff_var_boot_beta, fold_diff_patcor_boot_beta = bootstrap_fold_diff_var_ratio(
        beta_cv.sel(tendency_mode=modes_to_examine, predictor_mode=modes_to_examine), 
        beta_full.sel(tendency_mode=modes_to_examine, predictor_mode=modes_to_examine), 
        ensemble_size = n_ensemble,
        n_resamples = n_bootstraps,
        block_length = 1,
        rng_seed = 0,
        dim='seed'
    )

    # now plot them 
    n_modes = len(modes_to_examine)
    n_rows = n_modes + 1
    n_cols = n_modes 
    pidx = 0 

    panel_letters = [f"{chr(ord('a') + i)})" for i in range(n_rows * n_cols)]

    fig, axes = plt.subplots(
        nrows = n_rows,
        ncols = n_cols, 
        figsize=(4.3 * n_cols, 2.7 * n_rows),
        constrained_layout = False
    )
    axes = np.atleast_2d(axes)

    # add alpha left-axis label (alpha state idnenpendent yada yada)
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
        tp = fold_diff_patcor_boot_alpha
        field_filled = tp.sel(tendency_mode=tendency_mode)
        plot_fold_boot_distributions(
            field_filled,
            ax
        )
        ax.set_title("")
        ax.set_title(mode_labels[ci])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if ci == 0: 
            ax.set_ylabel(r"density ($\alpha$)")

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
            tp = fold_diff_patcor_boot_beta
            field_filled = tp.sel(tendency_mode=tendency_mode, predictor_mode=predictor_mode)
            plot_fold_boot_distributions(
                field_filled,
                ax
            )
            ax.set_title("")
            ax.set_xlabel("")
            if ci == 0:
                ax.set_ylabel(r"density ($\beta$)")
            else:
                ax.set_ylabel("")

            if ri == 3:
                ax.set_xlabel("Pattern Correlation")
            else:
                ax.set_xlabel("")
                

    plt.savefig("fold-sensitivity.png", dpi=300)
    plt.show()
