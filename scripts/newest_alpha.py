import xarray as xr 
import kgae 
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import gaussian_kde

def plot_seed_hist_kde_grid_by_modes(
    da,
    *,
    bins=25,
    kde_points=300,
    kde_bw="scott",          # "scott", "silverman", or float
    hist_alpha=0.25,
    kde_lw=1.6,
    density=True,
    sharex="col",
    sharey="row",
    figsize_per_panel=(3.0, 2.2),
    fold_labels=None,
    fold_linestyles=None,   # dict or list matching folds (e.g. {0:"-",1:"--",...})
    xlim=None,
    ylim=None,
    title_fmt="tend={tend}, pred={pred}",
):
    """
    Plot a grid of panels for an xarray.DataArray with dims:
      ('seed', 'fold', 'tendency_mode', 'predictor_mode').

    Layout:
      rows    -> tendency_mode
      columns -> predictor_mode

    Within each panel:
      - For each fold, plot a histogram + KDE across the 'seed' dimension.

    Parameters
    ----------
    da : xr.DataArray
        Must contain dims: seed, fold, tendency_mode, predictor_mode.
        Values should be scalar per seed/fold/mode pair (e.g., spatially-averaged beta or a point value).
    """

    # ---- checks ----
    required = {"seed", "fold", "tendency_mode", "predictor_mode"}
    missing = required - set(da.dims)
    if missing:
        print(f"da is missing required dims: {missing}. Got dims={da.dims}")
        if 'tendency_mode' in missing:
            da = da.expand_dims('tendency_mode')
        if 'predictor_mode' in missing:
            da = da.expand_dims('predictor_mode')
        if 'fold' in missing:
            da = da.expand_dims('fold')

    missing = required - set(da.dims)
    if missing: 
        raise ValueError("Could not rectify missing dims")

    # ---- coords / sizes ----
    tend_vals = da["tendency_mode"].values
    pred_vals = da["predictor_mode"].values
    fold_vals = da["fold"].values

    nrows = len(tend_vals)
    ncols = len(pred_vals)

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=sharex, sharey=sharey,
        squeeze=False
    )

    # ---- fold labeling / styling ----
    if fold_labels is None:
        fold_labels = {f: f"fold {int(f)}" if np.issubdtype(type(f), np.integer) else f"fold {f}" for f in fold_vals}
    elif isinstance(fold_labels, (list, tuple)):
        fold_labels = {f: fold_labels[i] for i, f in enumerate(fold_vals)}

    if fold_linestyles is None:
        # cycle a few common ones if multiple folds
        base = ["-", "--", ":", "-."]
        fold_linestyles = {f: base[i % len(base)] for i, f in enumerate(fold_vals)}
    elif isinstance(fold_linestyles, (list, tuple)):
        fold_linestyles = {f: fold_linestyles[i] for i, f in enumerate(fold_vals)}

    # ---- global x-range (optional) ----
    if xlim is None:
        vals = da.values
        vals = vals[np.isfinite(vals)]
        if vals.size:
            pad = 0.02 * (np.nanmax(vals) - np.nanmin(vals) + 1e-12)
            xlim = (np.nanmin(vals) - pad, np.nanmax(vals) + pad)

    # ---- plot ----
    for i, tend in enumerate(tend_vals):
        for j, pred in enumerate(pred_vals):
            ax = axes[i, j]

            # Pull (seed, fold) slice for this panel
            panel = da.sel(tendency_mode=tend, predictor_mode=pred)

            # Collect all fold data to pick a reasonable KDE support
            if xlim is not None:
                x_grid = np.linspace(xlim[0], xlim[1], kde_points)
            else:
                # fall back to data-derived range
                v = panel.values
                v = v[np.isfinite(v)]
                if v.size == 0:
                    x_grid = np.linspace(-1, 1, kde_points)
                else:
                    vmin, vmax = np.nanmin(v), np.nanmax(v)
                    pad = 0.02 * (vmax - vmin + 1e-12)
                    x_grid = np.linspace(vmin - pad, vmax + pad, kde_points)

            for f in fold_vals:
                x = panel.sel(fold=f).values
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue

                # Histogram
                ax.hist(
                    x,
                    bins=bins,
                    density=density,
                    alpha=hist_alpha,
                    label=None,  # keep legend clean; KDE carries labels
                )

                # KDE
                if x.size >= 2 and np.nanstd(x) > 0:
                    kde = gaussian_kde(x, bw_method=kde_bw)
                    y = kde(x_grid)
                    ax.plot(
                        x_grid, y,
                        linewidth=kde_lw,
                        linestyle=fold_linestyles.get(f, "-"),
                        label=fold_labels.get(f, f"fold {f}")
                    )
                else:
                    # If only one point (or zero variance), draw a thin marker line
                    ax.axvline(
                        float(x[0]),
                        linestyle=fold_linestyles.get(f, "-"),
                        linewidth=1.0,
                        label=fold_labels.get(f, f"fold {f}")
                    )

            # Titles / labels
            ax.set_title(title_fmt.format(tend=tend, pred=pred))
            if j == 0:
                ax.set_ylabel("density" if density else "count")

            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.grid(True, alpha=0.25)

    # Put one legend for the whole figure
    # (grab handles/labels from first axis that has them)
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        if len(h) > 0:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, axes


def cond_ztz_per_point(Z: xr.DataArray, eps: float = 0.0) -> xr.DataArray:
    """
    Compute condition number of ZTZ = Z @ Z.T using:
      - 'mode' as features
      - 'time' as samples
    Z dims must include ('mode','time'); any other dims are treated as broadcast dims.

    Returns: DataArray of cond(ZTZ) with dims = Z.dims minus ('mode','time').
    """
    if "mode" not in Z.dims or "time" not in Z.dims:
        raise ValueError(f"Z must have dims including ('mode','time'); got {Z.dims}")

    def _cond_ztz(z_mt: np.ndarray) -> np.float32:
        # z_mt: (mode, time)
        z_mt = np.asarray(z_mt, dtype=np.float64)

        # Handle NaNs by requiring all-finite
        if not np.isfinite(z_mt).all():
            return np.float32(np.nan)

        # ZTZ = (mode,time) @ (time,mode) -> (mode,mode)
        ztz = z_mt @ z_mt.T

        # Optional tiny ridge for numerical stability (set eps=0 to disable)
        #if eps > 0:
        #    ztz = ztz + eps * np.eye(ztz.shape[0], dtype=ztz.dtype)

        try:
            c = np.linalg.cond(ztz)  # 2-norm condition number
        except np.linalg.LinAlgError:
            c = np.inf

        return np.float32(c)

    return xr.apply_ufunc(
        _cond_ztz,
        Z,
        input_core_dims=[["mode", "time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).rename("cond_ZTZ")



# Example usage:
# fig, axes = plot_seed_hist_kde_grid_by_modes(beta_point_da)
# plt.show()


# Example:
# fig, axes = plot_fold_hist_kde_by_mode(beta_da)
# plt.show()


# --- example usage ---
# fig, axes = plot_fold_kde_by_mode(your_da)
# plt.show()
if __name__ == "__main__":


    experiment = 'large-ensemble-1940-2014'
    n_ensemble = 201
    boot_ens = 201
    n_boot = 1000

    ci = 0.9
    modes_to_examine = [5, 4, 3, 2, 1]
    n_modes = len(modes_to_examine)


   # Z = kgae.open_latents('xval', experiment=experiment, n_ensemble=n_ensemble)
   # Z1 = xr.concat([Z.isel(time=slice(i, None, 5)).drop('time') for i in range(5) ], 'fold')

   # cond_ZTZ = cond_ztz_per_point(Z1)
   # cond_ZTZf = cond_ztz_per_point(Z)
   # plot_seed_hist_kde_grid_by_modes(cond_ZTZ, title_fmt='byfold')
   # plot_seed_hist_kde_grid_by_modes(cond_ZTZf, title_fmt='all')

   # plt.show() 

   # alpha_cv, beta_cv = kgae.open_alphabetas('xval', experiment=experiment, n_ensemble=n_ensemble)
    alpha_full, beta_full = kgae.open_alphabetas('full', experiment=experiment, n_ensemble=n_ensemble)
   ## alpha_cv = alpha_cv.sel(tendency_mode = modes_to_examine)
   # beta_cv = beta_cv.sel(tendency_mode = modes_to_examine, predictor_mode = modes_to_examine)
    alpha_full = alpha_full.sel(tendency_mode = modes_to_examine)
    beta_full = beta_full.sel(tendency_mode = modes_to_examine, predictor_mode = modes_to_examine)
   # alpha_corr = xr.corr(alpha_cv.mean('fold'), alpha_full, dim=['lat', 'lon'])
   # beta_corr = xr.corr(beta_cv.mean('fold'), beta_full, dim=['lat', 'lon'])

   # plot_seed_hist_kde_grid_by_modes(alpha_corr)
   # plot_seed_hist_kde_grid_by_modes(beta_corr)
   # plt.show()
   # import sys; sys.exit()

    # bootstrap ensemble-mean estimate of size b 
    alpha_ests, beta_ests = [], []
    for _ in range(n_boot):
        idx = np.random.choice(n_ensemble, size=boot_ens)
        beta_ests.append( beta_full.isel(seed=idx).mean('seed') )
        alpha_ests.append( alpha_full.isel(seed=idx).mean('seed') )

    alpha_ests = xr.concat(alpha_ests, 'boot')
    beta_ests = xr.concat(beta_ests, 'boot')

    alpha_lo, alpha_hi = alpha_ests.quantile(0.025, 'boot'), alpha_ests.quantile(0.975, 'boot')
    beta_lo, beta_hi = beta_ests.quantile(0.025, 'boot'), beta_ests.quantile(0.975, 'boot')

    alpha_mask = (alpha_lo > 0) | (alpha_hi < 0)
    beta_mask = (beta_lo > 0) | (beta_hi < 0)

    alpha_full_mu = alpha_full.mean('seed')
    beta_full_mu = beta_full.mean('seed')

    alpha_full_mu_masked = alpha_full_mu.where( alpha_mask  )
    fig, ax = plt.subplots(nrows=1, ncols=len(modes_to_examine), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    for i, tendency_mode in enumerate(modes_to_examine):
        curmode = alpha_full_mu_masked.sel(tendency_mode=tendency_mode)
        p = curmode.plot(ax=ax[i])
        ax[i].coastlines()

    beta_full_mu_masked = beta_full_mu.where( beta_mask  )
    fig, ax = plt.subplots(nrows=len(modes_to_examine), ncols=len(modes_to_examine), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    for j, predictor_mode in enumerate(modes_to_examine):
        for i, tendency_mode in enumerate(modes_to_examine):
            curmode = beta_full_mu_masked.sel(tendency_mode=tendency_mode, predictor_mode=predictor_mode)
            p = curmode.plot(ax=ax[j][i])
            ax[j][i].coastlines()

    plt.show()