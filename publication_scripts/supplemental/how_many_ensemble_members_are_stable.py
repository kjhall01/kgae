import xarray as xr 
import kgae 
import matplotlib.pyplot as plt 
import numpy as np 


def plot_vs_ensemble_size(
    da: xr.DataArray,
    ax=None,
    dim: str = "ensemble_size",
    label: str | None = None,
    lw: float = 2.0,
    marker: str | None = None,
):
    """
    Plot a 1D DataArray (dim = ensemble_size) as a line on a given axis.

    Parameters
    ----------
    da : xr.DataArray
        Must have exactly one dimension (default: 'ensemble_size').
    ax : matplotlib axis, optional
        Axis to plot on. If None, a new figure/axis is created.
    dim : str
        Name of the dimension (default: 'ensemble_size').
    label : str, optional
        Legend label.
    lw : float
        Line width.
    marker : str, optional
        Marker style (e.g., 'o').

    Returns
    -------
    fig, ax
    """

    if dim not in da.dims:
        raise ValueError(f"DataArray must contain dimension {dim!r}. Got dims {da.dims}")

    if da.ndim != 1:
        raise ValueError(f"DataArray must be 1D. Got shape {da.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    x = da[dim].values
    y = da.values

    ax.plot(x, y, lw=lw, marker=marker, label=label)

    ax.set_xlabel(dim)
    ax.set_ylabel(da.name if da.name is not None else "value")
    ax.set_ylim(0,1.1)
    ax.axhline(0.95, linewidth=2, linestyle=":")
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend(frameon=False)

    return fig, ax



def bootstrap_rel_l2_dist(
        da_full,
        n_resamples = 1000,
        dim='seed', 
        block_length=1,
        rng_seed=-1,
        cl = 0.95,
        ensemble_size = 4
    ):

    rng = np.random.default_rng(rng_seed)

    full_ensemble_size = da_full.sizes[dim]
    
    ests, cors = [], []
    for _ in range(n_resamples):
        print(_)
       #idx = moving_block_bootstrap_indices(rng, full_ensemble_size, block_len=block_length)
        idx = rng.choice(full_ensemble_size, size=ensemble_size*2, replace=False) 
        idx2 = idx[:ensemble_size]
        idx = idx[ensemble_size:]

        full_boot_mean = da_full.isel(**{dim: idx}).mean(dim)
        full_boot_mean2 = da_full.isel(**{dim: idx2}).mean(dim)
        diff_mag = xr.corr(full_boot_mean, full_boot_mean2, dim=['lat', 'lon'])
        ests.append(diff_mag)

    return xr.concat(ests, 'boot').quantile(cl, 'boot')

def search_ensemble_sizes(
        da_full,
        n_resamples = 1000,
        dim='seed', 
        block_length=1,
        rng_seed=-1,
        ensemble_sizes = [3, 5, 7, 10, 15, 20,25, 30, 40, 50],
        cl=0.95
    ):
    ninety_fifth_pctl_bootstrapped_rel_l2_for_size = []
    for es in ensemble_sizes:
        print('STARTING ES: ', es)
        nfpbrl2fses = bootstrap_rel_l2_dist(
            da_full,
            ensemble_size=es,
            n_resamples=n_resamples,
            dim=dim,
            block_length=block_length,
            rng_seed=rng_seed,
            cl = cl 
        )
        ninety_fifth_pctl_bootstrapped_rel_l2_for_size.append(nfpbrl2fses)
    ninety_fifth_pctl_bootstrapped_rel_l2_for_size = xr.concat(ninety_fifth_pctl_bootstrapped_rel_l2_for_size, 'ensemble_size')
    return ninety_fifth_pctl_bootstrapped_rel_l2_for_size.assign_coords({'ensemble_size': ensemble_sizes})




if __name__ == "__main__":
    experiment = '../final_scripts/large-ensemble-2-1940-2014'
    n_ensemble = 100
    n_bootstraps = 1000
    overwrite = False
    cl = 0.05
    ensemble_sizes_to_search = range(4, 51, 2)

    modes_to_examine = [5, 4, 3]
    mode_labels = ["Decadal Mode", "Interannual Mode", "Quasibiennial Mode"]
    alpha_label = r"State-Independent Sensitivity $(\alpha$)"
    beta_labels = [ r"Decadal \nState-Dependence $(\beta)$", r"Interannual\nState-Dependence $(\beta)$", r"Quasibiennial\nState-Dependent $(\beta)$" ]

    # calculate fold-difference variance and optimizer-noise variance
    alpha_full, beta_full = kgae.open_alphabetas('full', n_ensemble=n_ensemble, experiment=experiment)
    alpha_cv, beta_cv = kgae.open_alphabetas('xval', n_ensemble=n_ensemble, experiment=experiment)
    
    alpha_ensemble_sizes = search_ensemble_sizes(
        alpha_full.sel(tendency_mode=modes_to_examine),
        n_resamples = n_bootstraps,
        block_length = 1,
        rng_seed = 0,
        dim='seed',
        ensemble_sizes= ensemble_sizes_to_search,
        cl = cl
    )

    beta_ensemble_sizes = search_ensemble_sizes(
        beta_full.sel(tendency_mode=modes_to_examine, predictor_mode=modes_to_examine),
        n_resamples = n_bootstraps,
        block_length = 1,
        rng_seed = 0,
        dim='seed',
        ensemble_sizes= ensemble_sizes_to_search,
        cl = cl
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
        figsize=(4.3 * n_cols, 2.6 * n_rows),
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
        tp = alpha_ensemble_sizes
        field_filled = tp.sel(tendency_mode=tendency_mode)
        plot_vs_ensemble_size(
            field_filled,
            ax
        )
        ax.set_title("")
        ax.set_title(mode_labels[ci])
        if ci == 0:
            ax.set_ylabel(r"Pattern Correlation ($\alpha$)")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")

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
            tp = beta_ensemble_sizes
            field_filled = tp.sel(tendency_mode=tendency_mode, predictor_mode=predictor_mode)
            plot_vs_ensemble_size(
                field_filled,
                ax
            )
            ax.set_title("")
            if ci == 0: 
                ax.set_ylabel(r"Pattern Correlation")
            else:
                ax.set_ylabel("")

            if ri==3:
                ax.set_xlabel("Ensemble Size")
            else:
                ax.set_xlabel("")


    plt.savefig("split-half-experiment.pdf")
    plt.show()
