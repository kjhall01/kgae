import numpy as np
import matplotlib.pyplot as plt

def plot_mode_histograms(xval_da, test_da, mode_indices=(1, 2),
                         mode_labels=("Interannual", "Decadal"),
                         bins=50, alpha=0.6, figsize=(12, 4),
                         colors=("tab:blue", "tab:orange")):
    """
    Plot 1D histograms of selected modes for cross-validation and test periods.

    Parameters
    ----------
    xval_da : xarray.DataArray
        Shape (time, mode, seed)
    test_da : xarray.DataArray
        Shape (time, mode, seed)
    mode_indices : tuple of int
        Indices of modes to plot
    mode_labels : tuple of str
        Labels for modes
    bins : int
        Number of bins
    alpha : float
        Transparency for histograms
    figsize : tuple
        Figure size
    colors : tuple
        Colors for (xval, test)
    """

    n_modes = len(mode_indices)
    fig, axes = plt.subplots(1, n_modes, figsize=figsize, sharey=True)
    if n_modes == 1:
        axes = [axes]

    for ax, mode_idx, label in zip(axes, mode_indices, mode_labels):
        # Collapse seeds into single array
        xval_data = xval_da[:, mode_idx, :].values.reshape(-1)
        test_data = test_da[:, mode_idx, :].values.reshape(-1)

        # Determine shared bins for comparability
        data_min = min(np.nanmin(xval_data), np.nanmin(test_data))
        data_max = max(np.nanmax(xval_data), np.nanmax(test_data))
        bins_edges = np.linspace(data_min, data_max, bins)

        # Plot histograms
        ax.hist(xval_data, bins=bins_edges, color=colors[0],
                alpha=alpha, density=True, label="Cross-validation")
        ax.hist(test_data, bins=bins_edges, color=colors[1],
                alpha=alpha, density=True, label="Test")

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    plt.suptitle("Distributions of Latent Modes")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_joint_histograms(
    xval_da, test_da, mode_indices=(1, 2),
    mode_labels=("Interannual", "Decadal"),
    bins=50, figsize=(12, 5),
    cmap="magma_r"
):
    """
    Plot CV and Test joint histograms side-by-side with separate colorbars.

    Parameters
    ----------
    xval_da, test_da : xarray.DataArray
        Shape (time, mode, seed)
    mode_indices : tuple
        Two mode indices to plot
    mode_labels : tuple
        Names of the two modes
    bins : int
        Number of bins for 2D histogram
    figsize : tuple
        Figure size
    cmap : str
        Colormap (default: viridis)
    """

    mode1, mode2 = mode_indices

    # Flatten time + seed, remove NaNs
    def flatten_da(da):
        d1 = da[:, mode1, :].values.astype(float).ravel()
        d2 = da[:, mode2, :].values.astype(float).ravel()
        mask = np.isfinite(d1) & np.isfinite(d2)
        return d1[mask], d2[mask]

    xval_x, xval_y = flatten_da(xval_da)
    test_x, test_y = flatten_da(test_da)

    # Shared bin edges (important!)
    xmin = min(xval_x.min(), test_x.min())
    xmax = max(xval_x.max(), test_x.max())
    ymin = min(xval_y.min(), test_y.min())
    ymax = max(xval_y.max(), test_y.max())

    bins_x = np.linspace(xmin, xmax, bins)
    bins_y = np.linspace(ymin, ymax, bins)

    # Histograms (density!)
    H_xval, xedges, yedges = np.histogram2d(
        xval_x, xval_y, bins=[bins_x, bins_y], density=True
    )
    H_test, _, _ = np.histogram2d(
        test_x, test_y, bins=[bins_x, bins_y], density=True
    )

    # Bin centers
    X, Y = np.meshgrid(
        (xedges[:-1] + xedges[1:]) / 2,
        (yedges[:-1] + yedges[1:]) / 2
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    # --- Cross-validation ---
    pcm0 = axes[0].pcolormesh(
        X, Y, H_xval.T, cmap=cmap, shading="auto", vmin=0, vmax=0.3
    )
    axes[0].set_title("Cross-validation")
    axes[0].set_xlabel(mode_labels[0])
    axes[0].set_ylabel(mode_labels[1])
    cbar0 = fig.colorbar(pcm0, ax=axes[0], pad=0.02)
    cbar0.set_label("Density (CV)")

    # --- Test ---
    pcm1 = axes[1].pcolormesh(
        X, Y, H_test.T, cmap=cmap, shading="auto"
    )
    axes[1].set_title("Test")
    axes[1].set_xlabel(mode_labels[0])
    cbar1 = fig.colorbar(pcm1, ax=axes[1], pad=0.02)
    cbar1.set_label("Density (Test)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
