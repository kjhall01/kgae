import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cartopy.crs as ccrs
import numpy as np

def plot_alpha_beta_maps(
    alpha,
    beta,
    alpha_sig,
    beta_sig,
    modes_to_examine,
    mode_labels,
    n_contours=11,
    cmap="coolwarm",
    height=-0.1
):
    """
    Plot alpha and beta maps with significance masking.

    Parameters
    ----------
    alpha : xarray.DataArray
        Alpha coefficients with dimensions (tendency_mode, lat, lon).
    beta : xarray.DataArray
        Beta coefficients with dimensions (tendency_mode, predictor_mode, lat, lon).
    alpha_sig : xarray.DataArray
        Significance mask for alpha with dimensions (tendency_mode, lat, lon).
    beta_sig : xarray.DataArray
        Significance mask for beta with dimensions (tendency_mode, predictor_mode, lat, lon).
    modes_to_examine : list
        List of mode indices to examine.
    mode_labels : list
        List of labels for the modes.
    n_contours : int, optional
        Number of contour levels to use in the plots. Default is 10.
    """
        
    proj = ccrs.PlateCarree(central_longitude=180)

    n_modes = len(modes_to_examine)
    fig = plt.figure(figsize=(16, 9))  # wider figure
    gs = gridspec.GridSpec(n_modes, n_modes + 1, figure=fig,
                        left=0.08, right=0.95, top=0.93, bottom=0.15,
                        wspace=0.05, hspace=0.15)  # leave space for ylabels/colorbars

    ax = np.empty((n_modes, n_modes+1), dtype=object)
    for i in range(n_modes):
        for j in range(n_modes+1):
            ax[i, j] = fig.add_subplot(gs[i, j], projection=proj)

    # ---------------------------------------------------------
    # Define contour levels (global, symmetric)
    # ---------------------------------------------------------
    alpha_max = np.nanmax(np.abs(alpha))
    beta_max  = np.nanmax(np.abs(beta))

    alpha_levels = np.linspace(-alpha_max, alpha_max, n_contours)
    beta_levels  = np.linspace(-beta_max,  beta_max,  n_contours)

    alpha_line_levels = alpha_levels[alpha_levels != 0]
    beta_line_levels  = beta_levels[beta_levels != 0]

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    for i, tendency_mode in enumerate(modes_to_examine):

        # ======================
        # ALPHA (first column)
        # ======================
        alpha_full = alpha.sel(tendency_mode=tendency_mode)
        alpha_mask = alpha_full.where(
            alpha_sig.sel(tendency_mode=tendency_mode)
        )

        # Shading (significant only)
        p_alpha = ax[i, 0].contourf(
            alpha.lon, alpha.lat, alpha_mask,
            levels=alpha_levels,
            cmap=cmap,
            extend="both",
            transform=proj
        )

        # Contours (full field)
        ax[i, 0].contour(
            alpha.lon, alpha.lat, alpha_full,
            levels=alpha_line_levels[alpha_line_levels > 0],
            colors="k", linewidths=0.8, linestyles="solid",
            transform=proj
        )

        ax[i, 0].contour(
            alpha.lon, alpha.lat, alpha_full,
            levels=alpha_line_levels[alpha_line_levels < 0],
            colors="k", linewidths=0.8, linestyles="dashed",
            transform=proj
        )

        ax[i, 0].contour(
            alpha.lon, alpha.lat, alpha_full,
            levels=[0], colors="k", linewidths=1.3,
            transform=proj
        )

        ax[i, 0].coastlines()
        ax[i, 0].text(
            -0.05, 0.5, mode_labels[i],
            transform=ax[i, 0].transAxes,
            rotation='vertical',
            va='center', ha='center'
        )

        if i == 0:
            ax[i, 0].set_title("Global ($\\alpha_i$)")

        # ======================
        # BETAS
        # ======================
        for j, mode in enumerate(modes_to_examine):

            beta_full = beta.sel(
                tendency_mode=tendency_mode,
                predictor_mode=mode
            )

            beta_mask = beta_full.where(
                beta_sig.sel(
                    tendency_mode=tendency_mode,
                    predictor_mode=mode
                )
            )

            # Shading (significant only)
            p_beta = ax[i, j + 1].contourf(
                beta.lon, beta.lat, beta_mask,
                levels=beta_levels,
                cmap=cmap,
                extend="both",
                transform=proj
            )

            # Contours (full field)
            ax[i, j + 1].contour(
                beta.lon, beta.lat, beta_full,
                levels=beta_line_levels[beta_line_levels > 0],
                colors="k", linewidths=0.7, linestyles="solid",
                transform=proj
            )

            ax[i, j + 1].contour(
                beta.lon, beta.lat, beta_full,
                levels=beta_line_levels[beta_line_levels < 0],
                colors="k", linewidths=0.7, linestyles="dashed",
                transform=proj
            )

            ax[i, j + 1].contour(
                beta.lon, beta.lat, beta_full,
                levels=[0], colors="k", linewidths=1.0,
                transform=proj
            )

            ax[i, j + 1].coastlines()

            if i == 0:
                ax[i, j + 1].set_title(f"State-Dependent ($\\beta_{{i,{j+1}}}$)")

    # ---------------------------------------------------------
    # Colorbars (below)
    # ---------------------------------------------------------
    #cax_alpha = fig.add_axes([0.10, 0.07, 0.22, 0.025])
    #cax_beta  = fig.add_axes([0.38, 0.07, 0.52, 0.025])

    # After creating your ax array from GridSpec:
    # ax is shape (n_modes, n_modes+1)

    # Use the first row (i=0) to compute positions
    alpha_ax_bbox = ax[0, 0].get_position()  # first column

    # Alpha colorbar: same width as first column, centered
    cax_alpha_left  = alpha_ax_bbox.x0
    cax_alpha_width = alpha_ax_bbox.width

    # Beta colorbar: span all beta columns (columns 1,2,3)
    beta_ax_start = ax[0, 1].get_position().x0
    beta_ax_end   = ax[0, -1].get_position().x1
    cax_beta_left  = beta_ax_start
    cax_beta_width = beta_ax_end - beta_ax_start

    cbar_height = 0.03
    cbar_bottom = height  # e.g., -0.1 or whatever you use

    cax_alpha = fig.add_axes([cax_alpha_left, cbar_bottom, cax_alpha_width, cbar_height])
    cax_beta  = fig.add_axes([cax_beta_left, cbar_bottom, cax_beta_width, cbar_height])

    num_ticks = n_contours//2 + 1 #if n_contours % 2 == 1 else (n_contours + 1)//2
    ticks_alpha = np.linspace(-np.nanmax(np.abs(alpha)), np.nanmax(np.abs(alpha)), num_ticks)
    ticks_beta  = np.linspace(-np.nanmax(np.abs(beta)),  np.nanmax(np.abs(beta)), num_ticks)

    cb_alpha = fig.colorbar(
        p_alpha, cax=cax_alpha, orientation="horizontal",
        label=r"$\alpha$  (K$\sigma_i^{-1}$)",
        ticks=ticks_alpha
    )
    cb_alpha.ax.tick_params(rotation=45)
    cb_alpha.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    cb_beta = fig.colorbar(
        p_beta, cax=cax_beta, orientation="horizontal",
        label=r"$\beta$  (K$\sigma_i^{-1}$$\sigma_j^{-1}$)",
        ticks=ticks_beta
    )
    cb_beta.ax.tick_params(rotation=45)
    cb_beta.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    #plt.tight_layout(rect=[0, 0.12, 1, 1])
    #plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_grid_maps(
    maps_da,
    sig_da,
    row_dim,
    col_dim,
    row_labels=None,
    col_labels=None,
    n_contours=11,
    cmap="coolwarm",
    central_longitude=180,
    figsize=(14, 8),
    cbar_height=0.03,
    cbar_bottom=0.08,
    cbar_label=None,
    title="",
):
    """
    Plot a grid of maps with significance masking.

    maps_da, sig_da: xarray.DataArray
        dims include row_dim, col_dim, lat, lon
    """

    proj = ccrs.PlateCarree(central_longitude=central_longitude)

    row_vals = maps_da.coords[row_dim].values
    col_vals = maps_da.coords[col_dim].values

    if row_labels is None: 
        row_labels = row_vals 

    if col_labels is None: 
        col_labels = col_vals 

    n_rows = len(row_vals)
    n_cols = len(col_vals)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.06, right=0.95, top=0.92, bottom=0.16,
        wspace=0.04, hspace=0.10
    )

    axes = np.empty((n_rows, n_cols), dtype=object)
    for ri in range(n_rows):
        for cj in range(n_cols):
            axes[ri, cj] = fig.add_subplot(gs[ri, cj], projection=proj)

    # ------------------------------------------------------------------
    # Color scale (global, symmetric)
    # ------------------------------------------------------------------
    vmax = np.nanmax(np.abs(maps_da.values))
    levels = np.linspace(-vmax, vmax, n_contours)
    pos_levels = levels[levels > 0]
    neg_levels = levels[levels < 0]

    last_p = None

    for ri, rv in enumerate(row_vals):
        for cj, cv in enumerate(col_vals):
            ax = axes[ri, cj]
            sel = {row_dim: rv, col_dim: cv}

            full = maps_da.sel(sel)
            masked = full.where(sig_da.sel(sel))

            last_p = ax.contourf(
                maps_da.lon, maps_da.lat, masked,
                levels=levels, cmap=cmap, extend="both",
                transform=proj
            )

            # contour lines from full field
            if len(pos_levels):
                ax.contour(
                    maps_da.lon, maps_da.lat, full,
                    levels=pos_levels, colors="k",
                    linewidths=0.6, linestyles="solid",
                    transform=proj
                )

            if len(neg_levels):
                ax.contour(
                    maps_da.lon, maps_da.lat, full,
                    levels=neg_levels, colors="k",
                    linewidths=0.6, linestyles="dashed",
                    transform=proj
                )

            ax.contour(
                maps_da.lon, maps_da.lat, full,
                levels=[0], colors="k",
                linewidths=0.9, transform=proj
            )

            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)

            ax.set_global()

            # remove tick labels inside grid
            ax.set_xticks([])
            ax.set_yticks([])

            # row labels (outside)
            if row_labels is not None and cj == 0:
                ax.text(
                    -0.08, 0.5, str(row_labels[ri]),
                    transform=ax.transAxes,
                    rotation=90, ha="center", va="center",
                    fontsize=11
                )

            # column titles
            if col_labels is not None and ri == 0:
                ax.set_title(str(col_labels[cj]), fontsize=11, pad=6)

    # ------------------------------------------------------------------
    # Shared horizontal colorbar
    # ------------------------------------------------------------------
    left = axes[0, 0].get_position().x0
    right = axes[0, -1].get_position().x1
    cax = fig.add_axes([left, cbar_bottom, right - left, cbar_height])

    cb = fig.colorbar(last_p, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb.ax.tick_params(labelsize=10)

    if cbar_label is not None:
        cb.set_label(cbar_label, fontsize=11)
    plt.suptitle(title)
    #plt.show()
