import xarray as xr 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path 
from scipy.stats import pearsonr 
from pandas.tseries.offsets import DateOffset
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

# ----------------------------
# Joint density (pretty)
# ----------------------------
def plot_joint_to_ax(z, ax_dist, *, mode_interannual="Interannual", mode_decadal="Decadal",
                     hist_bins=80, smooth_sigma=1.2):

    z = (z - z.mean("time")) / z.std("time")
    z_stack = z.stack(point=[d for d in z.dims if d != "mode"])

    zi = z_stack.sel(mode=mode_interannual).values
    zd = z_stack.sel(mode=mode_decadal).values
    ok = np.isfinite(zd) & np.isfinite(zi)
    x = zd[ok]  # decadal
    y = zi[ok]  # interannual

    H, xedges, yedges = np.histogram2d(x, y, bins=hist_bins, density=True)

    MIN_DENS = 1e-5
    Hs = H.astype(float)
   # if smooth_sigma and smooth_sigma > 0:
   #     Hs = gaussian_filter(Hs, sigma=smooth_sigma)

    Hs_masked = np.ma.masked_where(H < MIN_DENS, Hs)

    cmap_den = plt.get_cmap("magma_r").copy()
    cmap_den.set_bad("white")

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax_dist.imshow(
        Hs_masked.T, origin="lower", extent=extent,
        interpolation="bilinear", aspect="equal", cmap=cmap_den
    )

    ax_dist.set_xlabel("Decadal Mode")
    ax_dist.set_ylabel("Interannual Mode")

    ax_dist.set_xticks([-2, -1, 0, 1, 2])
    ax_dist.set_yticks([-2, -1, 0, 1, 2])

    ax_dist.grid(True, linestyle=":", linewidth=0.8, color="0.65")
    for sp in ax_dist.spines.values():
        sp.set_visible(False)

    return im



# ----------------------------
# xarray-aware month×lag corr (r and p)
# ----------------------------
def _pearsonr_nan(a, b):
    """a,b: 1D numpy arrays (time). Returns (r,p) with NaN handling."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan, np.nan
    r, p = pearsonr(a[ok], b[ok])
    return r, p

from scipy.stats import t as student_t

def month_lag_corr_xr(x, y, nlags=36):
    """
    x,y: DataArray with dim 'time' (may have other dims like seed/member/fold/etc.)
    Returns:
      r: DataArray(month, lag)
      p: DataArray(month, lag)
    Correlates x(t) with y(t+lag) for each calendar month subset.
    """
    # align on time first
    x, y = xr.align(x, y, join="inner")
    # collapse non-time dims (so we correlate ensemble-mean; change if you want seedwise)
    other_x = [d for d in x.dims if d != "time"]
    other_y = [d for d in y.dims if d != "time"]
    if other_x:
        x = x.stack(samp=other_x).mean("samp", skipna=True)
    if other_y:
        y = y.stack(samp=other_y).mean("samp", skipna=True)

    lags = np.arange(-nlags, nlags + 1)

    r_out = []
    p_out = []
    for m in range(1, 13):
        xm = x.where(x.time.dt.month == m, drop=True)

        r_l = []
        p_l = []
        for lag in lags:
            ym = y.shift(time=-lag).sel(time=xm.time)  # y(t+lag) aligned to xm.time

            a = xm.values
            b = ym.values
            ok = np.isfinite(a) & np.isfinite(b)
            n = int(ok.sum())

            if n < 6:
                r_l.append(np.nan)
                p_l.append(np.nan)
                continue

            aa = a[ok] - a[ok].mean()
            bb = b[ok] - b[ok].mean()
            denom = np.sqrt((aa**2).sum() * (bb**2).sum())
            r = float((aa * bb).sum() / denom) if denom > 0 else np.nan

            # p-value from t statistic
            if np.isfinite(r) and abs(r) < 1 and n > 2:
                tval = r * np.sqrt((n - 2) / (1 - r**2))
                p = 2 * student_t.sf(abs(tval), df=n - 2)
            else:
                p = np.nan

            r_l.append(r)
            p_l.append(p)

        r_out.append(r_l)
        p_out.append(p_l)

    r = xr.DataArray(r_out, coords={"month": np.arange(1, 13), "lag": lags}, dims=("month", "lag"))
    p = xr.DataArray(p_out, coords={"month": np.arange(1, 13), "lag": lags}, dims=("month", "lag"))
    return r, p



def plot_month_lag_to_ax(r: xr.DataArray, p: xr.DataArray, ax: plt.Axes, *, sig=0.05, nlags=36):
    """
    r,p: DataArray with dims ('month','lag') OR ('lag','month')
    """
    # Ensure correct orientation for plotting: rows=month, cols=lag
    if tuple(r.dims) != ("month", "lag"):
        r = r.transpose("month", "lag")
    if tuple(p.dims) != ("month", "lag"):
        p = p.transpose("month", "lag")

    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    levels  = np.linspace(-1, 1, 11)
    levels2 = np.linspace(-1, 1, 6)

    r_sig = r.where(p <= sig)

    X = r["lag"].values
    Y = np.arange(1, 13)  # month numbers

    cf = ax.contourf(X, Y, r_sig.values, levels=levels, cmap="RdBu_r", extend="both")
    cs = ax.contour(X, Y, r.values, levels=levels2, colors="k", linewidths=0.8,
                    negative_linestyles="dashed")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    ax.set_yticks([1, 4, 7, 10], labels=[months_lbl[j] for j in [0, 3, 6, 9]])
    ax.set_xticks(np.arange(-nlags, nlags + 1, 12))
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("")

  #  ax.text(0.02, 1.02, "y precedes x", transform=ax.transAxes,
  #          ha="left", va="bottom", fontsize=9)
  #  ax.text(0.98, 1.02, "y follows x", transform=ax.transAxes,
  #          ha="right", va="bottom", fontsize=9)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    return cf





if __name__ == "__main__":
    N=50
    run='basin.goodtest'

    if not Path('era5.latents.v1.nc').is_file():
        results_base = Path(f'~/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

        hs  = []
        tc3 = []
        for init in range(N):
            tc2 = []
            for split in range(5):
                tc = []
                for recursion in range(3):
                    encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'val_data.encodings.nc').encodings
                    tc.append(encodings)
                tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(3)})
                tc2.append(tc)
            tc2 = xr.concat(tc2, 'time').sortby('time')
            tc3.append(tc2)
        era5_latents = xr.concat(tc3, 'seed').assign_coords({'seed': np.arange(N)}).isel(recursion=0, drop=True)
        era5_latents.to_netcdf('era5.latents.v1.nc')
    era5_latents = xr.open_dataset('era5.latents.v1.nc')['encodings']

    iq50 = era5_latents.sel(mode="Interannual").mean('seed').quantile(0.5, 'time')
    dq50 = era5_latents.sel(mode='Decadal').mean('seed').quantile(0.5, 'time')

    if not Path('e3sm.latents.v1.nc').is_file():
        tc3 = []
        for rs in range(N):
            tc = []
            for split in range(5):
                tc5 = []
                for member in range(21):
                    print(f'rs{rs} split{split} member{member}')
                    ds = xr.open_dataset(f'~/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014/rs{rs}/split{split}/recursion0/e3sm/e3sm.mem{member}.encodings{split}.nc')
                    tc5.append(ds.encodings)
                tc5 = xr.concat(tc5, 'member').assign_coords({'member': np.arange(21)})
                tc.append(tc5)
            tc = xr.concat(tc, 'fold').assign_coords({'fold': np.arange(5)})
            tc3.append(tc)
        e3sm_latents = xr.concat(tc3, 'seed').assign_coords({'seed': np.arange(N)})
        e3sm_latents.to_netcdf('e3sm.latents.v1.nc')
    e3sm_latents = xr.open_dataset('e3sm.latents.v1.nc')['encodings']


    if not Path('cesm.latents.v1.nc').is_file():
        tc3 = []
        for rs in range(N):
            tc = []
            for split in range(5):
                tc5 = []
                for member in range(21):
                    print(f'rs{rs} split{split} member{member}')
                    ds = xr.open_dataset(f'~/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014/rs{rs}/split{split}/recursion0/cesm/cesm.mem{member}.encodings{split}.nc')
                    tc5.append(ds.encodings)
                tc5 = xr.concat(tc5, 'member').assign_coords({'member': np.arange(21)})
                tc.append(tc5)
            tc = xr.concat(tc, 'fold').assign_coords({'fold': np.arange(5)})
            tc3.append(tc)
        cesm_latents = xr.concat(tc3, 'seed').assign_coords({'seed': np.arange(N)})
        cesm_latents.to_netcdf('cesm.latents.v1.nc')
    cesm_latents = xr.open_dataset('cesm.latents.v1.nc')['encodings']


    # ----------------------------
    # Figure layout (no overlap)
    # ----------------------------
    fig = plt.figure(figsize=(13.0, 7.6), constrained_layout=True)

    # Make constrained_layout leave a little breathing room
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.08)

    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        height_ratios=[1.0, 0.07, 0.42, 0.07],   # give bottom row a bit more height
        width_ratios=[1, 1, 1]
    )

    # Top row: joint densities
    axH = [fig.add_subplot(gs[0, j]) for j in range(3)]
    caxH = fig.add_subplot(gs[1, 1])  # shared density colorbar

    # Bottom row: month–lag correlations
    axL = [fig.add_subplot(gs[2, j]) for j in range(3)]
    caxL = fig.add_subplot(gs[3, 1])  # shared corr colorbar


    das = [era5_latents, e3sm_latents, cesm_latents]
    titles = ["ERA5", "E3SM", "CESM"]

    # --- top row: joint densities
    im0 = None
    for j, (ax, da, ttl) in enumerate(zip(axH, das, titles)):
        im = plot_joint_to_ax(da, ax)
        ax.set_title(ttl, fontsize=11, pad=6)
        ax.set_aspect('auto')  # keeps squares without weird squeezing
        if im0 is None:
            im0 = im
            # grid lines at percentile thresholds
        if np.isfinite(dq50) :
            ax.axvline(dq50, color="k", lw=1.0)
        if np.isfinite(iq50) :
            ax.axhline(iq50, color="k", lw=1.0)

        # corners of the plot (axes coords)
        corner_labels = {
            (0.03, 0.99): "EP Warm Anomaly",  # top-left
            (0.03, 0.01): "CP Niña-like",  # bottom-left
            (0.97, 0.99): "CP Niño-like",  # top-right
            (0.97, 0.01): "EP Cold Anomaly",  # bottom-right
        }

        for (xA, yA), lab in corner_labels.items():
            ax.text(
                xA, yA, lab,
                transform=ax.transAxes,   # <-- axes coords
                ha="left" if xA < 0.5 else "right",
                va="top"  if yA > 0.5 else "bottom",
                fontsize=10, color="k",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2),
                zorder=10,
                clip_on=False,
            )


    cbH = fig.colorbar(im0, cax=caxH, orientation="horizontal")
    cbH.set_label("Joint density")
    cbH.ax.tick_params(labelsize=9)
    cbH.outline.set_visible(False)
    letters = ['d', 'e', 'f', 'a', 'b', 'c', ]

    # --- bottom row: month–lag correlations
    cf0 = None
    for j, (ax, da) in enumerate(zip(axL, das)):
        ia = da.sel(mode="Interannual")
        qb = da.sel(mode="Quasibiennial")

        r, p = month_lag_corr_xr(ia, qb, nlags=36)
        cf = plot_month_lag_to_ax(r, p, ax, sig=0.05, nlags=36)
        if cf0 is None:
            cf0 = cf

        # Make the panel compact & readable
        ax.set_title("")  # no per-panel title
        ax.tick_params(labelsize=9)

        # IMPORTANT: move the "precedes/follows" text into axes coords so it won't collide
        ax.text(0.02, 1.02, "QB precedes IA", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9)
        ax.text(0.98, 1.02, "QB follows IA", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9)
        ax.text(0.0, 1.15, f"{letters[j]})", transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", clip_on=False)

    cbL = fig.colorbar(cf0, cax=caxL, orientation="horizontal")
    cbL.set_label("Correlation Coefficient (r)")
    cbL.set_ticks(np.linspace(-1, 1, 6))
    cbL.outline.set_visible(False)
    

    for _,ax in enumerate(axH):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
            # ---- Panel labels
        ax.text(0.0, 1.0, f"{letters[_+3]})", transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", clip_on=False)
    plt.savefig('model_eval.png', dpi=300)
    plt.show()
