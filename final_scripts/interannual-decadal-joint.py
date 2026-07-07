import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec

# -----------------------------
# Bootstrap utilities
# -----------------------------
def circular_block_indices(n, block_len, rng):
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
    return idx.astype(int)

def _q33_67(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan, np.nan
    return np.nanquantile(x, 0.5), np.nanquantile(x, 0.5)

def _ci_excludes_zero(boot_maps, qlo=0.025, qhi=0.975):
    lo = np.nanquantile(boot_maps, qlo, axis=0)
    hi = np.nanquantile(boot_maps, qhi, axis=0)
    sig = (lo > 0) | (hi < 0)
    med = np.nanquantile(boot_maps, 0.5, axis=0)
    return med, lo, hi, sig

def add_panel_label(ax, label, fontsize=12, dx=0.0, dy=0.02):
    ax.text(
        0.0 + dx, 1.0 + dy, f"{label})",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=fontsize, fontweight="bold", clip_on=False
    )

# -----------------------------
# Compositing core
# -----------------------------
def composite_once_joint(
    ssta, z, i_mode, d_mode, other_modes,
    i_sign, d_sign,
    qlo=0.33, qhi=0.67
):
    """
    One composite mean map for a given joint quadrant with other-modes gated to middle third.
    i_sign, d_sign in {+1,-1} indicate high (>qhi) or low (<qlo).
    Returns (lat,lon) DataArray or None if too few samples.
    """
    zi = z.sel(mode=i_mode).values
    zd = z.sel(mode=d_mode).values

    i33, i67 = _q33_67(zi)
    d33, d67 = _q33_67(zd)
    if not (np.isfinite(i33) and np.isfinite(i67) and np.isfinite(d33) and np.isfinite(d67)):
        return None

    # thresholds for i,d quadrant
    if i_sign > 0:
        mi = zi > i67
    else:
        mi = zi < i33
    if d_sign > 0:
        md = zd > d67
    else:
        md = zd < d33

    gate = mi & md

    # other modes: middle third simultaneously
    #for m in other_modes:
    #    zk = z.sel(mode=m).values
    #    k33, k67 = _q33_67(zk)
    #    if not (np.isfinite(k33) and np.isfinite(k67)):
    #        return None
    #    gate &= (zk >= k33) & (zk <= k67)

    idx = np.where(gate)[0]
    if idx.size < 12:  # require at least a year of samples
        return None

    return ssta.isel(time=idx).mean("time", skipna=True)

def bootstrap_joint_composite(
    ssta, z, i_mode, d_mode, other_modes,
    i_sign, d_sign,
    n_boot=500, block_len=12, rng_seed=0
):
    """
    Returns dict with:
      med: (lat,lon) median composite
      sig: (lat,lon) CI excludes 0 mask
      lo/hi: CI bounds
    Bootstrap resamples time with circular blocks, recomputes percentiles each replicate.
    """
    s2, z2 = xr.align(ssta, z, join="inner")
    n = s2.sizes["time"]
    rng = np.random.default_rng(rng_seed)

    # preallocate boot maps as numpy for speed
    lat = s2["lat"].values
    lon = s2["lon"].values
    boot_maps = np.full((n_boot, lat.size, lon.size), np.nan, dtype=np.float32)

    for b in range(n_boot):
        idx = circular_block_indices(n, block_len, rng)
        s_b = s2.isel(time=idx)
        z_b = z2.isel(time=idx)

        m = composite_once_joint(
            s_b, z_b,
            i_mode=i_mode, d_mode=d_mode, other_modes=other_modes,
            i_sign=i_sign, d_sign=d_sign
        )
        if m is None:
            continue
        boot_maps[b, :, :] = m.values.astype(np.float32)

    med, lo, hi, sig = _ci_excludes_zero(boot_maps)
    med_da = xr.DataArray(med, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    lo_da  = xr.DataArray(lo,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    hi_da  = xr.DataArray(hi,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    sig_da = xr.DataArray(sig, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    return {"med": med_da, "lo": lo_da, "hi": hi_da, "sig": sig_da}

# -----------------------------
# Plotting helpers
# -----------------------------
def contour_sign(ax, da, levels, *, transform, color="k", lw=0.8):
    pos = [l for l in levels if l > 0]
    neg = [l for l in levels if l < 0]
    if neg:
        ax.contour(da["lon"], da["lat"], da, levels=neg, transform=transform,
                   colors=color, linewidths=lw, linestyles="--")
    if pos:
        ax.contour(da["lon"], da["lat"], da, levels=pos, transform=transform,
                   colors=color, linewidths=lw, linestyles="-")

def nice_sym_levels(data_list, n=17, robust_q=0.99):
    vals = []
    for da in data_list:
        v = da.values
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return np.linspace(-1, 1, n)
    v = np.concatenate(vals)
    vmax = float(np.nanquantile(np.abs(v), robust_q))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else 1.0
        if vmax == 0:
            vmax = 1.0
    return np.linspace(-vmax, vmax, n)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


def plot_joint_extremes_figure_v2(
    ssta, z_all,
    *,
    mode_interannual=4,
    mode_decadal=5,
    other_modes=None,
    n_boot=500,
    block_len=12,
    rng_seed=123,
    central_longitude=180,
    cmap="RdBu_r",
    # density controls
    hist_bins=80,
    smooth_sigma=1.2,   # in bin units
):
    """
    Same as previous, but:
      - colorbar under left maps (horizontal)
      - no titles on left maps or right density
      - right density panel ~square
      - use smoothed 2D histogram instead of hexbin
    """

    # ---- ensure z is (time, mode)
    z = z_all.mean('seed')
    if "time" not in z.dims:
        if "dim" in z.dims:
            z = z.rename({"dim": "time"})
        z = z.transpose("time", "mode")

    z_stack = z_all.rename({'time': 'time1'}).stack(time=('time1', 'seed'))

    if other_modes is None:
        other_modes = [m for m in z["mode"].values.tolist()
                       if m not in [mode_interannual, mode_decadal]]

    # ---- compute 4 quadrant boot composites
    quads = {
        "TL": dict(i_sign=+1, d_sign=-1),
        "BL": dict(i_sign=-1, d_sign=-1),
        "TR": dict(i_sign=+1, d_sign=+1),
        "BR": dict(i_sign=-1, d_sign=+1),
    }

    boot = {}
    for k, q in quads.items():
        boot[k] = bootstrap_joint_composite(
            ssta, z,
            i_mode=mode_interannual,
            d_mode=mode_decadal,
            other_modes=other_modes,
            i_sign=q["i_sign"],
            d_sign=q["d_sign"],
            n_boot=n_boot,
            block_len=block_len,
            rng_seed=rng_seed + (0 if k=="TL" else 1 if k=="TR" else 2 if k=="BL" else 3),
        )

    meds = [boot[k]["med"] for k in ["TL", "TR", "BL", "BR"]]
    fill_levels = np.linspace(-1.2, 1.2, 31)#nice_sym_levels(meds, n=21, robust_q=0.99)
    line_levels = np.linspace(-1.2, 1.2, 9)#nice_sym_levels(meds, n=9,  robust_q=0.99)

    # thresholds for grid lines
    zi = z.sel(mode=mode_interannual).values
    zd = z.sel(mode=mode_decadal).values
    i33, i67 = _q33_67(zi)
    d33, d67 = _q33_67(zd)

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree(central_longitude=central_longitude)

    # -----------------------------
    # Layout: 3 rows x 3 cols
    #   rows 0-1: left maps (2x2) + right density spans rows 0-1
    #   row 2:   left colorbar only (under maps), right cell empty
    # -----------------------------
    fig = plt.figure(figsize=(13.0, 6.8))
    gs = GridSpec(
        nrows=3, ncols=4, figure=fig,
        height_ratios=[1.0, 1.0, 0.10],
        width_ratios=[1.0, 1.0, 0.2, 1.6],
        wspace=0.10,
        hspace=0.10,
    )

    ax_TL = fig.add_subplot(gs[0, 0], projection=proj)
    ax_TR = fig.add_subplot(gs[0, 1], projection=proj)
    ax_BL = fig.add_subplot(gs[1, 0], projection=proj)
    ax_BR = fig.add_subplot(gs[1, 1], projection=proj)

    ax_dist = fig.add_subplot(gs[0:2, 3])  # spans rows 0-1
    cax_left  = fig.add_subplot(gs[2, 0:2])  # under left maps
    cax_right = fig.add_subplot(gs[2, 3])    # under right density
    # (rightmost bottom cell gs[2,2] intentionally unused)

    # panel letters
    add_panel_label(ax_TL, "a")
    add_panel_label(ax_TR, "b")
    add_panel_label(ax_BL, "c")
    add_panel_label(ax_BR, "d")
    add_panel_label(ax_dist, "e")

    # -----------------------------
    # Left maps
    # -----------------------------
    mappable = None
    order = [("TL", ax_TL), ("TR", ax_TR), ("BL", ax_BL), ("BR", ax_BR)]
    titles = {
        "TL": "EP Warm Anomaly",  # top-left
        "BL": "CP Niña-like",  # bottom-left
        "TR": "CP Niño-like",  # top-right
        "BR": "EP Cold Anomaly",  # bottom-right
    }
    for key, ax in order:
        ax.coastlines(linewidth=0.6)

        da = boot[key]["med"]

        ds = xr.open_dataset('climate_mode_patterns.nc').sel(mode=['E-Index', 'C-Index', 'ONI']).pattern
        print(key, xr.corr(da.sel(lat=slice(-20,20)), ds.sel(lat=slice(-20,20)), dim=['lat', 'lon']))

        sig = boot[key]["sig"]
        da_sig = da.where(sig)

        cf = ax.contourf(
            da["lon"], da["lat"], da_sig,
            levels=fill_levels, cmap=cmap, extend="both",
            transform=data_crs
        )
        if mappable is None:
            mappable = cf

        contour_sign(ax, da, line_levels, transform=data_crs, color="k", lw=0.8)

        # remove titles entirely (per request)
        ax.set_title(titles[key])

    # horizontal colorbar under left maps
    cb = fig.colorbar(mappable, cax=cax_left, orientation="horizontal")
    cb.set_label("SST anomaly composite (K)")
    cb.ax.tick_params(labelsize=9)

    # -----------------------------
    # Right: smooth 2D histogram + tic-tac-toe lines
    # -----------------------------
    # Remove any title
    ax_dist.set_title("")

    # data (x=decadal, y=interannual)
    zi = z_stack.sel(mode=mode_interannual).values
    zd = z_stack.sel(mode=mode_decadal).values
    ok = np.isfinite(zd) & np.isfinite(zi)
    x = zd[ok]
    y = zi[ok]

    # 2D histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=hist_bins, density=True)


    # Mask bins with too few points so they render as white
    MIN_COUNT = 0.0001
    H_masked = np.ma.masked_where(H < MIN_COUNT, H)

    Hs = H.astype(float)
    if gaussian_filter is not None and smooth_sigma and smooth_sigma > 0:
        Hs = gaussian_filter(Hs, sigma=smooth_sigma)
    Hs_masked = np.ma.masked_where(H < MIN_COUNT, Hs)

    cmap_den = plt.get_cmap("magma_r").copy()
    cmap_den.set_bad(color="white")

    # plot as image with continuous interpolation
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax_dist.imshow(
        Hs_masked.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",
        cmap=cmap_den
    )

    # grid lines at percentile thresholds
    if np.isfinite(d33) and np.isfinite(d67):
        ax_dist.axvline(d33, color="k", lw=1.0)
        ax_dist.axvline(d67, color="k", lw=1.0)
    if np.isfinite(i33) and np.isfinite(i67):
        ax_dist.axhline(i33, color="k", lw=1.0)
        ax_dist.axhline(i67, color="k", lw=1.0)

    corner_labels = {
        (0.03, 0.99): "EP Warm Anomaly",  # top-left
        (0.03, 0.01): "CP Niña-like",  # bottom-left
        (0.97, 0.99): "CP Niño-like",  # top-right
        (0.97, 0.01): "EP Cold Anomaly",  # bottom-right
    }


    for (xA, yA), lab in corner_labels.items():
        ax_dist.text(
            xA, yA, lab,
            transform=ax_dist.transAxes,   # <-- axes coords
            ha="left" if xA < 0.5 else "right",
            va="top"  if yA > 0.5 else "bottom",
            fontsize=12, color="k",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2),
            zorder=10,
            clip_on=False,
        )



    # optional middle-third box (keeps “tic-tac-toe” visually obvious)
    if np.isfinite(d33) and np.isfinite(d67) and np.isfinite(i33) and np.isfinite(i67):
        ax_dist.plot([d33, d67, d67, d33, d33],
                     [i33, i33, i67, i67, i33],
                     color="k", lw=0.8)

    ax_dist.set_xlabel("Decadal latent (z_d)")
    ax_dist.set_ylabel("Interannual latent (z_i)")

    # make it ~square
    # (set_box_aspect is best; fallback to equal aspect if older MPL)
    try:
        ax_dist.set_box_aspect(1)
    except Exception:
        ax_dist.set_aspect("equal", adjustable="box")

    # Make gridlines (no labels)
    gl = ax_dist.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.65")
    # Hide ticks + labels
    ax_dist.set_xticks([-2, -1, 0, 1, 2])
    ax_dist.set_yticks([-2, -1, 0, 1, 2])

    #ax_dist.set_xlabel("")
    #ax_dist.set_ylabel("")
    # Hide spines/border
    for sp in ax_dist.spines.values():
        sp.set_visible(False)

    # Density colorbar under the right panel (same "style" as left)
    cb_den = fig.colorbar(im, cax=cax_right, orientation="horizontal")
    cb_den.set_label("Joint density (counts per bin)")
    cb_den.ax.tick_params(labelsize=9)


    def add_xlabels_to_ax(axi, label):
        axi.text(
            0.5, -0.08, label,
            transform=axi.transAxes,
            ha="center",
            va="top",
            fontsize=10
        )
    
    def add_ylabels_to_ax(axi, label):
        axi.text(
            -0.08, 0.5, label,
            transform=axi.transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10
        )

    # tighten margins a bit
    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.08)
    add_ylabels_to_ax(ax_BL, "Negative Interannual")
    add_xlabels_to_ax(ax_BL, "Negative Decadal")
    add_xlabels_to_ax(ax_BR, "Positive Decadal")
    add_ylabels_to_ax(ax_TL, "Positive Interannual")
    return fig, boot



def month_histograms_by_quadrant(
    z: xr.DataArray,
    ia_mode: int,
    d_mode: int,
    *,
    normalize: str = "fraction",   # "fraction" (sum=1) or "zscore_vs_climo"
    ax=None,
    title_prefix="",
):
    """
    z: xr.DataArray with dims ('time','mode') and datetime-like time coord.
    Splits by medians of ia_mode (y) and d_mode (x), no other restrictions.

    Returns:
      out: dict with keys ['IA+ D-','IA- D-','IA+ D+','IA- D+'] each -> (12,) array
      months: 1..12
    """
    assert "time" in z.dims and "mode" in z.dims, "z must have dims ('time','mode')"
    # pull series
    zi = z.sel(mode=ia_mode)
    zd = z.sel(mode=d_mode)

    # medians (computed over available time)
    mi = float(zi.median("time", skipna=True).values)
    md = float(zd.median("time", skipna=True).values)

    # month index 1..12
    months = z["time"].dt.month

    # quadrant masks (IA on y, D on x)
    m_IAp = zi >  mi
    m_IAm = zi <= mi
    m_Dp  = zd >  md
    m_Dm  = zd <= md

    masks = {
        "IA+ D-": (m_IAp & m_Dm),
        "IA- D-": (m_IAm & m_Dm),
        "IA+ D+": (m_IAp & m_Dp),
        "IA- D+": (m_IAm & m_Dp),
    }

    # base climatology over ALL samples (for zscore option)
    base_counts = months.groupby(months).count().reindex(month=np.arange(1,13))
    base_frac = (base_counts / base_counts.sum()).values

    out = {}
    n_in_box = {}

    for name, m in masks.items():
        sel_months = months.where(m, drop=True)
        if sel_months.size == 0:
            counts = np.zeros(12, dtype=int)
        else:
            counts = sel_months.groupby(sel_months).count().reindex(month=np.arange(1,13)).fillna(0).astype(int).values

        n = int(np.sum(counts))
        n_in_box[name] = n

        if normalize == "fraction":
            vals = counts / n if n > 0 else np.full(12, np.nan)
        elif normalize == "zscore_vs_climo":
            # compare quadrant month fractions to full-sample climatology (simple z-ish normalization)
            frac = counts / n if n > 0 else np.full(12, np.nan)
            # avoid divide-by-zero: use binomial approx std for base frac
            # std ~ sqrt(p(1-p)/n) where n is quadrant size
            std = np.sqrt(base_frac * (1 - base_frac) / max(n, 1))
            std = np.where(std == 0, np.nan, std)
            vals = (frac - base_frac) / std
        else:
            raise ValueError("normalize must be 'fraction' or 'zscore_vs_climo'")

        out[name] = vals

    # ---- plotting ----
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.2))

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(12)

    # grouped bars (4 series)
    width = 0.20
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    order = ["IA+ D-","IA- D-","IA+ D+","IA- D+"]

    for off, key in zip(offsets, order):
        ax.bar(x + off, out[key], width=width, label=f"{key} (n={n_in_box[key]})")

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_xlim(-0.6, 11.6)

    if normalize == "fraction":
        ax.set_ylabel("Fraction of samples")
        ax.set_ylim(0, None)
    else:
        ax.set_ylabel("Std. dev. vs climatology")
        ax.axhline(0, color="0.5", linewidth=0.8)

    ax.set_title(f"{title_prefix}Month-of-year distribution by quadrant (median split)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)

    return out, np.arange(1,13)


if __name__ == "__main__":
    import kgae 
    import matplotlib.pyplot as plt 

    n_ensemble=100
    experiment = 'large-ensemble-2-1940-2014'

    z = kgae.open_latents('xval', experiment=experiment, n_ensemble=n_ensemble)#.mean('seed')
    ssta = kgae.open_references('xval', experiment=experiment)

    # -----------------------------
    # Script entry point
    # -----------------------------
    fig, boot = plot_joint_extremes_figure_v2(
        ssta, z, mode_interannual=4, mode_decadal=5,
        other_modes=[1,2,3], n_boot=1000, block_len=12
    )
    fig.savefig("joint_extremes_composites.png", dpi=300, bbox_inches="tight")
    plt.show()

    out, months = month_histograms_by_quadrant(
        z.mean('seed'), ia_mode=4, d_mode=5, normalize="fraction", title_prefix=""
    )
    plt.show()
