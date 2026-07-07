import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def add_panel_label(ax, label, dx=0.0, dy=0.02, fontsize=12):
    """
    Place panel label slightly above top-left corner of axis.
    dx, dy are figure-relative offsets in axis coordinates.
    """
    ax.text(
        0.0 + dx, 1.0 + dy,
        f"{label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
    )

# -----------------------------
# Bootstrap utilities
# -----------------------------
def circular_block_indices(n, block_len, rng):
    """Circular block bootstrap indices of length n."""
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend([(s + k) % n for k in range(block_len)])
    return np.array(idx[:n], dtype=int)

def _robust_sigma(x):
    """Std with safety; returns float."""
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s == 0.0:
        return np.nan
    return s

def composite_mean(field, mask_time):
    """
    field: DataArray (time, lat, lon)
    mask_time: boolean array shape (time,)
    returns: DataArray (lat, lon)
    """
    if mask_time.sum() < 3:
        return None
    return field.isel(time=np.where(mask_time)[0]).mean("time", skipna=True)

def composite_metrics_once(ssta, z, i_mode, other_modes, sigma_mult=1.0):
    """
    Compute (linear, symmetric) composite metrics once, using thresholds from z itself.
    Returns (lin, sym), each DataArray(lat, lon), or (None, None) if insufficient samples.
    """
    zi = z.sel(mode=i_mode).values
    sig_i = _robust_sigma(zi)
    if not np.isfinite(sig_i):
        return None, None

    # gating on other modes: |z_k| < sigma_k
    gate = np.ones(z.sizes["time"], dtype=bool)
    for m in other_modes:
        zk = z.sel(mode=m).values
        sig_k = _robust_sigma(zk)
        if not np.isfinite(sig_k):
            return None, None
        gate &= (np.abs(zk) < sigma_mult * sig_k)

    # masks
    pos = gate & (zi >  sigma_mult * sig_i)
    neg = gate & (zi < -sigma_mult * sig_i)
    neu = gate & (np.abs(zi) < sigma_mult * sig_i)
   
    a_pos = float(np.nanmean(zi[pos]))
    a_neg = float(np.nanmean(zi[neg]))
    a_neu = float(np.nanmean(zi[neu]))  # usually ~0


    mu_pos = composite_mean(ssta, pos)
    mu_neg = composite_mean(ssta, neg)
    mu_neu = composite_mean(ssta, neu)



    if (mu_pos is None) or (mu_neg is None) or (mu_neu is None):
        return None, None


    lin = (mu_pos - mu_neg) #/  (a_pos - a_neg)
    sym =  ( 
        (mu_pos - mu_neu) - 
        (mu_neu - mu_neg)
    ) #/ (a_pos - a_neg)
    return lin, sym

def bootstrap_composites(
    ssta, z, modes, *,
    n_boot=500, block_len=12, rng_seed=0, sigma_mult=1.0
):
    """
    modes: list of 3 mode labels (matching z.mode coords), e.g. [5,4,3]
    Returns dict:
      out["linear"][mode] -> DataArray(boot, lat, lon)
      out["symmetric"][mode] -> DataArray(boot, lat, lon)
      out["sig_linear"][mode] -> DataArray(lat, lon) boolean where 95% CI excludes 0
      out["sig_symmetric"][mode] -> DataArray(lat, lon) boolean
      out["ci_linear"][mode] -> DataArray(q, lat, lon) with q=[0.025,0.5,0.975]
      out["ci_symmetric"][mode] -> same
    """
    # align time
    ssta2, z2 = xr.align(ssta, z, join="inner")
    n = ssta2.sizes["time"]
    rng = np.random.default_rng(rng_seed)

    results = {"linear": {}, "symmetric": {}}

    for i_mode in modes:
        other_modes = [m for m in modes if m != i_mode]

        lin_list = []
        sym_list = []

        for b in range(n_boot):
            idx = circular_block_indices(n, block_len, rng)
            s_b = ssta2.isel(time=idx)
            z_b = z2.isel(time=idx)

            lin, sym = composite_metrics_once(
                s_b, z_b, i_mode=i_mode, other_modes=other_modes, sigma_mult=sigma_mult
            )

            if lin is None:
                # keep NaNs so quantiles ignore
                lin = xr.full_like(ssta2.isel(time=0), np.nan).drop_vars("time", errors="ignore")
                sym = xr.full_like(ssta2.isel(time=0), np.nan).drop_vars("time", errors="ignore")

            lin_list.append(lin)
            sym_list.append(sym)

        lin_boot = xr.concat(lin_list, dim="boot").assign_coords(boot=np.arange(n_boot))
        sym_boot = xr.concat(sym_list, dim="boot").assign_coords(boot=np.arange(n_boot))

        results["linear"][i_mode] = lin_boot
        results["symmetric"][i_mode] = sym_boot

    # CI + significance
    out = {
        "linear": results["linear"],
        "symmetric": results["symmetric"],
        "ci_linear": {},
        "ci_symmetric": {},
        "sig_linear": {},
        "sig_symmetric": {},
    }

    qs = [0.025, 0.5, 0.975]
    for i_mode in modes:
        lin = out["linear"][i_mode]
        sym = out["symmetric"][i_mode]

        ci_lin = lin.quantile(qs, dim="boot", skipna=True)
        ci_sym = sym.quantile(qs, dim="boot", skipna=True)

        # CI excludes zero
        lo_lin = ci_lin.sel(quantile=qs[0])
        hi_lin = ci_lin.sel(quantile=qs[2])
        sig_lin = (lo_lin > 0) | (hi_lin < 0)

        lo_sym = ci_sym.sel(quantile=qs[0])
        hi_sym = ci_sym.sel(quantile=qs[2])
        sig_sym = (lo_sym > 0) | (hi_sym < 0)

        out["ci_linear"][i_mode] = ci_lin
        out["ci_symmetric"][i_mode] = ci_sym
        out["sig_linear"][i_mode] = sig_lin
        out["sig_symmetric"][i_mode] = sig_sym

    return out

# -----------------------------
# Plotting
# -----------------------------
def _nice_levels(data_list, n_levels=13, robust=0.99):
    """Symmetric contour levels from pooled magnitudes."""
    vals = []
    for da in data_list:
        v = da.values
        vals.append(v[np.isfinite(v)])
    if not vals:
        return np.linspace(-1, 1, n_levels)
    v = np.concatenate(vals)
    if v.size == 0:
        return np.linspace(-1, 1, n_levels)
    vmax = float(np.nanquantile(np.abs(v), robust))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = float(np.nanmax(np.abs(v)))
        if vmax == 0 or not np.isfinite(vmax):
            vmax = 1.0
    return np.linspace(-vmax, vmax, n_levels)

def plot_six_panel_composites(
    ssta, z, *,
    modes, mode_titles,
    n_boot=500, block_len=12, rng_seed=0, sigma_mult=1.0,
    central_longitude=180,
    cmap='RdBu_r'
):
    """
    modes: list of 3 mode labels (matching z.mode coords)
    mode_titles: list of 3 strings for column titles
    """

    boot = bootstrap_composites(
        ssta, z, modes,
        n_boot=n_boot, block_len=block_len, rng_seed=rng_seed, sigma_mult=sigma_mult
    )
    # --- full-sample composites (no bootstrap) ---
    lin_full = []
    sym_full = []

    for m in modes:
        other = [mm for mm in modes if mm != m]
        lin, sym = composite_metrics_once(ssta, z, i_mode=m, other_modes=other)
        lin_full.append(lin)
        sym_full.append(sym)

    # medians
    lin_med = lin_full #[boot["ci_linear"][m].sel(quantile=0.5) for m in modes]
    sym_med = sym_full #[boot["ci_symmetric"][m].sel(quantile=0.5) for m in modes]

    # significance masks
    lin_sig = [boot["sig_linear"][m] for m in modes]
    sym_sig = [boot["sig_symmetric"][m] for m in modes]

    # levels per row
    s=2.5
    lev_lin = np.linspace(-s, s, 21)#_nice_levels(lin_med, n_levels=21, robust=0.99)
    lev_lin2 = np.linspace(-s, s, 7) #_nice_levels(lin_med, n_levels=7, robust=0.99)

    lev_sym = np.linspace(-s/2, s/2, 21)#_nice_levels(sym_med, n_levels=21, robust=0.99)
    lev_sym2 = np.linspace(-s/2, s/2, 7)#_nice_levels(sym_med, n_levels=7, robust=0.99)


    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    pc = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(12.2, 6.8))
    gs = gridspec.GridSpec(
        2, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        hspace=0.12,
        wspace=0.08,
    )

    letrs = ['a', 'b', 'c', 'd', 'e', 'f' ]
    axes = []
    for r in range(2):
        row_axes = []
        for c in range(3):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.coastlines(linewidth=0.7)
            add_panel_label(ax, letrs[r*3+c])
            row_axes.append(ax)
        axes.append(row_axes)

    # Dedicated colorbar axes keep all six map panels the same fixed size.
    cax_top = fig.add_subplot(gs[0, 3])
    cax_bot = fig.add_subplot(gs[1, 3])


    # helper for contours
    def contour_sign(ax, da, levels, **kw):
        pos = [l for l in levels if l > 0]
        neg = [l for l in levels if l < 0]
        cs_pos = ax.contour(da["lon"], da["lat"], da, levels=pos, transform=pc,
                            linewidths=0.9, linestyles="-", **kw)
        cs_neg = ax.contour(da["lon"], da["lat"], da, levels=neg, transform=pc,
                            linewidths=0.9, linestyles="--", **kw)
        return cs_pos, cs_neg

    # --- TOP ROW: linear (difference) ---
    mappable_top = None
    for j, m in enumerate(modes):
        ax = axes[0][j]
        da = lin_med[j]
        sig = lin_sig[j]

        da_sig = da.where(sig)

        # filled only where significant
        cf = ax.contourf(
            da["lon"], da["lat"], da_sig,
            levels=lev_lin, transform=pc, extend="both",
            cmap=cmap
        )
        if mappable_top is None:
            mappable_top = cf

        # line contours everywhere
        contour_sign(ax, da, lev_lin2, colors="k")

        ax.set_title(mode_titles[j])

    # --- TOP ROW colorbar (manual axis aligned to panels) ---
    cb_top = fig.colorbar(
        mappable_top,
        cax=cax_top,
        orientation="vertical",
    )
    cb_top.set_label(r"Linear response ($K$)")


    # --- BOTTOM ROW: symmetric (even) ---
    mappable_bot = None
    for j, m in enumerate(modes):
        ax = axes[1][j]
        da = sym_med[j]
        sig = sym_sig[j]

        da_sig = da.where(sig)

        cf = ax.contourf(
            da["lon"], da["lat"], da_sig,
            levels=lev_sym, transform=pc, extend="both",
            cmap=cmap
        )
        if mappable_bot is None:
            mappable_bot = cf

        contour_sign(ax, da, lev_sym2, colors="k")


    # --- BOTTOM ROW colorbar ---
    cb_bot = fig.colorbar(
        mappable_bot,
        cax=cax_bot,
        orientation="vertical",
    )
    cb_bot.set_label(r"Nonlinear response ($K$)")

    #fig.subplots_adjust(left=0.06, right=0.92, top=0.95, bottom=0.06)
    return fig, boot


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    import kgae 
    import matplotlib.pyplot as plt 

    n_ensemble=100
    experiment = './large-ensemble-2-1940-2014'

    z = kgae.open_latents('xval', experiment=experiment, n_ensemble=n_ensemble).mean('seed')
    ssta = kgae.open_references('xval', experiment=experiment)

    modes = [5,4,3]  # must match z.mode coords
    mode_titles = ["Decadal", "Interannual", "Quasibiennial"]
    fig, boot = plot_six_panel_composites(
        ssta, z,
        modes=modes,
        mode_titles=mode_titles,
        n_boot=100,
        block_len=12,
        rng_seed=123,
        sigma_mult=1,
        central_longitude=180,
    )
    fig.savefig("conditional_composites.png", dpi=300)
    plt.show()
