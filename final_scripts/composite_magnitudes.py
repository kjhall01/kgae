#!/usr/bin/env python3
"""
Conditional composite analysis with
**phase-restricted** (sign-constrained) linear/nonlinear responses.

What it computes (per target mode i, with optional gating on other modes):
-----------------------------------------------------------------------
We form *three magnitude bins within each sign*:

Positive phase bins (z_i > 0):
  lo:  0 < z < t1*sigma
  mid: t1*sigma <= z < t2*sigma
  hi:  z >= t2*sigma

Negative phase bins (z_i < 0):
  lo:  -t1*sigma < z < 0
  mid: -t2*sigma < z <= -t1*sigma
  hi:  z <= -t2*sigma

For each sign we compute:
  - Linear (phase-restricted slope):     L_pos, L_neg   [K / z]
  - Nonlinear (phase-restricted curvature proxy):
        NL_pos = slope(hi-mid) - slope(mid-lo)          [K / z^2]
        NL_neg = slope(hi-mid) - slope(mid-lo)          [K / z^2]

Bottom-row plot choices:
  * "nl_pos"  : nonlinearity within positive phase only
  * "nl_neg"  : nonlinearity within negative phase only
  * "nl_sum"  : NL_pos + NL_neg  (magnitude-driven curvature, sign-blind)
  * "nl_diff" : NL_pos - NL_neg  (phase-asymmetry contrast)

Plot:
  - 2 x 3 map figure (same style as before):
      top row: L_pos  (default)
      bottom row: nl_diff (default)
  - Filled contours only where bootstrap CI excludes 0, line contours everywhere
    (positive solid, negative dashed).
  - One colorbar per row.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec


# -----------------------------
# Small plot helper
# -----------------------------
def add_panel_label(ax, label, dx=0.0, dy=0.02, fontsize=12):
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
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s == 0.0:
        return np.nan
    return s


def composite_mean(field, mask_time, min_count=3):
    """
    field: DataArray (time, lat, lon)
    mask_time: boolean array shape (time,)
    returns: DataArray (lat, lon) or None
    """
    if int(mask_time.sum()) < min_count:
        return None
    return field.isel(time=np.where(mask_time)[0]).mean("time", skipna=True)


# -----------------------------
# Phase-binned metrics (core)
# -----------------------------
def composite_metrics_once_phase_bins(
    field,
    z,
    *,
    i_mode,
    other_modes,
    sigma_mult_gate=1.0,
    t1=0.75,
    t2=1.5,
    min_count=5,
):
    """
    Compute phase-restricted linear + nonlinear responses for one mode i.

    Returns dict of DataArray(lat, lon):
      lin_pos, lin_neg, nl_pos, nl_neg
    Or None if insufficient samples.
    """
    zi = z.sel(mode=i_mode).values
    sig_i = _robust_sigma(zi)
    if not np.isfinite(sig_i):
        return None

    # gating on other modes: |z_k| < sigma_mult_gate * sigma_k
    gate = np.ones(z.sizes["time"], dtype=bool)
    for m in other_modes:
        zk = z.sel(mode=m).values
        sig_k = _robust_sigma(zk)
        if not np.isfinite(sig_k):
            return None
        gate &= (np.abs(zk) < sigma_mult_gate * sig_k)

    # thresholds in raw latent units
    T1 = float(t1 * sig_i)
    T2 = float(t2 * sig_i)
    if not (T2 > T1 > 0):
        raise ValueError("Require t2 > t1 > 0")

    # bins: POSITIVE phase (z>0)
    pos_lo  = gate & (zi > 0.0) & (zi <  T1)
    pos_mid = gate & (zi >= T1) & (zi < T2)
    pos_hi  = gate & (zi >= T2)

    # bins: NEGATIVE phase (z<0)
    neg_lo  = gate & (zi < 0.0) & (zi > -T1)
    neg_mid = gate & (zi <= -T1) & (zi > -T2)
    neg_hi  = gate & (zi <= -T2)

    def _bin_stats(mask):
        n = int(mask.sum())
        if n < min_count:
            return None, None, n
        a = float(np.nanmean(zi[mask]))
        mu = composite_mean(field, mask, min_count=min_count)
        return a, mu, n

    a_pl, mu_pl, n_pl = _bin_stats(pos_lo)
    a_pm, mu_pm, n_pm = _bin_stats(pos_mid)
    a_ph, mu_ph, n_ph = _bin_stats(pos_hi)

    a_nl, mu_nl, n_nl = _bin_stats(neg_lo)
    a_nm, mu_nm, n_nm = _bin_stats(neg_mid)
    a_nh, mu_nh, n_nh = _bin_stats(neg_hi)

    if any(x is None for x in [mu_pl, mu_pm, mu_ph, mu_nl, mu_nm, mu_nh]):
        return None

    # linear slopes (hi - lo)
    denom_pos = (a_ph - a_pl)
    denom_neg = (a_nh - a_nl)
    if (not np.isfinite(denom_pos)) or (not np.isfinite(denom_neg)) or denom_pos == 0 or denom_neg == 0:
        return None

    lin_pos = (mu_ph - mu_pl) / denom_pos
    lin_neg = (mu_nh - mu_nl) / denom_neg

    # nonlinearity = difference of local slopes (hi-mid) - (mid-lo)
    d1p = (a_ph - a_pm); d0p = (a_pm - a_pl)
    d1n = (a_nh - a_nm); d0n = (a_nm - a_nl)
    if any((not np.isfinite(d)) or d == 0 for d in [d1p, d0p, d1n, d0n]):
        return None

    slope_p_hi = (mu_ph - mu_pm) / d1p
    slope_p_lo = (mu_pm - mu_pl) / d0p
    nl_pos = slope_p_hi - slope_p_lo  # ~ K/z^2

    slope_n_hi = (mu_nh - mu_nm) / d1n
    slope_n_lo = (mu_nm - mu_nl) / d0n
    nl_neg = slope_n_hi - slope_n_lo  # ~ K/z^2

    return {
        "lin_pos": lin_pos,
        "lin_neg": lin_neg,
        "nl_pos": nl_pos,
        "nl_neg": nl_neg,
        "counts_pos": (n_pl, n_pm, n_ph),
        "counts_neg": (n_nl, n_nm, n_nh),
    }


def bootstrap_phase_binned(
    field,
    z,
    modes,
    *,
    n_boot=500,
    block_len=12,
    rng_seed=0,
    sigma_mult_gate=1.0,
    t1=0.5,
    t2=1.0,
    min_count=5,
):
    """
    Returns dict:
      out[key][mode] -> DataArray(boot, lat, lon)
      out['ci_'+key][mode] -> DataArray(quantile, lat, lon) with quantile=[q0,q1,q2]
      out['sig_'+key][mode] -> DataArray(lat, lon) bool (CI excludes 0)

    keys include:
      lin_pos, lin_neg, nl_pos, nl_neg, nl_sum, nl_diff
    """
    field2, z2 = xr.align(field, z, join="inner")
    n = field2.sizes["time"]
    rng = np.random.default_rng(rng_seed)

    keys = ["lin_pos", "lin_neg", "nl_pos", "nl_neg", "nl_sum", "nl_diff"]
    results = {k: {} for k in keys}

    # a blank map for missing boot draws
    blank = xr.full_like(field2.isel(time=0), np.nan).drop_vars("time", errors="ignore")

    for i_mode in modes:
        other_modes = [m for m in modes if m != i_mode]
        lists = {k: [] for k in keys}

        for _ in range(n_boot):
            idx = circular_block_indices(n, block_len, rng)
            f_b = field2.isel(time=idx)
            z_b = z2.isel(time=idx)

            res = composite_metrics_once_phase_bins(
                f_b, z_b,
                i_mode=i_mode,
                other_modes=other_modes,
                sigma_mult_gate=sigma_mult_gate,
                t1=t1, t2=t2,
                min_count=min_count,
            )

            if res is None:
                for k in keys:
                    lists[k].append(blank)
            else:
                lin_pos = res["lin_pos"]
                lin_neg = res["lin_neg"]
                nl_pos = res["nl_pos"]
                nl_neg = res["nl_neg"]
                lists["lin_pos"].append(lin_pos)
                lists["lin_neg"].append(lin_neg)
                lists["nl_pos"].append(nl_pos)
                lists["nl_neg"].append(nl_neg)
                lists["nl_sum"].append(nl_pos + nl_neg)
                lists["nl_diff"].append(nl_pos - nl_neg)

        for k in keys:
            results[k][i_mode] = xr.concat(lists[k], dim="boot").assign_coords(boot=np.arange(n_boot))

    out = {k: results[k] for k in keys}
    qs = [0.25, 0.5, 0.75]

    for k in keys:
        out["ci_" + k] = {}
        out["sig_" + k] = {}

    for i_mode in modes:
        for k in keys:
            da = out[k][i_mode]
            ci = da.quantile(qs, dim="boot", skipna=True)
            lo = ci.sel(quantile=qs[0])
            hi = ci.sel(quantile=qs[2])
            sig = (lo > 0) | (hi < 0)
            out["ci_" + k][i_mode] = ci
            out["sig_" + k][i_mode] = sig

    return out


# -----------------------------
# Plotting helpers
# -----------------------------
def contour_sign(ax, da, levels, pc, **kw):
    pos = [l for l in levels if l > 0]
    neg = [l for l in levels if l < 0]
    if pos:
        ax.contour(da["lon"], da["lat"], da, levels=pos, transform=pc,
                   linewidths=0.9, linestyles="-", **kw)
    if neg:
        ax.contour(da["lon"], da["lat"], da, levels=neg, transform=pc,
                   linewidths=0.9, linestyles="--", **kw)


def _nice_levels_from_maps(maps, n_levels=21, robust=0.99):
    vals = []
    for da in maps:
        v = da.values
        vals.append(v[np.isfinite(v)])
    v = np.concatenate(vals) if vals else np.array([])
    if v.size == 0:
        return np.linspace(-1, 1, n_levels)
    vmax = float(np.nanquantile(np.abs(v), robust))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(v)))
        vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0) else vmax
    return np.linspace(-vmax, vmax, n_levels)


# -----------------------------
# Main plot function
# -----------------------------
def plot_six_panel_composites(
    ssta,
    z,
    *,
    modes,
    mode_titles,
    n_boot=500,
    block_len=12,
    rng_seed=0,
    sigma_mult=1.0,
    central_longitude=180,
    cmap="RdBu_r",
    # Magnitude-bin controls
    t1=0.5,
    t2=1.0,
    min_count=5,
    top_key="lin_pos",          # 'lin_pos' or 'lin_neg'
    bottom_key="nl_diff",       # 'nl_pos','nl_neg','nl_sum','nl_diff'
    # levels
    lev_top=None,
    lev_top2=None,
    lev_bot=None,
    lev_bot2=None,
):
    """
    Produces a 2x3 figure like before.

    Top row shows top_key (default: lin_pos).
    Bottom row shows bottom_key (default: nl_diff = NL(+) - NL(-)).
    """
    boot = bootstrap_phase_binned(
        ssta, z, modes,
        n_boot=n_boot, block_len=block_len, rng_seed=rng_seed,
        sigma_mult_gate=sigma_mult, t1=t1, t2=t2, min_count=min_count
    )

    # full-sample (no bootstrap) medians: use full-sample metrics
    top_maps = []
    bot_maps = []
    top_sig = []
    bot_sig = []

    for m in modes:
        other = [mm for mm in modes if mm != m]
        res = composite_metrics_once_phase_bins(
            ssta, z,
            i_mode=m, other_modes=other,
            sigma_mult_gate=sigma_mult, t1=t1, t2=t2, min_count=min_count
        )
        if res is None:
            blank = xr.full_like(ssta.isel(time=0), np.nan).drop_vars("time", errors="ignore")
            top_maps.append(blank)
            bot_maps.append(blank)
        else:
            # compute bottom derived key consistently
            nl_sum = res["nl_pos"] + res["nl_neg"]
            nl_diff = res["nl_pos"] - res["nl_neg"]
            derived = dict(res)
            derived["nl_sum"] = nl_sum
            derived["nl_diff"] = nl_diff
            top_maps.append(derived[top_key])
            bot_maps.append(derived[bottom_key])

        top_sig.append(boot["sig_" + top_key][m])
        bot_sig.append(boot["sig_" + bottom_key][m])

    # levels
    if lev_top is None:
        lev_top = _nice_levels_from_maps(top_maps, n_levels=21, robust=0.99)
    if lev_top2 is None:
        lev_top2 = _nice_levels_from_maps(top_maps, n_levels=7, robust=0.99)

    if lev_bot is None:
        lev_bot = _nice_levels_from_maps(bot_maps, n_levels=21, robust=0.99)
    if lev_bot2 is None:
        lev_bot2 = _nice_levels_from_maps(bot_maps, n_levels=7, robust=0.99)

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    pc = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(11.5, 6.8))
    gs = gridspec.GridSpec(2, 3, hspace=0.12, wspace=0.05)

    letters = ["a", "b", "c", "d", "e", "f"]
    axes = []
    for r in range(2):
        row_axes = []
        for c in range(3):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.coastlines(linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.7)
            add_panel_label(ax, letters[r * 3 + c])
            row_axes.append(ax)
        axes.append(row_axes)

    # --- TOP ROW ---
    mappable_top = None
    for j, m in enumerate(modes):
        ax = axes[0][j]
        da = top_maps[j]
        sig = top_sig[j]
        da_sig = da.where(sig)

        cf = ax.contourf(da["lon"], da["lat"], da_sig, levels=lev_top,
                         transform=pc, extend="both", cmap=cmap)
        if mappable_top is None:
            mappable_top = cf
        contour_sign(ax, da, lev_top2, pc, colors="k")
        ax.set_title(mode_titles[j])

    cb_top = fig.colorbar(mappable_top, ax=axes[0], orientation="vertical",
                          fraction=0.025, pad=0.02)
    cb_top.set_label(top_key.replace("_", " "))

    # --- BOTTOM ROW ---
    mappable_bot = None
    for j, m in enumerate(modes):
        ax = axes[1][j]
        da = bot_maps[j]
        sig = bot_sig[j]
        da_sig = da.where(sig)

        cf = ax.contourf(da["lon"], da["lat"], da_sig, levels=lev_bot,
                         transform=pc, extend="both", cmap=cmap)
        if mappable_bot is None:
            mappable_bot = cf
        contour_sign(ax, da, lev_bot2, pc, colors="k")

    cb_bot = fig.colorbar(mappable_bot, ax=axes[1], orientation="vertical",
                          fraction=0.025, pad=0.02)
    cb_bot.set_label(bottom_key.replace("_", " "))

    return fig, boot


# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    import kgae

    n_ensemble = 100
    experiment = "./large-ensemble-2-1940-2014"

    # Use seed-mean latents for the composite.
    z = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble).mean("seed")
    ssta = kgae.open_references("xval", experiment=experiment)

    modes = [5, 4, 3]
    mode_titles = ["Decadal", "Interannual", "Quasibiennial"]

    fig, boot = plot_six_panel_composites(
        ssta, z,
        modes=modes,
        mode_titles=mode_titles,
        n_boot=100,
        block_len=12,
        rng_seed=123,
        sigma_mult=1.5,          # gating on other modes
        central_longitude=180,
        # Bin thresholds in sigma units
        t1=0.5,
        t2=1.0,
        min_count=5,
        # Output fields to show
        top_key="lin_pos",       # linear within positive phase
        bottom_key="nl_diff",    # NL(+) - NL(-)  (phase-asymmetry of nonlinearity)
    )

    fig.savefig("conditional_composites_phase_binned.png", dpi=300, bbox_inches="tight")
    plt.show()
