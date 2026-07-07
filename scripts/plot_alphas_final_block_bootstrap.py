import os
import kgae
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

# ============================
# CACHE SETTINGS
# ============================
need_to_calculate = True
cache_file = "cache_alpha_and_linear_composites.nc"

experiment = "kgvae_cv_ema-1940-2014"
n_ensemble = 36
# ----------------------------
# Alpha plotting significance
# ----------------------------
sig_thresh_alpha = 0.1   # fraction of seeds significant for alpha fill
sig_thresh_alpha_hatch = 0.999   # fraction of seeds significant for alpha hatch
ci_interval=0.95
# ----------------------------
# Composite masking like betas:
# - bootstrap per seed @ 95% CI
# - fill where frac(sig_seed) >= comp_sig_thresh
# - hatch where frac(sig_seed) >= comp_sig_thresh_hatch
# ----------------------------
comp_sig_thresh = 0.5
comp_sig_thresh_hatch = 0.9

n_boot_comp = 200
ci_level_comp = 0.90      # seed-level bootstrap CI (95%)

# Composite definition settings
sigma_thresh_comp = 1.0
neutral_thresh_comp = 1.0

nline = 11
nlev = 21

# Plot ONLY these three alphas + their linear composites
tendency_modes = [5, 4, 3]  # Decadal, Interannual, Quasibiennial

tendency_mode_labels = {
    5: "Decadal Mode",
    4: "Interannual Mode",
    3: "Quasibiennial Mode",
}

alpha_row_label = r"Global Sensitivity $\alpha$"
comp_row_label = r"Linear Composite"

# ----------------------------
# helpers
# ----------------------------
def _ensure_dims_alpha(da: xr.DataArray) -> xr.DataArray:
    dims = list(da.dims)
    rename = {}
    if "mode" in dims and "tendency_mode" not in dims:
        rename["mode"] = "tendency_mode"
    da = da.rename(rename)

    needed = ["tendency_mode", "lat", "lon", "seed"]
    missing = [d for d in needed if d not in da.dims]
    if missing:
        raise ValueError(f"alpha dims missing {missing}; got {da.dims}")
    return da.transpose("tendency_mode", "lat", "lon", "seed")


# ----------------------------
# 2-year block bootstrap settings
# ----------------------------
block_len_months = 24   # 2-year blocks (monthly data)


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

def _bootstrap_sigmask_linear_comp_one_seed(
    X: xr.DataArray,              # (time, lat, lon)
    z_seed: xr.DataArray,         # (time, tendency_mode) for ONE seed
    mode_val,
    mode_dim="tendency_mode",
    sigma_thresh=1.0,
    neutral_thresh=1.0,
    n_boot=500,
    ci_level=0.95,
    rng_seed=0,
    block_len=24,                 # block length in months (moving-block bootstrap)
    recompute_thresholds=True,    # recompute std/neutral thresholds within each bootstrap
    min_samples=5,                # guard against empty composites
) -> xr.DataArray:
    """
    Seed-level significance for linear/symmetric composite:
        comp = 0.5 * (E[X | zi > +sigma_thresh*std_i, neutral] - E[X | zi < -sigma_thresh*std_i, neutral])

    Correct moving-block bootstrap:
      1) resample the full time series with contiguous blocks
      2) recompute thresholds/masks *within each bootstrap replicate* (optional but recommended)
      3) compute composite in that replicate
      4) CI from bootstrap distribution; significant if CI excludes 0

    Returns
    -------
    sigmask : xr.DataArray (lat, lon) boolean
    """
    rng = np.random.default_rng(rng_seed)
    X, z_seed = xr.align(X, z_seed, join="inner")

    # numpy views
    tN = X.sizes["time"]
    lat = X["lat"]
    lon = X["lon"]
    X_np = X.values.astype(np.float32)

    # Latent arrays: (time, L)
    modes = z_seed[mode_dim].values
    z_np = z_seed.transpose("time", mode_dim).values.astype(np.float32)
    L = z_np.shape[1]

    # Identify target and "other" mode indices
    try:
        i_mode = int(np.where(modes == mode_val)[0][0])
    except Exception as e:
        raise ValueError(f"mode_val={mode_val} not found in z_seed[{mode_dim}].") from e
    other_idx = np.array([k for k in range(L) if k != i_mode], dtype=int)

    # When thresholds are fixed across replicates, compute them once here.
    if not recompute_thresholds:
        zi = z_np[:, i_mode]
        std_i_fixed = float(np.nanstd(zi, ddof=0))
        if other_idx.size > 0:
            z_not = z_np[:, other_idx]
            std_not_fixed = np.nanstd(z_not, axis=0, ddof=0)
        else:
            std_not_fixed = None

    boot = np.full((n_boot, X.sizes["lat"], X.sizes["lon"]), np.nan, dtype=np.float32)

    for b in range(n_boot):
        # 1) resample time first (moving blocks)
        idx = moving_block_bootstrap_indices(rng, tN, block_len=block_len)

        X_b = X_np[idx, :, :]
        z_b = z_np[idx, :]  # (time, L)

        # 2) compute thresholds/masks within replicate
        zi_b = z_b[:, i_mode]

        if recompute_thresholds:
            std_i = float(np.nanstd(zi_b, ddof=0))
            # neutral condition on other modes
            if other_idx.size > 0:
                z_not_b = z_b[:, other_idx]
                std_not = np.nanstd(z_not_b, axis=0, ddof=0)
                neutral_b = np.all(np.abs(z_not_b) < (neutral_thresh * std_not), axis=1)
            else:
                neutral_b = np.ones(zi_b.shape[0], dtype=bool)
        else:
            std_i = std_i_fixed
            if other_idx.size > 0:
                z_not_b = z_b[:, other_idx]
                neutral_b = np.all(np.abs(z_not_b) < (neutral_thresh * std_not_fixed), axis=1)
            else:
                neutral_b = np.ones(zi_b.shape[0], dtype=bool)

        pos_b = (zi_b > (sigma_thresh * std_i)) & neutral_b
        neg_b = (zi_b < (-sigma_thresh * std_i)) & neutral_b

        # 3) composite for this replicate (guard empties)
        if pos_b.sum() < min_samples or neg_b.sum() < min_samples:
            # leave NaNs for this replicate; percentile ops will ignore NaNs
            continue

        pos_m = np.nanmean(X_b[pos_b, :, :], axis=0)
        neg_m = np.nanmean(X_b[neg_b, :, :], axis=0)

        # symmetric/linear composite scaling
        boot[b, :, :] = (pos_m - neg_m) / 2.0

    # 4) CI + significance mask
    alpha = 1.0 - ci_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    lo = np.nanpercentile(boot, lo_q, axis=0)
    hi = np.nanpercentile(boot, hi_q, axis=0)

    sigmask = (lo > 0) | (hi < 0)

    return xr.DataArray(sigmask, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

def linregress_with_r2(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, b, r2


def _stack_map(da_map: xr.DataArray) -> np.ndarray:
    return da_map.values.reshape(-1)


def _composite_linear_holding_others_neutral(
    reference_sst: xr.DataArray,
    latents: xr.DataArray,
    mode_coord_name="tendency_mode",
    sigma_thresh=1.0,
    neutral_thresh=1.0,
) -> xr.DataArray:
    """
    Per-seed linear/symmetric composite:
      E[X | zi > +1σ and others neutral] - E[X | zi < -1σ and others neutral]
    Returns: (tendency_mode, lat, lon, seed)
    """
    if "mode" in latents.dims and mode_coord_name not in latents.dims:
        latents = latents.rename({"mode": mode_coord_name})

    if not all(d in latents.dims for d in ["time", mode_coord_name, "seed"]):
        raise ValueError(f"latents dims must include ('time','{mode_coord_name}','seed'); got {latents.dims}")
    if "time" not in reference_sst.dims or not all(d in reference_sst.dims for d in ["lat", "lon"]):
        raise ValueError(f"reference_sst must have dims ('time','lat','lon'); got {reference_sst.dims}")

    reference_sst, latents = xr.align(reference_sst, latents, join="inner")

    modes = latents[mode_coord_name].values

    comps = []
    for mi in modes:
        zi = latents.sel({mode_coord_name: mi})          # (time, seed)
        sig_i = zi.std("time", ddof=0)

        z_not = latents.sel({mode_coord_name: [m for m in modes if m != mi]})
        if z_not.sizes.get(mode_coord_name, 0) > 0:
            sig_not = z_not.std("time", ddof=0)
            neutral_mask = (np.abs(z_not) < (neutral_thresh * sig_not)).all(mode_coord_name)
        else:
            neutral_mask = xr.ones_like(zi, dtype=bool)

        pos_mask = (zi > (sigma_thresh * sig_i)) & neutral_mask
        neg_mask = (zi < (-sigma_thresh * sig_i)) & neutral_mask

        pos_mean = reference_sst.where(pos_mask).mean("time")
        neg_mean = reference_sst.where(neg_mask).mean("time")
        comp = (pos_mean - neg_mean).expand_dims({mode_coord_name: [mi]}) / 2
        comps.append(comp)

    comp_all = xr.concat(comps, dim=mode_coord_name)
    return comp_all.transpose(mode_coord_name, "lat", "lon", "seed")


def _regress_composite_on_alpha_per_seed(alpha: xr.DataArray, comp: xr.DataArray):
    """
    For each mode and seed: spatial regression comp_map ~ m * alpha_map + b
    Badge = median across seeds (m_med, b_med, r2_med)
    """
    alpha, comp = xr.align(alpha, comp, join="inner")
    modes = alpha["tendency_mode"].values
    seeds = alpha["seed"].values

    out = {}
    for mi in modes:
        slopes, intercepts, r2s = [], [], []
        for s in seeds:
            a_map = alpha.sel(tendency_mode=mi, seed=s)
            c_map = comp.sel(tendency_mode=mi, seed=s)
            m, b, r2 = linregress_with_r2(_stack_map(a_map), _stack_map(c_map))
            slopes.append(m); intercepts.append(b); r2s.append(r2)

        slopes = np.asarray(slopes, dtype=float)
        intercepts = np.asarray(intercepts, dtype=float)
        r2s = np.asarray(r2s, dtype=float)

        out[int(mi)] = dict(
            slopes=slopes,
            intercepts=intercepts,
            r2s=r2s,
            badge=(float(np.nanmedian(slopes)),
                   float(np.nanmedian(intercepts)),
                   float(np.nanmedian(r2s))),
        )
    return out


# ============================
# LOAD OR CALCULATE
# ============================
if (not need_to_calculate) and os.path.exists(cache_file):
    ds = xr.open_dataset(cache_file)

    alpha_mean = ds["alpha_mean"]
    alpha_sig_frac = ds["alpha_sig_frac"]
    comp_mean = ds["comp_mean"]
    comp_sig_frac = ds["comp_sig_frac"]

    # regression stats
    reg_slopes = ds["reg_slopes"]           # (tendency_mode, seed)
    reg_intercepts = ds["reg_intercepts"]   # (tendency_mode, seed)
    reg_r2s = ds["reg_r2s"]                 # (tendency_mode, seed)

    # reconstruct badge dict
    reg_stats = {}
    for mi in ds["tendency_mode"].values:
        slopes = reg_slopes.sel(tendency_mode=mi).values
        intercepts = reg_intercepts.sel(tendency_mode=mi).values
        r2s = reg_r2s.sel(tendency_mode=mi).values
        reg_stats[int(mi)] = dict(
            slopes=slopes,
            intercepts=intercepts,
            r2s=r2s,
            badge=(float(np.nanmedian(slopes)),
                   float(np.nanmedian(intercepts)),
                   float(np.nanmedian(r2s))),
        )

else:
    # ----------------------------
    # load raw data
    # ----------------------------
    alphas, betas, alpha_sigs, beta_sigs = kgae.open_alphabetas(
        "xval", experiment=experiment, n_ensemble=n_ensemble
    )

    alpha_sigs = kgae.bootstrap_ci(alpha_sigs, ci_interval)


    reference_sst = kgae.open_references("xval", experiment=experiment).transpose("time", "lat", "lon")
    latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)

    alphas = _ensure_dims_alpha(alphas)
    alpha_sigs = _ensure_dims_alpha(alpha_sigs)

    alphas_sel = alphas.sel(tendency_mode=tendency_modes)
    alpha_sigs_sel = alpha_sigs.sel(tendency_mode=tendency_modes)

    # Compute composite on the same latent subset used for plotting.
    latents_sel = latents.sel(mode=tendency_modes)

    # linear composite per seed (for plotting mean + regression)
    comp_lin = _composite_linear_holding_others_neutral(
        reference_sst=reference_sst,
        latents=latents_sel,
        mode_coord_name="tendency_mode",
        sigma_thresh=sigma_thresh_comp,
        neutral_thresh=neutral_thresh_comp,
    ).sel(tendency_mode=tendency_modes)

    # regression: composite ON alpha (per seed); badge = median seed
    reg_stats = _regress_composite_on_alpha_per_seed(alphas_sel, comp_lin)

    # plot fields
    alpha_mean = alphas_sel.mean("seed")
    alpha_sig_frac = alpha_sigs_sel.mean("seed")
    comp_mean = comp_lin.mean("seed")

    # ----------------------------
    # Composite significance fraction across seeds using per-seed bootstrap at 95%.
    # ----------------------------
    if "mode" in latents_sel.dims and "tendency_mode" not in latents_sel.dims:
        latents_sel = latents_sel.rename({"mode": "tendency_mode"})
    if "seed" not in latents_sel.dims:
        raise ValueError(f"Expected latents to have 'seed' dim; got {latents_sel.dims}")

    reference_sst, latents_sel = xr.align(reference_sst, latents_sel, join="inner")
    seeds = latents_sel["seed"].values

    sig_masks = []
    for mi in tendency_modes:
        sig_by_seed = []
        for si, s in enumerate(seeds):
            z_s = latents_sel.sel(seed=s).transpose("time", "tendency_mode")
            sigmask = _bootstrap_sigmask_linear_comp_one_seed(
                X=reference_sst,
                z_seed=z_s,
                mode_val=mi,
                mode_dim="tendency_mode",
                sigma_thresh=sigma_thresh_comp,
                neutral_thresh=neutral_thresh_comp,
                n_boot=n_boot_comp,
                ci_level=ci_level_comp,
                rng_seed=10000 + int(mi) * 1000 + int(si),
                block_len=block_len_months,   # <--- add this
            )
            sig_by_seed.append(sigmask.expand_dims(seed=[s]))

        sig_by_seed = xr.concat(sig_by_seed, dim="seed")  # (seed, lat, lon)
        sig_masks.append(sig_by_seed.expand_dims(tendency_mode=[mi]))

    sig_masks = xr.concat(sig_masks, dim="tendency_mode")  # (mode, seed, lat, lon)
    comp_sig_frac = sig_masks.mean("seed")                 # (mode, lat, lon)

    # ----------------------------
    # SAVE CACHE
    # ----------------------------
    tm = alpha_mean["tendency_mode"]
    seed_coord = alphas_sel["seed"]

    reg_slopes = xr.DataArray(
        np.stack([reg_stats[int(mi)]["slopes"] for mi in tm.values], axis=0),
        coords={"tendency_mode": tm, "seed": seed_coord},
        dims=("tendency_mode", "seed"),
        name="reg_slopes",
    )
    reg_intercepts = xr.DataArray(
        np.stack([reg_stats[int(mi)]["intercepts"] for mi in tm.values], axis=0),
        coords={"tendency_mode": tm, "seed": seed_coord},
        dims=("tendency_mode", "seed"),
        name="reg_intercepts",
    )
    reg_r2s = xr.DataArray(
        np.stack([reg_stats[int(mi)]["r2s"] for mi in tm.values], axis=0),
        coords={"tendency_mode": tm, "seed": seed_coord},
        dims=("tendency_mode", "seed"),
        name="reg_r2s",
    )

    ds = xr.Dataset(
        data_vars=dict(
            alpha_mean=alpha_mean.astype(np.float32),
            alpha_sig_frac=alpha_sig_frac.astype(np.float32),
            comp_mean=comp_mean.astype(np.float32),
            comp_sig_frac=comp_sig_frac.astype(np.float32),
            reg_slopes=reg_slopes.astype(np.float32),
            reg_intercepts=reg_intercepts.astype(np.float32),
            reg_r2s=reg_r2s.astype(np.float32),
        ),
        attrs=dict(
            experiment=experiment,
            n_ensemble=n_ensemble,
            tendency_modes=str(tendency_modes),
            sigma_thresh_comp=sigma_thresh_comp,
            neutral_thresh_comp=neutral_thresh_comp,
            n_boot_comp=n_boot_comp,
            ci_level_comp=ci_level_comp,
        ),
    )
    ds.to_netcdf(cache_file)
    print(f"[cache] wrote {cache_file}")

# ============================
# PLOTTING (FAST ITERATION)
# ============================
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree(central_longitude=180)

alpha_vmax = float(np.nanmax(np.abs(alpha_mean.values)))
alpha_levels = np.linspace(-alpha_vmax, alpha_vmax, nlev)
alpha_line_levels = np.linspace(-alpha_vmax, alpha_vmax, nline)
alpha_line_levels = alpha_line_levels[alpha_line_levels != 0]

comp_vmax = float(np.nanmax(np.abs(comp_mean.values)))
comp_levels = np.linspace(-comp_vmax, comp_vmax, nlev)
comp_line_levels = np.linspace(-comp_vmax, comp_vmax, nline)
comp_line_levels = comp_line_levels[comp_line_levels != 0]

n_cols = len(tendency_modes)
n_rows = 2

fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(4.3 * n_cols, 2.6 * n_rows),
    subplot_kw={"projection": proj},
    constrained_layout=False
)
axs = np.atleast_2d(axs)

panel_letters = [f"{chr(ord('a') + i)})" for i in range(n_rows * n_cols)]
pidx = 0

mappable_alpha = None
mappable_comp = None

# ---- Row 0: alphas ----
for ci, mi in enumerate(tendency_modes):
    ax = axs[0, ci]
    ax.coastlines(linewidth=0.6)

    ax.text(
        0.0, 1.07, panel_letters[pidx],
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
    )
    pidx += 1

    field = alpha_mean.sel(tendency_mode=mi)
    frac_a = alpha_sig_frac.sel(tendency_mode=mi)

    # fill where frac >= sig_thresh_alpha
    field_filled = field.where(frac_a >= sig_thresh_alpha)

    mappable_alpha = field_filled.plot.contourf(
        ax=ax, transform=pc, levels=alpha_levels,
        add_colorbar=False, extend="both"
    )

    # hatch where frac >= sig_thresh_alpha_hatch
    hatch_mask = xr.where(frac_a >= sig_thresh_alpha_hatch, 1.0, np.nan)
    ax.contourf(
        hatch_mask["lon"], hatch_mask["lat"], hatch_mask.values,
        levels=[0.5, 1.5],
        hatches=["///"],
        colors="none",
        transform=pc,
        zorder=10
    )

    # contours
    pos_levels = [lv for lv in alpha_line_levels if lv > 0]
    neg_levels = [lv for lv in alpha_line_levels if lv < 0]
    if neg_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=neg_levels, colors="k", linestyles="dashed",
                   linewidths=0.7, transform=pc)
    if pos_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=pos_levels, colors="k", linestyles="solid",
                   linewidths=0.7, transform=pc)

    ax.set_title(tendency_mode_labels.get(int(mi), f"mode {mi}"), fontsize=11)




axs[0, 0].annotate(
    alpha_row_label,
    xy=(-0.07, 0.5), xycoords="axes fraction",
    rotation=90, ha="center", va="center",
    fontsize=10
)

# ---- Row 1: linear/symmetric composites ----
for ci, mi in enumerate(tendency_modes):
    ax = axs[1, ci]
    ax.coastlines(linewidth=0.6)

    ax.text(
        0.0, 1.07, panel_letters[pidx],
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
    )
    pidx += 1

    field = comp_mean.sel(tendency_mode=mi)
    frac = comp_sig_frac.sel(tendency_mode=mi)

    field_filled = field.where(frac >= comp_sig_thresh)

    mappable_comp = field_filled.plot.contourf(
        ax=ax, transform=pc, levels=comp_levels,
        add_colorbar=False, extend="both"
    )

    hatch_mask = xr.where(frac >= comp_sig_thresh_hatch, 1.0, np.nan)
    ax.contourf(
        hatch_mask["lon"], hatch_mask["lat"], hatch_mask.values,
        levels=[0.5, 1.5],
        hatches=["///"],
        colors="none",
        transform=pc,
        zorder=10
    )

    pos_levels = [lv for lv in comp_line_levels if lv > 0]
    neg_levels = [lv for lv in comp_line_levels if lv < 0]
    if neg_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=neg_levels, colors="k", linestyles="dashed",
                   linewidths=0.7, transform=pc)
    if pos_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=pos_levels, colors="k", linestyles="solid",
                   linewidths=0.7, transform=pc)

    ax.set_title("")

    # Regression badge goes on the composite panel.
    m_badge, b_badge, r2_badge = reg_stats[int(mi)]["badge"]
    ax.text(
        0.02, 0.02,
        f"y={m_badge:.2g}x+{b_badge:.2g} (R$^2$:{r2_badge:.2f})",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3)
    )

axs[1, 0].annotate(
    comp_row_label,
    xy=(-0.07, 0.5), xycoords="axes fraction",
    rotation=90, ha="center", va="center",
    fontsize=10
)

plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.06, right=0.88, top=0.93, bottom=0.10)

# alpha colorbar next to top row
if mappable_alpha is not None:
    row_boxes = [axs[0, j].get_position() for j in range(n_cols)]
    y0 = min(b.y0 for b in row_boxes)
    y1 = max(b.y1 for b in row_boxes)
    x1 = max(b.x1 for b in row_boxes)
    cax = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb = fig.colorbar(mappable_alpha, cax=cax, orientation="vertical")
    cb.set_label(r"sensitivity $(K$ $\sigma_i^{-1})$", fontsize=10)
    cb.formatter = FormatStrFormatter('%.3f')
    cb.update_ticks()

# composite colorbar next to bottom row
if mappable_comp is not None:
    row_boxes = [axs[1, j].get_position() for j in range(n_cols)]
    y0 = min(b.y0 for b in row_boxes)
    y1 = max(b.y1 for b in row_boxes)
    x1 = max(b.x1 for b in row_boxes)
    cax = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb = fig.colorbar(mappable_comp, cax=cax, orientation="vertical")
    cb.set_label(r"SSTA $(K)$", fontsize=10)
    cb.formatter = FormatStrFormatter('%.3f')
    cb.update_ticks()

#plt.savefig("alphas_and_composites_final.png", dpi=300)
plt.show()
