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
need_to_calculate = False
cache_file = "cache_betas_composites_regression.nc"

experiment = "kgvae_cv_ema-1940-2014"
n_ensemble = 36

# ----------------------------
# BETA significance thresholds (plotting only)
# ----------------------------
sig_thresh = 0.1 # percentage of ensemble members achieving significance to show
sig_thresh_hatch = 1.1 # percentage of ensemble members achieving significance to hatch
ci_interval = 0.9

# ----------------------------
# Composite masking like betas (seed-fraction of seedwise bootstrap significance)
# ----------------------------
comp_sig_thresh = 0.10
comp_sig_thresh_hatch = 0.50

n_boot_comp = 200
ci_level_comp = 0.9

sigma_thresh_comp = 1.0
neutral_thresh_comp = 1.0

# plotting
nline = 11
nlev = 21

# ONLY plot these betas / composites
tendency_modes = [5, 4]     # columns
predictor_modes = [5, 4]    # rows

tendency_mode_labels = {5: "Decadal Mode", 4: "Interannual Mode"}
beta_row_labels = {5: r"Decadal-State Dependence", 4: r"Interannual-State Dependence"}
composite_label = "Nonlinear Composite"

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
        raise ValueError(f"alpha dims missing {missing}; got dims={da.dims}")
    return da.transpose("tendency_mode", "lat", "lon", "seed")

import numpy as np

# ----------------------------
# 2-year moving block bootstrap
# ----------------------------
BLOCK_LEN_MONTHS = 24

def moving_block_bootstrap_indices(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    """
    Moving-block bootstrap indices for a length-n time series.
    Draw contiguous blocks of length block_len with replacement.
    """
    if block_len <= 1:
        return rng.integers(0, n, size=n)

    if n <= block_len:
        start = int(rng.integers(0, n))
        return (np.arange(n) + start) % n

    n_starts = n - block_len + 1
    n_blocks = int(np.ceil(n / block_len))

    starts = rng.integers(0, n_starts, size=n_blocks)
    idx = np.concatenate(
        [np.arange(s, s + block_len) for s in starts],
        axis=0
    )
    return idx[:n]


def _ensure_dims_beta(da: xr.DataArray) -> xr.DataArray:
    dims = list(da.dims)
    rename = {}
    if "mode" in dims and "tendency_mode" not in dims:
        rename["mode"] = "tendency_mode"
    if "pred_mode" in dims and "predictor_mode" not in dims:
        rename["pred_mode"] = "predictor_mode"
    da = da.rename(rename)
    needed = ["tendency_mode", "predictor_mode", "lat", "lon", "seed"]
    missing = [d for d in needed if d not in da.dims]
    if missing:
        raise ValueError(f"beta dims missing {missing}; got dims={da.dims}")
    return da.transpose("tendency_mode", "predictor_mode", "lat", "lon", "seed")


def _rename_mode_dim(latents: xr.DataArray, mode_coord_name="tendency_mode") -> xr.DataArray:
    if "mode" in latents.dims and mode_coord_name not in latents.dims:
        latents = latents.rename({"mode": mode_coord_name})
    return latents


def _stack_map(da_map: xr.DataArray) -> np.ndarray:
    return da_map.values.reshape(-1)


def _spatial_multi_regress(y_map: xr.DataArray, X_maps: list[xr.DataArray]):
    y = _stack_map(y_map)
    Xcols = [_stack_map(Xm) for Xm in X_maps]
    Xmat = np.vstack(Xcols).T

    good = np.isfinite(y) & np.all(np.isfinite(Xmat), axis=1)
    y = y[good]
    Xmat = Xmat[good, :]

    n = y.size
    if n < 50:
        return np.full(1 + len(X_maps), np.nan), np.nan, int(n)

    A = np.column_stack([np.ones(n), Xmat])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)

    yhat = A @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef, float(r2), int(n)


def _median_badge_from_seed_stats(coefs: np.ndarray, r2s: np.ndarray):
    return np.nanmedian(coefs, axis=0), float(np.nanmedian(r2s))

def _pure_even_sigmask_one_seed(
    X: xr.DataArray,              # (time, lat, lon)
    z_seed: xr.DataArray,         # (time, tendency_mode) for ONE seed
    mode_val,
    mode_dim="tendency_mode",
    sigma_thresh=1.0,
    neutral_thresh=1.0,
    n_boot=500,
    ci_level=0.95,
    rng_seed=0,
    block_len=24,                 # block length in months
    recompute_thresholds=True,    # recompute std/neutral thresholds within each replicate
    min_samples=5,                # guard against empty regimes
):
    """
    PURE-EVEN asymmetric estimator:
        comp = E[X | zi > +k*std_i, others neutral]
             + E[X | zi < -k*std_i, others neutral]
             - 2 * E[X | |zi| <  k*std_i, others neutral]

    Correct moving-block bootstrap:
      1) resample the full time series with contiguous blocks
      2) recompute thresholds/masks *within each bootstrap replicate* (optional but recommended)
      3) compute composite in that replicate
      4) CI from bootstrap distribution; significant if CI excludes 0

    Returns
    -------
    comp_point : xr.DataArray (lat, lon)
        Point estimate from the original (non-bootstrapped) series.
    sigmask : xr.DataArray (lat, lon) boolean
        True where bootstrap CI excludes 0.
    """
    rng = np.random.default_rng(rng_seed)
    X, z_seed = xr.align(X, z_seed, join="inner")

    # ---- point estimate (original series; keep as-is) ----
    zi = z_seed.sel({mode_dim: mode_val})  # (time,)
    std_i = float(zi.std("time", ddof=0).values)

    other_modes = [m for m in z_seed[mode_dim].values if m != mode_val]
    if len(other_modes) > 0:
        z_not = z_seed.sel({mode_dim: other_modes})      # (time, other_mode)
        std_not = z_not.std("time", ddof=0)
        others_neutral = (np.abs(z_not) < (neutral_thresh * std_not)).all(mode_dim)
    else:
        others_neutral = xr.ones_like(zi, dtype=bool)

    pos_mask = (zi > ( sigma_thresh * std_i)) & others_neutral
    neg_mask = (zi < (-sigma_thresh * std_i)) & others_neutral
    neu_mask = (np.abs(zi) < ( sigma_thresh * std_i)) & others_neutral

    pos_mean = X.where(pos_mask).mean("time")
    neg_mean = X.where(neg_mask).mean("time")
    neu_mean = X.where(neu_mask).mean("time")
    comp_point = pos_mean + neg_mean - 2.0 * neu_mean

    # ---- bootstrap (block resample time first, then recompute masks) ----
    tN = X.sizes["time"]
    lat = X["lat"]
    lon = X["lon"]

    X_np = X.values.astype(np.float32)
    modes = z_seed[mode_dim].values
    z_np = z_seed.transpose("time", mode_dim).values.astype(np.float32)  # (time, L)
    L = z_np.shape[1]

    try:
        i_mode = int(np.where(modes == mode_val)[0][0])
    except Exception as e:
        raise ValueError(f"mode_val={mode_val} not found in z_seed[{mode_dim}].") from e
    other_idx = np.array([k for k in range(L) if k != i_mode], dtype=int)

    # fixed thresholds if not recomputing per replicate
    if not recompute_thresholds:
        zi0 = z_np[:, i_mode]
        std_i_fixed = float(np.nanstd(zi0, ddof=0))
        if other_idx.size > 0:
            std_not_fixed = np.nanstd(z_np[:, other_idx], axis=0, ddof=0)
        else:
            std_not_fixed = None

    boot = np.full((n_boot, X.sizes["lat"], X.sizes["lon"]), np.nan, dtype=np.float32)

    for b in range(n_boot):
        idx = moving_block_bootstrap_indices(rng, tN, block_len=block_len)

        X_b = X_np[idx, :, :]      # (time, lat, lon)
        z_b = z_np[idx, :]         # (time, L)

        zi_b = z_b[:, i_mode]

        if recompute_thresholds:
            std_i_b = float(np.nanstd(zi_b, ddof=0))
            if other_idx.size > 0:
                z_not_b = z_b[:, other_idx]
                std_not_b = np.nanstd(z_not_b, axis=0, ddof=0)
                others_neutral_b = np.all(np.abs(z_not_b) < (neutral_thresh * std_not_b), axis=1)
            else:
                others_neutral_b = np.ones(zi_b.shape[0], dtype=bool)
        else:
            std_i_b = std_i_fixed
            if other_idx.size > 0:
                z_not_b = z_b[:, other_idx]
                others_neutral_b = np.all(np.abs(z_not_b) < (neutral_thresh * std_not_fixed), axis=1)
            else:
                others_neutral_b = np.ones(zi_b.shape[0], dtype=bool)

        pos_b = (zi_b > ( sigma_thresh * std_i_b)) & others_neutral_b
        neg_b = (zi_b < (-sigma_thresh * std_i_b)) & others_neutral_b
        neu_b = (np.abs(zi_b) < ( sigma_thresh * std_i_b)) & others_neutral_b

        # guard against empty regimes
        if (pos_b.sum() < min_samples) or (neg_b.sum() < min_samples) or (neu_b.sum() < min_samples):
            continue

        pos_m = np.nanmean(X_b[pos_b, :, :], axis=0)
        neg_m = np.nanmean(X_b[neg_b, :, :], axis=0)
        neu_m = np.nanmean(X_b[neu_b, :, :], axis=0)

        boot[b, :, :] = pos_m + neg_m - 2.0 * neu_m

    alpha = 1.0 - ci_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    lo = np.nanpercentile(boot, lo_q, axis=0)
    hi = np.nanpercentile(boot, hi_q, axis=0)

    sigmask = (lo > 0) | (hi < 0)
    sigmask = xr.DataArray(sigmask, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    return comp_point, sigmask



# ============================
# LOAD OR CALCULATE
# ============================
if (not need_to_calculate) and os.path.exists(cache_file):
    ds = xr.open_dataset(cache_file)

    beta_mean = ds["beta_mean"]
    beta_sig_frac = ds["beta_sig_frac"]

    comp_mean = ds["comp_mean"]
    comp_sig_frac = ds["comp_sig_frac"]

    # regression badge terms
    badge_coef = ds["badge_coef"]   # (tendency_mode, coef)
    badge_r2 = ds["badge_r2"]       # (tendency_mode,)

else:
    # ----------------------------
    # load data
    # ----------------------------
    alphas, betas, alpha_sigs, beta_sigs = kgae.open_alphabetas(
        "xval", experiment=experiment, n_ensemble=n_ensemble
    )
    alphas = _ensure_dims_alpha(alphas)
    betas = _ensure_dims_beta(betas)
    beta_sigs = _ensure_dims_beta(beta_sigs)
    beta_sigs = kgae.bootstrap_ci(beta_sigs, ci_interval)

    betas_sel = betas.sel(tendency_mode=tendency_modes, predictor_mode=predictor_modes)
    beta_sigs_sel = beta_sigs.sel(tendency_mode=tendency_modes, predictor_mode=predictor_modes)

    beta_mean = betas_sel.mean("seed")
    beta_sig_frac = beta_sigs_sel.mean("seed")

    # Optional predictors for regression.
    alphas_sel = alphas.sel(tendency_mode=tendency_modes)

    reference_sst = kgae.open_references("xval", experiment=experiment).transpose("time", "lat", "lon")
    latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)
    latents = _rename_mode_dim(latents, mode_coord_name="tendency_mode")

    if "seed" not in latents.dims:
        raise ValueError(f"Expected latents to have a 'seed' dim; got dims={latents.dims}")

    latents_sel = latents.sel(tendency_mode=tendency_modes).transpose("time", "tendency_mode", "seed")
    reference_sst, latents_sel = xr.align(reference_sst, latents_sel, join="inner")
    seeds = latents_sel["seed"].values

    # ----------------------------
    # Compute composite per seed + seedwise sigmask => comp_mean + comp_sig_frac.
    # ----------------------------
    comp_point_by_seed = []
    sigmask_by_seed = []

    for mi in tendency_modes:
        cmaps = []
        smaps = []
        for si, s in enumerate(seeds):
            z_s = latents_sel.sel(seed=s).transpose("time", "tendency_mode")
            c_point, s95 = _pure_even_sigmask_one_seed(
                X=reference_sst,
                z_seed=z_s,
                mode_val=mi,
                mode_dim="tendency_mode",
                sigma_thresh=sigma_thresh_comp,
                neutral_thresh=neutral_thresh_comp,
                n_boot=n_boot_comp,
                ci_level=ci_level_comp,
                rng_seed=50000 + int(mi) * 1000 + int(si),
            )
            cmaps.append(c_point.expand_dims(seed=[s]))
            smaps.append(s95.expand_dims(seed=[s]))

        cmaps = xr.concat(cmaps, dim="seed")  # (seed, lat, lon)
        smaps = xr.concat(smaps, dim="seed")  # (seed, lat, lon)
        comp_point_by_seed.append(cmaps.expand_dims(tendency_mode=[mi]))
        sigmask_by_seed.append(smaps.expand_dims(tendency_mode=[mi]))

    comp_point_by_seed = xr.concat(comp_point_by_seed, dim="tendency_mode")  # (mode, seed, lat, lon)
    sigmask_by_seed = xr.concat(sigmask_by_seed, dim="tendency_mode")        # (mode, seed, lat, lon)

    comp_mean = comp_point_by_seed.mean("seed")     # (mode, lat, lon)
    comp_sig_frac = sigmask_by_seed.mean("seed")    # (mode, lat, lon)

    # ----------------------------
    # regression badge:
    # regress each seed's composite y_map on ALL FOUR beta panels (a–d) + intercept
    # a = beta(tendency=5, predictor=5)
    # b = beta(tendency=4, predictor=5)
    # c = beta(tendency=5, predictor=4)
    # d = beta(tendency=4, predictor=4)
    # ----------------------------
    badge_coef_list = []
    badge_r2_list = []

    for mi_target in tendency_modes:
        coefs = np.full((len(seeds), 5), np.nan, dtype=float)  # [b0, wa, wb, wc, wd]
        r2s = np.full((len(seeds),), np.nan, dtype=float)

        for si, s in enumerate(seeds):
            # target map: this seed’s point composite for mode mi_target
            y_map = comp_point_by_seed.sel(tendency_mode=mi_target, seed=s)

            # predictors: the 2x2 beta grid for this seed (a–d)
            Xa = betas_sel.sel(seed=s, tendency_mode=5, predictor_mode=5)  # a
            Xb = betas_sel.sel(seed=s, tendency_mode=4, predictor_mode=5)  # b
            Xc = betas_sel.sel(seed=s, tendency_mode=5, predictor_mode=4)  # c
            Xd = betas_sel.sel(seed=s, tendency_mode=4, predictor_mode=4)  # d

            coef, r2, _ = _spatial_multi_regress(y_map, [Xa, Xb, Xc, Xd])
            coefs[si, :] = coef
            r2s[si] = r2

        coef_med, r2_med = _median_badge_from_seed_stats(coefs, r2s)
        badge_coef_list.append(coef_med)
        badge_r2_list.append(r2_med)

    badge_coef = xr.DataArray(
        np.vstack(badge_coef_list),
        coords={"tendency_mode": tendency_modes, "coef": ["b0", "wa", "wb", "wc", "wd"]},
        dims=("tendency_mode", "coef"),
        name="badge_coef",
    )
    badge_r2 = xr.DataArray(
        np.asarray(badge_r2_list),
        coords={"tendency_mode": tendency_modes},
        dims=("tendency_mode",),
        name="badge_r2",
    )

    # ----------------------------
    # SAVE CACHE
    # ----------------------------
    ds = xr.Dataset(
        data_vars=dict(
            beta_mean=beta_mean.astype(np.float32),
            beta_sig_frac=beta_sig_frac.astype(np.float32),
            comp_mean=comp_mean.astype(np.float32),
            comp_sig_frac=comp_sig_frac.astype(np.float32),
            badge_coef=badge_coef.astype(np.float32),
            badge_r2=badge_r2.astype(np.float32),
        ),
        attrs=dict(
            experiment=experiment,
            n_ensemble=n_ensemble,
            tendency_modes=str(tendency_modes),
            predictor_modes=str(predictor_modes),
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

beta_vmax = float(np.nanmax(np.abs(beta_mean.values)))
beta_levels = np.linspace(-beta_vmax, beta_vmax, nlev)
beta_line_levels = np.linspace(-beta_vmax, beta_vmax, nline)
beta_line_levels = beta_line_levels[beta_line_levels != 0]

comp_vmax = float(np.nanmax(np.abs(comp_mean.values)))
comp_levels = np.linspace(-comp_vmax, comp_vmax, nlev)
comp_line_levels = np.linspace(-comp_vmax, comp_vmax, nline)
comp_line_levels = comp_line_levels[comp_line_levels != 0]

n_cols = len(tendency_modes)
n_beta_rows = len(predictor_modes)
n_rows = n_beta_rows + 1
panel_letters = [f"{chr(ord('a') + i)})" for i in range(n_rows * n_cols)]

fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(4.3 * n_cols, 2.6 * n_rows),
    subplot_kw={"projection": proj},
    constrained_layout=False
)
axs = np.atleast_2d(axs)

mappable_beta = None
mappable_comp = None
pidx = 0

# ---- betas ----
for ri, pj in enumerate(predictor_modes):
    for ci, mi in enumerate(tendency_modes):
        ax = axs[ri, ci]
        ax.coastlines(linewidth=0.6)

        ax.text(
            0.0, 1.07, panel_letters[pidx],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
        )
        pidx += 1

        fieldb = beta_mean.sel(tendency_mode=mi, predictor_mode=pj)
        frac = beta_sig_frac.sel(tendency_mode=mi, predictor_mode=pj)

        fieldb_filled = fieldb.where(frac >= sig_thresh)

        m = fieldb_filled.plot.contourf(
            ax=ax, transform=pc, levels=beta_levels,
            add_colorbar=False, extend="both"
        )
        mappable_beta = m

        hatch_mask = xr.where(frac >= sig_thresh_hatch, 1.0, np.nan)
        ax.contourf(
            hatch_mask["lon"], hatch_mask["lat"], hatch_mask.values,
            levels=[0.5, 1.5],
            hatches=["///"],
            colors="none",
            transform=pc,
            zorder=10
        )

        pos_levels = [lv for lv in beta_line_levels if lv > 0]
        neg_levels = [lv for lv in beta_line_levels if lv < 0]
        if neg_levels:
            ax.contour(fieldb["lon"], fieldb["lat"], fieldb.values,
                       levels=neg_levels, colors="k", linestyles="dashed",
                       linewidths=0.7, transform=pc)
        if pos_levels:
            ax.contour(fieldb["lon"], fieldb["lat"], fieldb.values,
                       levels=pos_levels, colors="k", linestyles="solid",
                       linewidths=0.7, transform=pc)

        if ri == 0:
            ax.set_title(tendency_mode_labels.get(int(mi), f"mode {mi}"), fontsize=11)
        else:
            ax.set_title("")

        if ci == 0:
            ax.annotate(
                beta_row_labels.get(int(pj), f"predictor {pj}"),
                xy=(-0.06, 0.5), xycoords="axes fraction",
                rotation=90, ha="center", va="center",
                fontsize=10
            )

# ---- composite row ----
comp_row = n_rows - 1
for ci, mi in enumerate(tendency_modes):
    ax = axs[comp_row, ci]
    ax.coastlines(linewidth=0.6)

    ax.text(
        0.0, 1.07, panel_letters[pidx],
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
    )
    pidx += 1

    field = comp_mean.sel(tendency_mode=mi)
    frac = comp_sig_frac.sel(tendency_mode=mi)

    field_filled = field.where(frac >= comp_sig_thresh)

    m = field_filled.plot.contourf(
        ax=ax, transform=pc, levels=comp_levels,
        add_colorbar=False, extend="both"
    )
    mappable_comp = m

    hatch_mask = xr.where(frac >= comp_sig_thresh_hatch, 1.0, np.nan)
    ax.contourf(
        hatch_mask["lon"], hatch_mask["lat"], hatch_mask.values,
        levels=[0.5, 1.5],
        hatches=["///"],
        colors="none",
        transform=pc,
        zorder=10
    )

    # contours
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

    # badge
    coef = badge_coef.sel(tendency_mode=mi).values
    r2 = float(badge_r2.sel(tendency_mode=mi).values)

    b0, wa, wb, wc, wd = coef
    ax.text(
        0.02, 0.02,
        f"y={wa:.2g}a + {wb:.2g}b + {wc:.2g}c + {wd:.2g}d + {b0:.2g} (R$^2$ {r2:.2f})",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3)
    )

    if ci == 0:
        ax.annotate(
            composite_label,
            xy=(-0.06, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=10
        )

plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.06, right=0.88, top=0.95, bottom=0.06)

# colorbars
if mappable_beta is not None:
    beta_boxes = [axs[r, c].get_position() for r in range(n_beta_rows) for c in range(n_cols)]
    y0 = min(b.y0 for b in beta_boxes)
    y1 = max(b.y1 for b in beta_boxes)
    x1 = max(b.x1 for b in beta_boxes)
    cax_beta = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb = fig.colorbar(mappable_beta, cax=cax_beta, orientation="vertical")
    cb.set_label(r"$\beta$ $(K$ $\sigma_i^{-1}$ $\sigma_j^{-1})$", fontsize=10)
    cb.formatter = FormatStrFormatter('%.3f')
    cb.update_ticks()

if mappable_comp is not None:
    comp_boxes = [axs[comp_row, c].get_position() for c in range(n_cols)]
    y0 = min(b.y0 for b in comp_boxes)
    y1 = max(b.y1 for b in comp_boxes)
    x1 = max(b.x1 for b in comp_boxes)
    cax_comp = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb = fig.colorbar(mappable_comp, cax=cax_comp, orientation="vertical")
    cb.set_label(r"composite mean $(K)$", fontsize=10)
    cb.formatter = FormatStrFormatter('%.3f')
    cb.update_ticks()

plt.savefig("betas_and_composites_final.png", dpi=300)
plt.show()
