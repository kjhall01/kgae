# combined_pure_even_and_cross_composites_sigma.py
import kgae
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

# ----------------------------
# user settings
# ----------------------------
experiment = "kgvae_cv1940-2014"
n_ensemble = 100  # used for open_* calls

# --- SIGMA thresholds (set these) ---
# Extremes for i (self / diagonal)
sigma_thresh_self = 1.0

# Extremes for i and j (cross / off-diagonal)
sigma_i = 1.0
sigma_j = 1.0

# Neutrality thresholds (hold OTHER modes within +/- neutral_thresh * std)
neutral_thresh_self = 1.0
neutral_thresh_cross = 1.0

# bootstrap / CI / masking
n_boot_self = 200
ci_level_self = 0.50
mask_if_ci_excludes_zero_self = True

n_boot_cross = 200
ci_level_cross = 0.50
mask_if_ci_excludes_zero_cross = True

# plotting
nline = 11
nlev = 21

# modes
modes = [5, 4, 3, 2, 1]
mode_labels = {5: "Decadal", 4: "Interannual", 3: "Quasibiennial", 2: "Biennial", 1: "Other"}

# which data rows
X_row_labels = {"obs": "Observed", "rec": "Reconstruction", "res": "Residual"}


# ----------------------------
# helpers
# ----------------------------
def _rename_mode_dim(latents: xr.DataArray, mode_coord_name="tendency_mode") -> xr.DataArray:
    if "mode" in latents.dims and mode_coord_name not in latents.dims:
        latents = latents.rename({"mode": mode_coord_name})
    return latents


def _infer_ens_dim(da: xr.DataArray):
    for cand in ["seed", "ensemble", "member"]:
        if cand in da.dims:
            return cand
    return None


def _strip_nondim_coords(da: xr.DataArray) -> xr.DataArray:
    # drops scalar coords like 'time', 'tendency_mode', 'predictor_mode' that can break concat
    return da.reset_coords(drop=True)


def _pure_even_bootstrap_ensemble_mean_sigma(
    X: xr.DataArray,                 # (time, lat, lon)
    z: xr.DataArray,                 # (time, tendency_mode) ENSEMBLE MEAN
    mode_val,
    mode_dim="tendency_mode",
    sigma_thresh=1.0,
    neutral_thresh=1.0,
    n_boot=100,
    ci_level=0.90,
    rng_seed=0,
):
    """
    Sigma-based pure-even composite:
      (E[X|zi>t, neutral] + E[X|zi<-t, neutral]) - 2*E[X|neutral]
    where t = sigma_thresh * std(zi).

    "neutral" means: for all other modes k != i, |z_k| < neutral_thresh * std(z_k).
    """
    rng = np.random.default_rng(rng_seed)

    # Align time
    X, z = xr.align(X, z, join="inner")

    zi = z.sel({mode_dim: mode_val})
    std_i = float(zi.std("time", ddof=0).values)

    other_modes = [m for m in z[mode_dim].values if m != mode_val]
    if len(other_modes) > 0:
        z_not = z.sel({mode_dim: other_modes})
        std_not = z_not.std("time", ddof=0)
        neutral_mask = (np.abs(z_not) < (neutral_thresh * std_not)).all(mode_dim)
    else:
        neutral_mask = xr.ones_like(zi, dtype=bool)

    pos_mask = (zi > (sigma_thresh * std_i)) & neutral_mask
    neg_mask = (zi < (-sigma_thresh * std_i)) & neutral_mask

    # point estimate
    pos_mean = X.where(pos_mask).mean("time")
    neg_mean = X.where(neg_mask).mean("time")
    base_mean = X.where(neutral_mask).mean("time")
    comp_mean = (pos_mean + neg_mean) - 2.0 * base_mean

    # bootstrap over time indices
    tN = X.sizes["time"]
    pos_np = pos_mask.values
    neg_np = neg_mask.values
    neu_np = neutral_mask.values
    X_np = X.values  # (time, lat, lon)

    boot = np.full((n_boot, X.sizes["lat"], X.sizes["lon"]), np.nan, dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, tN, size=tN)

        X_b = X_np[idx, :, :]
        pos_b = pos_np[idx]
        neg_b = neg_np[idx]
        neu_b = neu_np[idx]

        pos_m = np.nanmean(X_b[pos_b, :, :], axis=0) if pos_b.any() else np.full((X.sizes["lat"], X.sizes["lon"]), np.nan, np.float32)
        neg_m = np.nanmean(X_b[neg_b, :, :], axis=0) if neg_b.any() else np.full((X.sizes["lat"], X.sizes["lon"]), np.nan, np.float32)
        base_m = np.nanmean(X_b[neu_b, :, :], axis=0) if neu_b.any() else np.full((X.sizes["lat"], X.sizes["lon"]), np.nan, np.float32)

        boot[b, :, :] = (pos_m + neg_m) - 2.0 * base_m

    # CI + significance
    alpha = 1.0 - ci_level
    lo_qp = 100 * (alpha / 2)
    hi_qp = 100 * (1 - alpha / 2)
    comp_lo = np.nanpercentile(boot, lo_qp, axis=0)
    comp_hi = np.nanpercentile(boot, hi_qp, axis=0)
    sig_mask = (comp_lo > 0) | (comp_hi < 0)

    sig_mask = xr.DataArray(sig_mask, coords={"lat": X["lat"], "lon": X["lon"]}, dims=("lat", "lon"))
    return comp_mean, sig_mask


def _pure_cross_interaction_bootstrap_ensemble_mean_sigma(
    X: xr.DataArray,     # (time, lat, lon)
    z: xr.DataArray,     # (time, tendency_mode)
    mode_i,
    mode_j,
    mode_dim="tendency_mode",
    sigma_i=1.0,
    sigma_j=1.0,
    neutral_thresh=1.0,
    n_boot=100,
    ci_level=0.90,
    rng_seed=0,
):
    """
    Sigma-based "pure cross" estimator:

      S(i | j+) = E(i+,j+) + E(i-,j+) - 2 E(n_i,j+)
      S(i | j-) = E(i+,j-) + E(i-,j-) - 2 E(n_i,j-)
      I_pure(i,j) = S(i|j+) - S(i|j-)

    Definitions (sigma-based):
      i+ : zi > +sigma_i * std(zi)
      i- : zi < -sigma_i * std(zi)
      n_i : |zi| < neutral_thresh * std(zi)
      j+ : zj > +sigma_j * std(zj)
      j- : zj < -sigma_j * std(zj)

    All masks also enforce neutrality on other modes k != i,j:
      |z_k| < neutral_thresh * std(z_k)
    """
    rng = np.random.default_rng(rng_seed)

    # Align time
    X, z = xr.align(X, z, join="inner")

    zi = z.sel({mode_dim: mode_i})
    zj = z.sel({mode_dim: mode_j})

    std_i = float(zi.std("time", ddof=0).values)
    std_j = float(zj.std("time", ddof=0).values)

    # neutrality on all other modes (exclude i and j)
    other_modes = [m for m in z[mode_dim].values if (m != mode_i and m != mode_j)]
    if len(other_modes) > 0:
        z_not = z.sel({mode_dim: other_modes})
        std_not = z_not.std("time", ddof=0)
        neutral_others = (np.abs(z_not) < (neutral_thresh * std_not)).all(mode_dim)
    else:
        neutral_others = xr.ones_like(zi, dtype=bool)

    # i states
    i_pos = zi > +sigma_i * std_i
    i_neg = zi < -sigma_i * std_i
    i_neu = np.abs(zi) < (neutral_thresh * std_i)

    # j states
    j_pos = zj > +sigma_j * std_j
    j_neg = zj < -sigma_j * std_j

    # masks (all include neutral_others)
    m_ip_jp = i_pos & j_pos & neutral_others
    m_in_jp = i_neg & j_pos & neutral_others
    m_ni_jp = i_neu & j_pos & neutral_others

    m_ip_jn = i_pos & j_neg & neutral_others
    m_in_jn = i_neg & j_neg & neutral_others
    m_ni_jn = i_neu & j_neg & neutral_others

    # point estimate
    E_ip_jp = X.where(m_ip_jp).mean("time")
    E_in_jp = X.where(m_in_jp).mean("time")
    E_ni_jp = X.where(m_ni_jp).mean("time")

    E_ip_jn = X.where(m_ip_jn).mean("time")
    E_in_jn = X.where(m_in_jn).mean("time")
    E_ni_jn = X.where(m_ni_jn).mean("time")

    S_jp = (E_ip_jp + E_in_jp) - 2.0 * E_ni_jp
    S_jn = (E_ip_jn + E_in_jn) - 2.0 * E_ni_jn
    I_mean = S_jp - S_jn

    # bootstrap over time indices
    tN = X.sizes["time"]
    X_np = X.values

    ip_jp = m_ip_jp.values
    in_jp = m_in_jp.values
    ni_jp = m_ni_jp.values
    ip_jn = m_ip_jn.values
    in_jn = m_in_jn.values
    ni_jn = m_ni_jn.values

    boot = np.full((n_boot, X.sizes["lat"], X.sizes["lon"]), np.nan, dtype=np.float32)

    for b in range(n_boot):
        idx = rng.integers(0, tN, size=tN)
        X_b = X_np[idx, :, :]

        ip_jp_b = ip_jp[idx]
        in_jp_b = in_jp[idx]
        ni_jp_b = ni_jp[idx]

        ip_jn_b = ip_jn[idx]
        in_jn_b = in_jn[idx]
        ni_jn_b = ni_jn[idx]

        def _mmean(mask):
            return np.nanmean(X_b[mask, :, :], axis=0) if mask.any() else np.full((X.sizes["lat"], X.sizes["lon"]), np.nan, np.float32)

        E_ip_jp_b = _mmean(ip_jp_b)
        E_in_jp_b = _mmean(in_jp_b)
        E_ni_jp_b = _mmean(ni_jp_b)

        E_ip_jn_b = _mmean(ip_jn_b)
        E_in_jn_b = _mmean(in_jn_b)
        E_ni_jn_b = _mmean(ni_jn_b)

        S_jp_b = (E_ip_jp_b + E_in_jp_b) - 2.0 * E_ni_jp_b
        S_jn_b = (E_ip_jn_b + E_in_jn_b) - 2.0 * E_ni_jn_b

        boot[b, :, :] = S_jp_b - S_jn_b

    # CI + significance
    alpha = 1.0 - ci_level
    lo_qp = 100 * (alpha / 2)
    hi_qp = 100 * (1 - alpha / 2)
    lo = np.nanpercentile(boot, lo_qp, axis=0)
    hi = np.nanpercentile(boot, hi_qp, axis=0)
    sig_mask = (lo > 0) | (hi < 0)

    sig_mask = xr.DataArray(sig_mask, coords={"lat": X["lat"], "lon": X["lon"]}, dims=("lat", "lon"))
    return I_mean, sig_mask


# ----------------------------
# load data (obs + latents + recon)
# ----------------------------
reference_sst = kgae.open_references("xval", experiment=experiment).transpose("time", "lat", "lon")

latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)
latents = _rename_mode_dim(latents, mode_coord_name="tendency_mode")
latents_sel = latents.sel(tendency_mode=modes)

ens_dim_lat = _infer_ens_dim(latents_sel)
if ens_dim_lat is None:
    raise ValueError(f"Could not infer ensemble dim for latents; dims={latents_sel.dims}")
zbar = latents_sel.mean(ens_dim_lat).transpose("time", "tendency_mode")

reconstructions = kgae.open_reconstructions("xval", experiment=experiment, n_ensemble=n_ensemble)
ens_dim_rec = _infer_ens_dim(reconstructions)
if ens_dim_rec is None:
    raise ValueError(f"Could not infer ensemble dim for reconstructions; dims={reconstructions.dims}")
Xhat = reconstructions.mean(ens_dim_rec).transpose("time", "lat", "lon")
del reconstructions

# align time
reference_sst, Xhat, zbar = xr.align(reference_sst, Xhat, zbar, join="inner")
R = reference_sst - Xhat

X_by_row = {"obs": reference_sst, "rec": Xhat, "res": R}

# ----------------------------
# compute diagonal (pure-even) and off-diagonal (pure-cross) maps
# ----------------------------
diag_maps = {}
diag_sigs = {}
cross_maps = {}
cross_sigs = {}

for rk, X in X_by_row.items():
    # diagonal
    comps = []
    sigs = []
    for mi in modes:
        comp_mean, sig_mask = _pure_even_bootstrap_ensemble_mean_sigma(
            X=X,
            z=zbar,
            mode_val=mi,
            mode_dim="tendency_mode",
            sigma_thresh=sigma_thresh_self,
            neutral_thresh=neutral_thresh_self,
            n_boot=n_boot_self,
            ci_level=ci_level_self,
            rng_seed=100 + (0 if rk == "obs" else 1 if rk == "rec" else 2) + int(mi),
        )
        comps.append(comp_mean.expand_dims({"tendency_mode": [mi]}))
        sigs.append(sig_mask.expand_dims({"tendency_mode": [mi]}))

    diag_maps[rk] = xr.concat(comps, dim="tendency_mode").transpose("tendency_mode", "lat", "lon")
    diag_sigs[rk] = xr.concat(sigs, dim="tendency_mode").transpose("tendency_mode", "lat", "lon")

    # cross
    maps_ij = []
    sigs_ij = []
    for i in modes:
        row_i = []
        row_sig = []
        for j in modes:
            if i == j:
                nan_map = xr.full_like(X.isel(time=0, drop=True), np.nan)
                nan_sig = xr.full_like(X.isel(time=0, drop=True), False).astype(bool)
                row_i.append(nan_map.expand_dims({"predictor_mode": [j]}))
                row_sig.append(nan_sig.expand_dims({"predictor_mode": [j]}))
                continue

            I_mean, sig_mask = _pure_cross_interaction_bootstrap_ensemble_mean_sigma(
                X=X,
                z=zbar,
                mode_i=i,
                mode_j=j,
                mode_dim="tendency_mode",
                sigma_i=sigma_i,
                sigma_j=sigma_j,
                neutral_thresh=neutral_thresh_cross,
                n_boot=n_boot_cross,
                ci_level=ci_level_cross,
                rng_seed=2000 + (0 if rk == "obs" else 1 if rk == "rec" else 2) * 100 + int(i) * 10 + int(j),
            )
            row_i.append(I_mean.expand_dims({"predictor_mode": [j]}))
            row_sig.append(sig_mask.expand_dims({"predictor_mode": [j]}))

        row_i = xr.concat(row_i, dim="predictor_mode")
        row_sig = xr.concat(row_sig, dim="predictor_mode")
        maps_ij.append(row_i.expand_dims({"tendency_mode": [i]}))
        sigs_ij.append(row_sig.expand_dims({"tendency_mode": [i]}))

    cross_maps[rk] = xr.concat(maps_ij, dim="tendency_mode").transpose("tendency_mode", "predictor_mode", "lat", "lon")
    cross_sigs[rk] = xr.concat(sigs_ij, dim="tendency_mode").transpose("tendency_mode", "predictor_mode", "lat", "lon")


# ----------------------------
# plotting: diagonal and off-diagonal get separate colorbars
# ----------------------------
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree(central_longitude=180)

def plot_combined_grid(row_key: str):
    n = len(modes)

    maps = []
    sigs = []
    for j in modes:
        row_maps = []
        row_sigs = []
        for i in modes:
            if i == j:
                field = diag_maps[row_key].sel(tendency_mode=i)
                sigm  = diag_sigs[row_key].sel(tendency_mode=i)
            else:
                field = cross_maps[row_key].sel(tendency_mode=i, predictor_mode=j)
                sigm  = cross_sigs[row_key].sel(tendency_mode=i, predictor_mode=j)

            field = _strip_nondim_coords(field)
            sigm  = _strip_nondim_coords(sigm)

            row_maps.append(field.expand_dims({"tendency_mode": [i]}))
            row_sigs.append(sigm.expand_dims({"tendency_mode": [i]}))

        row_maps = xr.concat(row_maps, dim="tendency_mode")
        row_sigs = xr.concat(row_sigs, dim="tendency_mode")

        maps.append(row_maps.expand_dims({"predictor_mode": [j]}))
        sigs.append(row_sigs.expand_dims({"predictor_mode": [j]}))

    combined = xr.concat(maps, dim="predictor_mode").transpose("predictor_mode", "tendency_mode", "lat", "lon")
    combined_sig = xr.concat(sigs, dim="predictor_mode").transpose("predictor_mode", "tendency_mode", "lat", "lon")

    diag_vals, off_vals = [], []
    for j in modes:
        for i in modes:
            v = combined.sel(predictor_mode=j, tendency_mode=i).values
            (diag_vals if i == j else off_vals).append(v.ravel())

    diag_vals = np.concatenate(diag_vals) if diag_vals else np.array([np.nan])
    off_vals  = np.concatenate(off_vals)  if off_vals  else np.array([np.nan])

    vmax_diag = float(np.nanmax(np.abs(diag_vals))) if np.isfinite(np.nanmax(np.abs(diag_vals))) else 1.0
    vmax_off  = float(np.nanmax(np.abs(off_vals)))  if np.isfinite(np.nanmax(np.abs(off_vals)))  else 1.0

    levels_diag = np.linspace(-vmax_diag, vmax_diag, nlev)
    levels_off  = np.linspace(-vmax_off,  vmax_off,  nlev)

    line_levels_diag = np.linspace(-vmax_diag, vmax_diag, nline)
    line_levels_off  = np.linspace(-vmax_off,  vmax_off,  nline)
    line_levels_diag = line_levels_diag[line_levels_diag != 0]
    line_levels_off  = line_levels_off[line_levels_off != 0]

    fig, axs = plt.subplots(
        n, n,
        figsize=(4.3 * n, 2.6 * n),
        subplot_kw={"projection": proj},
        constrained_layout=False
    )
    axs = np.atleast_2d(axs)

    mappable_diag = None
    mappable_off = None
    panel_letters = [f"{chr(ord('a') + k)})" for k in range(n * n)]
    pidx = 0

    for ri, j in enumerate(modes):
        for ci, i in enumerate(modes):
            ax = axs[ri, ci]
            ax.coastlines(linewidth=0.6)

            ax.text(
                0.0, 1.08, panel_letters[pidx],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
            )
            pidx += 1

            field = combined.sel(predictor_mode=j, tendency_mode=i)
            sigmask = combined_sig.sel(predictor_mode=j, tendency_mode=i)

            # panel-specific masks already encoded in sigmask
            use_mask = (mask_if_ci_excludes_zero_self if (i == j) else mask_if_ci_excludes_zero_cross)
            field_filled = field.where(sigmask) if use_mask else field

            is_diag = (i == j)
            levels = levels_diag if is_diag else levels_off
            line_levels = line_levels_diag if is_diag else line_levels_off

            m = field_filled.plot.contourf(
                ax=ax, transform=pc, levels=levels,
                add_colorbar=False, extend="both"
            )
            if is_diag:
                mappable_diag = m
            else:
                mappable_off = m

            pos_levels = [lv for lv in line_levels if lv > 0]
            neg_levels = [lv for lv in line_levels if lv < 0]
            if neg_levels:
                ax.contour(field["lon"], field["lat"], field.values,
                           levels=neg_levels, colors="k", linestyles="dashed",
                           linewidths=0.7, transform=pc)
            if pos_levels:
                ax.contour(field["lon"], field["lat"], field.values,
                           levels=pos_levels, colors="k", linestyles="solid",
                           linewidths=0.7, transform=pc)

            if ri == 0:
                ax.set_title(mode_labels.get(i, f"mode {i}"), fontsize=11)
            else:
                ax.set_title("")

            if ci == 0:
                ax.annotate(
                    f"j = {mode_labels.get(j, str(j))}",
                    xy=(-0.05, 0.5), xycoords="axes fraction",
                    rotation=90, ha="center", va="center",
                    fontsize=10
                )

            ax.text(
                0.98, 0.02, "self" if is_diag else "cross",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.70, edgecolor="none", pad=1.5)
            )

    plt.suptitle(
        f"{X_row_labels.get(row_key, row_key)} (sigma thresholds): diagonal = pure-even; off-diagonal = pure-cross",
        y=0.995, fontsize=12
    )
    plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.06, right=0.88, top=0.95, bottom=0.05)

    boxes = [axs[r, c].get_position() for r in range(n) for c in range(n)]
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    x1 = max(b.x1 for b in boxes)

    bar_w = 0.015
    gap = 0.015
    total_h = (y1 - y0)

    if mappable_diag is not None:
        cax1 = fig.add_axes([x1 + 0.01, y0 + 0.52 * total_h, bar_w, 0.48 * total_h - gap])
        cb1 = fig.colorbar(mappable_diag, cax=cax1, orientation="vertical")
        cb1.set_label("Diagonal (self) composite SST anomaly (K)", fontsize=10)
        cb1.formatter = FormatStrFormatter('%.3f')
        cb1.update_ticks()

    if mappable_off is not None:
        cax2 = fig.add_axes([x1 + 0.01, y0, bar_w, 0.48 * total_h - gap])
        cb2 = fig.colorbar(mappable_off, cax=cax2, orientation="vertical")
        cb2.set_label("Off-diagonal (cross) composite SST anomaly (K)", fontsize=10)
        cb2.formatter = FormatStrFormatter('%.3f')
        cb2.update_ticks()

    plt.show()


# ----------------------------
# requested plotting order
# ----------------------------
plot_combined_grid("obs")
plot_combined_grid("rec")
plot_combined_grid("res")
