import kgae
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

experiment = "kgvae_cv1940-2014"
n_ensemble = 95  # used for open_* calls

# --- composite settings (ENSEMBLE-MEAN VERSION) ---
sigma_thresh = 1.0
neutral_thresh = 1.0
n_boot = 100
ci_level = 0.90
mask_if_ci_excludes_zero = True

# plotting
nline = 11
nlev = 21

# modes (columns)
tendency_modes = [5, 4, 3]
tendency_mode_labels = {5: "Decadal", 4: "Interannual", 3: "Quasibiennial"}

row_labels = {
    0: "Observed (pure-even)",
    1: "Reconstruction (pure-even)",
    2: "Residual (pure-even)",
}


# ----------------------------
# helpers
# ----------------------------
def _rename_mode_dim(latents: xr.DataArray, mode_coord_name="tendency_mode") -> xr.DataArray:
    if "mode" in latents.dims and mode_coord_name not in latents.dims:
        latents = latents.rename({"mode": mode_coord_name})
    return latents


def _pure_even_bootstrap_ensemble_mean(
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
    Pure-even composite:
      (E[X|zi>t, neutral] + E[X|zi<-t, neutral]) - 2*E[X|neutral]
    Bootstraps over time indices.

    Returns:
      comp_mean(lat,lon)  [xarray]
      sig_mask(lat,lon)   [xarray bool]
    """
    rng = np.random.default_rng(rng_seed)

    # Align time
    X, z = xr.align(X, z, join="inner")

    zi = z.sel({mode_dim: mode_val})
    sig_i = float(zi.std("time", ddof=0).values)

    other_modes = [m for m in z[mode_dim].values if m != mode_val]
    if len(other_modes) > 0:
        z_not = z.sel({mode_dim: other_modes})
        sig_not = z_not.std("time", ddof=0)
        neutral_mask = (np.abs(z_not) < (neutral_thresh * sig_not)).all(mode_dim)
    else:
        neutral_mask = xr.ones_like(zi, dtype=bool)

    pos_mask = (zi > (sigma_thresh * sig_i)) & neutral_mask
    neg_mask = (zi < (-sigma_thresh * sig_i)) & neutral_mask

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
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)
    comp_lo = np.nanpercentile(boot, lo_q, axis=0)
    comp_hi = np.nanpercentile(boot, hi_q, axis=0)
    sig_mask = (comp_lo > 0) | (comp_hi < 0)

    sig_mask = xr.DataArray(sig_mask, coords={"lat": X["lat"], "lon": X["lon"]}, dims=("lat", "lon"))
    return comp_mean, sig_mask


def _infer_ens_dim(da: xr.DataArray):
    for cand in ["seed", "ensemble", "member"]:
        if cand in da.dims:
            return cand
    return None


# ----------------------------
# load data (obs + latents + recon)
# ----------------------------
reference_sst = kgae.open_references("xval", experiment=experiment).transpose("time", "lat", "lon")

latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)
latents = _rename_mode_dim(latents, mode_coord_name="tendency_mode")
latents_sel = latents.sel(tendency_mode=tendency_modes)
print(latents.tendency_mode.values, latents_sel.tendency_mode.values)
# ensemble-mean latents
ens_dim_lat = _infer_ens_dim(latents_sel)
if ens_dim_lat is None:
    raise ValueError(f"Could not infer ensemble dim for latents; dims={latents_sel.dims}")
zbar = latents_sel.mean(ens_dim_lat).transpose("time", "tendency_mode")

# reconstructions and ensemble mean
reconstructions = kgae.open_reconstructions("xval", experiment=experiment, n_ensemble=n_ensemble)
ens_dim_rec = _infer_ens_dim(reconstructions)
if ens_dim_rec is None:
    raise ValueError(f"Could not infer ensemble dim for reconstructions; dims={reconstructions.dims}")
Xhat = reconstructions.mean(ens_dim_rec).transpose("time", "lat", "lon")

# align time
reference_sst, Xhat, zbar = xr.align(reference_sst, Xhat, zbar, join="inner")
R = reference_sst - Xhat

# ----------------------------
# compute composites on ensemble means
# ----------------------------
X_by_row = {"obs": reference_sst, "rec": Xhat, "res": R}

comp_maps = {}
sig_masks = {}

for rk, X in X_by_row.items():
    comps = []
    sigs = []
    for mi in tendency_modes:
        comp_mean, sig_mask = _pure_even_bootstrap_ensemble_mean(
            X=X,
            z=zbar,
            mode_val=mi,
            mode_dim="tendency_mode",
            sigma_thresh=sigma_thresh,
            neutral_thresh=neutral_thresh,
            n_boot=n_boot,
            ci_level=ci_level,
            rng_seed=100 + (0 if rk == "obs" else 1 if rk == "rec" else 2) + int(mi),
        )
        comps.append(comp_mean.expand_dims({"tendency_mode": [mi]}))
        sigs.append(sig_mask.expand_dims({"tendency_mode": [mi]}))

    comp_maps[rk] = xr.concat(comps, dim="tendency_mode").transpose("tendency_mode", "lat", "lon")
    sig_masks[rk] = xr.concat(sigs, dim="tendency_mode").transpose("tendency_mode", "lat", "lon")

# ----------------------------
# plotting
# ----------------------------
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree(central_longitude=180)

n_cols = len(tendency_modes)
n_rows = 3
panel_letters = [f"{chr(ord('a') + i)})" for i in range(n_rows * n_cols)]

# shared color scale
all_vals = np.concatenate([
    comp_maps["obs"].values.ravel(),
    comp_maps["rec"].values.ravel(),
    comp_maps["res"].values.ravel(),
])
vmax = float(np.nanmax(np.abs(all_vals)))
levels = np.linspace(-vmax, vmax, nlev)
line_levels = np.linspace(-vmax, vmax, nline)
line_levels = line_levels[line_levels != 0]

fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(4.3 * n_cols, 2.6 * n_rows),
    subplot_kw={"projection": proj},
    constrained_layout=False
)
axs = np.atleast_2d(axs)

mappable = None
panel_idx = 0
row_order = [("obs", 0), ("rec", 1), ("res", 2)]

for (rk, ri) in row_order:
    for ci, mi in enumerate(tendency_modes):
        ax = axs[ri, ci]
        ax.coastlines(linewidth=0.6)

        ax.text(
            0.0, 1.08, panel_letters[panel_idx],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
        )
        panel_idx += 1

        field = comp_maps[rk].sel(tendency_mode=mi)
        sigmask = sig_masks[rk].sel(tendency_mode=mi)
        field_filled = field.where(sigmask) if mask_if_ci_excludes_zero else field

        mappable = field_filled.plot.contourf(
            ax=ax, transform=pc, levels=levels,
            add_colorbar=False, extend="both"
        )

        # unmasked line contours
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
            ax.set_title(tendency_mode_labels.get(mi, f"mode {mi}"), fontsize=11)
        else:
            ax.set_title("")

        if ci == 0:
            ax.annotate(
                row_labels.get(ri, ""),
                xy=(-0.05, 0.5), xycoords="axes fraction",
                rotation=90, ha="center", va="center",
                fontsize=10
            )

plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.05, right=0.88, top=0.95, bottom=0.06)

if mappable is not None:
    boxes = [axs[r, c].get_position() for r in range(n_rows) for c in range(n_cols)]
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    cax = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cb.set_label("Pure-even composite SST anomaly (K)", fontsize=10)
    cb.formatter = FormatStrFormatter('%.3f')
    cb.update_ticks()

plt.show()
