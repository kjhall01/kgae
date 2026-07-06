# obs_linear_and_pureeven_composites.py
import kgae
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import FormatStrFormatter

# ----------------------------
# user settings
# ----------------------------
experiment = "kgvae_cv1940-2014"
n_ensemble = 100  # for open_latents

modes = [5, 4, 3]  # columns
mode_labels = {5: "Decadal Mode", 4: "Interannual Mode"}

sigma_thresh = 1.0
neutral_thresh = 1.0

n_boot = 500
ci_level = 0.9
mask_if_ci_excludes_zero = True

# plotting
nlev = 21
nline = 11

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


def _boot_ci_excludes_zero(boot: np.ndarray, ci_level: float) -> np.ndarray:
    """boot: (n_boot, lat, lon) -> bool(lat,lon) where CI excludes 0."""
    alpha = 1.0 - ci_level
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)
    lo = np.nanpercentile(boot, lo_q, axis=0)
    hi = np.nanpercentile(boot, hi_q, axis=0)
    return (lo > 0) | (hi < 0)


def _bootstrap_obs_composites(
    X: xr.DataArray,   # (time, lat, lon)
    z: xr.DataArray,   # (time, tendency_mode) ensemble-mean latents
    mode_val,
    mode_dim="tendency_mode",
    sigma_thresh=1.0,
    neutral_thresh=1.0,
    n_boot=500,
    ci_level=0.90,
    rng_seed=0,
):
    """
    Two estimators computed with the SAME masks:
      1) linear:   E[X|i+] - E[X|i-]
      2) pure-even: (E[X|i+] + E[X|i-]) - 2*E[X|neutral_others]
         where "neutral_others" means all k!=i satisfy |z_k| < neutral_thresh*std_k

    Bootstraps over time indices. Returns:
      lin_mean, lin_sigmask, pe_mean, pe_sigmask   (each lat,lon)
    """
    rng = np.random.default_rng(rng_seed)

    X, z = xr.align(X, z, join="inner")

    zi = z.sel({mode_dim: mode_val})
    std_i = float(zi.std("time", ddof=0).values)

    other_modes = [m for m in z[mode_dim].values if m != mode_val]
    if len(other_modes) > 0:
        z_not = z.sel({mode_dim: other_modes})
        std_not = z_not.std("time", ddof=0)
        neutral_others = (np.abs(z_not) < (neutral_thresh * std_not)).all(mode_dim)
    else:
        neutral_others = xr.ones_like(zi, dtype=bool)

    pos_mask = (zi > (sigma_thresh * std_i)) & neutral_others
    neg_mask = (zi < (-sigma_thresh * std_i)) & neutral_others

    # point estimates
    pos_mean = X.where(pos_mask).mean("time")
    neg_mean = X.where(neg_mask).mean("time")
    base_mean = X.where(neutral_others).mean("time")

    lin_mean = pos_mean - neg_mean
    pe_mean = (pos_mean + neg_mean) - 2.0 * base_mean

    # bootstrap (fast numpy)
    tN = X.sizes["time"]
    X_np = X.values
    pos_np = pos_mask.values
    neg_np = neg_mask.values
    neu_np = neutral_others.values

    boot_lin = np.full((n_boot, X.sizes["lat"], X.sizes["lon"]), np.nan, dtype=np.float32)
    boot_pe = np.full((n_boot, X.sizes["lat"], X.sizes["lon"]), np.nan, dtype=np.float32)

    for b in range(n_boot):
        idx = rng.integers(0, tN, size=tN)
        X_b = X_np[idx, :, :]

        pos_b = pos_np[idx]
        neg_b = neg_np[idx]
        neu_b = neu_np[idx]

        pos_m = np.nanmean(X_b[pos_b, :, :], axis=0) if pos_b.any() else np.full((X.sizes["lat"], X.sizes["lon"]), np.nan, np.float32)
        neg_m = np.nanmean(X_b[neg_b, :, :], axis=0) if neg_b.any() else np.full((X.sizes["lat"], X.sizes["lon"]), np.nan, np.float32)
        neu_m = np.nanmean(X_b[neu_b, :, :], axis=0) if neu_b.any() else np.full((X.sizes["lat"], X.sizes["lon"]), np.nan, np.float32)

        boot_lin[b, :, :] = pos_m - neg_m
        boot_pe[b, :, :] = (pos_m + neg_m) - 2.0 * neu_m

    lin_sig = _boot_ci_excludes_zero(boot_lin, ci_level=ci_level)
    pe_sig = _boot_ci_excludes_zero(boot_pe, ci_level=ci_level)

    lin_sig = xr.DataArray(lin_sig, coords={"lat": X["lat"], "lon": X["lon"]}, dims=("lat", "lon"))
    pe_sig = xr.DataArray(pe_sig, coords={"lat": X["lat"], "lon": X["lon"]}, dims=("lat", "lon"))
    return lin_mean, lin_sig, pe_mean, pe_sig


# ----------------------------
# load data: obs + ensemble-mean latents
# ----------------------------
reference_sst = kgae.open_references("xval", experiment=experiment).transpose("time", "lat", "lon")

latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)
latents = _rename_mode_dim(latents, mode_coord_name="tendency_mode")

ens_dim = _infer_ens_dim(latents)
if ens_dim is None:
    raise ValueError(f"Could not infer ensemble dim for latents; dims={latents.dims}")

zbar = latents.mean(ens_dim).transpose("time", "tendency_mode")

# align time
reference_sst, zbar = xr.align(reference_sst, zbar, join="inner")

# ----------------------------
# compute composites for each mode
# ----------------------------
lin_maps = {}
lin_sigs = {}
pe_maps = {}
pe_sigs = {}

for mi in modes:
    lin_m, lin_sig, pe_m, pe_sig = _bootstrap_obs_composites(
        X=reference_sst,
        z=zbar,
        mode_val=mi,
        mode_dim="tendency_mode",
        sigma_thresh=sigma_thresh,
        neutral_thresh=neutral_thresh,
        n_boot=n_boot,
        ci_level=ci_level,
        rng_seed=1000 + int(mi),
    )
    lin_maps[mi] = lin_m
    lin_sigs[mi] = lin_sig
    pe_maps[mi] = pe_m
    pe_sigs[mi] = pe_sig

# ----------------------------
# plotting: 2 rows x len(modes) cols
# ----------------------------
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree(central_longitude=180)

# separate color scales per row
lin_vals = np.concatenate([lin_maps[mi].values.ravel() for mi in modes])
pe_vals = np.concatenate([pe_maps[mi].values.ravel() for mi in modes])

vmax_lin = float(np.nanmax(np.abs(lin_vals)))
vmax_pe = float(np.nanmax(np.abs(pe_vals)))

levels_lin = np.linspace(-vmax_lin, vmax_lin, nlev)
levels_pe = np.linspace(-vmax_pe, vmax_pe, nlev)

line_levels_lin = np.linspace(-vmax_lin, vmax_lin, nline)
line_levels_pe = np.linspace(-vmax_pe, vmax_pe, nline)
line_levels_lin = line_levels_lin[line_levels_lin != 0]
line_levels_pe = line_levels_pe[line_levels_pe != 0]

fig, axs = plt.subplots(
    2, len(modes),
    figsize=(4.3 * len(modes), 2.6 * 2),
    subplot_kw={"projection": proj},
    constrained_layout=False
)
axs = np.atleast_2d(axs)

mappable_top = None
mappable_bot = None

for ci, mi in enumerate(modes):
    # --- top row: linear (pos - neg)
    ax = axs[0, ci]
    ax.coastlines(linewidth=0.6)

    field = lin_maps[mi]
    sig = lin_sigs[mi]
    field_filled = field.where(sig) if mask_if_ci_excludes_zero else field

    m = field_filled.plot.contourf(
        ax=ax, transform=pc, levels=levels_lin,
        add_colorbar=False, extend="both"
    )
    mappable_top = m

    pos_levels = [lv for lv in line_levels_lin if lv > 0]
    neg_levels = [lv for lv in line_levels_lin if lv < 0]
    if neg_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=neg_levels, colors="k", linestyles="dashed",
                   linewidths=0.7, transform=pc)
    if pos_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=pos_levels, colors="k", linestyles="solid",
                   linewidths=0.7, transform=pc)

    ax.set_title(mode_labels.get(mi, f"mode {mi}"), fontsize=11)

    if ci == 0:
        ax.annotate(
            r"Linear: $E[X|i^+]-E[X|i^-]$",
            xy=(-0.06, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=10
        )

    # --- bottom row: pure-even (pos + neg - 2*neutral)
    ax = axs[1, ci]
    ax.coastlines(linewidth=0.6)

    field = pe_maps[mi]
    sig = pe_sigs[mi]
    field_filled = field.where(sig) if mask_if_ci_excludes_zero else field

    m = field_filled.plot.contourf(
        ax=ax, transform=pc, levels=levels_pe,
        add_colorbar=False, extend="both"
    )
    mappable_bot = m

    pos_levels = [lv for lv in line_levels_pe if lv > 0]
    neg_levels = [lv for lv in line_levels_pe if lv < 0]
    if neg_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=neg_levels, colors="k", linestyles="dashed",
                   linewidths=0.7, transform=pc)
    if pos_levels:
        ax.contour(field["lon"], field["lat"], field.values,
                   levels=pos_levels, colors="k", linestyles="solid",
                   linewidths=0.7, transform=pc)

    if ci == 0:
        ax.annotate(
            r"Pure-even: $E[X|i^+]+E[X|i^-]-2E[X|\mathrm{neutral}]$",
            xy=(-0.06, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=10
        )

plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.06, right=0.88, top=0.95, bottom=0.06)

# colorbars: one for each row
if mappable_top is not None:
    row_boxes = [axs[0, c].get_position() for c in range(len(modes))]
    y0 = min(b.y0 for b in row_boxes)
    y1 = max(b.y1 for b in row_boxes)
    x1 = max(b.x1 for b in row_boxes)
    cax = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb = fig.colorbar(mappable_top, cax=cax, orientation="vertical")
    cb.set_label("Obs linear composite (K)", fontsize=10)
    cb.formatter = FormatStrFormatter('%.3f')
    cb.update_ticks()

if mappable_bot is not None:
    row_boxes = [axs[1, c].get_position() for c in range(len(modes))]
    y0 = min(b.y0 for b in row_boxes)
    y1 = max(b.y1 for b in row_boxes)
    x1 = max(b.x1 for b in row_boxes)
    cax = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb = fig.colorbar(mappable_bot, cax=cax, orientation="vertical")
    cb.set_label("Obs pure-even composite (K)", fontsize=10)
    cb.formatter = FormatStrFormatter('%.3f')
    cb.update_ticks()

plt.show()
