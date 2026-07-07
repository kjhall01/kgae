import kgae
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

experiment = "kgvae_cv1940-2014"
n_ensemble = 100

# Mask rule recommendation:
# - show filled colors where at least 3/5 seeds significant (>=0.6)
# - still draw unmasked contour lines everywhere to show structure even if not "robust"
sig_thresh = 0.25
# line contour levels (skip 0): fewer looks cleaner
nline = 11
# contour levels
nlev = 21


# Set these lists to select specific modes; otherwise they auto-trim below.
tendency_modes = [5,4,3]          # e.g. [0, 1, 2] or ["dec1","dec2",...]
predictor_modes = [5,4,3]        # e.g. [0, 1, 2]


# --- labels ---
# columns (tendency modes)
tendency_mode_labels = {
    tendency_modes[0]: "Decadal",
    tendency_modes[1]: "Interannual",
    tendency_modes[2]: "Quasibiennial",
}

# rows (alpha + each predictor_mode row for betas)
alpha_row_label = r"State Independent"
beta_row_labels = {
    predictor_modes[0]: r"Decadal-State Dependence",
    predictor_modes[1]: r"Interannual-State Dependence",
    predictor_modes[2]: r"Quasibiennial-State Dependence",
}



# ----------------------------
# helpers
# ----------------------------
def _ensure_dims_alpha(da: xr.DataArray) -> xr.DataArray:
    """Return alpha-like array with dims ('tendency_mode','lat','lon','seed') when possible."""
    dims = list(da.dims)
    # common swaps: ('seed','tendency_mode','lat','lon') etc.
    rename = {}
    if "mode" in dims and "tendency_mode" not in dims:
        rename["mode"] = "tendency_mode"
    da = da.rename(rename)

    needed = ["tendency_mode", "lat", "lon", "seed"]
    missing = [d for d in needed if d not in da.dims]
    if missing:
        raise ValueError(f"alpha dims missing {missing}; got dims={da.dims}")
    return da.transpose("tendency_mode", "lat", "lon", "seed")


def _ensure_dims_beta(da: xr.DataArray) -> xr.DataArray:
    """Return beta-like array with dims ('tendency_mode','predictor_mode','lat','lon','seed') when possible."""
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
    """Flatten lat/lon with NaNs removed."""
    # da_map dims: lat, lon
    arr = da_map.values.reshape(-1)
    return arr


def _composite_linear_holding_others_neutral(
    reference_sst: xr.DataArray,
    latents: xr.DataArray,
    mode_coord_name="tendency_mode",
    sigma_thresh=1.0,
    neutral_thresh=1.0,
) -> xr.DataArray:
    """
    Compute per-seed linear composite map for each mode:
      E[X | z_i > +1σ and |z_not_i|<1σ] - E[X | z_i < -1σ and |z_not_i|<1σ]

    Returns: comp(mode, lat, lon, seed) in same physical units as reference_sst.
    """
    # normalize latent dim names
    if "mode" in latents.dims and mode_coord_name not in latents.dims:
        latents = latents.rename({"mode": mode_coord_name})

    if not all(d in latents.dims for d in ["time", mode_coord_name, "seed"]):
        raise ValueError(f"latents dims must include ('time','{mode_coord_name}','seed'); got {latents.dims}")
    if "time" not in reference_sst.dims or not all(d in reference_sst.dims for d in ["lat", "lon"]):
        raise ValueError(f"reference_sst must have dims ('time','lat','lon'); got {reference_sst.dims}")

    # align time
    reference_sst, latents = xr.align(reference_sst, latents, join="inner")

    modes = latents[mode_coord_name].values
    seeds = latents["seed"].values

    comps = []
    for mi in modes:
        # z_i(time, seed)
        zi = latents.sel({mode_coord_name: mi})
        # sigma_i(seed) from time std
        sig_i = zi.std("time", ddof=0)

        # all other modes (time, other_mode, seed)
        z_not = latents.sel({mode_coord_name: [m for m in modes if m != mi]})
        # neutral mask (time, seed): all other modes within ±neutral_thresh*sigma
        if z_not.sizes.get(mode_coord_name, 0) > 0:
            sig_not = z_not.std("time", ddof=0)
            neutral_mask = (np.abs(z_not) < (neutral_thresh * sig_not)).all(mode_coord_name)
        else:
            # only one mode exists
            neutral_mask = xr.ones_like(zi, dtype=bool)

        pos_mask = (zi > (sigma_thresh * sig_i)) & neutral_mask
        neg_mask = (zi < (-sigma_thresh * sig_i)) & neutral_mask

        # composite per seed: mean over time where mask true
        # xarray trick: mask -> where -> mean
        pos_mean = reference_sst.where(pos_mask).mean("time")
        neg_mean = reference_sst.where(neg_mask).mean("time")
        comp = (pos_mean - neg_mean).expand_dims({mode_coord_name: [mi]})
        comps.append(comp)

    comp_all = xr.concat(comps, dim=mode_coord_name)  # (mode, lat, lon, seed)
    return comp_all.transpose(mode_coord_name, "lat", "lon", "seed")

def _regress_composite_on_alpha_per_seed(alpha: xr.DataArray, comp: xr.DataArray):
    """
    alpha: (tendency_mode, lat, lon, seed)
    comp:  (tendency_mode, lat, lon, seed)

    Returns dict with per-mode arrays of slopes/intercepts/r2 across seeds
    plus a 'badge' tuple that uses the MEDIAN across seeds.
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

            x = _stack_map(a_map)
            y = _stack_map(c_map)
            m, b, r2 = linregress_with_r2(x, y)
            slopes.append(m); intercepts.append(b); r2s.append(r2)

        slopes = np.asarray(slopes, dtype=float)
        intercepts = np.asarray(intercepts, dtype=float)
        r2s = np.asarray(r2s, dtype=float)

        # MEDIAN-across-seeds badge values (ignore NaNs)
        m_med = float(np.nanmedian(slopes))
        b_med = float(np.nanmedian(intercepts))
        r2_med = float(np.nanmedian(r2s))

        out[mi] = dict(
            slopes=slopes,
            intercepts=intercepts,
            r2s=r2s,
            badge=(m_med, b_med, r2_med),
        )
    return out



def _ci_str(arr, qlo=2.5, qhi=97.5, fmt="{:.3g}"):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "nan–nan"
    lo = np.percentile(arr, qlo)
    hi = np.percentile(arr, qhi)
    return f"{fmt.format(lo)}–{fmt.format(hi)}"


# ----------------------------
# load data
# ----------------------------
alphas, betas, alpha_sigs, beta_sigs = kgae.open_alphabetas(
    "xval", experiment=experiment, n_ensemble=n_ensemble
)
reference_sst = kgae.open_references("xval", experiment=experiment)
#reconstructions = kgae.open_reconstructions("xval", experiment=experiment, n_ensemble=n_ensemble)
latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)

print("latents mode coord name(s):", latents.dims)
if "tendency_mode" in latents.coords:
    print("latents tendency_mode values:", latents["tendency_mode"].values)
if "mode" in latents.coords:
    print("latents mode values:", latents["mode"].values)

print("alphas tendency_mode values:", alphas["tendency_mode"].values)



# standardize dims
alphas = _ensure_dims_alpha(alphas)
alpha_sigs = _ensure_dims_alpha(alpha_sigs)
betas = _ensure_dims_beta(betas)
beta_sigs = _ensure_dims_beta(beta_sigs)



# filter dataarrays
alphas_sel = alphas.sel(tendency_mode=tendency_modes)
alpha_sigs_sel = alpha_sigs.sel(tendency_mode=tendency_modes)
betas_sel = betas.sel(tendency_mode=tendency_modes, predictor_mode=predictor_modes)
beta_sigs_sel = beta_sigs.sel(tendency_mode=tendency_modes, predictor_mode=predictor_modes)

latents_sel = latents.sel(mode=tendency_modes)

n_cols = len(tendency_modes)
n_beta_rows = len(predictor_modes)
n_rows = 1 + n_beta_rows

# panel lettering
panel_letters = [f"{chr(ord('a') + i)})" for i in range(n_rows * n_cols)]

# ----------------------------
# compute composites and regression badges (per seed + mean)
# ----------------------------
# composite linear response holding others neutral
comp_lin = _composite_linear_holding_others_neutral(
    reference_sst=reference_sst,
    latents=latents_sel,
    mode_coord_name="tendency_mode",
    sigma_thresh=1.0,
    neutral_thresh=1.0,
).sel(tendency_mode=tendency_modes)

# regress comp_lin on alpha per seed to get CIs + mean-map regression for badge
alpha_for_reg = alphas_sel
reg_stats = _regress_composite_on_alpha_per_seed(alpha_for_reg, comp_lin)

# print confidence intervals
print("\nComposite-on-alpha regression (per-seed) 95% CI (2.5–97.5 pct):")
for mi in tendency_modes:
    st = reg_stats[mi]
    print(f"  mode {mi}: slope CI { _ci_str(st['slopes']) }, "
          f"intercept CI { _ci_str(st['intercepts']) }, "
          f"R^2 CI { _ci_str(st['r2s'], fmt='{:.3f}') }")
    m_badge, b_badge, r2_badge = st["badge"]
    print(f"        median-seed badge: slope={m_badge:.3g}, intercept={b_badge:.3g}, R^2={r2_badge:.3f}")

# ----------------------------
# plotting
# ----------------------------
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree(central_longitude=180)

alpha_mean = alphas_sel.mean("seed")
beta_mean = betas_sel.mean("seed")

# significance: fraction of seeds significant at each gridpoint
alpha_sig_frac = alpha_sigs_sel.mean("seed")
beta_sig_frac = beta_sigs_sel.mean("seed")



# shared color scales
alpha_vmax = float(np.nanmax(np.abs(alpha_mean.values)))
beta_vmax = float(np.nanmax(np.abs(beta_mean.values)))


alpha_levels = np.linspace(-alpha_vmax, alpha_vmax, nlev)
beta_levels = np.linspace(-beta_vmax, beta_vmax, nlev)


alpha_line_levels = np.linspace(-alpha_vmax, alpha_vmax, nline)
alpha_line_levels = alpha_line_levels[alpha_line_levels != 0]
beta_line_levels = np.linspace(-beta_vmax, beta_vmax, nline)
beta_line_levels = beta_line_levels[beta_line_levels != 0]

fig, axs = plt.subplots(
    n_rows, n_cols,
    figsize=(4.3 * n_cols, 2.6 * n_rows),
    subplot_kw={"projection": proj},
    constrained_layout=False
)

# make axs always 2D
axs = np.atleast_2d(axs)

mappable_alpha = None
mappable_beta = None
panel_idx = 0 

for ci, mi in enumerate(tendency_modes):
    # ---- row 0: alpha
    ax = axs[0, ci]
    ax.coastlines(linewidth=0.6)

    ax.text(
        0.0, 1.08, panel_letters[panel_idx],
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
    )
    panel_idx += 1

    field = alpha_mean.sel(tendency_mode=mi)
    sigmask = alpha_sig_frac.sel(tendency_mode=mi) >= sig_thresh
    field_filled = field.where(sigmask)

    mappable_alpha = field_filled.plot.contourf(
        ax=ax, transform=pc, levels=alpha_levels,
        add_colorbar=False, extend="both"
    )
    ax.set_title("")
    ax.set_title(tendency_mode_labels.get(mi, f"mode {mi}"), fontsize=11)

    # line contours (unmasked) with linestyle by sign
    # positive: solid black, negative: dashed black
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

    # regression badge from mean-map regression
    m_badge, b_badge, r2_badge = reg_stats[mi]["badge"]
    ax.text(
        0.02, 0.02,
        f"y={m_badge:.2g}x+{b_badge:.2g} (R$^2$:{r2_badge:.2f})",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=3)
    )

    # ---- beta rows
    for ri, pj in enumerate(predictor_modes, start=1):
        print(f"ri {ri} pj {pj} ci {ci} mi {mi}")
        ax = axs[ri, ci]
        ax.coastlines(linewidth=0.6)

        ax.text(
            0.02, 1.08, panel_letters[panel_idx],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
        )
        panel_idx += 1

        fieldb = beta_mean.sel(tendency_mode=mi, predictor_mode=pj)
        sigmaskb = beta_sig_frac.sel(tendency_mode=mi, predictor_mode=pj) >= sig_thresh
        fieldb_filled = fieldb.where(sigmaskb)

        mappable_beta = fieldb_filled.plot.contourf(
            ax=ax, transform=pc, levels=beta_levels,
            add_colorbar=False, extend="both"
        )
        ax.set_title("")

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

        if ci == 0:
            ax.annotate(
                beta_row_labels.get(pj, f"predictor {pj}"),
                xy=(-0.05, 0.5), xycoords="axes fraction",
                rotation=90, ha="center", va="center",
                fontsize=10
            )

# row label for alpha
axs[0, 0].annotate(
    alpha_row_label,
    xy=(-0.05, 0.5), xycoords="axes fraction",
    rotation=90, ha="center", va="center",
    fontsize=10
)
# ---- colorbars: one for alpha row, one for all beta rows
# manually place them on the right so they don't overlap

# Tight-ish layout first; we'll add cbar axes in figure coords after.
plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.05, right=0.88, top=0.95, bottom=0.06)

# alpha colorbar next to top row
if mappable_alpha is not None:
    # get bounding box of top row axes in figure coords
    row0_boxes = [axs[0, j].get_position() for j in range(n_cols)]
    y0 = min(b.y0 for b in row0_boxes)
    y1 = max(b.y1 for b in row0_boxes)
    x1 = max(b.x1 for b in row0_boxes)
    cax_alpha = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb1 = fig.colorbar(mappable_alpha, cax=cax_alpha, orientation="vertical")
    cb1.set_label(r"$\alpha (K\sigma_i^{-1})$", fontsize=10)
    cb1.formatter = FormatStrFormatter('%.3f')
    cb1.update_ticks()

# beta colorbar spanning beta rows
if mappable_beta is not None and n_beta_rows > 0:
    beta_boxes = [axs[r, j].get_position() for r in range(1, n_rows) for j in range(n_cols)]
    y0 = min(b.y0 for b in beta_boxes)
    y1 = max(b.y1 for b in beta_boxes)
    x1 = max(axs[0, j].get_position().x1 for j in range(n_cols))  # align with alpha cbar
    cax_beta = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
    cb2 = fig.colorbar(mappable_beta, cax=cax_beta, orientation="vertical")
    cb2.set_label(r"$\beta (K\sigma_i^{-1}\sigma_j^{-1})$", fontsize=10)
    cb2.formatter = FormatStrFormatter('%.3f')
    cb2.update_ticks()

plt.show()
