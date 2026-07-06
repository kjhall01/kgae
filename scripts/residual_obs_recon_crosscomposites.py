import kgae
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

experiment = "kgvae_cv1940-2014"
n_ensemble = 95  # used for open_* calls

# --- PURE cross-mode nonlinearity settings (ENSEMBLE-MEAN VERSION) ---
sigma_i = 1.0          # threshold for i high/low (in std units)
sigma_j = 1.0          # threshold for j high/low (in std units)
neutral_thresh = 1.0   # hold all OTHER modes within +/- neutral_thresh * sigma
n_boot = 100
ci_level = 0.6
mask_if_ci_excludes_zero = True

# plotting
nline = 11
nlev = 21

# modes to analyze
modes = [5, 4, 3]
mode_labels = {5: "Decadal", 4: "Interannual", 3: "Quasibiennial"}

# grid layout:
# cols = tendency_mode i (whose self-nonlinearity is being modulated)
# rows = predictor_mode j (the modulating mode)
tendency_modes = modes
predictor_modes = modes

# which data rows
X_row_labels = {
    "obs": "Observed",
    "rec": "Reconstruction",
    "res": "Residual",
}


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


def _pure_cross_interaction_bootstrap_ensemble_mean(
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
    Implements the user's "pure cross" estimator:

      S(i | j+) = [ (E(i+,j+) - E(n_i,j+)) - (E(n_i,j+) - E(i-,j+)) ]
               = E(i+,j+) + E(i-,j+) - 2 E(n_i,j+)

      S(i | j-) = E(i+,j-) + E(i-,j-) - 2 E(n_i,j-)

      I_pure(i,j) = S(i|j+) - S(i|j-)

    All conditions also enforce neutrality on all other modes k != i,j:
      |z_k| < neutral_thresh * std_k

    Returns:
      I_mean(lat,lon) [xarray]
      sig_mask(lat,lon) [xarray bool] where bootstrap CI excludes 0
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
    i_neu = np.abs(zi) < (neutral_thresh * std_i)  # n_i baseline: i neutral

    # j states
    j_pos = zj > +sigma_j * std_j
    j_neg = zj < -sigma_j * std_j

    # masks (all include neutral_others)
    # j+
    m_ip_jp = i_pos & j_pos & neutral_others
    m_in_jp = i_neg & j_pos & neutral_others
    m_ni_jp = i_neu & j_pos & neutral_others

    # j-
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

    S_jp = (E_ip_jp + E_in_jp) - 2.0 * E_ni_jp # this is the difference between moving into ipos from ineu  during jpos  and moving in to ineu from ineg during jpos 
    S_jn = (E_ip_jn + E_in_jn) - 2.0 * E_ni_jn

    I_mean = S_jp - S_jn

    # bootstrap over time indices (fast numpy)
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

        # j+
        ip_jp_b = ip_jp[idx]
        in_jp_b = in_jp[idx]
        ni_jp_b = ni_jp[idx]

        # j-
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
    lo_q = 100 * (alpha / 2)
    hi_q = 100 * (1 - alpha / 2)
    lo = np.nanpercentile(boot, lo_q, axis=0)
    hi = np.nanpercentile(boot, hi_q, axis=0)
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

# align time
reference_sst, Xhat, zbar = xr.align(reference_sst, Xhat, zbar, join="inner")
R = reference_sst - Xhat

X_by_row = {"obs": reference_sst, "rec": Xhat, "res": R}

# ----------------------------
# compute pure-cross maps for each row and each ordered pair (i,j)
# ----------------------------
pure_cross_maps = {}
pure_cross_sigs = {}

for rk, X in X_by_row.items():
    maps_ij = []
    sigs_ij = []
    for i in tendency_modes:
        row_i = []
        row_sig = []
        for j in predictor_modes:
            if i == j:
                nan_map = xr.full_like(X.isel(time=0), np.nan)
                nan_sig = xr.full_like(X.isel(time=0), False).astype(bool)
                row_i.append(nan_map.expand_dims({"predictor_mode": [j]}))
                row_sig.append(nan_sig.expand_dims({"predictor_mode": [j]}))
                continue

            I_mean, sig_mask = _pure_cross_interaction_bootstrap_ensemble_mean(
                X=X,
                z=zbar,
                mode_i=i,
                mode_j=j,
                mode_dim="tendency_mode",
                sigma_i=sigma_i,
                sigma_j=sigma_j,
                neutral_thresh=neutral_thresh,
                n_boot=n_boot,
                ci_level=ci_level,
                rng_seed=2000 + (0 if rk == "obs" else 1 if rk == "rec" else 2) * 100 + int(i) * 10 + int(j),
            )
            row_i.append(I_mean.expand_dims({"predictor_mode": [j]}))
            row_sig.append(sig_mask.expand_dims({"predictor_mode": [j]}))

        row_i = xr.concat(row_i, dim="predictor_mode")
        row_sig = xr.concat(row_sig, dim="predictor_mode")
        maps_ij.append(row_i.expand_dims({"tendency_mode": [i]}))
        sigs_ij.append(row_sig.expand_dims({"tendency_mode": [i]}))

    pure_cross_maps[rk] = xr.concat(maps_ij, dim="tendency_mode").transpose("tendency_mode", "predictor_mode", "lat", "lon")
    pure_cross_sigs[rk] = xr.concat(sigs_ij, dim="tendency_mode").transpose("tendency_mode", "predictor_mode", "lat", "lon")

# ----------------------------
# plotting: for each row_key, grid rows=j, cols=i
# ----------------------------
proj = ccrs.PlateCarree(central_longitude=180)
pc = ccrs.PlateCarree(central_longitude=180)

def plot_pure_cross_grid(row_key: str):
    data = pure_cross_maps[row_key]
    sig = pure_cross_sigs[row_key]

    n_cols = len(tendency_modes)
    n_rows = len(predictor_modes)

    vmax = float(np.nanmax(np.abs(data.values)))
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
    panel_letters = [f"{chr(ord('a') + k)})" for k in range(n_rows * n_cols)]
    pidx = 0

    for ri, j in enumerate(predictor_modes):
        for ci, i in enumerate(tendency_modes):
            ax = axs[ri, ci]
            ax.coastlines(linewidth=0.6)

            ax.text(
                0.0, 1.08, panel_letters[pidx],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
            )
            pidx += 1

            field = data.sel(tendency_mode=i, predictor_mode=j)
            sigmask = sig.sel(tendency_mode=i, predictor_mode=j)
            field_filled = field.where(sigmask) if mask_if_ci_excludes_zero else field

            mappable = field_filled.plot.contourf(
                ax=ax, transform=pc, levels=levels,
                add_colorbar=False, extend="both"
            )

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

    plt.suptitle(
        f"{X_row_labels.get(row_key, row_key)} pure cross-mode nonlinearity: "
        r"$I_{ij} = S(i|j+) - S(i|j-)$, "
        r"$S(i|j\pm)=E(i+,j\pm)+E(i-,j\pm)-2E(n_i,j\pm)$",
        y=0.995, fontsize=12
    )

    plt.subplots_adjust(wspace=0.03, hspace=0.10, left=0.06, right=0.88, top=0.95, bottom=0.05)

    if mappable is not None:
        boxes = [axs[r, c].get_position() for r in range(n_rows) for c in range(n_cols)]
        y0 = min(b.y0 for b in boxes)
        y1 = max(b.y1 for b in boxes)
        x1 = max(b.x1 for b in boxes)
        cax = fig.add_axes([x1 + 0.01, y0, 0.015, y1 - y0])
        cb = fig.colorbar(mappable, cax=cax, orientation="vertical")
        cb.set_label("Pure cross interaction composite SST anomaly (K)", fontsize=10)
        cb.formatter = FormatStrFormatter('%.3f')
        cb.update_ticks()

    plt.show()

# call as needed
plot_pure_cross_grid("obs")
plot_pure_cross_grid("rec")
plot_pure_cross_grid("res")
