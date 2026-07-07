#!/usr/bin/env python3
"""
Ensemble (seed/split) regression + seed-bootstrap significance for:
  - R^2 maps
  - alpha maps
  - beta maps

Workflow:
1) Load/anomalize SST and build the exact feature template from that SST.
2) For each trained model (seed*/split*/model.pkl):
   - encode SST -> latents (apply flip_signs, scale by sigma over time)
   - compute jacobian tendencies via kgae.jacobian (scale by sigma)
   - build quadratic predictors from selected modes (unique z_i z_j terms)
   - run kgae.regress_tendencies_r2 on (predictors, tendencies)
3) For each seed: average across splits (split-mean).
4) Bootstrap across seeds (i.i.d.) to get:
   - mean field
   - CI (lo/hi)
   - significance mask (CI excludes 0) for alpha/beta
   - significance mask for R^2 relative to a baseline (optional) OR >0 (default)
5) Plot masked mean maps (filled where significant) + contour lines everywhere.

Notes:
- This assumes all models share the same training feature mask/domain as the SST stacking.
- If not, a feature-length mismatch indicates that the template should be built
  using the same mask used in training.
"""

import glob, re, pickle
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import kgae

# -----------------------------
# USER SETTINGS
# -----------------------------
MODEL_GLOB = "./large-ensemble-2-1940-2014/seed*/split*/model.pkl"
ERA5_SST_PATH = "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"

# Tendencies/J slices to model (these are "tendency modes" in the target)
TENDENCY_MODES = [5, 4, 3]  # e.g. [5,4] (Decadal, Interannual)

# Latent modes to use in predictor construction
LATENT_MODES_FOR_PREDICTORS = [5, 4]  # build z_i z_j among these

# Bootstrap over seeds
N_BOOT = 500
RNG_SEED = 0
CI_LO, CI_HI = 2.5, 97.5

# Plot levels (tune)
R2_LEVELS = np.linspace(0, 1, 11)
CMAP = "RdBu_r"
CENTRAL_LON = 180

# Significance rule for coefficients: CI excludes 0
# For R^2: by default we just show mean (and optionally mask by CI > 0)
MASK_R2_BY_CI_POSITIVE = False  # set True to only shade where lower CI > 0


# -----------------------------
# helpers
# -----------------------------
def parse_seed_split(path: str):
    parts = str(path).replace("\\", "/").split("/")
    seed = int([p for p in parts if p.startswith("seed")][0].replace("seed", ""))
    split = int([p for p in parts if p.startswith("split")][0].replace("split", ""))
    return seed, split

def build_feature_template(sst_anom: xr.DataArray) -> xr.DataArray:
    sst_stack = (
        sst_anom
        .stack(feature=("lat", "lon"))
        .dropna("feature", how="any")
        .transpose("time", "feature")
    )
    return sst_stack

def unique_pair_labels(modes):
    labels = []
    pairs = []
    for i in modes:
        for j in modes:
            lab = f"z{i}z{j}"
            lab2 = f"z{j}z{i}"
            if (lab not in labels) and (lab2 not in labels):
                labels.append(lab)
                pairs.append((i, j))
    return pairs, labels



# -----------------------------
# Bootstrap across seeds (i.i.d.)
# -----------------------------
def bootstrap_ci(members: np.ndarray, n_boot: int, rng: np.random.Generator,
                      ci_lo: float = 2.5, ci_hi: float = 97.5, chunk: int = 100):
    """
    members: (n_seed, n_feature) per-seed split-mean maps.
    Returns: mean, lo, hi arrays (n_feature,)
    """
    members = np.asarray(members)
    S, F = members.shape
    mean = members.mean(axis=0)

    boot_means = np.empty((n_boot, F), dtype=np.float32)
    written = 0
    while written < n_boot:
        b = min(chunk, n_boot - written)
        idx = rng.integers(0, S, size=(b, S))
        for k in range(b):
            boot_means[written + k] = members[idx[k]].mean(axis=0)
        written += b

    lo = np.percentile(boot_means, ci_lo, axis=0)
    hi = np.percentile(boot_means, ci_hi, axis=0)
    return mean, lo, hi


def sig_mask_from_ci(lo, hi):
    return (lo > 0) | (hi < 0)

def match_pred_to_tendency(pred_name: str, tm_label) -> bool:
    # extract all integers after 'z' from predictor label
    z_ids = [int(x) for x in re.findall(r"z(\d+)", str(pred_name))]
    try:
        tm = int(tm_label)
    except Exception:
        return str(tm_label) in str(pred_name)
    return tm in z_ids

import numpy as np
import xarray as xr

def bootstrap_xr_over_seed_2d(da_seed: xr.DataArray, *, n_boot: int, rng,
                             ci_lo: float = 2.5, ci_hi: float = 97.5,
                             seed_dim: str = "seed", chunk: int = 10):
    """
    Bootstrap CI over `seed_dim` for a 2D DataArray shaped (seed, space).
    Returns mean/lo/hi as DataArrays shaped (space,).
    """
    members = np.asarray(da_seed.values)  # (S, F)
    mean, lo, hi = bootstrap_ci(members, n_boot=n_boot, rng=rng,
                                     ci_lo=ci_lo, ci_hi=ci_hi, chunk=chunk)
    mean_da = xr.DataArray(mean, dims=("space",), coords={"space": da_seed["space"]})
    lo_da   = xr.DataArray(lo,   dims=("space",), coords={"space": da_seed["space"]})
    hi_da   = xr.DataArray(hi,   dims=("space",), coords={"space": da_seed["space"]})
    return mean_da, lo_da, hi_da


def bootstrap_alpha_beta_r2_sliced(alpha_seed, beta_seed, r2_seed, *,
                                  n_boot: int, rng,
                                  ci_lo: float = 2.5, ci_hi: float = 97.5,
                                  chunk: int = 10,
                                  seed_dim: str = "seed"):
    """
    alpha_seed dims: (seed, tendency_mode, lat, lon)
    beta_seed  dims: (seed, tendency_mode, predictor_mode, lat, lon)
    r2_seed    dims: (seed, tendency_mode, lat, lon)

    Returns:
      alpha_mean/lo/hi: (tendency_mode, lat, lon)
      beta_mean/lo/hi : (tendency_mode, predictor_mode, lat, lon)
      r2_mean/lo/hi   : (tendency_mode, lat, lon)
    """
    # ---- stack space once ----
    alpha_s = alpha_seed.stack(space=("lat", "lon"))
    beta_s  = beta_seed.stack(space=("lat", "lon"))
    r2_s    = r2_seed.stack(space=("lat", "lon"))

    tm_vals = alpha_s["tendency_mode"].values
    pm_vals = beta_s["predictor_mode"].values

    # ---- alpha: loop over tendency_mode ----
    a_mean_list, a_lo_list, a_hi_list = [], [], []
    for tm in tm_vals:
        da = alpha_s.sel(tendency_mode=tm)  # (seed, space)
        mean_da, lo_da, hi_da = bootstrap_xr_over_seed_2d(
            da, n_boot=n_boot, rng=rng, ci_lo=ci_lo, ci_hi=ci_hi, chunk=chunk, seed_dim=seed_dim
        )
        a_mean_list.append(mean_da.assign_coords(tendency_mode=tm).expand_dims("tendency_mode"))
        a_lo_list.append(lo_da.assign_coords(tendency_mode=tm).expand_dims("tendency_mode"))
        a_hi_list.append(hi_da.assign_coords(tendency_mode=tm).expand_dims("tendency_mode"))

    alpha_mean = xr.concat(a_mean_list, dim="tendency_mode").unstack("space")
    alpha_lo   = xr.concat(a_lo_list,   dim="tendency_mode").unstack("space")
    alpha_hi   = xr.concat(a_hi_list,   dim="tendency_mode").unstack("space")

    # ---- beta: loop over (tendency_mode, predictor_mode) ----
    b_mean_chunks, b_lo_chunks, b_hi_chunks = [], [], []
    for tm in tm_vals:
        mean_rows, lo_rows, hi_rows = [], [], []
        for pm in pm_vals:
            da = beta_s.sel(tendency_mode=tm, predictor_mode=pm)  # (seed, space)
            mean_da, lo_da, hi_da = bootstrap_xr_over_seed_2d(
                da, n_boot=n_boot, rng=rng, ci_lo=ci_lo, ci_hi=ci_hi, chunk=chunk, seed_dim=seed_dim
            )
            mean_rows.append(mean_da.assign_coords(predictor_mode=pm).expand_dims("predictor_mode"))
            lo_rows.append(lo_da.assign_coords(predictor_mode=pm).expand_dims("predictor_mode"))
            hi_rows.append(hi_da.assign_coords(predictor_mode=pm).expand_dims("predictor_mode"))

        b_mean_tm = xr.concat(mean_rows, dim="predictor_mode").assign_coords(tendency_mode=tm).expand_dims("tendency_mode")
        b_lo_tm   = xr.concat(lo_rows,   dim="predictor_mode").assign_coords(tendency_mode=tm).expand_dims("tendency_mode")
        b_hi_tm   = xr.concat(hi_rows,   dim="predictor_mode").assign_coords(tendency_mode=tm).expand_dims("tendency_mode")

        b_mean_chunks.append(b_mean_tm)
        b_lo_chunks.append(b_lo_tm)
        b_hi_chunks.append(b_hi_tm)

    beta_mean = xr.concat(b_mean_chunks, dim="tendency_mode").unstack("space")
    beta_lo   = xr.concat(b_lo_chunks,   dim="tendency_mode").unstack("space")
    beta_hi   = xr.concat(b_hi_chunks,   dim="tendency_mode").unstack("space")

    # ---- r2: loop over tendency_mode ----
    r_mean_list, r_lo_list, r_hi_list = [], [], []
    for tm in tm_vals:
        da = r2_s.sel(tendency_mode=tm)  # (seed, space)
        mean_da, lo_da, hi_da = bootstrap_xr_over_seed_2d(
            da, n_boot=n_boot, rng=rng, ci_lo=ci_lo, ci_hi=ci_hi, chunk=chunk, seed_dim=seed_dim
        )
        r_mean_list.append(mean_da.assign_coords(tendency_mode=tm).expand_dims("tendency_mode"))
        r_lo_list.append(lo_da.assign_coords(tendency_mode=tm).expand_dims("tendency_mode"))
        r_hi_list.append(hi_da.assign_coords(tendency_mode=tm).expand_dims("tendency_mode"))

    r2_mean = xr.concat(r_mean_list, dim="tendency_mode").unstack("space")
    r2_lo   = xr.concat(r_lo_list,   dim="tendency_mode").unstack("space")
    r2_hi   = xr.concat(r_hi_list,   dim="tendency_mode").unstack("space")

    return alpha_mean, alpha_lo, alpha_hi, beta_mean, beta_lo, beta_hi, r2_mean, r2_lo, r2_hi

# -----------------------------
# Load/anomalize SST
# -----------------------------
ds = xr.open_dataset(ERA5_SST_PATH)
sst = ds.sst
if "latitude" in sst.dims:
    sst = sst.rename({"latitude": "lat"})
if "longitude" in sst.dims:
    sst = sst.rename({"longitude": "lon"})

ssta, *_ = kgae.global_detrend(sst, deg=2)
sst_anom, _monthly_clim = kgae.remove_climo(ssta)

sst_stack = build_feature_template(sst_anom)
template_feature = sst_stack["feature"]
time_coord = sst_stack["time"]

# land/invalid mask for plotting
valid_mask = xr.where(np.isfinite(sst_anom.mean("time")), 1.0, np.nan)

print("SST stack:", sst_stack.shape, "n_feature=", sst_stack.sizes["feature"])

# -----------------------------
# Loop over models -> per seed/split outputs
# -----------------------------
paths = sorted(glob.glob(MODEL_GLOB))
if not paths:
    raise FileNotFoundError(f"No models found for MODEL_GLOB={MODEL_GLOB}")
print(f"Found {len(paths)} models")

# store: dict[seed] -> list of per-split DataArrays
per_seed_alpha = {}
per_seed_beta  = {}
per_seed_r2    = {}

pairs, pred_labels = unique_pair_labels(LATENT_MODES_FOR_PREDICTORS)
print("Predictor terms:", pred_labels)

for p in paths[:1]:
    seed, split = parse_seed_split(p)
    print("Loading", p)
    with open(p, "rb") as f:
        model = pickle.load(f)
    model.eval()
    try:
        model.to("cpu")
    except Exception:
        pass

    # --- encode latents (time, L) ---
    with torch.no_grad():
        x = torch.tensor(sst_stack.values, dtype=torch.float32)
        lat = model.encoder(x)
        if getattr(model, "is_variational", False):
            lat = lat[:, :model.latent_dim]
    lat = lat.detach().cpu().numpy()

    latents = xr.DataArray(
        lat,
        coords={"time": time_coord.values, "mode": np.arange(1, model.latent_dim + 1)},
        dims=("time", "mode"),
    ).sortby("time")

    # apply sign convention
    latents = latents * model.flip_signs

    # Scale by sigma.
    sig = latents.std("time")
    mu = latents.mean("time")
    print(mu)
    # (optional) standardize: 
    latents = (latents - mu) / sig

    # --- compute tendencies = dD/dz via kgae.jacobian ---
    tens = kgae.jacobian(model, torch.tensor(sst_stack.values, dtype=torch.float32))
    tens = xr.DataArray(
        tens.detach().cpu().numpy(),
        coords={
            "time": time_coord.values,
            "mode": np.arange(1, model.latent_dim + 1),
            "feature": template_feature,
        },
        dims=("time", "mode", "feature"),
    ).sortby("time")


    # restrict tendency modes (target)
    T = tens.unstack("feature").sortby(["time", "lat", "lon"]).sel(mode=TENDENCY_MODES)

    for mo in TENDENCY_MODES: 
        # --- build predictors: quadratic terms among chosen latent modes ---
        pred_terms = []
        pred_labels= []
        for (i, j) in pairs:
            zi = latents.sel(mode=i, drop=True)
            if i == j: 
                pred_terms.append(zi**2)
                pred_terms.append(zi)
                pred_labels.append(f"z{i}z{i}")
                pred_labels.append(f"z{i}")

            else:
                zj = latents.sel(mode=j, drop=True)
                if f"z{i}z{j}" not in pred_labels and f"z{j}z{i}" not in pred_labels:
                    pred_terms.append(zi * zj)
                    pred_labels.append(f"z{i}z{j}")

        predictors = xr.concat(pred_terms, dim="mode").assign_coords(mode=pred_labels).transpose("time", "mode")
        #other_mode = TENDENCY_MODES[
        #                    TENDENCY_MODES.index([_ for _ in TENDENCY_MODES if _ != mo][0])
        #            ]
        #predictors = xr.concat(
        #    [ 
        #        latents.sel(mode=mo, drop=True)**2, 
        #        latents.sel(mode=mo, drop=True) * \
        #            latents.sel(mode=other_mode, drop=True),
        #        latents.sel(mode=other_mode, drop=True) **2 
        #    ],
        #    'mode'
        #).assign_coords({'mode': [f'z{mo}^2', f'z{mo}z{other_mode}', f'z{other_mode}^2']}).transpose('time', 'mode')

        #predictors = (latents.sel(mode=LATENT_MODES_FOR_PREDICTORS)**2).assign_coords({'mode': [f"z{m}" for m in LATENT_MODES_FOR_PREDICTORS]})
        predictors = predictors - predictors.mean('time')
        
        # --- regress (uses all predictors provided) ---
        alpha, beta, r2, *_ = kgae.regress_tendencies_r2(predictors, T)



    # collect
    per_seed_alpha.setdefault(seed, []).append(alpha)
    per_seed_beta.setdefault(seed, []).append(beta)
    per_seed_r2.setdefault(seed, []).append(r2)

# -----------------------------
# Split-mean within each seed
# -----------------------------
seed_list = sorted(per_seed_alpha.keys())
print("Seeds:", seed_list)

alpha_seed = xr.concat([xr.concat(per_seed_alpha[s], dim="split").mean("split") for s in seed_list], dim="seed")
beta_seed  = xr.concat([xr.concat(per_seed_beta[s],  dim="split").mean("split") for s in seed_list], dim="seed")
r2_seed    = xr.concat([xr.concat(per_seed_r2[s],    dim="split").mean("split") for s in seed_list], dim="seed")

alpha_seed = alpha_seed.assign_coords(seed=seed_list)
beta_seed  = beta_seed.assign_coords(seed=seed_list)
r2_seed    = r2_seed.assign_coords(seed=seed_list)
alpha_seed.to_netcdf('alpha_seed.nc')
beta_seed.to_netcdf('beta_seed.nc')
r2_seed.to_netcdf('r2_seed.nc')

# -----------------------------
# Bootstrap across seeds -> mean/CI/masks
# -----------------------------
rng = np.random.default_rng(RNG_SEED)

alpha_mean, alpha_lo, alpha_hi, beta_mean, beta_lo, beta_hi, r2_mean, r2_lo, r2_hi = \
    bootstrap_alpha_beta_r2_sliced(
        alpha_seed, beta_seed, r2_seed,
        n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI, chunk=10
    )

alpha_sig = (alpha_lo > 0) | (alpha_hi < 0)
beta_sig  = (beta_lo  > 0) | (beta_hi  < 0)


if MASK_R2_BY_CI_POSITIVE:
    r2_sig = (r2_lo > 0)
else:
    r2_sig = xr.ones_like(r2_mean, dtype=bool)

# apply valid ocean mask
alpha_mean = alpha_mean * valid_mask
beta_mean  = beta_mean * valid_mask
r2_mean    = r2_mean * valid_mask

alpha_sig = alpha_sig.where(np.isfinite(valid_mask))
beta_sig  = beta_sig.where(np.isfinite(valid_mask))
r2_sig    = r2_sig.where(np.isfinite(valid_mask))

# -----------------------------
# Plot R^2 (masked by r2_sig if enabled)
# -----------------------------
proj = ccrs.PlateCarree(central_longitude=CENTRAL_LON)
pc   = ccrs.PlateCarree(central_longitude=CENTRAL_LON)
print(r2_mean.mean(['lat', 'lon']))
p = (r2_mean.where(r2_sig)).plot(
    col="tendency_mode", col_wrap=3, cmap="viridis",
    subplot_kws={"projection": proj},
    levels=R2_LEVELS, add_colorbar=True
)
for ax in p.axes.flatten():
    ax.coastlines()
plt.suptitle("Bootstrapped seed-mean $R^2$ (masked if enabled)", y=1.02)
plt.show()

# -----------------------------
# Plot alpha + selected beta rows per tendency column (only predictors involving that tendency)
# Use mode-label matching logic.
# -----------------------------
pred_names = beta_mean["predictor_mode"].values
tm_labels  = beta_mean["tendency_mode"].values
n_tm = beta_mean.sizes["tendency_mode"]

pred_idx_by_tm = []
max_rows = 0
for tm in range(n_tm):
    idx = [i for i, pn in enumerate(pred_names)]# if match_pred_to_tendency(pn, tm_labels[tm])]
    pred_idx_by_tm.append(idx)
    max_rows = max(max_rows, len(idx))
if max_rows == 0:
    raise ValueError("No predictors matched any tendency mode. Check predictor_mode labels (need z<id> tokens).")
fig = plt.figure(figsize=(4.0 * n_tm, 2.7 * (max_rows + 1)))
axes = fig.subplots(nrows=max_rows + 1, ncols=n_tm, subplot_kw={"projection": proj})
axes = np.atleast_2d(axes)

# ---------- ALPHA ROW ----------
m_alpha = None

for tm in range(n_tm):
    ax = axes[0, tm]

    da = alpha_mean.isel(tendency_mode=tm)
    sig = alpha_sig.isel(tendency_mode=tm)

    vmax = float(np.nanmax(np.abs(da)))
    fill_levels = np.linspace(-vmax, vmax, 21)
    line_levels = np.linspace(-vmax, vmax, 7)

    # filled significant
    m = ax.contourf(
        da.lon, da.lat, da.where(sig),
        levels=fill_levels,
        cmap=CMAP,
        transform=pc
    )

    # unfilled contours
    ax.contour(
        da.lon, da.lat, da,
        levels=line_levels[line_levels != 0],
        colors="k",
        linewidths=0.7,
        linestyles=["dashed" if lev < 0 else "solid" for lev in line_levels if lev != 0],
        transform=pc
    )

    # zero contour
    ax.contour(
        da.lon, da.lat, da,
        levels=[0],
        colors="k",
        linewidths=1.2,
        transform=pc
    )

    if m_alpha is None:
        m_alpha = m

    ax.coastlines()
    ax.set_title(f"alpha: tm={tm_labels[tm]}")

# colorbar aligned to top row
cb1 = fig.colorbar(
    m_alpha,
    ax=axes[0, :],
    orientation="vertical",
    fraction=0.035,
    pad=0.02
)
cb1.set_label("alpha")


# ---------- BETA ROWS ----------
m_beta = None

for tm in range(n_tm):
    idxs = pred_idx_by_tm[tm]

    for r in range(max_rows):
        ax = axes[r + 1, tm]

        if r >= len(idxs):
            ax.set_axis_off()
            continue

        pm = idxs[r]

        da = beta_mean.isel(tendency_mode=tm, predictor_mode=pm)
        sig = beta_sig.isel(tendency_mode=tm, predictor_mode=pm)

        vmax = float(np.nanmax(np.abs(da)))
        fill_levels = np.linspace(-vmax, vmax, 21)
        line_levels = np.linspace(-vmax, vmax, 7)

        # filled significant
        m = ax.contourf(
            da.lon, da.lat, da.where(sig),
            levels=fill_levels,
            cmap=CMAP,
            transform=pc
        )

        # line contours
        ax.contour(
            da.lon, da.lat, da,
            levels=line_levels[line_levels != 0],
            colors="k",
            linewidths=0.7,
            linestyles=["dashed" if lev < 0 else "solid" for lev in line_levels if lev != 0],
            transform=pc
        )

        # zero contour
        ax.contour(
            da.lon, da.lat, da,
            levels=[0],
            colors="k",
            linewidths=1.2,
            transform=pc
        )

        if m_beta is None:
            m_beta = m

        ax.coastlines()
        ax.set_title(str(pred_names[pm]), fontsize=9)

# colorbar aligned to bottom rows
cb2 = fig.colorbar(
    m_beta,
    ax=axes[1:, :],
    orientation="vertical",
    fraction=0.035,
    pad=0.02
)
cb2.set_label("beta")

#plt.tight_layout()
plt.show()
