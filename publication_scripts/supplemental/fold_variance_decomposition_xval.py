#!/usr/bin/env python3
"""
Fold-stability diagnostics using *decoder-probe* quantities, with NO full-models.

Reference pattern:
  For each seed and quantity, the reference is the mean across all 5 folds:
      X_ref(seed, quantity) = mean_fold X(seed, fold, quantity)

Then for each fold we compare the *ensemble-mean fold map* to the *ensemble-mean reference map*
(using the same bootstrap-resampled seeds) and compute pattern correlation.

This matches your previous workflow, but replaces "full model" with "CV fold-mean reference".

Outputs:
  - A multi-panel figure of bootstrap pattern-correlation distributions by fold, for each quantity.

Quantities computed (maps):
  - lin_dec, lin_ia, lin_qb                 [K / z]
  - curv_dec, curv_ia                       [K / z^2]
  - mod_dec_to_ia, mod_ia_to_dec, mod_qb_to_ia [K / z^3]
"""

import glob
import pickle
from collections import defaultdict

import numpy as np
import xarray as xr
import torch

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# -----------------------------
# USER SETTINGS
# -----------------------------
EXPERIMENT = "large-ensemble-2-1940-2014"
MODEL_GLOB_CV = f"/Users/kylehall/Desktop/kgae/final_scripts/{EXPERIMENT}/seed*/split*/model.pkl"

ERA5_SST_PATH = "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"
TRAINING_PERIOD = ("1940-01-01", "2014-12-31")

LATENT_DIM = 5
MODE_DEC, MODE_IA, MODE_QB = 5, 4, 3

# Probe step sizes (latent units)
A_STEP = 1.0

# Bootstrap across seeds
N_BOOTSTRAPS = 1000
RNG_SEED = 0

# Plot/KDE settings
BINS = "fd"
KDE_BW = "scott"
HIST_ALPHA = 0.25
KDE_LW = 2.0
PCOR_XLIM = (0.0, 1.0)


# ============================================================
# Plot helper (your original, lightly simplified)
# ============================================================
def plot_fold_boot_distributions(
    da: xr.DataArray,
    ax=None,
    boot_dim: str = "boot",
    fold_dim: str = "fold",
    bins: int | str = "fd",
    kde_points: int = 400,
    kde_bw: float | str = "scott",
    hist_alpha: float = 0.25,
    kde_lw: float = 2.0,
    clip_to_data: bool = True,
):
    """
    Overlay per-fold bootstrap distributions along `boot_dim`:
      - histogram (density=True)
      - KDE curve

    da must have dims (boot, fold) (order can be swapped).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if boot_dim not in da.dims or fold_dim not in da.dims:
        raise ValueError(f"da must have dims including {boot_dim!r} and {fold_dim!r}. Got {da.dims}.")

    x = da.transpose(fold_dim, boot_dim)

    all_vals = x.values.reshape(-1)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError("No finite values found in da.")

    xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    xgrid = np.linspace(xmin - pad, xmax + pad, kde_points)

    folds = x[fold_dim].values
    for f in folds:
        vals = x.sel({fold_dim: f}).values
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            continue

        ax.hist(vals, bins=bins, density=True, alpha=hist_alpha, label=f"fold {int(f)}")
        kde = gaussian_kde(vals, bw_method=kde_bw)
        y = kde(xgrid)
        if clip_to_data:
            y = np.where((xgrid >= vals.min()) & (xgrid <= vals.max()), y, np.nan)
        ax.plot(xgrid, y, lw=kde_lw)

    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    if PCOR_XLIM is not None:
        ax.set_xlim(*PCOR_XLIM)
    return fig, ax


# ============================================================
# Bootstrap: fold vs reference (seed-mean across folds)
# ============================================================
def bootstrap_fold_vs_ref_patcor(
    da_fold: xr.DataArray,
    da_ref: xr.DataArray,
    *,
    n_resamples: int = 1000,
    seed_dim: str = "seed",
    rng_seed: int = 0,
) -> xr.DataArray:
    """
    da_fold: (seed, fold, lat, lon)
    da_ref : (seed, lat, lon)   (here: mean across folds for each seed)

    Returns:
      patcor_boot: (boot, fold)
    computed as corr( mean_seed(da_fold[idx]), mean_seed(da_ref[idx]) ) over lat/lon.
    """
    rng = np.random.default_rng(rng_seed)
    S = da_fold.sizes[seed_dim]

    cors = []
    for b in range(n_resamples):
        if b % 50 == 0:
            print("boot", b, "/", n_resamples)
        idx = rng.integers(0, S, size=S)  # resample seeds with replacement

        fold_mean = da_fold.isel({seed_dim: idx}).mean(seed_dim)  # (fold, lat, lon)
        ref_mean  = da_ref.isel({seed_dim: idx}).mean(seed_dim)   # (lat, lon)

        cor = xr.corr(fold_mean, ref_mean, dim=["lat", "lon"])    # (fold,)
        cors.append(cor)

    return xr.concat(cors, dim="boot").assign_coords(boot=np.arange(n_resamples))


# ============================================================
# Decoder-probe utilities
# ============================================================
def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def decode_model(model, z_np: np.ndarray) -> np.ndarray:
    """
    Decode latent vector z_np (LATENT_DIM,) -> flat numpy (n_feature,).
    Applies model.flip_signs internally.
    """
    device = next(model.parameters()).device
    sgn = np.asarray(model.flip_signs.values, dtype=np.float32)  # (L,)
    z = _to_tensor((z_np * sgn)[None, :], device=device)

    if hasattr(model, "decode") and callable(getattr(model, "decode")):
        x = model.decode(z)
    elif hasattr(model, "decoder") and callable(getattr(model, "decoder")):
        x = model.decoder(z)
    else:
        raise AttributeError("Model has neither callable .decode nor callable .decoder")

    return x.detach().cpu().numpy().squeeze()


def linear_response_single_mode(model, *, latent_dim: int, i_mode0: int, a: float) -> np.ndarray:
    """[D(+a e_i) - D(-a e_i)] / (2a)  (K/z)"""
    zp = np.zeros(latent_dim, dtype=np.float32); zp[i_mode0] = +a
    zm = np.zeros(latent_dim, dtype=np.float32); zm[i_mode0] = -a
    return (decode_model(model, zp) - decode_model(model, zm)) / (2.0 * a)


def even_curvature_single_mode(model, *, latent_dim: int, i_mode0: int, a: float) -> np.ndarray:
    """[D(+a)+D(-a)-2D(0)] / a^2  (K/z^2)"""
    z0 = np.zeros(latent_dim, dtype=np.float32)
    zp = np.zeros(latent_dim, dtype=np.float32); zp[i_mode0] = +a
    zm = np.zeros(latent_dim, dtype=np.float32); zm[i_mode0] = -a
    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a * a)


def curvature_target_given_bg(
    model,
    *,
    latent_dim: int,
    i_target0: int,
    j_bg0: int,
    z_bg: float,
    a_target: float,
) -> np.ndarray:
    """
    curv_i(z_j=z_bg) = [D(z_bg,+a_i)+D(z_bg,-a_i)-2D(z_bg,0)] / a_i^2
    """
    z0 = np.zeros(latent_dim, dtype=np.float32)
    z0[j_bg0] = z_bg
    z0[i_target0] = 0.0

    zp = np.zeros(latent_dim, dtype=np.float32)
    zp[j_bg0] = z_bg
    zp[i_target0] = +a_target

    zm = np.zeros(latent_dim, dtype=np.float32)
    zm[j_bg0] = z_bg
    zm[i_target0] = -a_target

    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a_target * a_target)


def linear_response_of_curvature(
    model,
    *,
    latent_dim: int,
    i_target0: int,
    j_bg0: int,
    a_target: float,
    a_bg: float,
) -> np.ndarray:
    """
    S_{i<-j} = [curv_i(z_j=+a_bg) - curv_i(z_j=-a_bg)] / (2 a_bg)  (K/z^3)
    """
    cp = curvature_target_given_bg(
        model, latent_dim=latent_dim, i_target0=i_target0, j_bg0=j_bg0, z_bg=+a_bg, a_target=a_target
    )
    cm = curvature_target_given_bg(
        model, latent_dim=latent_dim, i_target0=i_target0, j_bg0=j_bg0, z_bg=-a_bg, a_target=a_target
    )
    return (cp - cm) / (2.0 * a_bg)


# ============================================================
# IO helpers
# ============================================================
def parse_seed_split(p: str) -> tuple[int, int]:
    parts = str(p).split("/")
    seed = int([s for s in parts if s.startswith("seed")][0].replace("seed", ""))
    split = int([s for s in parts if s.startswith("split")][0].replace("split", ""))
    return seed, split


def build_feature_template(era5_path: str, training_period: tuple[str, str]) -> xr.DataArray:
    sst = xr.open_dataset(era5_path).sst.sel(time=slice(training_period[0], training_period[1]))
    if "latitude" in sst.dims:
        sst = sst.rename({"latitude": "lat"})
    if "longitude" in sst.dims:
        sst = sst.rename({"longitude": "lon"})
    sst_feat = (
        sst.stack(feature=("lat", "lon"))
           .dropna("feature", how="any")
           .transpose("time", "feature")
    )
    feat = sst_feat["feature"]
    return xr.DataArray(
        np.zeros(feat.size, dtype=np.float32),
        dims=("feature",),
        coords={"feature": feat},
        name="template",
    )


# ============================================================
# Compute decoder-probe maps for CV models
# ============================================================
def compute_decoder_probe_maps_cv(
    model_glob: str,
    template: xr.DataArray,
    *,
    latent_dim: int,
    a_step: float,
    mode_dec: int,
    mode_ia: int,
    mode_qb: int,
) -> xr.DataArray:
    """
    Returns xr.DataArray with dims:
      (seed, fold, quantity, lat, lon)
    with fold = split index parsed from path.
    """
    paths = sorted(glob.glob(model_glob))
    if not paths:
        raise FileNotFoundError(f"No models found for model_glob={model_glob}")

    n_feat = template.sizes["feature"]
    i_dec0, i_ia0, i_qb0 = mode_dec - 1, mode_ia - 1, mode_qb - 1

    quantities = [
        "lin_dec", "lin_ia", "lin_qb",
        "curv_dec", "curv_ia",
        "mod_dec_to_ia", "mod_ia_to_dec", "mod_qb_to_ia",
    ]

    # store[(seed, fold)][quantity] -> list of flat maps (in case duplicates)
    store = defaultdict(lambda: defaultdict(list))

    for p in paths:
        seed, fold = parse_seed_split(p)

        with open(p, "rb") as f:
            model = pickle.load(f)
        model.eval()
        try:
            model.to(torch.device("cpu"))
        except Exception:
            pass

        test0 = decode_model(model, np.zeros(latent_dim, dtype=np.float32))
        if test0.shape[0] != n_feat:
            raise ValueError(
                f"Decoded length {test0.shape[0]} != n_feature {n_feat}. "
                "ERA5 feature mask/domain mismatch for this model."
            )

        q = {}
        q["lin_dec"] = linear_response_single_mode(model, latent_dim=latent_dim, i_mode0=i_dec0, a=a_step)
        q["lin_ia"]  = linear_response_single_mode(model, latent_dim=latent_dim, i_mode0=i_ia0,  a=a_step)
        q["lin_qb"]  = linear_response_single_mode(model, latent_dim=latent_dim, i_mode0=i_qb0,  a=a_step)

        q["curv_dec"] = even_curvature_single_mode(model, latent_dim=latent_dim, i_mode0=i_dec0, a=a_step)
        q["curv_ia"]  = even_curvature_single_mode(model, latent_dim=latent_dim, i_mode0=i_ia0,  a=a_step)

        q["mod_dec_to_ia"] = linear_response_of_curvature(
            model, latent_dim=latent_dim, i_target0=i_ia0, j_bg0=i_dec0, a_target=a_step, a_bg=a_step
        )
        q["mod_ia_to_dec"] = linear_response_of_curvature(
            model, latent_dim=latent_dim, i_target0=i_dec0, j_bg0=i_ia0, a_target=a_step, a_bg=a_step
        )
        q["mod_qb_to_ia"] = linear_response_of_curvature(
            model, latent_dim=latent_dim, i_target0=i_ia0, j_bg0=i_qb0, a_target=a_step, a_bg=a_step
        )

        for name in quantities:
            store[(seed, fold)][name].append(q[name])

    # assemble to xarray
    entries = []
    seeds = []
    folds = []

    for (seed, fold), qdict in store.items():
        qstack = []
        for name in quantities:
            arrs = qdict[name]
            qstack.append(np.stack(arrs, axis=0).mean(axis=0))  # (n_feat,)
        qstack = np.stack(qstack, axis=0)  # (quantity, n_feat)

        da = xr.DataArray(
            qstack,
            dims=("quantity", "feature"),
            coords={"quantity": quantities, "feature": template["feature"]},
        ).unstack("feature")  # -> (quantity, lat, lon)

        entries.append(da.transpose("quantity", "lat", "lon"))
        seeds.append(seed)
        folds.append(fold)

    out = xr.concat(entries, dim="entry").assign_coords(entry=np.arange(len(entries)))
    out = out.assign_coords(seed=("entry", seeds), fold=("entry", folds))
    out = out.set_index(entry=["seed", "fold"]).unstack("entry")  # -> seed, fold, quantity, lat, lon

    # sort for sanity
    out = out.sortby("seed").sortby("fold")
    return out


# ============================================================
# Main plotting logic
# ============================================================
def main():
    template = build_feature_template(ERA5_SST_PATH, TRAINING_PERIOD)

    print("Computing decoder-probe maps (CV)...")
    da_cv = compute_decoder_probe_maps_cv(
        MODEL_GLOB_CV, template,
        latent_dim=LATENT_DIM, a_step=A_STEP,
        mode_dec=MODE_DEC, mode_ia=MODE_IA, mode_qb=MODE_QB
    )
    print(da_cv)

    # Reference = mean across all folds for each seed+quantity
    da_ref = da_cv.mean("fold")  # (seed, quantity, lat, lon)

    # Grid of quantities to plot
    grid = [
        ["lin_dec", "lin_ia", "lin_qb"],
        ["curv_dec", "curv_ia", None],
        ["mod_dec_to_ia", "mod_ia_to_dec", "mod_qb_to_ia"],
    ]
    titles = {
        "lin_dec": "Linear: Decadal",
        "lin_ia":  "Linear: Interannual",
        "lin_qb":  "Linear: Quasibiennial",
        "curv_dec":"Even curvature: Decadal",
        "curv_ia": "Even curvature: Interannual",
        "mod_dec_to_ia": "Modulation: DEC → IA",
        "mod_ia_to_dec": "Modulation: IA → DEC",
        "mod_qb_to_ia":  "Modulation: QB → IA",
    }

    # Bootstrap pattern-correlation distributions (boot, fold) per quantity
    patcor_boot = {}
    for row in grid:
        for q in row:
            if q is None:
                continue
            print("\nBootstrapping:", q)
            pc = bootstrap_fold_vs_ref_patcor(
                da_fold=da_cv.sel(quantity=q),   # (seed, fold, lat, lon)
                da_ref=da_ref.sel(quantity=q),   # (seed, lat, lon)
                n_resamples=N_BOOTSTRAPS,
                rng_seed=RNG_SEED,
            )
            pc.name = q
            patcor_boot[q] = pc

    # Plot
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.3 * ncols, 2.7 * nrows),
        constrained_layout=False
    )
    axes = np.atleast_2d(axes)

    panel_letters = [f"{chr(ord('a') + i)})" for i in range(nrows * ncols)]
    pidx = 0

    for ri in range(nrows):
        for ci in range(ncols):
            ax = axes[ri, ci]
            q = grid[ri][ci]

            ax.text(
                0.0, 1.07, panel_letters[pidx],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2)
            )
            pidx += 1

            if q is None:
                ax.axis("off")
                continue

            plot_fold_boot_distributions(
                patcor_boot[q],
                ax=ax,
                bins=BINS,
                kde_bw=KDE_BW,
                hist_alpha=HIST_ALPHA,
                kde_lw=KDE_LW,
            )
            ax.set_title(titles.get(q, q))
            ax.set_xlabel("" if ri < nrows - 1 else "Pattern Correlation")
            ax.set_ylabel("density" if ci == 0 else "")

            # keep legends only on rightmost panel of each row
            if ci != ncols - 1:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()
            else:
                ax.legend(ncols=2, fontsize=9, frameon=False)

    plt.tight_layout()
    out = "fold-sensitivity-decoder-probes_refIsFoldMean.pdf"
    plt.savefig(out)
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()