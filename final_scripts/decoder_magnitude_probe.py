#!/usr/bin/env python3
"""
Two-panel decoder probe:

LEFT:  linear response (w.r.t. DEC) of the nonlinear response (even curvature) to IA
    S_{IA<-DEC} = [ N_IA(z_DEC=+A_DEC) - N_IA(z_DEC=-A_DEC) ] / (2 A_DEC)

RIGHT: linear response (w.r.t. IA) of the nonlinear response (even curvature) to DEC
    S_{DEC<-IA} = [ N_DEC(z_IA=+A_IA) - N_DEC(z_IA=-A_IA) ] / (2 A_IA)

Where
    N_i(z_bg) = [ D(z_bg, z_i=+A_i) + D(z_bg, z_i=-A_i) - 2 D(z_bg, z_i=0) ] / (A_i^2)

All other latents set to 0.

Bootstrap:
- Load all CV models seed*/split*/model.pkl
- Split-mean within each seed
- i.i.d. bootstrap across seeds to form CI
- Filled contours only where CI excludes 0; black contours everywhere (solid +, dashed -)
- Separate colorbar for each panel
"""

import glob
import pickle
import numpy as np
import xarray as xr
import torch

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -----------------------------
# USER SETTINGS
# -----------------------------
MODEL_GLOB = "~/Desktop/kgae/final_scripts/large-ensemble-2-1940-2014/seed*/split*/model.pkl"

ERA5_SST_PATH = "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"
TRAINING_PERIOD = ("1940-01-01", "2014-12-31")

LATENT_DIM = 5

# Mode labels (1-based)
MODE_DEC = 5          # "Decadal"
MODE_IA  = 4          # "Interannual"
MODE_QB = 3  # Quasibiennial mode

# Amplitudes in latent units
A_DEC = 1.0
A_IA  = 1.0
A_QB = 1.0

# Bootstrap across seeds
N_BOOT = 1000
RNG_SEED = 123
CI_LO, CI_HI = 2.5, 97.5

# Plot
central_longitude = 180
cmap = "RdBu_r"
ADD_STATES = False


# -----------------------------
# Small plot helpers
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

def parse_seed_split(p: str) -> tuple[int, int]:
    parts = str(p).split("/")
    seed = int([s for s in parts if s.startswith("seed")][0].replace("seed", ""))
    split = int([s for s in parts if s.startswith("split")][0].replace("split", ""))
    return seed, split


# -----------------------------
# Feature template (stack/unstack)
# -----------------------------
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


# -----------------------------
# Decoder probing core
# -----------------------------
def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

@torch.no_grad()
def decode_model(model, z_np: np.ndarray) -> np.ndarray:
    """
    Decode latent vector z_np (LATENT_DIM,) -> flat numpy (n_feature,).
    Applies model.flip_signs internally (assumes z_np is in reference orientation).
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

def nonlinear_response_target_given_bg(model, *, i_target: int, j_bg: int, z_bg: float,
                                      a_target: float) -> np.ndarray:
    """
    N_i(z_bg) = [D(z_bg, +a_i) + D(z_bg, -a_i) - 2 D(z_bg, 0)] / a_i^2
    All other latents = 0.
    i_target, j_bg are 0-based indices.
    """
    z0 = np.zeros(LATENT_DIM, dtype=np.float32)
    z0[j_bg] = z_bg
    z0[i_target] = 0.0

    zp = np.zeros(LATENT_DIM, dtype=np.float32)
    zp[j_bg] = z_bg
    zp[i_target] = +a_target

    zm = np.zeros(LATENT_DIM, dtype=np.float32)
    zm[j_bg] = z_bg
    zm[i_target] = -a_target

    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a_target * a_target)

def linear_response_of_nonlinear(model, *, i_target: int, j_bg: int, a_target: float, a_bg: float) -> np.ndarray:
    """
    S_{i<-j} = [N_i(+a_bg) - N_i(-a_bg)] / (2 a_bg)
    """
    Np = nonlinear_response_target_given_bg(model, i_target=i_target, j_bg=j_bg, z_bg=+a_bg, a_target=a_target)
    Nm = nonlinear_response_target_given_bg(model, i_target=i_target, j_bg=j_bg, z_bg=-a_bg, a_target=a_target)
    return (Np - Nm) / (2.0 * a_bg)


# -----------------------------
# Bootstrap across seeds (i.i.d.)
# -----------------------------
def bootstrap_ci_seed(members: np.ndarray, n_boot: int, rng: np.random.Generator,
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

def _nice_levels_pair(da_left, da_right, n_levels=21, robust=0.99):
    v = np.concatenate([
        da_left.values[np.isfinite(da_left.values)],
        da_right.values[np.isfinite(da_right.values)],
    ])
    if v.size == 0:
        return np.linspace(-1, 1, n_levels), np.linspace(-1, 1, 7)
    vmax = float(np.nanquantile(np.abs(v), robust))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else 1.0
        if vmax == 0:
            vmax = 1.0
    return np.linspace(-vmax, vmax, n_levels), np.linspace(-vmax, vmax, 7)


# -----------------------------
# Main
# -----------------------------
def main():
    rng = np.random.default_rng(RNG_SEED)

    template = build_feature_template(ERA5_SST_PATH, TRAINING_PERIOD)
    n_feat = template.sizes["feature"]
    print(f"Feature template built: n_feature={n_feat}")

    paths = sorted(glob.glob(MODEL_GLOB))
    if not paths:
        raise FileNotFoundError(f"No models found for MODEL_GLOB={MODEL_GLOB}")
    print(f"Found {len(paths)} model files.")

    # 0-based indices
    i_dec = int(MODE_DEC) - 1
    i_ia  = int(MODE_IA)  - 1
    i_qb = int(MODE_QB) - 1

    # per_seed maps: seed -> list(split_maps)
    per_seed_left = {}   # S_{IA<-DEC}
    per_seed_right = {}  # S_{DEC<-IA}
    per_seed_third = {}

    # Loop over all models
    for p in paths:
        seed_idx, split_idx = parse_seed_split(p)
        print("Loading:", p)

        with open(p, "rb") as f:
            model = pickle.load(f)
        model.eval()
        try:
            model.to(torch.device("cpu"))
        except Exception:
            pass

        # feature-length check
        test = decode_model(model, np.zeros(LATENT_DIM, dtype=np.float32))
        if test.shape[0] != n_feat:
            raise ValueError(
                f"Decoded length {test.shape[0]} != n_feature {n_feat}. "
                "ERA5 feature mask/training domain mismatch for this model."
            )

        left_map = linear_response_of_nonlinear(model, i_target=i_ia, j_bg=i_dec, a_target=A_IA, a_bg=A_DEC)
        right_map = linear_response_of_nonlinear(model, i_target=i_dec, j_bg=i_ia, a_target=A_DEC, a_bg=A_IA)

        third_map = linear_response_of_nonlinear(
            model,
            i_target=i_ia,
            j_bg=i_qb,
            a_target=A_IA,
            a_bg=A_QB
        )

        per_seed_third.setdefault(seed_idx, []).append(third_map)
        per_seed_left.setdefault(seed_idx, []).append(left_map)
        per_seed_right.setdefault(seed_idx, []).append(right_map)

    # common seeds
    seed_list = sorted(set(per_seed_left.keys()) & set(per_seed_right.keys()))
    print(f"Seeds represented in both panels: {len(seed_list)}")
    if len(seed_list) == 0:
        raise RuntimeError("No common seeds across panels.")

    # split-mean within seed => members arrays (n_seed, n_feature)
    left_members = np.stack([np.stack(per_seed_left[s], axis=0).mean(axis=0) for s in seed_list], axis=0)
    right_members = np.stack([np.stack(per_seed_right[s], axis=0).mean(axis=0) for s in seed_list], axis=0)
    third_members = np.stack([np.stack(per_seed_third[s], axis=0).mean(axis=0) for s in seed_list], axis=0)

    # bootstrap => mean + sig
    left_mean, left_lo, left_hi = bootstrap_ci_seed(left_members, n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)
    right_mean, right_lo, right_hi = bootstrap_ci_seed(right_members, n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)
    third_mean, third_lo, third_hi = bootstrap_ci_seed(third_members, n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)

    left_da = xr.DataArray(left_mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
    left_sig = ((xr.DataArray(left_lo, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature") > 0) |
                (xr.DataArray(left_hi, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature") < 0))

    right_da = xr.DataArray(right_mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
    right_sig = ((xr.DataArray(right_lo, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature") > 0) |
                 (xr.DataArray(right_hi, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature") < 0))

    third_da = xr.DataArray(third_mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
    third_sig = ((xr.DataArray(third_lo, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature") > 0) |
                 (xr.DataArray(third_hi, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature") < 0))


    # levels (per panel, but chosen consistently using pooled values)
    vals = np.concatenate([
        left_da.values.flatten(),
        right_da.values.flatten(),
        third_da.values.flatten()
    ])

    vmax = np.nanquantile(np.abs(vals),0.99)
    levels = np.linspace(-vmax,vmax,21)
    levels2 = np.linspace(-vmax,vmax,7)

        # -----------------------------
    # Plot
    # -----------------------------
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    pc   = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(12,3.8))
    n=10
    gs = fig.add_gridspec(
        n, 6,
        #height_ratios=[1,1,1,1, 0.08]   # bottom row reserved for colorbar
    )

    ax1 = fig.add_subplot(gs[:-1,:2], projection=proj)
    ax2 = fig.add_subplot(gs[:-1,2:4], projection=proj)
    ax3 = fig.add_subplot(gs[:-1,4:], projection=proj)

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.7)

    panels = [
        (left_da,left_sig,r"$\text{Decadal } \rightarrow \text{ Interannual}$"),
        (right_da,right_sig,r"$\text{Interannual } \rightarrow \text{ Decadal}$"),
        (third_da,third_sig,r"$\text{Quasibiennial }\rightarrow \text{ Interannual}$"),
    ]

    for ax,(da,sig,title) in zip(axes,panels):

        cf = ax.contourf(
            da.lon,da.lat,
            da.where(sig),
            levels=levels,
            cmap=cmap,
            transform=pc,
            extend="both"
        )

        contour_sign(ax,da,levels2,pc,colors="k")

        ax.set_title(title)

    cax = fig.add_subplot(gs[-1,1:5])
    pos = cax.get_position()

    cax.set_position([
        pos.x0,
        pos.y0+0.1,
        pos.width,
        pos.height * 0.5
    ])

    cbar = fig.colorbar(
        cf,
        cax=cax,
        orientation="horizontal",
        aspect=200
    )

    cbar.set_label(r"Cross-mode nonlinear sensitivity ($K\ z_i^{-2}z^{-1}_j$)")

    labels = ["a)", "b)", "c)"]

    for ax, lab in zip(axes, labels):
        ax.text(
            0.0, 1.12,
            lab,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold"
        )

    #plt.tight_layout()
    out = "decoder_probe_crossmode.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()