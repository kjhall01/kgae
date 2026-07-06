#!/usr/bin/env python3
"""
Quick sanity-figure: plot the *raw decoded fields* at z=+a and z=-a for each mode.

For each mode (columns): two panels
  top:  D(+a e_mode)
  bottom: D(-a e_mode)

- Loads all CV models (seed*/split*/model.pkl).
- For each seed: average across splits (split-mean).
- Then average across seeds (simple mean) to get one representative field.
  (No bootstrap/significance here — this is just for visual inspection.)
- One shared colorbar per column (optional toggle), symmetric levels.

Output: decoder_probe_raw_zplus_zminus.png
"""

import glob
import pickle
import numpy as np
import xarray as xr
import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# -----------------------------
# USER SETTINGS
# -----------------------------
MODEL_GLOB = "/Users/kylehall/Desktop/kgae/final_scripts/large-ensemble-2-1940-2014/seed*/split*/model.pkl"

ERA5_SST_PATH = "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"
TRAINING_PERIOD = ("1940-01-01", "2014-12-31")

LATENT_DIM = 5

modes = [5, 4, 3]  # Decadal, Interannual, Quasibiennial
mode_titles = ["Decadal", "Interannual", "Quasibiennial"]

A_STEP = 1.0

central_longitude = 180
cmap = "RdBu_r"
ADD_STATES = False

# Plot levels: if None, auto from robust quantile
LEVELS = None          # e.g. np.linspace(-1.0, 1.0, 21)
ROBUST_Q = 0.99        # used if LEVELS is None

# Colorbar behavior
ONE_COLORBAR_PER_COL = True   # if False: one shared bar for whole figure


# -----------------------------
# Helpers
# -----------------------------
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

def robust_levels_from_fields(fields, q=0.99, n=21):
    vals = []
    for da in fields:
        v = da.values
        vals.append(v[np.isfinite(v)])
    v = np.concatenate(vals) if vals else np.array([])
    if v.size == 0:
        return np.linspace(-1, 1, n)
    vmax = float(np.nanquantile(np.abs(v), q))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else 1.0
        if vmax == 0:
            vmax = 1.0
    return np.linspace(-vmax, vmax, n)


# -----------------------------
# Main
# -----------------------------
def main():
    template = build_feature_template(ERA5_SST_PATH, TRAINING_PERIOD)
    n_feat = template.sizes["feature"]
    print(f"Template: n_feature={n_feat}")

    paths = sorted(glob.glob(MODEL_GLOB))
    if not paths:
        raise FileNotFoundError(f"No models found for MODEL_GLOB={MODEL_GLOB}")
    print(f"Found {len(paths)} model files.")

    # per_seed[m]["plus"|"minus"] = list of (n_feat,) over splits
    per_seed = {m: {"plus": {}, "minus": {}} for m in modes}

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

        # feature length check
        test = decode_model(model, np.zeros(LATENT_DIM, dtype=np.float32))
        if test.shape[0] != n_feat:
            raise ValueError(
                f"Decoded length {test.shape[0]} != template n_feature {n_feat}. "
                "ERA5 feature mask/domain mismatch for this model."
            )

        for m in modes:
            j0 = int(m) - 1

            zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[j0] = +A_STEP
            zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[j0] = -A_STEP

            x_plus = decode_model(model, zp*0)
            x_minus = decode_model(model, zm*0)

            per_seed[m]["plus"].setdefault(seed_idx, []).append(x_plus)
            per_seed[m]["minus"].setdefault(seed_idx, []).append(x_minus)

    # seeds represented (intersection across modes)
    seed_sets = []
    for m in modes:
        seed_sets.append(set(per_seed[m]["plus"].keys()) & set(per_seed[m]["minus"].keys()))
    seed_list = sorted(set.intersection(*seed_sets))
    print(f"Seeds represented (all modes): {len(seed_list)}")
    if len(seed_list) == 0:
        raise RuntimeError("No common seeds across modes.")

    # fold-mean within seed then mean across seeds => one field per (mode, sign)
    plus_fields = []
    minus_fields = []
    plus_da = []
    minus_da = []

    for m in modes:
        plus_members = np.stack([np.stack(per_seed[m]["plus"][s], axis=0).mean(axis=0) for s in seed_list], axis=0)
        minus_members = np.stack([np.stack(per_seed[m]["minus"][s], axis=0).mean(axis=0) for s in seed_list], axis=0)

        x_plus_mean = plus_members.mean(axis=0)   # mean across seeds
        x_minus_mean = minus_members.mean(axis=0)

        da_p = xr.DataArray(x_plus_mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_m = xr.DataArray(x_minus_mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")

        plus_da.append(da_p)
        minus_da.append(da_m)
        plus_fields.append(da_p)
        minus_fields.append(da_m)

    # shared contour levels
    if LEVELS is None:
        lev = robust_levels_from_fields(plus_fields + minus_fields, q=ROBUST_Q, n=21)
    else:
        lev = LEVELS

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    pc = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(3.6 * len(modes), 6.0))
    gs = gridspec.GridSpec(2, len(modes), hspace=0.10, wspace=0.05)

    axes = [[None]*len(modes) for _ in range(2)]
    for c in range(len(modes)):
        axes[0][c] = fig.add_subplot(gs[0, c], projection=proj)
        axes[1][c] = fig.add_subplot(gs[1, c], projection=proj)

    for r in range(2):
        for c in range(len(modes)):
            ax = axes[r][c]
            ax.coastlines(linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.7)
            if ADD_STATES:
                ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.6)

    # titles
    for c, title in enumerate(mode_titles):
        axes[0][c].set_title(title)

    # row labels (left)
    axes[0][0].text(-0.08, 0.5, rf"$D(+{A_STEP}\,e_j)$",
                    transform=axes[0][0].transAxes, ha="right", va="center",
                    rotation=90, fontsize=11)
    axes[1][0].text(-0.08, 0.5, rf"$D(-{A_STEP}\,e_j)$",
                    transform=axes[1][0].transAxes, ha="right", va="center",
                    rotation=90, fontsize=11)

    # plot
    mappables = [[None]*len(modes) for _ in range(2)]
    for c in range(len(modes)):
        cf1 = axes[0][c].contourf(plus_da[c]["lon"], plus_da[c]["lat"], plus_da[c],
                                 levels=lev, transform=pc, extend="both", cmap=cmap)
        cf2 = axes[1][c].contourf(minus_da[c]["lon"], minus_da[c]["lat"], minus_da[c],
                                 levels=lev, transform=pc, extend="both", cmap=cmap)
        mappables[0][c] = cf1
        mappables[1][c] = cf2

    # colorbars
    if ONE_COLORBAR_PER_COL:
        for c in range(len(modes)):
            cb = fig.colorbar(mappables[0][c], ax=[axes[0][c], axes[1][c]],
                              orientation="vertical", fraction=0.040, pad=0.02)
            cb.set_label("K")
    else:
        cb = fig.colorbar(mappables[0][0], ax=[axes[r][c] for r in range(2) for c in range(len(modes))],
                          orientation="vertical", fraction=0.030, pad=0.02)
        cb.set_label("K")

    out = "decoder_probe_raw_zplus_zminus.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()