#!/usr/bin/env python3
"""
Distribution-weighted (data-supported) decoder curvature probe (4x3) using xarray feature unstacking,
rewritten to use explicit "D(0)"-style second differences.

Interpretation:
- Row 1 (alpha-like):   E_z[ ∂D/∂z_j (z) ]  via central first difference in z_j around each sampled z.
- Rows 2–4 (beta-like): E_z[ ∂²D/∂z_j∂z_k (z) ] using a stencil that explicitly includes the "neutral"
  point in the *k-direction* (i.e., D(z ± a e_j, z_k=0)):

  For j != k (mixed partial):
      ∂²D/∂z_j∂z_k ≈ (1/(2a)) * { [D(z+a e_j + b e_k) - D(z+a e_j) - (D(z+a e_j) - D(z+a e_j - b e_k))]/b
                                 -[D(z-a e_j + b e_k) - D(z-a e_j) - (D(z-a e_j) - D(z-a e_j - b e_k))]/b }
                      which simplifies to:
      ∂²D/∂z_j∂z_k ≈ ( D(z+a e_j + b e_k) - 2 D(z+a e_j) + D(z+a e_j - b e_k)
                     -D(z-a e_j + b e_k) + 2 D(z-a e_j) - D(z-a e_j - b e_k) ) / (2 a b^2)

  For j == k (pure second derivative):
      ∂²D/∂z_j² ≈ ( D(z + a e_j) - 2 D(z) + D(z - a e_j) ) / a^2

Notes:
- This is *not* the minimal 4-corner mixed-partial stencil; it is a "D(0)-included" formulation.
- It should be numerically stable when b is small, but uses 6 decoder evals per batch for off-diagonals.
- If you want symmetric j<->k by construction, use the 4-corner mixed central difference instead.
"""

import glob
import pickle
import numpy as np
import torch
import xarray as xr
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
A = 1   # step for first-derivative direction (j)
B = 1   # step for second-difference direction (k) in the D(0) stencil

N_SAMPLES = 500
RANDOM_SEED = 0
BATCH = 256

# Mode indices (your current mapping)
MODE = {"QB": 2, "IA": 3, "DEC": 4}
COL_ORDER = ["DEC", "IA", "QB"]
ROW_CONTEXT = [None, "DEC", "IA", "QB"]

# Plot style
PROJ = ccrs.PlateCarree(central_longitude=180)
CMAP = "RdBu_r"
ROBUST = True
ROBUST_PCT = (2, 98)
ADD_STATES = False


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
# Decoder batch helpers
# -----------------------------
def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

@torch.no_grad()
def decode_batch(model, Z_np: np.ndarray) -> np.ndarray:
    """
    Decode a batch of latent vectors in *reference* coords Z_np, applying model.flip_signs internally
    to move into model coords.
    Z_np: (n, latent_dim)
    returns: (n, n_feature)
    """
    device = next(model.parameters()).device
    s = np.asarray(model.flip_signs.values, dtype=np.float32)  # (L,)
    Zt = _to_tensor(Z_np * s[None, :], device=device)

    if hasattr(model, "decode") and callable(getattr(model, "decode")):
        X = model.decode(Zt)
    elif hasattr(model, "decoder") and callable(getattr(model, "decoder")):
        X = model.decoder(Zt)
    else:
        raise AttributeError("Model has neither callable .decode nor callable .decoder")

    return X.detach().cpu().numpy()


# -----------------------------
# Finite differences (data-supported)
# -----------------------------
def alpha_like_Ez(model, Zs: np.ndarray, j: int, a: float, batch: int) -> np.ndarray:
    """
    E_z[ ∂D/∂z_j (z) ] via central first difference:
      (D(z + a e_j) - D(z - a e_j)) / (2a)
    """
    n = Zs.shape[0]
    ej = np.zeros((1, LATENT_DIM), dtype=np.float32)
    ej[0, j] = 1.0

    acc = None
    seen = 0

    for s in range(0, n, batch):
        Z0 = Zs[s:s+batch].astype(np.float32)
        Xp = decode_batch(model, Z0 + a*ej)
        Xm = decode_batch(model, Z0 - a*ej)

        local = (Xp - Xm) / (2.0 * a)
        mean_local = local.mean(axis=0)

        acc = mean_local if acc is None else (acc + mean_local)
        seen += 1

    return acc / seen


def beta_diag_Ez_D0(model, Zs: np.ndarray, j: int, a: float, batch: int) -> np.ndarray:
    """
    Diagonal curvature using explicit neutral:
      ∂²D/∂z_j² ≈ (D(z+a e_j) - 2D(z) + D(z-a e_j)) / a^2
    """
    n = Zs.shape[0]
    ej = np.zeros((1, LATENT_DIM), dtype=np.float32)
    ej[0, j] = 1.0

    acc = None
    seen = 0

    for s in range(0, n, batch):
        Z0 = Zs[s:s+batch].astype(np.float32)
        X0 = decode_batch(model, Z0)
        Xp = decode_batch(model, Z0 + a*ej)
        Xm = decode_batch(model, Z0 - a*ej)

        local = (Xp - 2.0*X0 + Xm) / (2*a)
        mean_local = local.mean(axis=0)

        acc = mean_local if acc is None else (acc + mean_local)
        seen += 1

    return acc / seen


def beta_offdiag_Ez_D0(model, Zs: np.ndarray, j: int, k: int, a: float, b: float, batch: int) -> np.ndarray:
    """
    Off-diagonal mixed partial using a D(0)-style stencil in the k-direction (explicit neutral at k=0),
    and differencing that curvature between z_j = +a and z_j = -a:

      H_jk(z) = ∂²D/∂z_j∂z_k

    Approximate ∂D/∂z_j at (z ± b e_k) and at (z), then take a second difference in k of that slope:

      H_jk(z) ≈ [ ∂D/∂z_j(z + b e_k) - 2 ∂D/∂z_j(z) + ∂D/∂z_j(z - b e_k) ] / b^2

    where ∂D/∂z_j(u) ≈ [D(u + a e_j) - D(u - a e_j)]/(2a)

    Expanding yields 6 decoder evaluations per basepoint u=z:
      D(z±a e_j ± b e_k), D(z±a e_j)

    Final closed form:
      H_jk(z) ≈ ( D(z+a e_j + b e_k) - 2 D(z+a e_j) + D(z+a e_j - b e_k)
                -D(z-a e_j + b e_k) + 2 D(z-a e_j) - D(z-a e_j - b e_k) ) / (2 a b^2)
    """
    if j == k:
        raise ValueError("Use beta_diag_Ez_D0 for diagonal terms j==k.")

    n = Zs.shape[0]
    ej = np.zeros((1, LATENT_DIM), dtype=np.float32); ej[0, j] = 1.0
    ek = np.zeros((1, LATENT_DIM), dtype=np.float32); ek[0, k] = 1.0

    acc = None
    seen = 0

    for s in range(0, n, batch):
        Z0 = Zs[s:s+batch].astype(np.float32)

        # +a branch (explicit neutral in k via -2 D(z+a e_j))
        Xp_p = decode_batch(model, Z0 + a*ej + b*ek)
        Xp_0 = decode_batch(model, Z0 + a*ej)
        Xp_m = decode_batch(model, Z0 + a*ej - b*ek)

        # -a branch
        Xm_p = decode_batch(model, Z0 - a*ej + b*ek)
        Xm_0 = decode_batch(model, Z0 - a*ej)
        Xm_m = decode_batch(model, Z0 - a*ej - b*ek)

        local = ((Xp_p - 2.0*Xp_0 + Xp_m)/(2*b) - (Xm_p - 2.0*Xm_0 + Xm_m)/(2*b))/(a*2) #/ (2.0 * a * (b*b))
        mean_local = local.mean(axis=0)

        acc = mean_local if acc is None else (acc + mean_local)
        seen += 1

    return acc / seen


# -----------------------------
# Plot helpers
# -----------------------------
def robust_vlim(da2: xr.DataArray) -> tuple[float, float]:
    v = da2.values
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if ROBUST:
        lo, hi = np.percentile(finite, list(ROBUST_PCT))
        m = float(max(abs(lo), abs(hi)))
        if m == 0:
            m = 1.0
        return (-m, m)
    m = float(np.nanmax(np.abs(v)))
    if m == 0:
        m = 1.0
    return (-m, m)

def add_map_features(ax):
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.7)
    if ADD_STATES:
        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.6)

def pcolormesh_latlon(ax, da_latlon: xr.DataArray, vmin: float, vmax: float):
    lon = da_latlon["lon"].values
    lat = da_latlon["lat"].values
    return ax.pcolormesh(lon, lat, da_latlon.values, transform=PROJ,
                         cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto")


# -----------------------------
# Main
# -----------------------------
def main():
    import kgae  # you already have this in your environment
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) feature template
    template = build_feature_template(ERA5_SST_PATH, TRAINING_PERIOD)
    n_feat = template.sizes["feature"]
    print(f"Feature template built: n_feature={n_feat}")

    # 2) load empirical latent states Z(t) (reference coord system)
    Z = kgae.open_latents('xval', experiment='large-ensemble-2-1940-2014', n_ensemble=100).mean('seed')
    print(Z.mean('time'), Z.std('time'))
    sigma = Z.std('time').values

    Z = Z.transpose("time", "mode")
    if Z.sizes["mode"] != LATENT_DIM:
        raise ValueError(f"Expected LATENT_DIM={LATENT_DIM}, got {Z.sizes['mode']}")

    T = Z.sizes["time"]
    print(f"Loaded latents: time={T}, mode={Z.sizes['mode']}")

    # sample z(t)
    idx = rng.integers(0, T, size=N_SAMPLES)
    Zs = Z.isel(time=idx).values.astype(np.float32)  # (N_SAMPLES, latent_dim)

    # 3) collect models
    paths = sorted(glob.glob(MODEL_GLOB))
    if not paths:
        raise FileNotFoundError(f"No models found for MODEL_GLOB={MODEL_GLOB}")
    print(f"Found {len(paths)} model(s). Computing ensemble-mean data-supported derivatives.")

    accum = {(r, c): [] for r in ROW_CONTEXT for c in COL_ORDER}

    for p in paths:
        print("Loading:", p)
        with open(p, "rb") as f:
            model = pickle.load(f)
        model.eval()
        try:
            model.to(torch.device("cpu"))
        except Exception:
            pass

        # feature-length consistency check at z=0
        test0 = decode_batch(model, np.zeros((1, LATENT_DIM), dtype=np.float32)).squeeze()
        if test0.shape[0] != n_feat:
            raise ValueError(
                f"Decoded length {test0.shape[0]} != n_feature {n_feat}. "
                "Feature mask/training domain differs from ERA5 template."
            )

        # Row 1: alpha-like
        for c in COL_ORDER:
            j = MODE[c]
            accum[(None, c)].append(alpha_like_Ez(model, Zs, j=j, a=A*sigma[j], batch=BATCH))

        # Rows 2–4: beta-like using D(0)-style stencils
        for r in ["DEC", "IA", "QB"]:
            k = MODE[r]
            for c in COL_ORDER:
                j = MODE[c]
                if j == k:
                    accum[(r, c)].append(beta_diag_Ez_D0(model, Zs, j=j, a=B*sigma[j], batch=BATCH))
                else:
                    accum[(r, c)].append(beta_offdiag_Ez_D0(model, Zs, j=j, k=k, a=A*sigma[j], b=B*sigma[k], batch=BATCH))

    # ensemble mean -> maps
    mean_maps = {}
    for key, lst in accum.items():
        arr = np.stack(lst, axis=0).mean(axis=0)
        da_feat = xr.DataArray(arr, dims=("feature",), coords={"feature": template["feature"]})
        mean_maps[key] = da_feat.unstack("feature")

    # plot
    fig, axes = plt.subplots(
        nrows=4, ncols=3, figsize=(12.5, 11),
        subplot_kw={"projection": PROJ},
        constrained_layout=True,
    )

    row_titles = {
        None: r"$\mathbb{E}_z[\partial D/\partial z_j]$ (alpha-like)",
        "DEC": r"$\mathbb{E}_z[\partial^2 D/\partial z_j \partial z_{DEC}]$ (beta-like, D(0) stencils)",
        "IA":  r"$\mathbb{E}_z[\partial^2 D/\partial z_j \partial z_{IA}]$ (beta-like, D(0) stencils)",
        "QB":  r"$\mathbb{E}_z[\partial^2 D/\partial z_j \partial z_{QB}]$ (beta-like, D(0) stencils)",
    }
    col_titles = {"QB": "QB", "IA": "IA", "DEC": "Decadal"}

    for i, r in enumerate(ROW_CONTEXT):
        for j, c in enumerate(COL_ORDER):
            ax = axes[i, j]
            da = mean_maps[(r, c)]

            vmin, vmax = robust_vlim(da)
            im = pcolormesh_latlon(ax, da, vmin=vmin, vmax=vmax)
            add_map_features(ax)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(col_titles[c], fontsize=11)
            if j == 0:
                ax.text(
                    0.01, 0.98, row_titles[r],
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                )

            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Data-supported decoder derivatives with explicit D(0) stencils (A={A}, B={B}, z-samples={N_SAMPLES}, models={len(paths)})",
        fontsize=13
    )

    out = "decoder_state_dependence_4x3_DATA_SUPPORTED_D0_STENCILS.png"
    fig.savefig(out, dpi=200)
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()