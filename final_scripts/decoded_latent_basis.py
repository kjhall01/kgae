#!/usr/bin/env python3
"""
Nine-panel decoder-probing figure (3 rows x 3 cols).

Row 1 (top):   alpha-like linear response
    alpha_j ≈ [D(+a e_j) - D(-a e_j)] / (2a)

Row 2 (middle): even curvature ("nonlinear response" / diag beta)
    beta_jj ≈ [D(+a e_j) + D(-a e_j) - 2D(0)] / a^2

Row 3 (bottom): *cubic* diagnostic = "linear response of nonlinear response to itself"
    S_j ≈ [ N_j(+A) - N_j(-A) ] / (2A)
where
    N_j(z0) = [D(z0+δ) + D(z0-δ) - 2D(z0)] / δ^2   (curvature around basepoint z0)
and z0 = ±A along the SAME mode j.

This bottom-row quantity approximates ∂^3 D / ∂z_j^3 (up to finite-difference error),
i.e., phase-asymmetry of the curvature / cubic nonlinearity.

Bootstrap:
- Loads all CV models (seed*/split*/model.pkl).
- For each seed: average across splits (split-mean).
- i.i.d. bootstrap across seeds to form CI at each gridpoint.
- Filled contours only where CI excludes 0; black contours everywhere (neg dashed).
- One colorbar per row (right side).
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
MODEL_GLOB = "~/Desktop/kgae/final_scripts/large-ensemble-2-1940-2014/seed*/split*/model.pkl"

ERA5_SST_PATH = "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"
TRAINING_PERIOD = ("1940-01-01", "2014-12-31")

LATENT_DIM = 5

modes = [5, 4, 3]  # Decadal, Interannual, Quasibiennial
mode_titles = ["Decadal", "Interannual", "Quasibiennial"]

# Probe amplitudes
A_STEP = 1.0          # a for alpha + even curvature
CUBIC_A = 1.0         # A for the basepoint z0 = ±A in the cubic diagnostic
CUBIC_DELTA = 0.25    # δ for curvature around basepoint (smaller than A)

# Seed bootstrap
N_BOOT = 1000
RNG_SEED = 123
CI_LO, CI_HI = 2.5, 97.5

# Levels (adjust as needed)
LEV_ALPHA  = np.linspace(-0.35, 0.35, 21)
LEV_ALPHA2 = np.linspace(-0.35, 0.35, 7)

LEV_BETA   = np.linspace(-0.005, 0.005, 21)
LEV_BETA2  = np.linspace(-0.005, 0.005, 7)

# Cubic term typically smaller; start with something conservative and adjust after one run
LEV_CUBIC  = np.linspace(-0.1, 0.1, 21)
LEV_CUBIC2 = np.linspace(-0.1, 0.1, 7)

central_longitude = 180
cmap = "RdBu_r"
ADD_STATES = False


# -----------------------------
# Small plot helpers
# -----------------------------
def add_panel_label(ax, letter, *, x=0.02, y=1.1):
    ax.text(
        x, y, f"{letter})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11,
        weight='bold',
        bbox=dict(facecolor="white", alpha=0.0, edgecolor="none", pad=2),
        zorder=20
    )

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
# Decoder probing math
# -----------------------------
def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

@torch.no_grad()
def decode_model(model, z_np: np.ndarray) -> np.ndarray:
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

def alpha_probe(model, j0: int, a: float = A_STEP) -> np.ndarray:
    """alpha_j ≈ [D(+a e_j) - D(-a e_j)] / (2a)"""
    zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[j0] = +a
    zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[j0] = -a
    return (decode_model(model, zp) - decode_model(model, zm)) / (2.0 * a)

def beta_diag_probe(model, j0: int, a: float = A_STEP) -> np.ndarray:
    """beta_jj ≈ [D(+a)+D(-a)-2D(0)] / a^2"""
    z0 = np.zeros(LATENT_DIM, dtype=np.float32)
    zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[j0] = +a
    zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[j0] = -a
    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a * a)

def curvature_at_basepoint(model, j0: int, z0_val: float, delta: float = CUBIC_DELTA) -> np.ndarray:
    """
    N_j(z0) ≈ [D(z0+δ) + D(z0-δ) - 2D(z0)] / δ^2
    with z0 along the SAME mode j (others 0).
    """
    z0 = np.zeros(LATENT_DIM, dtype=np.float32); z0[j0] = z0_val
    zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[j0] = z0_val + delta
    zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[j0] = z0_val - delta
    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (delta * delta)

def cubic_self_probe(model, j0: int, A: float = CUBIC_A, delta: float = CUBIC_DELTA) -> np.ndarray:
    """
    S_j ≈ [ N_j(+A) - N_j(-A) ] / (2A)
    This is the "linear response of the nonlinear response to itself" (cubic).
    """
    Np = curvature_at_basepoint(model, j0=j0, z0_val=+A, delta=delta)
    Nm = curvature_at_basepoint(model, j0=j0, z0_val=-A, delta=delta)
    return (Np - Nm) / (2.0 * A)


# -----------------------------
# Bootstrap across seeds (i.i.d.)
# -----------------------------
def bootstrap_ci_seed(
    members: np.ndarray, n_boot: int, rng: np.random.Generator,
    ci_lo: float = 2.5, ci_hi: float = 97.5, chunk: int = 100
):
    members = np.asarray(members)
    S, F = members.shape
    mean = members.mean(axis=0)

    boot_means = np.empty((n_boot, F), dtype=np.float32)
    written = 0
    while written < n_boot:
        b = min(chunk, n_boot - written)
        idx = rng.integers(0, S, size=(b, S))
        for i in range(b):
            boot_means[written + i] = members[idx[i]].mean(axis=0)
        written += b

    lo = np.percentile(boot_means, ci_lo, axis=0)
    hi = np.percentile(boot_means, ci_hi, axis=0)
    return mean, lo, hi


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

    per_seed_alpha = {m: {} for m in modes}
    per_seed_beta  = {m: {} for m in modes}
    per_seed_cubic = {m: {} for m in modes}

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

        test = decode_model(model, np.zeros(LATENT_DIM, dtype=np.float32))
        if test.shape[0] != n_feat:
            raise ValueError(
                f"Decoded length {test.shape[0]} != n_feature {n_feat}. "
                "ERA5 feature mask/training domain mismatch for this model."
            )

        for m in modes:
            j0 = int(m) - 1
            a = alpha_probe(model, j0=j0, a=A_STEP)
            b = beta_diag_probe(model, j0=j0, a=A_STEP)
            c = cubic_self_probe(model, j0=j0, A=CUBIC_A, delta=CUBIC_DELTA)

            per_seed_alpha[m].setdefault(seed_idx, []).append(a)
            per_seed_beta[m].setdefault(seed_idx, []).append(b)
            per_seed_cubic[m].setdefault(seed_idx, []).append(c)

    seed_list = sorted(set().union(*[set(d.keys()) for d in per_seed_alpha.values()]))
    print(f"Seeds represented: {len(seed_list)}")

    alpha_members, beta_members, cubic_members = {}, {}, {}
    for m in modes:
        alpha_members[m] = np.stack([np.stack(per_seed_alpha[m][s], axis=0).mean(axis=0) for s in seed_list], axis=0)
        beta_members[m]  = np.stack([np.stack(per_seed_beta[m][s],  axis=0).mean(axis=0) for s in seed_list], axis=0)
        cubic_members[m] = np.stack([np.stack(per_seed_cubic[m][s], axis=0).mean(axis=0) for s in seed_list], axis=0)

    alpha_mean, alpha_sig = [], []
    beta_mean,  beta_sig  = [], []
    cubic_mean, cubic_sig = [], []

    for m in modes:
        mean, lo, hi = bootstrap_ci_seed(alpha_members[m], n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)
        da_mean = xr.DataArray(mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_lo   = xr.DataArray(lo,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_hi   = xr.DataArray(hi,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        alpha_mean.append(da_mean)
        alpha_sig.append((da_lo > 0) | (da_hi < 0))

        mean, lo, hi = bootstrap_ci_seed(beta_members[m], n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)
        da_mean = xr.DataArray(mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_lo   = xr.DataArray(lo,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_hi   = xr.DataArray(hi,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        beta_mean.append(da_mean)
        beta_sig.append((da_lo > 0) | (da_hi < 0))

        mean, lo, hi = bootstrap_ci_seed(cubic_members[m], n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)
        da_mean = xr.DataArray(mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_lo   = xr.DataArray(lo,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_hi   = xr.DataArray(hi,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        cubic_mean.append(da_mean)
        cubic_sig.append((da_lo > 0) | (da_hi < 0))

    # -----------------------------
    # Plot
    # -----------------------------
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    pc   = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(11.5, 9.3))
    gs = gridspec.GridSpec(3, len(modes), hspace=0.12, wspace=0.05)

    letrs = [chr(ord('a') + k) for k in range(3 * len(modes))]

    axes = []
    k = 0
    for r in range(3):
        row_axes = []
        for c in range(len(modes)):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.coastlines(linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.7)
            if ADD_STATES:
                ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.6)
            add_panel_label(ax, letrs[k])
            k += 1
            row_axes.append(ax)
        axes.append(row_axes)

    # --- ROW 1: alpha ---
    mappable1 = None
    for j in range(len(modes)):
        ax = axes[0][j]
        da = alpha_mean[j]
        sig = alpha_sig[j]
        da_sig = da.where(sig)

        cf = ax.contourf(da["lon"], da["lat"], da_sig,
                         levels=LEV_ALPHA, transform=pc, extend="both", cmap=cmap)
        if mappable1 is None:
            mappable1 = cf
        contour_sign(ax, da, LEV_ALPHA2, pc, colors="k")
        ax.set_title(mode_titles[j])

    cb1 = fig.colorbar(mappable1, ax=axes[0], orientation="vertical", fraction=0.025, pad=0.02)
    cb1.set_label(r"Linear response ($K z^{-1}$)")

    # --- ROW 2: even curvature (beta diag) ---
    mappable2 = None
    for j in range(len(modes)):
        ax = axes[1][j]
        da = beta_mean[j]
        sig = beta_sig[j]
        da_sig = da.where(sig)

        cf = ax.contourf(da["lon"], da["lat"], da_sig,
                         levels=LEV_BETA, transform=pc, extend="both", cmap=cmap)
        if mappable2 is None:
            mappable2 = cf
        contour_sign(ax, da, LEV_BETA2, pc, colors="k")

    cb2 = fig.colorbar(mappable2, ax=axes[1], orientation="vertical", fraction=0.025, pad=0.02)
    cb2.set_label(r"Nonlinear response ($K z^{-2}$)")

    # --- ROW 3: cubic self-modulation ---
    mappable3 = None
    for j in range(len(modes)):
        ax = axes[2][j]
        da = cubic_mean[j]
        sig = cubic_sig[j]
        da_sig = da.where(sig)

        cf = ax.contourf(da["lon"], da["lat"], da_sig,
                         levels=LEV_CUBIC, transform=pc, extend="both", cmap=cmap)
        if mappable3 is None:
            mappable3 = cf
        contour_sign(ax, da, LEV_CUBIC2, pc, colors="k")

    cb3 = fig.colorbar(mappable3, ax=axes[2], orientation="vertical", fraction=0.025, pad=0.02)
    cb3.set_label(r"Self state-dependence ($K z^{-3}$)")

    out = "decoder_patterns.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()
