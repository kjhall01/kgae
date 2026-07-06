#!/usr/bin/env python3
"""
Decoder-probing figure: (N+1) x N grid.

Row 0 (top): alpha-like linear response for each mode i
    alpha_i ≈ [D(+a_i e_i) - D(-a_i e_i)] / (2 a_i)

Rows 1..N: beta-like "state-dependence" for J_i ≡ dD/dz_i
    Panel (row=i+1, col=j): B_ij estimated as modulation of J_i by z_j
    using a *pure i-active basepoint* z* = ± a_i e_i, with z_j toggled.

Directional interaction probe (your preferred construction):
    1) Define J_i(z) ≈ [D(z + δ_i e_i) - D(z - δ_i e_i)] / (2 δ_i)
    2) Then B_ij(z*) ≈ [J_i(z* + a_j e_j) - J_i(z* - a_j e_j)] / (2 a_j)
    3) Average over z* = +a_i e_i and z* = -a_i e_i

Diagonal panels (i=j): uses the simpler self-curvature probe around 0:
    B_ii ≈ [D(+a_i e_i) + D(-a_i e_i) - 2 D(0)] / (a_i^2)

Bootstrap:
- Loads all CV models (seed*/split*/model.pkl).
- For each seed: average across splits (split-mean).
- i.i.d. bootstrap across seeds to form 95% CI at each gridpoint.
- Filled contours only where CI excludes 0; black contours everywhere
  (solid positive, dashed negative).
- One colorbar for alpha row, one for beta block.

Note on scaling:
- a_i, a_j, δ_i are in *latent units* (post flip_signs alignment).
- If you want "per-sigma" meaning, set a_i = 1 to mean 1σ *in your standardized latent*,
  and δ_i small (e.g., 0.25).
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

# Column order (1-based labels shown in plot)
modes = [5, 4, 3]
mode_titles = ["Decadal", "Interannual", "Quasibiennial"]

# Probe step sizes (latent units)
A_STEP = 1      # a_i and a_j
DELTA_I = 0.1    # δ_i for Jacobian finite difference in i-direction (should be smaller)

# Bootstrap across seeds
N_BOOT = 500
RNG_SEED = 123
CI_LO, CI_HI = 2.5, 97.5

# Plot settings
LEV_ALPHA  = np.linspace(-0.35, 0.35, 21)
LEV_ALPHA2 = np.linspace(-0.35, 0.35, 7)

quantity = 'z2'
lev_beta = 0.1 if quantity == 'z2' else 0.001


central_longitude = 180
cmap = "RdBu_r"
ADD_STATES = False


# -----------------------------
# Small plot helpers
# -----------------------------
def add_panel_label(ax, letter, *, x=0.02, y=1.08):
    ax.text(
        x, y, f"{letter})",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10,
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

def alpha_probe(model, i0: int, a: float = A_STEP) -> np.ndarray:
    """alpha_i ≈ [D(+a e_i) - D(-a e_i)] / (2a)"""
    zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[i0] = +a
    zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[i0] = -a
    return (decode_model(model, zp) - decode_model(model, zm)) / (2.0 * a)

def beta_self_probe(model, i0: int, a: float = A_STEP) -> np.ndarray:
    """B_ii (self) ≈ [D(+a e_i) + D(-a e_i) - 2D(0)] / a^2"""
    z0 = np.zeros(LATENT_DIM, dtype=np.float32)
    zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[i0] = +a
    zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[i0] = -a
    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a * a)

def J_i_probe(model, z_base: np.ndarray, i0: int, delta: float = DELTA_I) -> np.ndarray:
    """J_i(z) ≈ [D(z+δ e_i) - D(z-δ e_i)] / (2δ)"""
    zp = z_base.copy(); zp[i0] += delta
    zm = z_base.copy(); zm[i0] -= delta
    return (decode_model(model, zp) - decode_model(model, zm)) / (2.0*delta)

def beta_interaction_probe_pure_i(model, i0: int, j0: int,
                                 a: float = A_STEP, delta: float = DELTA_I) -> np.ndarray:
    """
    Directional interaction: B_ij as "modulation of J_i by z_j" at *pure i-active* basepoint.

    For s in {+1, -1}, define z* = s a e_i (pure i active, others 0).
    Then:
        B_ij(z*) ≈ [ J_i(z* + a e_j) - J_i(z* - a e_j) ] / (2a)
    Return average over s=+1 and s=-1.
    """
   # if i0 == j0:
    #    return beta_self_probe(model, i0=i0, a=a)

    out = 0.0
    for s in (+1.0, -1.0):
        zstar = np.zeros(LATENT_DIM, dtype=np.float32)
        zstar[i0] = s * a

        z_plus = zstar.copy();  z_plus[j0] += a
        z_minus = zstar.copy(); z_minus[j0] -= a

        J_plus = J_i_probe(model, z_plus,  i0=i0, delta=delta)
        J_minus= J_i_probe(model, z_minus, i0=i0, delta=delta)

        out = out + (J_plus - J_minus) / (2.0*a ) if quantity != 'z2' else out + s*(J_plus - J_minus) / (2.0 *a) 

    return out / 2.0


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


# -----------------------------
# Plotting
# -----------------------------
def _nice_levels(data_list, n_levels=13, robust=0.99):
    vals = []
    for row in data_list:
        for da in row:
            v = da.values
            vals.append(v[np.isfinite(v)])
    if not vals:
        return np.linspace(-1, 1, n_levels)
    v = np.concatenate(vals)
    if v.size == 0:
        return np.linspace(-1, 1, n_levels)
    vmax = float(np.nanquantile(np.abs(v), robust))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = float(np.nanmax(np.abs(v)))
        if vmax == 0 or not np.isfinite(vmax):
            vmax = 1.0
    return np.linspace(-vmax, vmax, n_levels)

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

    N = len(modes)

    # per_seed_alpha[m][seed] = [arrays over splits]
    # per_seed_beta[(i_mode, j_mode)][seed] = [arrays over splits]
    per_seed_alpha = {m: {} for m in modes}
    per_seed_beta = {(mi, mj): {} for mi in modes for mj in modes}

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

        # --- alphas ---
        for m in modes:
            i0 = int(m) - 1
            a_map = alpha_probe(model, i0=i0, a=A_STEP)
            per_seed_alpha[m].setdefault(seed_idx, []).append(a_map)

        # --- betas (directional modulation of J_i by z_j at pure i basepoint) ---
        for mi in modes:
            j0 = int(mi) - 1
            for mj in modes:
                i0 = int(mj) - 1
                b_map = beta_interaction_probe_pure_i(model, i0=i0, j0=j0, a=A_STEP, delta=DELTA_I)
                per_seed_beta[(mi, mj)].setdefault(seed_idx, []).append(b_map)

    # Seeds represented
    seed_list = sorted(set().union(*[set(d.keys()) for d in per_seed_alpha.values()]))
    print(f"Seeds represented: {len(seed_list)}")

    # Split-mean within each seed => members arrays
    alpha_members = {}
    beta_members = {}

    for m in modes:
        alpha_members[m] = np.stack(
            [np.stack(per_seed_alpha[m][s], axis=0).mean(axis=0) for s in seed_list],
            axis=0
        )

    for mi in modes:
        for mj in modes:
            beta_members[(mi, mj)] = np.stack(
                [np.stack(per_seed_beta[(mi, mj)][s], axis=0).mean(axis=0) for s in seed_list],
                axis=0
            )

    # Bootstrap => mean + significance
    alpha_mean = []
    alpha_sig = []

    # store betas in (N,N) lists aligned to modes order
    beta_mean = [[None]*N for _ in range(N)]
    beta_sig  = [[None]*N for _ in range(N)]

    # --- alphas ---
    for m in modes:
        mean, lo, hi = bootstrap_ci_seed(alpha_members[m], n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)
        da_mean = xr.DataArray(mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_lo   = xr.DataArray(lo,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_hi   = xr.DataArray(hi,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        alpha_mean.append(da_mean)
        alpha_sig.append((da_lo > 0) | (da_hi < 0))

    # --- betas ---
    for r, mi in enumerate(modes):
        for c, mj in enumerate(modes):
            mean, lo, hi = bootstrap_ci_seed(beta_members[(mi, mj)], n_boot=N_BOOT, rng=rng, ci_lo=CI_LO, ci_hi=CI_HI)
            da_mean = xr.DataArray(mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
            da_lo   = xr.DataArray(lo,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
            da_hi   = xr.DataArray(hi,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
            beta_mean[r][c] = da_mean
            beta_sig[r][c]  = (da_lo > 0) | (da_hi < 0)

    # -----------------------------
    # Plot
    # -----------------------------
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    pc   = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(2.55*N + 1.6, 3.15*(N+1)))
    gs = gridspec.GridSpec(N+1, N, hspace=0.10, wspace=0.03)

    letters = [chr(ord('a') + k) for k in range((N+1)*N)]

    axes = [[None]*N for _ in range(N+1)]
    k = 0
    for r in range(N+1):
        for c in range(N):
            ax = fig.add_subplot(gs[r, c], projection=proj)
            ax.coastlines(linewidth=0.7)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.7)
            if ADD_STATES:
                ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.6)
            add_panel_label(ax, letters[k])
            axes[r][c] = ax
            k += 1

    # ---- Row 0: alphas ----
    mappable_alpha = None
    for c in range(N):
        ax = axes[0][c]
        da = alpha_mean[c]
        sig = alpha_sig[c]
        da_sig = da.where(sig)

        cf = ax.contourf(da["lon"], da["lat"], da_sig,
                         levels=LEV_ALPHA, transform=pc, extend="both", cmap=cmap)
        if mappable_alpha is None:
            mappable_alpha = cf

        contour_sign(ax, da, LEV_ALPHA2, pc, colors="k")
        ax.set_title(mode_titles[c])

    # Alpha row label on left-most axis
    axes[0][0].text(-0.08, 0.5, r"$\alpha_i$",
                    transform=axes[0][0].transAxes,
                    ha="right", va="center", rotation=90, fontsize=11)


    LEV_BETA   = _nice_levels(beta_mean, n_levels=21)
    LEV_BETA2  = _nice_levels(beta_mean, n_levels=7)

    # ---- Beta block: rows 1..N, cols 0..N-1 ----
    mappable_beta = None
    for r in range(1, N+1):
        i = r - 1  # i-index in modes
        for c in range(N):
            ax = axes[r][c]
            da = beta_mean[i][c]
            sig = beta_sig[i][c]
            da_sig = da.where(sig)

            cf = ax.contourf(da["lon"], da["lat"], da_sig,
                             levels=LEV_BETA, transform=pc, extend="both", cmap=cmap)
            if mappable_beta is None:
                mappable_beta = cf

            contour_sign(ax, da, LEV_BETA2, pc, colors="k")

        # row label on left-most panel
        axes[r][0].text(-0.08, 0.5, rf"$\beta_{{{mode_titles[i][0]}\,\cdot}}$",
                        transform=axes[r][0].transAxes,
                        ha="right", va="center", rotation=90, fontsize=11)

    # Column labels for beta block (optional): put j labels just above beta block
    for c in range(N):
        axes[1][c].text(0.5, 1.10, rf"$z_{{{mode_titles[c][0]}}}$",
                        transform=axes[1][c].transAxes,
                        ha="center", va="bottom", fontsize=10)

    # ---- Colorbars ----
    cb1 = fig.colorbar(mappable_alpha, ax=[axes[0][c] for c in range(N)],
                       orientation="vertical", fraction=0.028, pad=0.02)
    cb1.set_label(r"Linear response  ($\partial D/\partial z_i$)", fontsize=10)

    cb2 = fig.colorbar(mappable_beta, ax=[axes[r][c] for r in range(1, N+1) for c in range(N)],
                       orientation="vertical", fraction=0.028, pad=0.02)
    cb2.set_label(r"State dependence  ($\partial^2 D/\partial z_j \partial z_i$)", fontsize=10)

    out = "decoder_probe_grid_alpha_betaIJ_pureI.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)
    plt.show()


if __name__ == "__main__":
    main()