#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
import torch

# -----------------------------
# USER SETTINGS
# -----------------------------
# Experiment / data
EXPERIMENT = "./large-ensemble-2-1940-2014"   # for regression + composites (edit)
N_ENSEMBLE_REG = 100                            # used by kgae.open_alphabetas (edit)
N_ENSEMBLE_LATENTS = 100                      # used by kgae.open_latents (edit)

# Decoder-probe models live on disk. Override these with environment variables
# when running from a different local data layout.
MODEL_GLOB = os.environ.get(
    "KGAE_MODEL_GLOB",
    str(Path.home() / "Desktop/kgae/final_scripts/large-ensemble-2-1940-2014/seed*/split*/model.pkl"),
)
ERA5_SST_PATH = os.environ.get(
    "KGAE_ERA5_SST_PATH",
    str(Path.home() / "Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"),
)
TRAINING_PERIOD = ("1940-01-01", "2014-12-31")
LATENT_DIM = 5

# Modes to compare (3-mode case)
MODES = [5, 4, 3]  # Decadal, Interannual, Quasibiennial ordering for this analysis
MODE_TITLES = ["Decadal", "Interannual", "Quasibiennial"]

# --- masking / significance
MASK_MODE = "none"           # "none" or "sig_intersection"
USE_COSLAT_WEIGHTS = True

# Regression sig bootstrap settings, matching plot_alpha_betas.py.
REG_SIG_N_BOOT = 200
REG_SIG_CI = 0.90
REG_SIG_DIM = "seed"

# Composite bootstrap settings, matching composites.py.
COMP_N_BOOT = 200
COMP_BLOCK_LEN = 12
COMP_SIGMA_MULT = 1.0
COMP_RNG_SEED = 123

# Probe bootstrap settings, matching decode_latent_basis_4x3.py.
PROBE_N_BOOT = 300
PROBE_RNG_SEED = 123
PROBE_A_STEP = 1.0
PROBE_DELTA_I = 0.25
PROBE_CI_LO, PROBE_CI_HI = 2.5, 97.5

OUT_DIR = Path("./corr_outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================
# Utilities: grid alignment, weights, pattern correlation
# ============================================================
def _to_180(lon: xr.DataArray) -> xr.DataArray:
    return ((lon + 180) % 360) - 180

def ensure_lon_180(da: xr.DataArray) -> xr.DataArray:
    if "lon" not in da.coords:
        return da
    lon = da["lon"]
    if float(lon.max()) > 180:
        da = da.assign_coords(lon=_to_180(lon))
    return da.sortby("lon")

def area_weights_coslat(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return w

def align_on_common_grid(a: xr.DataArray, b: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    a = ensure_lon_180(a)
    b = ensure_lon_180(b)
    a2, b2 = xr.align(a, b, join="inner")
    return a2, b2

def apply_pair_mask(
    a: xr.DataArray,
    b: xr.DataArray,
    mask_a: Optional[xr.DataArray],
    mask_b: Optional[xr.DataArray],
    mask_mode: str,
) -> Tuple[xr.DataArray, xr.DataArray]:
    if mask_mode == "none":
        return a, b
    if mask_mode != "sig_intersection":
        raise ValueError(f"Unknown MASK_MODE={mask_mode}")
    if mask_a is None or mask_b is None:
        return a, b
    ma, mb = align_on_common_grid(mask_a.astype(bool), mask_b.astype(bool))
    keep = ma & mb
    return a.where(keep), b.where(keep)

def pattern_corr(a: xr.DataArray, b: xr.DataArray, w_lat: Optional[xr.DataArray]) -> float:
    a, b = align_on_common_grid(a, b)
    if w_lat is None:
        w = xr.ones_like(a)
    else:
        w = w_lat.broadcast_like(a)

    m = xr.ufuncs.isfinite(a) & xr.ufuncs.isfinite(b) & xr.ufuncs.isfinite(w)
    if int(m.sum()) < 50:
        return np.nan

    a1 = a.where(m); b1 = b.where(m); w1 = w.where(m)
    wsum = w1.sum()

    am = (a1 * w1).sum() / wsum
    bm = (b1 * w1).sum() / wsum

    cov = ((a1 - am) * (b1 - bm) * w1).sum() / wsum
    va = (((a1 - am) ** 2) * w1).sum() / wsum
    vb = (((b1 - bm) ** 2) * w1).sum() / wsum

    denom = np.sqrt(va * vb)
    if float(denom) == 0.0:
        return np.nan
    return float(cov / denom)


# ============================================================
# Regression alpha/beta + sig masks (from plot_alpha_betas.py)
# ============================================================
def moving_block_bootstrap_indices(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    if n <= block_len:
        start = int(rng.integers(0, n))
        return (np.arange(n) + start) % n
    n_starts = n - block_len + 1
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n_starts, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)
    return idx[:n]

def bootstrap_pct_lt_zero(
    da: xr.DataArray,
    dim: str,
    n_resamples: int,
    block_length: int,
    rng_seed: int,
) -> xr.DataArray:
    rng = np.random.default_rng(rng_seed)
    n = da.sizes[dim]
    ests = []
    for _ in range(n_resamples):
        idx = moving_block_bootstrap_indices(rng, n, block_len=block_length)
        ests.append(da.isel(**{dim: idx}).mean(dim))
    ests = xr.concat(ests, dim="boot")
    pct_lt_0 = (ests < 0).mean("boot").where(np.isfinite(ests.mean("boot")))
    return pct_lt_0

def reg_sig_mask_from_pct(pct_lt_0: xr.DataArray, ci: float) -> xr.DataArray:
    # Matches the masking criterion in plot_alpha_betas.py.
    return (pct_lt_0 > ci + (1 - ci) / 2) | (pct_lt_0 < (1 - ci) / 2)

def load_regression_fields() -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
    import kgae  # local project

    alpha_full, beta_full = kgae.open_alphabetas("xval", experiment=EXPERIMENT, n_ensemble=N_ENSEMBLE_REG)
    # Match plotting reductions: mean over seed and fold for point estimate.
    alpha_mean = alpha_full.mean("seed").mean("fold")
    beta_mean  = beta_full.mean("seed").mean("fold")

    # only keep the 3 modes we want (both dims)
    alpha_mean = alpha_mean.sel(tendency_mode=MODES)
    beta_mean  = beta_mean.sel(tendency_mode=MODES, predictor_mode=MODES)

    # Build masks (optional; expensive)
    alpha_sig = None
    beta_sig = None
    if MASK_MODE == "sig_intersection":
        # Bootstrap along seed.
        alpha_pct = bootstrap_pct_lt_zero(alpha_full.mean("fold").sel(tendency_mode=MODES),
                                          dim=REG_SIG_DIM, n_resamples=REG_SIG_N_BOOT,
                                          block_length=1, rng_seed=0)
        beta_pct = bootstrap_pct_lt_zero(beta_full.mean("fold").sel(tendency_mode=MODES, predictor_mode=MODES),
                                         dim=REG_SIG_DIM, n_resamples=REG_SIG_N_BOOT,
                                         block_length=1, rng_seed=0)
        alpha_sig = reg_sig_mask_from_pct(alpha_pct, ci=REG_SIG_CI)
        beta_sig  = reg_sig_mask_from_pct(beta_pct,  ci=REG_SIG_CI)

    fields = {
        "alpha_reg": ensure_lon_180(alpha_mean),
        "beta_reg": ensure_lon_180(beta_mean),
    }
    masks = {}
    if alpha_sig is not None:
        masks["alpha_reg_sig"] = ensure_lon_180(alpha_sig)
    if beta_sig is not None:
        masks["beta_reg_sig"] = ensure_lon_180(beta_sig)
    return fields, masks


# ============================================================
# Composites (from composites.py)
# ============================================================
def _robust_sigma(x: np.ndarray) -> float:
    s = float(np.nanstd(x))
    if not np.isfinite(s) or s == 0.0:
        return np.nan
    return s

def circular_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend([(s + k) % n for k in range(block_len)])
    return np.array(idx[:n], dtype=int)

def composite_mean(field: xr.DataArray, mask_time: np.ndarray) -> Optional[xr.DataArray]:
    if mask_time.sum() < 3:
        return None
    return field.isel(time=np.where(mask_time)[0]).mean("time", skipna=True)

def composite_metrics_once(
    ssta: xr.DataArray, z: xr.DataArray, i_mode: int, other_modes: List[int], sigma_mult: float = 1.0
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    # Matches composites.py.
    zi = z.sel(mode=i_mode).values
    sig_i = _robust_sigma(zi)
    if not np.isfinite(sig_i):
        return None, None

    gate = np.ones(z.sizes["time"], dtype=bool)
    for m in other_modes:
        zk = z.sel(mode=m).values
        sig_k = _robust_sigma(zk)
        if not np.isfinite(sig_k):
            return None, None
        gate &= (np.abs(zk) < sigma_mult * sig_k)

    pos = gate & (zi >  sigma_mult * sig_i)
    neg = gate & (zi < -sigma_mult * sig_i)
    neu = gate & (np.abs(zi) < sigma_mult * sig_i)

    mu_pos = composite_mean(ssta, pos)
    mu_neg = composite_mean(ssta, neg)
    mu_neu = composite_mean(ssta, neu)
    if (mu_pos is None) or (mu_neg is None) or (mu_neu is None):
        return None, None

    lin = (mu_pos - mu_neg)
    sym = (mu_pos - 2.0 * mu_neu + mu_neg)  # algebraic simplification of the symmetric form
    return lin, sym

def bootstrap_composites(
    ssta: xr.DataArray, z: xr.DataArray, modes: List[int],
    n_boot: int, block_len: int, rng_seed: int, sigma_mult: float
) -> Dict[str, Dict[int, xr.DataArray]]:
    ssta2, z2 = xr.align(ssta, z, join="inner")
    n = ssta2.sizes["time"]
    rng = np.random.default_rng(rng_seed)

    out = {"lin_boot": {}, "sym_boot": {}}
    for i_mode in modes:
        other = [m for m in modes if m != i_mode]
        lin_list, sym_list = [], []
        for _ in range(n_boot):
            idx = circular_block_indices(n, block_len, rng)
            s_b = ssta2.isel(time=idx)
            z_b = z2.isel(time=idx)
            lin, sym = composite_metrics_once(s_b, z_b, i_mode=i_mode, other_modes=other, sigma_mult=sigma_mult)
            if lin is None:
                lin = xr.full_like(ssta2.isel(time=0), np.nan).drop_vars("time", errors="ignore")
                sym = xr.full_like(ssta2.isel(time=0), np.nan).drop_vars("time", errors="ignore")
            lin_list.append(lin)
            sym_list.append(sym)
        out["lin_boot"][i_mode] = xr.concat(lin_list, dim="boot")
        out["sym_boot"][i_mode] = xr.concat(sym_list, dim="boot")
    return out

def load_composite_fields() -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
    import kgae  # local project

    z = kgae.open_latents("xval", experiment=EXPERIMENT, n_ensemble=N_ENSEMBLE_LATENTS).mean("seed")
    ssta = kgae.open_references("xval", experiment=EXPERIMENT)

    # align to three modes and compute full-sample composites
    z = z.sel(mode=MODES)
    ssta, z = xr.align(ssta, z, join="inner")

    lin_maps = []
    sym_maps = []
    for m in MODES:
        other = [mm for mm in MODES if mm != m]
        lin, sym = composite_metrics_once(ssta, z, i_mode=m, other_modes=other, sigma_mult=COMP_SIGMA_MULT)
        lin_maps.append(lin)
        sym_maps.append(sym)

    comp_lin = xr.concat(lin_maps, dim="mode").assign_coords(mode=MODES)
    comp_sym = xr.concat(sym_maps, dim="mode").assign_coords(mode=MODES)

    masks = {}
    if MASK_MODE == "sig_intersection":
        boot = bootstrap_composites(
            ssta, z, MODES, n_boot=COMP_N_BOOT, block_len=COMP_BLOCK_LEN,
            rng_seed=COMP_RNG_SEED, sigma_mult=COMP_SIGMA_MULT
        )
        # Use 95% CI by quantiles.
        qs = [0.025, 0.975]
        lin_sig = []
        sym_sig = []
        for m in MODES:
            linq = boot["lin_boot"][m].quantile(qs, dim="boot", skipna=True)
            symq = boot["sym_boot"][m].quantile(qs, dim="boot", skipna=True)
            lin_sig.append((linq.sel(quantile=qs[0]) > 0) | (linq.sel(quantile=qs[1]) < 0))
            sym_sig.append((symq.sel(quantile=qs[0]) > 0) | (symq.sel(quantile=qs[1]) < 0))
        masks["comp_lin_sig"] = xr.concat(lin_sig, dim="mode").assign_coords(mode=MODES)
        masks["comp_sym_sig"] = xr.concat(sym_sig, dim="mode").assign_coords(mode=MODES)

    fields = {
        "comp_lin": ensure_lon_180(comp_lin),
        "comp_nonlin": ensure_lon_180(comp_sym),
    }
    for k in list(masks.keys()):
        masks[k] = ensure_lon_180(masks[k])
    return fields, masks


# ============================================================
# Decoder probes (from decode_latent_basis_4x3.py)
# ============================================================
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

def build_feature_template(era5_path: str, training_period: Tuple[str, str]) -> xr.DataArray:
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

def parse_seed_split(p: str) -> Tuple[int, int]:
    parts = str(p).split("/")
    seed = int([s for s in parts if s.startswith("seed")][0].replace("seed", ""))
    split = int([s for s in parts if s.startswith("split")][0].replace("split", ""))
    return seed, split

def alpha_probe(model, i0: int, a: float) -> np.ndarray:
    zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[i0] = +a
    zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[i0] = -a
    return (decode_model(model, zp) - decode_model(model, zm)) / (2.0 * a)

def beta_self_probe(model, i0: int, a: float) -> np.ndarray:
    z0 = np.zeros(LATENT_DIM, dtype=np.float32)
    zp = np.zeros(LATENT_DIM, dtype=np.float32); zp[i0] = +a
    zm = np.zeros(LATENT_DIM, dtype=np.float32); zm[i0] = -a
    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a * a)

def J_i_probe(model, z_base: np.ndarray, i0: int, delta: float) -> np.ndarray:
    zp = z_base.copy(); zp[i0] += delta
    zm = z_base.copy(); zm[i0] -= delta
    return (decode_model(model, zp) - decode_model(model, zm)) / (2.0 * delta)

def beta_interaction_probe_pure_i(model, i0: int, j0: int, a: float, delta: float) -> np.ndarray:
    if i0 == j0:
        return beta_self_probe(model, i0=i0, a=a)
    out = 0.0
    for s in (+1.0, -1.0):
        zstar = np.zeros(LATENT_DIM, dtype=np.float32)
        zstar[i0] = s * a

        z_plus = zstar.copy();  z_plus[j0] += a
        z_minus = zstar.copy(); z_minus[j0] -= a

        J_plus = J_i_probe(model, z_plus,  i0=i0, delta=delta)
        J_minus= J_i_probe(model, z_minus, i0=i0, delta=delta)
        out = out + (J_plus - J_minus) / (2.0 * a)
    return out / 2.0

def bootstrap_ci_seed(members: np.ndarray, n_boot: int, rng: np.random.Generator,
                      ci_lo: float, ci_hi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = np.asarray(members)
    S, F = members.shape
    mean = members.mean(axis=0)
    idx = rng.integers(0, S, size=(n_boot, S))
    boot_means = members[idx].mean(axis=1)  # (n_boot, F)
    lo = np.percentile(boot_means, ci_lo, axis=0)
    hi = np.percentile(boot_means, ci_hi, axis=0)
    return mean, lo, hi

def load_probe_fields() -> Tuple[Dict[str, xr.DataArray], Dict[str, xr.DataArray]]:
    template = build_feature_template(ERA5_SST_PATH, TRAINING_PERIOD)
    n_feat = template.sizes["feature"]

    paths = sorted(glob.glob(MODEL_GLOB))
    if not paths:
        raise FileNotFoundError(f"No models found for MODEL_GLOB={MODEL_GLOB}")

    rng = np.random.default_rng(PROBE_RNG_SEED)

    # store per-seed list over splits
    per_seed_alpha: Dict[int, Dict[int, List[np.ndarray]]] = {m: {} for m in MODES}
    per_seed_beta: Dict[Tuple[int, int], Dict[int, List[np.ndarray]]] = {(mi, mj): {} for mi in MODES for mj in MODES}

    for p in paths:
        seed_idx, split_idx = parse_seed_split(p)
        with open(p, "rb") as f:
            model = pickle.load(f)
        model.eval()
        try:
            model.to(torch.device("cpu"))
        except Exception:
            pass

        test = decode_model(model, np.zeros(LATENT_DIM, dtype=np.float32))
        if test.shape[0] != n_feat:
            raise ValueError(f"Decoded length {test.shape[0]} != n_feature {n_feat} (mask mismatch).")

        # alpha
        for m in MODES:
            i0 = int(m) - 1
            a_map = alpha_probe(model, i0=i0, a=PROBE_A_STEP)
            per_seed_alpha[m].setdefault(seed_idx, []).append(a_map)

        # beta
        for mi in MODES:
            i0 = int(mi) - 1
            for mj in MODES:
                j0 = int(mj) - 1
                b_map = beta_interaction_probe_pure_i(model, i0=i0, j0=j0, a=PROBE_A_STEP, delta=PROBE_DELTA_I)
                per_seed_beta[(mi, mj)].setdefault(seed_idx, []).append(b_map)

    seed_list = sorted(set().union(*[set(d.keys()) for d in per_seed_alpha.values()]))

    # split-mean per seed => members
    alpha_members = {
        m: np.stack([np.stack(per_seed_alpha[m][s], axis=0).mean(axis=0) for s in seed_list], axis=0)
        for m in MODES
    }
    beta_members = {
        (mi, mj): np.stack([np.stack(per_seed_beta[(mi, mj)][s], axis=0).mean(axis=0) for s in seed_list], axis=0)
        for mi in MODES for mj in MODES
    }

    # bootstrap => mean + sig
    alpha_maps = []
    alpha_sig = []
    for m in MODES:
        mean, lo, hi = bootstrap_ci_seed(alpha_members[m], n_boot=PROBE_N_BOOT, rng=rng,
                                         ci_lo=PROBE_CI_LO, ci_hi=PROBE_CI_HI)
        da_mean = xr.DataArray(mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_lo   = xr.DataArray(lo,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        da_hi   = xr.DataArray(hi,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
        alpha_maps.append(da_mean)
        alpha_sig.append((da_lo > 0) | (da_hi < 0))

    beta_maps = np.empty((len(MODES), len(MODES)), dtype=object)
    beta_sigs = np.empty((len(MODES), len(MODES)), dtype=object)
    for i, mi in enumerate(MODES):
        for j, mj in enumerate(MODES):
            mean, lo, hi = bootstrap_ci_seed(beta_members[(mi, mj)], n_boot=PROBE_N_BOOT, rng=rng,
                                             ci_lo=PROBE_CI_LO, ci_hi=PROBE_CI_HI)
            da_mean = xr.DataArray(mean, dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
            da_lo   = xr.DataArray(lo,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
            da_hi   = xr.DataArray(hi,   dims=("feature",), coords={"feature": template["feature"]}).unstack("feature")
            beta_maps[i, j] = da_mean
            beta_sigs[i, j] = (da_lo > 0) | (da_hi < 0)

    alpha_probe_da = xr.concat(alpha_maps, dim="mode").assign_coords(mode=MODES)
    beta_probe_da = xr.DataArray(
        np.stack([np.stack([beta_maps[i, j].values for j in range(len(MODES))], axis=0)
                  for i in range(len(MODES))], axis=0),
        dims=("i_mode", "j_mode", "lat", "lon"),
        coords={"i_mode": MODES, "j_mode": MODES,
                "lat": alpha_probe_da["lat"], "lon": alpha_probe_da["lon"]},
        name="beta_probe"
    )
    alpha_probe_sig = xr.concat(alpha_sig, dim="mode").assign_coords(mode=MODES)
    beta_probe_sig = xr.DataArray(
        np.stack([np.stack([beta_sigs[i, j].values for j in range(len(MODES))], axis=0)
                  for i in range(len(MODES))], axis=0),
        dims=("i_mode", "j_mode", "lat", "lon"),
        coords={"i_mode": MODES, "j_mode": MODES,
                "lat": alpha_probe_da["lat"], "lon": alpha_probe_da["lon"]},
        name="beta_probe_sig"
    )

    fields = {
        "alpha_probe": ensure_lon_180(alpha_probe_da),
        "beta_probe": ensure_lon_180(beta_probe_da),
    }
    masks = {}
    if MASK_MODE == "sig_intersection":
        masks["alpha_probe_sig"] = ensure_lon_180(alpha_probe_sig)
        masks["beta_probe_sig"] = ensure_lon_180(beta_probe_sig)
    return fields, masks


# ============================================================
# Build map lists + correlation matrices + LaTeX
# ============================================================
@dataclass
class MapItem:
    key: str
    label: str
    da: xr.DataArray
    sig: Optional[xr.DataArray] = None

def corr_matrix(items: List[MapItem], w_lat: Optional[xr.DataArray]) -> xr.DataArray:
    n = len(items)
    M = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            a = items[i].da
            b = items[j].da
            a2, b2 = apply_pair_mask(a, b, items[i].sig, items[j].sig, MASK_MODE)
            M[i, j] = pattern_corr(a2, b2, w_lat=w_lat)
    return xr.DataArray(M, dims=("row", "col"),
                        coords={"row": [it.label for it in items],
                                "col": [it.label for it in items]})

def to_latex_table(corr: xr.DataArray, caption: str, label: str, bold_thresh: float = 0.7) -> str:
    rows = list(corr["row"].values)
    cols = list(corr["col"].values)

    def fmt(x: float) -> str:
        if not np.isfinite(x):
            return "-"
        s = f"{x:.2f}"
        if abs(x) >= bold_thresh:
            return r"\textbf{" + s + "}"
        return s

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    col_spec = "l" + "c" * len(cols)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for r, rname in enumerate(rows):
        vals = [fmt(float(corr.values[r, c])) for c in range(len(cols))]
        lines.append(rname + " & " + " & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
def main():
    # Load/recompute everything
    reg_fields, reg_masks = load_regression_fields()
    probe_fields, probe_masks = load_probe_fields()
    comp_fields, comp_masks = load_composite_fields()

    # ---- Restrict to Decadal + Interannual only ----
    MODES2 = [MODES[0], MODES[1]]           # assumes MODES = [D, I, Q] in that order
    TITLES2 = [MODE_TITLES[0], MODE_TITLES[1]]  # ["Decadal","Interannual"]

    # weights
    sample = reg_fields["alpha_reg"].sel(tendency_mode=MODES2[0])
    w_lat = area_weights_coslat(sample["lat"]) if USE_COSLAT_WEIGHTS else None

    items: List[MapItem] = []

    # --------------------------
    # Composites: Lin + NL (D,I)
    # --------------------------
    for k, m in enumerate(MODES2):
        name = TITLES2[k]
        items.append(MapItem(
            key=f"comp_lin_{m}",
            label=rf"Comp Lin ${name[0]}$",
            da=comp_fields["comp_lin"].sel(mode=m),
            sig=(comp_masks.get("comp_lin_sig").sel(mode=m) if "comp_lin_sig" in comp_masks else None),
        ))
    for k, m in enumerate(MODES2):
        name = TITLES2[k]
        items.append(MapItem(
            key=f"comp_nl_{m}",
            label=rf"Comp NL ${name[0]}$",
            da=comp_fields["comp_nonlin"].sel(mode=m),
            sig=(comp_masks.get("comp_nonlin_sig").sel(mode=m) if "comp_nonlin_sig" in comp_masks else None),
        ))

    # --------------------------
    # Regression: alpha (D,I)
    # --------------------------
    for k, m in enumerate(MODES2):
        name = TITLES2[k]
        items.append(MapItem(
            key=f"alpha_reg_{m}",
            label=rf"Reg $\alpha_{{{name[0]}}}$",
            da=reg_fields["alpha_reg"].sel(tendency_mode=m),
            sig=(reg_masks.get("alpha_reg_sig").sel(tendency_mode=m) if "alpha_reg_sig" in reg_masks else None),
        ))

    # --------------------------
    # Regression: beta (2x2) => DD, DI, ID, II
    # --------------------------
    for i, mi in enumerate(MODES2):
        for j, mj in enumerate(MODES2):
            li = TITLES2[i][0]; lj = TITLES2[j][0]
            items.append(MapItem(
                key=f"beta_reg_{mi}_{mj}",
                label=rf"Reg $\beta_{{{li}{lj}}}$",
                da=reg_fields["beta_reg"].sel(tendency_mode=mi, predictor_mode=mj),
                sig=(reg_masks.get("beta_reg_sig").sel(tendency_mode=mi, predictor_mode=mj)
                     if "beta_reg_sig" in reg_masks else None),
            ))

    # --------------------------
    # Probes: alpha (D,I)
    # --------------------------
    for k, m in enumerate(MODES2):
        name = TITLES2[k]
        items.append(MapItem(
            key=f"alpha_probe_{m}",
            label=rf"Probe $\alpha_{{{name[0]}}}$",
            da=probe_fields["alpha_probe"].sel(mode=m),
            sig=(probe_masks.get("alpha_probe_sig").sel(mode=m) if "alpha_probe_sig" in probe_masks else None),
        ))

    # --------------------------
    # Probes: beta (2x2) => DD, DI, ID, II
    # --------------------------
    for i, mi in enumerate(MODES2):
        for j, mj in enumerate(MODES2):
            li = TITLES2[i][0]; lj = TITLES2[j][0]
            items.append(MapItem(
                key=f"beta_probe_{mi}_{mj}",
                label=rf"Probe $\beta_{{{li}{lj}}}$",
                da=probe_fields["beta_probe"].sel(i_mode=mi, j_mode=mj),
                sig=(probe_masks.get("beta_probe_sig").sel(i_mode=mi, j_mode=mj)
                     if "beta_probe_sig" in probe_masks else None),
            ))

    # ---- Correlation matrix ----
    corr = corr_matrix(items, w_lat=w_lat)

    # Save
    corr.to_netcdf(OUT_DIR / f"corr_DI_only_mask-{MASK_MODE}.nc")
    corr.to_pandas().to_csv(OUT_DIR / f"corr_DI_only_mask-{MASK_MODE}.csv")

    # LaTeX
    tex = to_latex_table(
        corr,
        caption=rf"Area-weighted spatial pattern correlations for Decadal (D) and Interannual (I) only: composites (Lin/NL), regression ($\alpha,\beta$), and decoder probes ($\alpha,\beta$). Masking: {MASK_MODE}.",
        label=f"tab:corr_DI_only_{MASK_MODE}",
        bold_thresh=0.7
    )
    (OUT_DIR / f"corr_DI_only_mask-{MASK_MODE}.tex").write_text(tex)

    print("Wrote:", OUT_DIR / f"corr_DI_only_mask-{MASK_MODE}.tex")

if __name__ == "__main__":
    main()
