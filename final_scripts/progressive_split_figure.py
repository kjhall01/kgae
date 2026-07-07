#!/usr/bin/env python3
"""
progressive_split_figure_aies.py

AIES-ready figure generation from progressive-split experiment outputs.

Key design/logic choices:
- Ensemble mean over seeds is the statistical object: seed-mean ALWAYS happens first.
- Time bootstrap uses 12-month block bootstrap.
- For each bootstrap replicate, ALL statistics for that period are computed from the SAME resampled time indices.
- Missing files tolerated: partial splits/seed/fold are skipped.

Figure layout:
- 2x2 grid (top): Wasserstein distance (left col) + pattern correlation vs split0 (right col)
- 1 full-width panel (bottom): normalized MSE gaps (shift + generalization; two normalizations)
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance


# -----------------------
# CONFIG
# -----------------------
ROOT = Path("progressive-split")

MODE_INTERANNUAL = 4
MODE_DECADAL = 5
PC_INTERANNUAL = 1
PC_DECADAL = 2

# For publication use 200–500; 400 is a good default.
N_BOOT = 1000
BLOCK = 12
RNG_SEED = 123

# Minimum requirements
MIN_FOLDS_FOR_WASS = 1
MIN_SEEDS_FOR_PATTERN = 1
MIN_FOLDS_FOR_MSE = 1

# Colors (model identity)
COLOR_KGAE = "#1f77b4"   # blue
COLOR_PCA  = "#ff7f0e"   # orange

# --- Styles: unique patterns (avoid repeats) ---
STYLE = {
    # WD
    "wd_within": dict(linestyle="-",  linewidth=2.2),
    "wd_shift":  dict(linestyle="--", linewidth=2.2),

    # Pattern corr (unique)
    "alpha_diag": dict(linestyle="-.", linewidth=2.2),               # dash-dot
    "beta_cross": dict(linestyle=":",  linewidth=2.4),               # dotted
    "eof":        dict(linestyle=(0, (5, 2, 1, 2)), linewidth=2.2),   # long dash-dot-ish

    # Gaps (use markers too; keep linestyles limited)
    "gap_cv":     dict(linestyle="-",  linewidth=2.2),
    "gap_within": dict(linestyle="--", linewidth=2.2),
}

MARK = {
    "shift": "o",   # circle
    "gen":   "s",   # square
}

# File names
FILE_TRAIN_LAT = "train_latents.nc"
FILE_VAL_LAT   = "val_latents.nc"
FILE_TEST_LAT  = "test_latents.nc"

FILE_TRAIN_REC = "train_reconstructions_fixed.nc"
FILE_VAL_REC   = "val_reconstructions.nc"
FILE_TEST_REC  = "test_reconstructions.nc"

FILE_TRAIN_REF = "train_references.nc"
FILE_VAL_REF   = "val_references.nc"
FILE_TEST_REF  = "test_references.nc"

FILE_PCA_TRAIN_LAT = "pca_train_latents.nc"
FILE_PCA_VAL_LAT   = "pca_val_latents.nc"
FILE_PCA_TEST_LAT  = "pca_test_latents.nc"

FILE_PCA_TRAIN_REC = "pca_train_reconstructions_fixed.nc"
FILE_PCA_VAL_REC   = "pca_val_reconstructions.nc"
FILE_PCA_TEST_REC  = "pca_test_reconstructions.nc"

FILE_FULL_ALPHA = "full_alpha.nc"
FILE_FULL_BETA  = "full_beta.nc"
FILE_FULL_PCA_EOFS = "full_pca_eofs.nc"


# -----------------------
# Utilities
# -----------------------
def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def parse_split_dirname(name: str) -> Optional[pd.Timestamp]:
    m = re.match(r"^(\d{4})-(\d{1,2})$", name)
    if not m:
        return None
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)


def try_open_dataarray_any(path: Path) -> Optional[xr.DataArray]:
    if not path.exists():
        return None
    try:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 1:
            return next(iter(ds.data_vars.values()))
        for key in ["sst", "pc", "eof", "alpha", "beta"]:
            if key in ds:
                return ds[key]
        return next(iter(ds.data_vars.values()))
    except Exception as e:
        warn(f"Failed opening {path}: {e}")
        return None


def load_field(sd: Path, fold: int, fname: str) -> Optional[xr.DataArray]:
    fp = sd / f"fold{fold}" / fname
    return try_open_dataarray_any(fp)


def load_latent(sd: Path, fold: int, fname: str, mode: int) -> Optional[xr.DataArray]:
    fp = sd / f"fold{fold}" / fname
    da = try_open_dataarray_any(fp)
    if da is None:
        return None
    try:
        return da.sel(mode=mode)
    except Exception:
        return None


def load_full_alpha(sd: Path) -> Optional[xr.DataArray]:
    fp = sd / FILE_FULL_ALPHA
    return try_open_dataarray_any(fp)


def load_full_beta(sd: Path) -> Optional[xr.DataArray]:
    fp = sd / FILE_FULL_BETA
    return try_open_dataarray_any(fp)


def load_full_pca_eof(seed0_dir: Path, pc: int) -> Optional[xr.DataArray]:
    fp = seed0_dir / FILE_FULL_PCA_EOFS
    da = try_open_dataarray_any(fp)
    if da is None:
        return None
    try:
        return da.sel(mode=pc)
    except Exception:
        return None


def ensemble_mean_over_seeds(arrs: List[xr.DataArray], dim: str = "seed") -> Optional[xr.DataArray]:
    """
    Ensemble mean with union of coords (missing -> NaN, mean(skipna=True)).
    This is the right behavior when some seeds are missing or have different time coords.
    """
    if len(arrs) == 0:
        return None
    try:
        cat = xr.concat(arrs, dim=dim)  # union on coords, NaNs where missing
        return cat.mean(dim, skipna=True)
    except Exception:
        return None


def merge_folds_disjoint(arrs: List[xr.DataArray]) -> Optional[xr.DataArray]:
    """For validation: each fold holds out distinct times → concatenate gives monthly series."""
    if len(arrs) == 0:
        return None
    return xr.concat(arrs, dim="time").sortby("time")


def merge_folds_overlap_mean(arrs: List[xr.DataArray]) -> Optional[xr.DataArray]:
    """
    For train/test: the same times appear across folds → average by time.
    This produces a single monthly series.
    """
    if len(arrs) == 0:
        return None
    x = xr.concat(arrs, dim="time")
    return x.groupby("time").mean("time").sortby("time")


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=k)
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
    return idx


def summarise_boot(b: Optional[np.ndarray]) -> np.ndarray:
    if b is None or len(b) == 0 or not np.isfinite(b).any():
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    b = b[np.isfinite(b)]
    if len(b) == 1:
        return np.array([b[0], b[0], b[0]], dtype=float)
    return np.array([np.nanmedian(b), np.nanquantile(b, 0.05), np.nanquantile(b, 0.95)], dtype=float)


def bootstrap_wasserstein_1d(x: np.ndarray, y: np.ndarray, n_boot: int, block: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    WD bootstrap over time for two 1D series.
    Each replicate draws a block-bootstrap index within each series, then computes WD.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return None
    if len(x) < block or len(y) < block:
        return np.array([wasserstein_distance(x, y)], dtype=float)  # point estimate only

    out = np.empty(n_boot, dtype=float)
    nx, ny = len(x), len(y)
    for i in range(n_boot):
        ix = block_bootstrap_indices(nx, block, rng)
        iy = block_bootstrap_indices(ny, block, rng)
        out[i] = wasserstein_distance(x[ix], y[iy])
    return out


def spatial_corr(a: xr.DataArray, b: xr.DataArray) -> float:
    try:
        a2, b2 = xr.align(a, b, join="inner")
        av = a2.stack(z=("lat", "lon")).values
        bv = b2.stack(z=("lat", "lon")).values
        m = np.isfinite(av) & np.isfinite(bv)
        if m.sum() < 3:
            return np.nan
        av = av[m] - av[m].mean()
        bv = bv[m] - bv[m].mean()
        denom = np.sqrt(np.sum(av**2) * np.sum(bv**2))
        if denom == 0:
            return np.nan
        return float(np.sum(av * bv) / denom)
    except Exception:
        return np.nan


def select_pattern(da: xr.DataArray, tendency_mode: int, predictor_mode: int) -> Optional[xr.DataArray]:
    """
    Robust selection for alpha/beta fields with tendency_mode/predictor_mode dims (or fallbacks).
    Returns lat-lon DataArray if possible.
    """
    if da is None:
        return None
    try:
        x = da
        if "tendency_mode" in x.dims:
            x = x.sel(tendency_mode=tendency_mode)
        elif "mode" in x.dims:
            x = x.sel(mode=tendency_mode)

        if "predictor_mode" in x.dims:
            x = x.sel(predictor_mode=predictor_mode)
        elif "pred_mode" in x.dims:
            x = x.sel(pred_mode=predictor_mode)

        if ("lat" not in x.dims or "lon" not in x.dims) and ("feature" in x.dims):
            try:
                x = x.unstack("feature")
            except Exception:
                pass

        if ("lat" in x.dims) and ("lon" in x.dims):
            return x
        return None
    except Exception:
        return None


@dataclass
class SplitPaths:
    split_date: pd.Timestamp
    split_dir: Path
    seed_dirs: List[Path]


def discover_splits(root: Path) -> List[SplitPaths]:
    splits: List[SplitPaths] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        ts = parse_split_dirname(d.name)
        if ts is None:
            continue
        seed_dirs = sorted([p for p in d.iterdir() if p.is_dir() and p.name.startswith("seed")])
        splits.append(SplitPaths(ts, d, seed_dirs))
    splits.sort(key=lambda s: s.split_date)
    return splits


def compute_wasserstein_panels(splits: List[SplitPaths], mode: int, pc: int) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Fold-consistent WD:

    For each split and fold:
      - KGAE: compute ensemble-mean latent series for train/val/test (seed-mean first)
      - PCA: seed0 latent series for train/val/test
      - bootstrap WD (12-month block) within that fold for:
          train vs val (within)
          val vs test (shift)

    Then for each bootstrap replicate, average across folds (only folds with valid draws).
    Summarize across replicates -> [median, 5%, 95%].
    """
    rng = np.random.default_rng(RNG_SEED)

    out = {"kgae": {"train_val": [], "val_test": []},
           "pca":  {"train_val": [], "val_test": []}}

    for sp in splits:
        seed0 = sp.split_dir / "seed0"

        # collect fold-level bootstrap draws
        k_tv_draws, k_vt_draws = [], []
        p_tv_draws, p_vt_draws = [], []

        for fold in range(5):
            # ---- KGAE: ensemble mean over seeds FIRST (per fold) ----
            kt_list, kv_list, kq_list = [], [], []
            for sd in sp.seed_dirs:
                kt = load_latent(sd, fold, FILE_TRAIN_LAT, mode)
                kv = load_latent(sd, fold, FILE_VAL_LAT, mode)
                kq = load_latent(sd, fold, FILE_TEST_LAT, mode)
                if kt is None or kv is None or kq is None:
                    continue
                kt_list.append(kt); kv_list.append(kv); kq_list.append(kq)

            ktm = ensemble_mean_over_seeds(kt_list)
            kvm = ensemble_mean_over_seeds(kv_list)
            kqm = ensemble_mean_over_seeds(kq_list)

            if ktm is not None and kvm is not None:
                b = bootstrap_wasserstein_1d(ktm.values, kvm.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    k_tv_draws.append(b)

            if kvm is not None and kqm is not None:
                b = bootstrap_wasserstein_1d(kvm.values, kqm.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    k_vt_draws.append(b)

            # ---- PCA: seed0 only (per fold) ----
            pt = load_latent(seed0, fold, FILE_PCA_TRAIN_LAT, pc)
            pv = load_latent(seed0, fold, FILE_PCA_VAL_LAT, pc)
            pq = load_latent(seed0, fold, FILE_PCA_TEST_LAT, pc)

            if pt is not None and pv is not None:
                b = bootstrap_wasserstein_1d(pt.values, pv.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    p_tv_draws.append(b)

            if pv is not None and pq is not None:
                b = bootstrap_wasserstein_1d(pv.values, pq.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    p_vt_draws.append(b)

        def foldmean_summary(draws: List[np.ndarray]) -> np.ndarray:
            if len(draws) == 0:
                return np.array([np.nan, np.nan, np.nan], dtype=float)
            # shape: (n_folds, N_BOOT)
            D = np.vstack(draws)
            # average across folds per replicate -> (N_BOOT,)
            m = np.nanmean(D, axis=0)
            return summarise_boot(m)

        out["kgae"]["train_val"].append(foldmean_summary(k_tv_draws))
        out["kgae"]["val_test"].append(foldmean_summary(k_vt_draws))
        out["pca"]["train_val"].append(foldmean_summary(p_tv_draws))
        out["pca"]["val_test"].append(foldmean_summary(p_vt_draws))

    for model in out:
        for key in out[model]:
            out[model][key] = np.vstack(out[model][key])
    return out



# -----------------------
# Pattern corr computation (NO CIs, ensemble mean only)
# -----------------------
def compute_patterncorr_panels(splits: List[SplitPaths]) -> Dict[str, np.ndarray]:
    """
    Pattern corr vs split0:
      Interannual row: alpha diag (4<-4), beta cross (4<-5), EOF1
      Decadal row:     alpha diag (5<-5), beta cross (5<-4), EOF2
    """
    out = {k: [] for k in ["a44", "b45", "eof1", "a55", "b54", "eof2"]}

    sp0 = splits[0]
    seed0_0 = sp0.split_dir / "seed0"

    # split0 alpha mean
    alpha0_list = [a for sd in sp0.seed_dirs if (a := load_full_alpha(sd)) is not None]
    alpha0 = None
    if len(alpha0_list) >= MIN_SEEDS_FOR_PATTERN:
        try:
            alpha0 = xr.concat(xr.align(*alpha0_list, join="inner"), dim="seed").mean("seed")
        except Exception:
            alpha0 = None

    # split0 beta mean
    beta0_list = [b for sd in sp0.seed_dirs if (b := load_full_beta(sd)) is not None]
    beta0 = None
    if len(beta0_list) >= MIN_SEEDS_FOR_PATTERN:
        try:
            beta0 = xr.concat(xr.align(*beta0_list, join="inner"), dim="seed").mean("seed")
        except Exception:
            beta0 = None

    ref_a44 = select_pattern(alpha0, MODE_INTERANNUAL, MODE_INTERANNUAL) if alpha0 is not None else None
    ref_a55 = select_pattern(alpha0, MODE_DECADAL,     MODE_DECADAL)     if alpha0 is not None else None
    ref_b45 = select_pattern(beta0,  MODE_INTERANNUAL, MODE_DECADAL)     if beta0 is not None else None
    ref_b54 = select_pattern(beta0,  MODE_DECADAL,     MODE_INTERANNUAL) if beta0 is not None else None

    ref_eof1 = load_full_pca_eof(seed0_0, PC_INTERANNUAL)
    ref_eof2 = load_full_pca_eof(seed0_0, PC_DECADAL)

    for sp in splits:
        seed0 = sp.split_dir / "seed0"

        alpha_list = [a for sd in sp.seed_dirs if (a := load_full_alpha(sd)) is not None]
        beta_list  = [b for sd in sp.seed_dirs if (b := load_full_beta(sd))  is not None]

        alpha_mean = None
        beta_mean = None

        if len(alpha_list) >= MIN_SEEDS_FOR_PATTERN:
            try:
                alpha_mean = xr.concat(xr.align(*alpha_list, join="inner"), dim="seed").mean("seed")
            except Exception:
                alpha_mean = None

        if len(beta_list) >= MIN_SEEDS_FOR_PATTERN:
            try:
                beta_mean = xr.concat(xr.align(*beta_list, join="inner"), dim="seed").mean("seed")
            except Exception:
                beta_mean = None

        a44 = select_pattern(alpha_mean, MODE_INTERANNUAL, MODE_INTERANNUAL) if alpha_mean is not None else None
        a55 = select_pattern(alpha_mean, MODE_DECADAL,     MODE_DECADAL)     if alpha_mean is not None else None
        b45 = select_pattern(beta_mean,  MODE_INTERANNUAL, MODE_DECADAL)     if beta_mean  is not None else None
        b54 = select_pattern(beta_mean,  MODE_DECADAL,     MODE_INTERANNUAL) if beta_mean  is not None else None

        eof1 = load_full_pca_eof(seed0, PC_INTERANNUAL)
        eof2 = load_full_pca_eof(seed0, PC_DECADAL)

        out["a44"].append(spatial_corr(a44, ref_a44) if (a44 is not None and ref_a44 is not None) else np.nan)
        out["a55"].append(spatial_corr(a55, ref_a55) if (a55 is not None and ref_a55 is not None) else np.nan)
        out["b45"].append(spatial_corr(b45, ref_b45) if (b45 is not None and ref_b45 is not None) else np.nan)
        out["b54"].append(spatial_corr(b54, ref_b54) if (b54 is not None and ref_b54 is not None) else np.nan)
        out["eof1"].append(spatial_corr(eof1, ref_eof1) if (eof1 is not None and ref_eof1 is not None) else np.nan)
        out["eof2"].append(spatial_corr(eof2, ref_eof2) if (eof2 is not None and ref_eof2 is not None) else np.nan)

    for k in out:
        out[k] = np.asarray(out[k], dtype=float)
    return out


# -----------------------
# MSE gaps computation (bootstrap-consistent within each period)
# -----------------------
def per_time_space_mse_series(rec: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    rec2, ref2 = xr.align(rec, ref, join="inner")
    return ((rec2 - ref2) ** 2).mean(dim=[d for d in rec2.dims if d != "time"])


def per_time_space_moments(ref: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    m1 = ref.mean(dim=[d for d in ref.dims if d != "time"])
    m2 = (ref ** 2).mean(dim=[d for d in ref.dims if d != "time"])
    return m1, m2


def bootstrap_stats_for_period(e: np.ndarray, m1: np.ndarray, m2: np.ndarray,
                               n_boot: int, block: int, rng: np.random.Generator) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    e = np.asarray(e); m1 = np.asarray(m1); m2 = np.asarray(m2)
    e = e[np.isfinite(e)]
    # For moments, we need matching length; safest is to filter jointly by finite mask on all 3
    # (using the original indices).
    # Here, assume they share same length and filter with combined mask:
    # (we rebuild with combined mask below)
    # If lengths differ, bail.
    if not (len(e) == len(m1) == len(m2)):
        # fall back: require equal lengths from the start (should be true)
        n = min(len(e), len(m1), len(m2))
        e, m1, m2 = e[:n], m1[:n], m2[:n]

    mask = np.isfinite(e) & np.isfinite(m1) & np.isfinite(m2)
    e, m1, m2 = e[mask], m1[mask], m2[mask]
    n = len(e)
    if n < block:
        return None, None

    mse_b = np.empty(n_boot, float)
    var_b = np.empty(n_boot, float)

    for i in range(n_boot):
        idx = block_bootstrap_indices(n, block, rng)
        mse_b[i] = float(np.mean(e[idx]))
        mu = float(np.mean(m1[idx]))
        ex2 = float(np.mean(m2[idx]))
        var_b[i] = ex2 - mu * mu

    return mse_b, var_b

def compute_mse_gaps(splits: List[SplitPaths]) -> Dict[str, np.ndarray]:
    """
    Fold-consistent gap statistics:

    For each split and fold:
      - use that fold's references and reconstructions
      - KGAE recon = seed-mean first (within the fold)
      - PCA recon = seed0
      - compute per-time spatial MSE series and ref moments series within fold
      - bootstrap (12-month blocks) within each period using SAME time indices for mse and variance
      - compute 4 gap metrics per bootstrap replicate

    Then: average metrics across folds per replicate -> fold-mean bootstrap distribution.
    Summarize -> (4 metrics, 3 summary stats).
    """
    rng = np.random.default_rng(RNG_SEED + 7)

    out_k = []
    out_p = []

    for sp in splits:
        seed0 = sp.split_dir / "seed0"

        # per-fold bootstrap matrices (N_BOOT, 4)
        mats_k = []
        mats_p = []

        for fold in range(5):
            tr_ref = load_field(seed0, fold, FILE_TRAIN_REF)
            va_ref = load_field(seed0, fold, FILE_VAL_REF)
            te_ref = load_field(seed0, fold, FILE_TEST_REF)
            if tr_ref is None or va_ref is None or te_ref is None:
                continue

            # ---- KGAE fold recon: ensemble mean over seeds FIRST ----
            tr_list, va_list, te_list = [], [], []
            for sd in sp.seed_dirs:
                tr = load_field(sd, fold, FILE_TRAIN_REC)
                va = load_field(sd, fold, FILE_VAL_REC)
                te = load_field(sd, fold, FILE_TEST_REC)
                if tr is None or va is None or te is None:
                    continue
                tr_list.append(tr); va_list.append(va); te_list.append(te)

            trm = ensemble_mean_over_seeds(tr_list)
            vam = ensemble_mean_over_seeds(va_list)
            tem = ensemble_mean_over_seeds(te_list)

            # ---- PCA fold recon: seed0 ----
            ptr = load_field(seed0, fold, FILE_PCA_TRAIN_REC)
            pva = load_field(seed0, fold, FILE_PCA_VAL_REC)
            pte = load_field(seed0, fold, FILE_PCA_TEST_REC)

            def fold_mat(rec_tr, rec_va, rec_te, ref_tr, ref_va, ref_te) -> Optional[np.ndarray]:
                if any(v is None for v in [rec_tr, rec_va, rec_te, ref_tr, ref_va, ref_te]):
                    return None

                # align within fold (should be consistent)
                rec_tr, ref_tr = xr.align(rec_tr, ref_tr, join="inner")
                rec_va, ref_va = xr.align(rec_va, ref_va, join="inner")
                rec_te, ref_te = xr.align(rec_te, ref_te, join="inner")

                # per-time spatial MSE
                e_tr = per_time_space_mse_series(rec_tr, ref_tr).values
                e_va = per_time_space_mse_series(rec_va, ref_va).values
                e_te = per_time_space_mse_series(rec_te, ref_te).values

                # per-time spatial moments for var(space-time) with time bootstrap
                m1_tr, m2_tr = per_time_space_moments(ref_tr); m1_tr = m1_tr.values; m2_tr = m2_tr.values
                m1_va, m2_va = per_time_space_moments(ref_va); m1_va = m1_va.values; m2_va = m2_va.values
                m1_te, m2_te = per_time_space_moments(ref_te); m1_te = m1_te.values; m2_te = m2_te.values

                mse_tr, var_tr = bootstrap_stats_for_period(e_tr, m1_tr, m2_tr, N_BOOT, BLOCK, rng)
                mse_va, var_va = bootstrap_stats_for_period(e_va, m1_va, m2_va, N_BOOT, BLOCK, rng)
                mse_te, var_te = bootstrap_stats_for_period(e_te, m1_te, m2_te, N_BOOT, BLOCK, rng)

                if mse_tr is None or mse_va is None or mse_te is None:
                    return None

                # 4 metrics per replicate
                shift_cv     = (mse_te - mse_va) / var_va
                shift_within = (mse_te / var_te) - (mse_va / var_va)
                gen_cv       = (mse_va - mse_tr) / var_va
                gen_within   = (mse_va / var_va) - (mse_tr / var_tr)

                return np.vstack([shift_cv, shift_within, gen_cv, gen_within]).T  # (N_BOOT,4)

            mk = fold_mat(trm, vam, tem, tr_ref, va_ref, te_ref)
            if mk is not None and mk.shape == (N_BOOT, 4):
                mats_k.append(mk)

            mp = fold_mat(ptr, pva, pte, tr_ref, va_ref, te_ref)
            if mp is not None and mp.shape == (N_BOOT, 4):
                mats_p.append(mp)

        def foldmean_summarize(mats: List[np.ndarray]) -> np.ndarray:
            if len(mats) == 0:
                return np.full((4, 3), np.nan)
            # mats: list of (N_BOOT,4) -> stack (n_folds, N_BOOT, 4)
            M = np.stack(mats, axis=0)
            # average across folds per replicate -> (N_BOOT,4)
            Mm = np.nanmean(M, axis=0)

            med = np.nanmedian(Mm, axis=0)
            lo  = np.nanquantile(Mm, 0.05, axis=0)
            hi  = np.nanquantile(Mm, 0.95, axis=0)
            return np.vstack([med, lo, hi]).T  # (4,3)

        out_k.append(foldmean_summarize(mats_k))
        out_p.append(foldmean_summarize(mats_p))

    return {"kgae": np.stack(out_k, axis=0), "pca": np.stack(out_p, axis=0)}

# -----------------------
# Plot helper
# -----------------------
def plot_ci(ax, x, summary, style, label=None, zorder=3, alpha_fill=0.14):
    med, lo, hi = summary[:, 0], summary[:, 1], summary[:, 2]
    ax.plot(x, med, label=label, zorder=zorder, **style)
    ax.fill_between(x, lo, hi, color=style.get("color", "0.2"), alpha=alpha_fill, linewidth=0, zorder=zorder-1)


def label_last(ax, x, y, text, color, dx=0.12, dy=0.0):
    """Inline label at the last finite point (journal-friendly; reduces legend clutter)."""
    y = np.asarray(y)
    m = np.isfinite(y)
    if not m.any():
        return
    i = np.where(m)[0][-1]
    ax.text(x[i] + dx, y[i] + dy, text, color=color, fontsize=8,
            va="center", ha="left", clip_on=False)


# -----------------------
# Main
# -----------------------
def main():

    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    import numpy as np

    if not ROOT.exists():
        raise SystemExit(f"Root directory not found: {ROOT.resolve()}")

    splits = discover_splits(ROOT)
    if len(splits) == 0:
        raise SystemExit("No split directories found.")

    x_dates = [sp.split_date for sp in splits]
    x = np.arange(len(splits))

    w_inter = compute_wasserstein_panels(splits, MODE_INTERANNUAL, PC_INTERANNUAL)
    w_deca  = compute_wasserstein_panels(splits, MODE_DECADAL,     PC_DECADAL)
    patt    = compute_patterncorr_panels(splits)
    gaps    = compute_mse_gaps(splits)


    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
    })

    # Double-column journal-friendly aspect.
    fig = plt.figure(figsize=(8.5, 9.2))
    gs = GridSpec(
        nrows=4, ncols=2,
        height_ratios=[1.05, 1.05, 0.95, 0.95],
        hspace=0.35, wspace=0.25,
        figure=fig
    )

    ax_w_i = fig.add_subplot(gs[0, 0])
    ax_p_i = fig.add_subplot(gs[0, 1])
    ax_w_d = fig.add_subplot(gs[1, 0])
    ax_p_d = fig.add_subplot(gs[1, 1])

    ax_gap_shift = fig.add_subplot(gs[2, :])   # e
    ax_gap_gen   = fig.add_subplot(gs[3, :])   # f

    def add_panel_letter(ax, letter):
        ax.text(
            -0.02, 1.05, f"{letter})",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=11, fontweight="bold",
            clip_on=False
        )

    # -----------------------------
    # Panel a: WD Interannual / PC1
    # -----------------------------
    plot_ci(ax_w_i, x, w_inter["kgae"]["train_val"], dict(color=COLOR_KGAE, **STYLE["wd_within"]))
    plot_ci(ax_w_i, x, w_inter["kgae"]["val_test"],  dict(color=COLOR_KGAE, **STYLE["wd_shift"]))
    plot_ci(ax_w_i, x, w_inter["pca"]["train_val"],  dict(color=COLOR_PCA,  **STYLE["wd_within"]))
    plot_ci(ax_w_i, x, w_inter["pca"]["val_test"],   dict(color=COLOR_PCA,  **STYLE["wd_shift"]))

    #ax_w_i.set_title("Latent distribution distance (Interannual / PC1)")
    ax_w_i.set_ylabel("Wasserstein distance")
    ax_w_i.set_xlim(x.min(), x.max())
    add_panel_letter(ax_w_i, "a")

    # -----------------------------
    # Panel b: WD Decadal / PC2
    # -----------------------------
    plot_ci(ax_w_d, x, w_deca["kgae"]["train_val"], dict(color=COLOR_KGAE, **STYLE["wd_within"]))
    plot_ci(ax_w_d, x, w_deca["kgae"]["val_test"],  dict(color=COLOR_KGAE, **STYLE["wd_shift"]))
    plot_ci(ax_w_d, x, w_deca["pca"]["train_val"],  dict(color=COLOR_PCA,  **STYLE["wd_within"]))
    plot_ci(ax_w_d, x, w_deca["pca"]["val_test"],   dict(color=COLOR_PCA,  **STYLE["wd_shift"]))

    #ax_w_d.set_title("Latent distribution distance (Decadal / PC2)")
    ax_w_d.set_ylabel("Wasserstein distance")
    ax_w_d.set_xlim(x.min(), x.max())
    add_panel_letter(ax_w_d, "b")

    # -----------------------------
    # Panel c: Pattern correlations (Interannual row)
    # -----------------------------
    ax_p_i.plot(x, patt["a44"],  color=COLOR_KGAE, **STYLE["alpha_diag"])
    #ax_p_i.plot(x, patt["b45"],  color=COLOR_KGAE, **STYLE["beta_cross"])
    ax_p_i.plot(x, patt["eof1"], color=COLOR_PCA,  **STYLE["eof"])

    #ax_p_i.set_title("Pattern corr. vs split-0 (Interannual)")
    ax_p_i.set_ylabel("Spatial correlation")
    ax_p_i.set_xlim(x.min(), x.max())
    add_panel_letter(ax_p_i, "c")

    # -----------------------------
    # Panel d: Pattern correlations (Decadal row)
    # -----------------------------
    ax_p_d.plot(x, patt["a55"],  color=COLOR_KGAE, **STYLE["alpha_diag"])
    #ax_p_d.plot(x, patt["b54"],  color=COLOR_KGAE, **STYLE["beta_cross"])
    ax_p_d.plot(x, patt["eof2"], color=COLOR_PCA,  **STYLE["eof"])

    #ax_p_d.set_title("Pattern corr. vs split-0 (Decadal)")
    ax_p_d.set_ylabel("Spatial correlation")
    ax_p_d.set_xlim(x.min(), x.max())
    add_panel_letter(ax_p_d, "d")

    # -----------------------------
    # Bottom panels: gaps
    # Use marker+linestyle combos (distinct) instead of trying to create 4 unique linestyles
    # -----------------------------
    def plot_shift(ax, model_key, color):
        arr = gaps[model_key]
        # shift_cv (circle, solid)
        plot_ci(ax, x, arr[:, 0, :], dict(color=color, marker=MARK["shift"], markevery=1, **STYLE["gap_cv"]))
        # shift_within (circle, dashed)
        plot_ci(ax, x, arr[:, 1, :], dict(color=color, marker=MARK["shift"], markevery=1, **STYLE["gap_within"]))

    def plot_gen(ax, model_key, color):
        arr = gaps[model_key]
        # gen_cv (square, solid)
        plot_ci(ax, x, arr[:, 2, :], dict(color=color, marker=MARK["gen"], markevery=1, **STYLE["gap_cv"]))
        # gen_within (square, dashed)
        plot_ci(ax, x, arr[:, 3, :], dict(color=color, marker=MARK["gen"], markevery=1, **STYLE["gap_within"]))

    plot_shift(ax_gap_shift, "kgae", COLOR_KGAE)
    plot_shift(ax_gap_shift, "pca",  COLOR_PCA)
    ax_gap_shift.axhline(0, color="0.7", linewidth=1)
    #ax_gap_shift.set_title("Normalized reconstruction shift gaps (Val→Test; 12-month block bootstrap)")
    ax_gap_shift.set_ylabel("Normalized MSE gap")
    ax_gap_shift.set_xlim(x.min(), x.max())
    add_panel_letter(ax_gap_shift, "e")

    plot_gen(ax_gap_gen, "kgae", COLOR_KGAE)
    plot_gen(ax_gap_gen, "pca",  COLOR_PCA)
    ax_gap_gen.axhline(0, color="0.7", linewidth=1)
   # ax_gap_gen.set_title("Normalized reconstruction generalization gaps (Train→Val; 12-month block bootstrap)")
    ax_gap_gen.set_ylabel("Normalized MSE gap")
    ax_gap_gen.set_xlim(x.min(), x.max())
    ax_gap_gen.set_xlabel("Split date (training end)")
    add_panel_letter(ax_gap_gen, "f")

    # -----------------------------
    # X ticks: equally spaced, YYYY-MM
    # x is already equally spaced; choose a constant stride for readability
    # -----------------------------
    n = len(x_dates)
    tick_step = max(1, int(np.ceil(n / 8)))  # roughly 8 labels across
    tick_idx = np.arange(0, n, tick_step)
    tick_labels = [x_dates[i].strftime("%Y-%m") for i in tick_idx]

    for ax in (ax_w_i, ax_p_i, ax_w_d, ax_p_d, ax_gap_shift, ax_gap_gen):
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")

    # -----------------------------
    # One clean legend below everything (no overlap)
    # -----------------------------
    handles = [
        # Model colors
        Line2D([0], [0], color=COLOR_KGAE, linewidth=3, label="KGAE"),
        Line2D([0], [0], color=COLOR_PCA,  linewidth=3, label="PCA"),

        # Top panels encodings
        Line2D([0], [0], color="k", **STYLE["wd_within"], label="Train–Val WD"),
        Line2D([0], [0], color="k", **STYLE["wd_shift"],  label="Val–Test WD"),
        Line2D([0], [0], color="k", **STYLE["alpha_diag"], label="α (pattern corr)"),
       # Line2D([0], [0], color="k", **STYLE["beta_cross"], label="β (pattern corr)"),
        Line2D([0], [0], color="k", **STYLE["eof"],        label="EOF (PCA)"),

        # Bottom panels encodings
        Line2D([0], [0], color="k", marker=MARK["shift"], **STYLE["gap_cv"],     label="XVal-Test NMSE Gap (xval norm)"),
        Line2D([0], [0], color="k", marker=MARK["shift"], **STYLE["gap_within"], label="XVal-Test NMSE Gap (self-norm)"),
        Line2D([0], [0], color="k", marker=MARK["gen"],   **STYLE["gap_cv"],     label="Train-Val MSE Gap (val norm)"),
        Line2D([0], [0], color="k", marker=MARK["gen"],   **STYLE["gap_within"], label="Train-Val MSE Gap (self-norm)"),
    ]

    # Reserve bottom margin for legend
    fig.subplots_adjust(left=0.075, right=0.995, top=0.95, bottom=0.16)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=3.0,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.02),
    )

    # ---- Save (avoid cutoff) ----
    outpath = Path("progressive_split_figure_final.png")
    outpdf  = Path("progressive_split_figure_final.pdf")
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outpdf, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote: {outpath.resolve()}")
    print(f"Wrote: {outpdf.resolve()}")


if __name__ == "__main__":
    main()
