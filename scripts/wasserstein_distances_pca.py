#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hashlib
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance

import torch
import torch.nn as nn
import kgae


# ============================================================
# Cache utilities
# ============================================================

def _stable_hash(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _npz_save(path: Path, **arrays):
    _ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)

def _npz_load(path: Path):
    return dict(np.load(path, allow_pickle=True))

def _json_save(path: Path, obj: dict):
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


# ============================================================
# Preprocessing / stacking
# ============================================================

def preprocess_train_test_contiguous(sst, frac_train, deg=2):
    split_index = int(sst.sizes["time"] * frac_train)
    train_raw = sst.isel(time=slice(None, split_index)).sortby("time")
    test_raw  = sst.isel(time=slice(split_index, None)).sortby("time")

    train_dt, fit, gwm, p = kgae.global_detrend(train_raw, deg=deg)
    train_anom, monthly_clim = kgae.remove_climo(train_dt)

    # Apply train detrend to test
    test_dt = test_raw - xr.polyval(test_raw["time"], p.polyfit_coefficients)

    # Apply train monthly climatology to test (yearly monthly mean - climo)
    dim = "time"
    monthly = test_dt
    toconcat = []
    for year in sorted({pd.Timestamp(t).year for t in monthly[dim].values}):
        ds_yearly = (
            monthly.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            .groupby(f"{dim}.month")
            .mean()
            - monthly_clim
        )
        ds_yearly = ds_yearly.assign_coords(
            {"month": [pd.Timestamp(year, m, 1) for m in ds_yearly.month.values]}
        ).rename({"month": dim})
        toconcat.append(ds_yearly)

    test_anom = xr.concat(toconcat, dim).sortby(dim).sortby("time")
    return train_anom, test_anom

def stack_time_feature(da):
    stacked = (
        da.stack(feature=("lat", "lon"))
        .dropna("feature", how="any")
        .transpose("time", "feature")
    )
    return stacked.values, stacked.feature, stacked.time.values

def build_reference_pattern(train_anom):
    oni = train_anom.sel(lat=slice(-5, 5), lon=slice(10, 60)).mean(("lat", "lon"))
    corr = xr.corr(train_anom, oni, dim="time")
    ref = corr.stack(feature=("lat", "lon")).dropna("feature", how="any").values
    return ref


# ============================================================
# Block bootstrap helpers
# ============================================================

def circular_block_indices(n, block_len, rng):
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend([(s + k) % n for k in range(block_len)])
    return np.array(idx[:n], dtype=int)

def ws_blockboot_1d(x_tr, x_te, n_boot, block_len, rng):
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        it = circular_block_indices(len(x_tr), block_len, rng)
        ie = circular_block_indices(len(x_te), block_len, rng)
        out[b] = wasserstein_distance(x_tr[it], x_te[ie])
    return out


# ============================================================
# Correlation utilities (robust)
# ============================================================

def corrcoef_finite(a, b, min_n=2000, eps=1e-12):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    n_ok = int(ok.sum())
    if n_ok < min_n:
        return np.nan
    aa = a[ok]; bb = b[ok]
    if float(np.std(aa)) < eps or float(np.std(bb)) < eps:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])

def canonical_alpha(alpha, n_components):
    """
    Return alpha in canonical shape (lat, lon, mode).
    Accepts (mode, lat, lon) or (lat, lon, mode).
    """
    alpha = np.asarray(alpha)
    if alpha.ndim != 3:
        raise ValueError(f"alpha must be 3D, got shape {alpha.shape}")
    if alpha.shape[0] == n_components and alpha.shape[-1] != n_components:
        alpha = np.moveaxis(alpha, 0, -1)  # (lat, lon, mode)
    if alpha.shape[-1] != n_components:
        raise ValueError(f"Expected alpha last axis to be n_components={n_components}, got {alpha.shape}")
    return alpha

def canonical_beta(beta, n_components):
    """
    Return beta in canonical shape (lat, lon, tendency_mode, predictor_mode).

    Accepts common permutations:
      - (lat, lon, tend, pred)  [preferred]
      - (tend, pred, lat, lon)
      - (lat, lon, pred, tend) (will be rejected unless you reorder before saving)
    """
    beta = np.asarray(beta)
    if beta.ndim != 4:
        raise ValueError(f"beta must be 4D, got shape {beta.shape}")

    # if leading axes are tend/pred
    if beta.shape[0] == n_components and beta.shape[1] == n_components and beta.shape[2] != n_components:
        beta = np.moveaxis(beta, (0, 1), (-2, -1))  # (lat,lon,tend,pred) if it was (tend,pred,lat,lon)

    if beta.shape[-2] != n_components or beta.shape[-1] != n_components:
        raise ValueError(
            f"Expected last two axes to be (tend,pred)=({n_components},{n_components}), got {beta.shape}"
        )
    # now should be (lat,lon,tend,pred)
    return beta


# ============================================================
# Reconstruction metric helpers
# ============================================================

def mse_timeseries(X, Xhat):
    # X, Xhat: (time, feature)
    e = (X - Xhat) ** 2
    return e.mean(axis=1).astype(np.float32)

def energy_timeseries(X):
    # proxy for mean spatial variance when mean is ~0 (anomalies)
    return (X ** 2).mean(axis=1).astype(np.float32)

def pca_cv_test_recon_ts(X_tr, X_te, n_components=2, n_splits=5):
    """
    Mimic the KGAE fold logic for PCA reconstruction:
      - Build a CV reconstruction time series by training PCA on (train / val) and reconstructing val
      - Build a test reconstruction by averaging reconstructions from each fold-trained PCA

    Returns:
      mse_cv_ts_pca (n_train,)
      eng_cv_ts_pca (n_train,)
      mse_te_ts_pca (n_test,)
      eng_te_ts_pca (n_test,)
    """
    n_tr = X_tr.shape[0]
    n_te = X_te.shape[0]

    mse_cv = np.full(n_tr, np.nan, dtype=np.float32)
    eng_cv = np.full(n_tr, np.nan, dtype=np.float32)

    # test recon: average reconstructions across folds (like KGAE fold-mean)
    X_te_hat_sum = np.zeros_like(X_te, dtype=np.float32)

    for split in range(n_splits):
        val_idx = np.arange(split, n_tr, n_splits)
        train_idx = np.setdiff1d(np.arange(n_tr), val_idx)

        pca = PCA(n_components=n_components)
        pca.fit(X_tr[train_idx])

        # reconstruct val
        pcs_val = pca.transform(X_tr[val_idx])
        X_val_hat = pca.inverse_transform(pcs_val).astype(np.float32)

        mse_cv[val_idx] = mse_timeseries(X_tr[val_idx].astype(np.float32), X_val_hat)
        eng_cv[val_idx] = energy_timeseries(X_tr[val_idx].astype(np.float32))

        # reconstruct test (accumulate for fold-mean)
        pcs_te = pca.transform(X_te)
        X_te_hat = pca.inverse_transform(pcs_te).astype(np.float32)
        X_te_hat_sum += X_te_hat

    X_te_hat_mean = (X_te_hat_sum / float(n_splits)).astype(np.float32)

    mse_te = mse_timeseries(X_te.astype(np.float32), X_te_hat_mean)
    eng_te = energy_timeseries(X_te.astype(np.float32))

    return mse_cv.astype(np.float32), eng_cv.astype(np.float32), mse_te.astype(np.float32), eng_te.astype(np.float32)


def boot_mean_from_ts(ts, n_boot, block_len, rng):
    n = len(ts)
    out = np.empty(n_boot, dtype=np.float32)
    for b in range(n_boot):
        idx = circular_block_indices(n, block_len, rng)
        out[b] = float(np.mean(ts[idx]))
    return out

def boot_recon_gaps(
    mse_cv_ts, mse_te_ts,
    eng_cv_ts, eng_te_ts,
    n_boot, block_len, rng,
    var_ref_cv=None
):
    """
    Bootstrap distributions for:
      - MSE gap: mse_te - mse_cv
      - NMSE_within gap: (mse_te/var_te) - (mse_cv/var_cv)
      - NMSE_refcv gap: (mse_te - mse_cv) / var_ref_cv
    Variance proxy uses mean(eng_ts) in the resample (anomalies assumed mean ~0).
    """
    mse_cv_b = boot_mean_from_ts(mse_cv_ts, n_boot, block_len, rng)
    mse_te_b = boot_mean_from_ts(mse_te_ts, n_boot, block_len, rng)

    var_cv_b = boot_mean_from_ts(eng_cv_ts, n_boot, block_len, rng)
    var_te_b = boot_mean_from_ts(eng_te_ts, n_boot, block_len, rng)

    var_cv_b = np.where(var_cv_b == 0, np.nan, var_cv_b)
    var_te_b = np.where(var_te_b == 0, np.nan, var_te_b)

    gap_mse = mse_te_b - mse_cv_b
    gap_nmse_within = (mse_te_b / var_te_b) - (mse_cv_b / var_cv_b)

    if var_ref_cv is None or var_ref_cv == 0:
        gap_nmse_refcv = np.full_like(gap_mse, np.nan, dtype=np.float32)
    else:
        gap_nmse_refcv = gap_mse / float(var_ref_cv)

    return gap_mse, gap_nmse_within, gap_nmse_refcv


# ============================================================
# KGAE CV latents + alpha/beta + recon metrics
# ============================================================

def kgae_cv_latents_alpha_beta_recon(
    train_anom, test_anom, reference_pattern,
    modes_to_examine=(4, 5),
    n_epochs=150,
    n_splits=5,
    random_seed=0,
    delta=1e-2,
    device="cpu",
):
    """
    Returns:
      z_sel: xval latents (time, mode_sel)
      z_te:  test latents (time, mode_sel) fold-mean
      alpha: alpha map (lat,lon,mode_sel)
      beta:  beta map  (lat,lon,tendency_mode, predictor_mode) restricted to mode_sel pairs
      recon metrics:
        mse_cv_ts (time,), eng_cv_ts (time,)
        mse_te_ts (time,), eng_te_ts (time,)  (test fold-mean)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    splits = [train_anom.isel(time=slice(i, None, n_splits)) for i in range(n_splits)]

    xval_latents = []
    xval_tendencies = []

    mse_cv_list = []
    eng_cv_list = []

    test_latents_list = []
    mse_te_folds = []
    eng_te_folds = []

    for split in range(n_splits):
        train_fold = xr.concat([splits[j] for j in range(n_splits) if j != split], dim="time").sortby("time")
        val_fold = splits[split].sortby("time")

        train_splits = [train_fold.isel(time=slice(j, None, n_splits - 1)) for j in range(n_splits - 1)]
        training_data = []
        for t in train_splits:
            X, _, _ = stack_time_feature(t)
            training_data.append(X)

        X_val, feat, t_val = stack_time_feature(val_fold)
        X_test, feat_te, t_te = stack_time_feature(test_anom)

        oae = kgae.KGAE(
            input_dim=training_data[0].shape[1],
            latent_dim=5,
            is_variational=True,
            hidden_layers=[248, 248],
            activation=nn.Tanh,
            power_spectrum_smoothing_kernel=7,
            device=device
        )
        oae.fit(training_data=training_data, val_data=X_val, num_epochs=n_epochs)
        oae.eval()

        # sorting-based alignment (ensures modes 4/5 are interannual/decadal)
        oae.sort_latents_by_frequency(X_val, reference_pattern=reference_pattern)

        # train stats for standardization
        X_train_full = np.vstack(training_data)
        with torch.no_grad():
            z_train = oae.encoder(torch.tensor(X_train_full, dtype=torch.float32))
            if oae.is_variational:
                z_train = z_train[:, :oae.latent_dim]
        z_train = z_train.detach().cpu().numpy() * oae.flip_signs.values
        mu = z_train.mean(axis=0)
        sig = z_train.std(axis=0)
        sig[sig == 0] = 1.0

        # ---- val latents + recon ----
        with torch.no_grad():
            z_val = oae.encoder(torch.tensor(X_val, dtype=torch.float32))
            if oae.is_variational:
                z_val = z_val[:, :oae.latent_dim]
            X_val_hat = oae.decoder(z_val)

        z_val = z_val.detach().cpu().numpy() * oae.flip_signs.values
        z_val = (z_val - mu) / sig
        X_val_hat = X_val_hat.detach().cpu().numpy()

        z_val_da = xr.DataArray(
            z_val,
            coords={"time": t_val, "mode": np.arange(1, oae.latent_dim + 1)},
            dims=("time", "mode"),
        ).sortby("time")
        xval_latents.append(z_val_da)

        mse_cv_list.append(
            xr.DataArray(mse_timeseries(X_val, X_val_hat), coords={"time": t_val}, dims=("time",)).sortby("time")
        )
        eng_cv_list.append(
            xr.DataArray(energy_timeseries(X_val), coords={"time": t_val}, dims=("time",)).sortby("time")
        )

        # tendencies via jacobian (still needed to get alpha/beta)
        tend = kgae.jacobian(oae, torch.tensor(X_val, dtype=torch.float32), delta=delta, device=device)
        tend = tend.detach().cpu().numpy().astype(np.float32)

        tend_da = xr.DataArray(
            tend,
            coords={"time": t_val, "mode": np.arange(1, oae.latent_dim + 1), "feature": feat},
            dims=("time", "mode", "feature"),
        ).sortby("time")
        xval_tendencies.append(tend_da)

        # ---- test latents + recon (per fold) ----
        with torch.no_grad():
            z_te = oae.encoder(torch.tensor(X_test, dtype=torch.float32))
            if oae.is_variational:
                z_te = z_te[:, :oae.latent_dim]
            X_te_hat = oae.decoder(z_te)

        z_te_np = z_te.detach().cpu().numpy() * oae.flip_signs.values
        z_te_np = (z_te_np - mu) / sig
        X_te_hat = X_te_hat.detach().cpu().numpy()

        test_latents_list.append(
            xr.DataArray(
                z_te_np,
                coords={"time": t_te, "mode": np.arange(1, oae.latent_dim + 1)},
                dims=("time", "mode"),
            ).sortby("time")
        )

        mse_te_folds.append(mse_timeseries(X_test, X_te_hat))
        eng_te_folds.append(energy_timeseries(X_test))

    # concat CV time series
    xval_latents = xr.concat(xval_latents, dim="time").sortby("time")
    xval_tendencies = xr.concat(xval_tendencies, dim="time").sortby("time")

    mse_cv = xr.concat(mse_cv_list, dim="time").sortby("time")
    eng_cv = xr.concat(eng_cv_list, dim="time").sortby("time")

    # fold-mean test latents and recon metrics
    test_latents = xr.concat(test_latents_list, dim="fold").mean("fold")
    mse_te = np.mean(np.stack(mse_te_folds, axis=0), axis=0).astype(np.float32)
    eng_te = np.mean(np.stack(eng_te_folds, axis=0), axis=0).astype(np.float32)

    # select only modes of interest
    modes = list(modes_to_examine)
    z_sel = xval_latents.sel(mode=modes).sortby("time")
    tend_sel = xval_tendencies.sel(mode=modes).sortby("time")

    # alpha/beta using official function
    t_sel_unstack = tend_sel.unstack("feature").sortby(["time", "lat", "lon"])
    alpha, beta, r2_train_boot, r2_cv_da, r2_test_boot = kgae.regress_tendencies(z_sel, t_sel_unstack)

    if "tendency_mode" in alpha.dims:
        alpha = alpha.rename({"tendency_mode": "mode"})
    alpha = alpha.sel(mode=modes).transpose("lat", "lon", "mode")

    # beta dims expected: (lat, lon, tendency_mode, predictor_mode)
    if "tendency_mode" in beta.dims or "predictor_mode" in beta.dims:
        beta = beta.rename(
            {k: v for k, v in [("tendency_mode", "tendency_mode"), ("predictor_mode", "predictor_mode")] if k in beta.dims}
        )
    # select to modes_to_examine on both axes and order consistently
    if "tendency_mode" not in beta.dims or "predictor_mode" not in beta.dims:
        raise RuntimeError(f"Unexpected beta dims from kgae.regress_tendencies: {beta.dims}")
    beta = beta.sel(tendency_mode=modes, predictor_mode=modes).transpose("lat", "lon", "tendency_mode", "predictor_mode")

    return (
        z_sel,
        test_latents.sel(mode=modes),
        alpha,
        beta,
        mse_cv, eng_cv,
        xr.DataArray(mse_te, coords={"time": test_anom.time.values}, dims=("time",)).sortby("time"),
        xr.DataArray(eng_te, coords={"time": test_anom.time.values}, dims=("time",)).sortby("time"),
    )


# ============================================================
# Caching wrappers
# ============================================================

def cache_paths(cache_dir: Path, prefix: str, params: dict):
    h = _stable_hash(params)
    base = cache_dir / prefix / h
    return {
        "base": base,
        "meta": base / "meta.json",
        "preproc": base / "preproc.npz",
        "pca": base / "pca.npz",
        "kgae": base / "kgae",
        "summary": base / "summary.npz",
        "stability": base / "stability.npz",
        "recon": base / "recon.npz",
    }

def cache_preproc(cachefile: Path, train_anom, test_anom):
    X_tr, feat, t_tr = stack_time_feature(train_anom)
    X_te, _, t_te = stack_time_feature(test_anom)
    feat_lat = feat["lat"].values.astype(np.float32)
    feat_lon = feat["lon"].values.astype(np.float32)

    _npz_save(
        cachefile,
        X_tr=X_tr.astype(np.float32),
        X_te=X_te.astype(np.float32),
        t_tr=t_tr.astype("datetime64[ns]"),
        t_te=t_te.astype("datetime64[ns]"),
        feat_lat=feat_lat,
        feat_lon=feat_lon,
    )

def load_preproc(cachefile: Path):
    return _npz_load(cachefile)

def save_pca(cachefile: Path, pcs_tr, pcs_te, loadings_flat, mu, sig):
    _npz_save(
        cachefile,
        pcs_tr=pcs_tr.astype(np.float32),
        pcs_te=pcs_te.astype(np.float32),
        loadings=loadings_flat.astype(np.float32),
        mu=mu.astype(np.float32),
        sig=sig.astype(np.float32),
    )

def load_pca(cachefile: Path):
    return _npz_load(cachefile)

def save_kgae_seed(cache_dir: Path, seed: int,
                   z_tr_da, z_te_da,
                   alpha_map, beta_map,
                   mse_cv, eng_cv, mse_te, eng_te):
    """
    Save everything needed for:
      - WS (z_tr, z_te)
      - alpha/beta stability (alpha map + beta map)
      - reconstruction degradation (mse/energy time series)
    """
    _ensure_dir(cache_dir)

    alpha_map = alpha_map.transpose("lat", "lon", "mode")
    alpha = alpha_map.values.astype(np.float32)
    lat = alpha_map["lat"].values.astype(np.float32)
    lon = alpha_map["lon"].values.astype(np.float32)
    mode = alpha_map["mode"].values.astype(np.int32)

    beta_map = beta_map.transpose("lat", "lon", "tendency_mode", "predictor_mode")
    beta = beta_map.values.astype(np.float32)
    tend_mode = beta_map["tendency_mode"].values.astype(np.int32)
    pred_mode = beta_map["predictor_mode"].values.astype(np.int32)

    _npz_save(
        cache_dir / f"seed{seed:03d}.npz",
        # latents
        z_tr=z_tr_da.transpose("time", "mode").values.astype(np.float32),
        z_te=z_te_da.transpose("time", "mode").values.astype(np.float32),
        t_tr_lat=z_tr_da["time"].values.astype("datetime64[ns]"),
        t_te_lat=z_te_da["time"].values.astype("datetime64[ns]"),
        # alpha map
        alpha=alpha,
        lat=lat,
        lon=lon,
        mode=mode,
        # beta map
        beta=beta,
        tend_mode=tend_mode,
        pred_mode=pred_mode,
        # recon metric time series
        mse_cv_ts=mse_cv.values.astype(np.float32),
        eng_cv_ts=eng_cv.values.astype(np.float32),
        t_cv=mse_cv["time"].values.astype("datetime64[ns]"),
        mse_te_ts=mse_te.values.astype(np.float32),
        eng_te_ts=eng_te.values.astype(np.float32),
        t_te=mse_te["time"].values.astype("datetime64[ns]"),
    )

def load_kgae_seed(cache_dir: Path, seed: int):
    return _npz_load(cache_dir / f"seed{seed:03d}.npz")

def _fraction_params(frac_train, deg, n_components, modes_to_examine, seeds, n_epochs, device, n_boot, block_len):
    return dict(
        frac=float(frac_train),
        deg=int(deg),
        n_components=int(n_components),
        modes=tuple(int(m) for m in modes_to_examine),
        seeds=tuple(int(s) for s in seeds),
        n_epochs=int(n_epochs),
        device=str(device),
        delta=1e-2,
        n_boot=int(n_boot),
        block_len=int(block_len),
    )


# ============================================================
# Main experiment with caching
#   - NO alpha bootstrap
#   - ADD beta pattern-stability (non-bootstrapped)
#   - KEEP WS bootstrap and recon gap bootstrap
# ============================================================

def progressive_split_experiment_cached(
    sst_path="~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc",
    cache_dir="~/Desktop/kgae_ws_cache",
    fractions=np.linspace(0.5, 0.9, 9),
    n_components=2,
    modes_to_examine=(4, 5),
    seeds=(0, 1, 2),
    n_boot=500,
    block_len=12,
    n_epochs=150,
    device="cpu",
    deg=2,
    cache_level="full",        # "summary" | "preproc+pca" | "full"
):
    """
    Returns:
      results_pca, results_kgae:
         Wasserstein results (median/5-95)
      stability:
         PCA loading corr (deterministic wrt PCA fit)
         KGAE alpha corr (non-boot; seed-spread summarized)
         KGAE beta corr  (non-boot; seed-spread summarized) for each (tendency,predictor) pair
      recon:
         reconstruction degradation distributions summarized (median/5-95)
           - gap_mse
           - gap_nmse_within
           - gap_nmse_refcv
         AND (if available):
           - pca_gap_mse_*
           - pca_gap_nmse_within_*
           - pca_gap_nmse_refcv_*
    """
    print(f"cache_level={cache_level}")

    fractions = [round(float(f), 4) for f in fractions]
    cache_dir = Path(os.path.expanduser(cache_dir))

    # Summary cache (depends on ALL fractions and bootstrap sizes)
    summary_params = dict(
        fractions=fractions,
        n_components=int(n_components),
        modes=tuple(int(m) for m in modes_to_examine),
        seeds=tuple(int(s) for s in seeds),
        n_boot=int(n_boot),
        block_len=int(block_len),
        n_epochs=int(n_epochs),
        deg=int(deg),
        device=str(device),
    )
    summary_paths = cache_paths(cache_dir, "progressive_split_summary", summary_params)

    # ---------- Summary-only ----------
    if cache_level == "summary":
        if summary_paths["summary"].exists() and summary_paths["stability"].exists() and summary_paths["recon"].exists():
            d = np.load(summary_paths["summary"])
            s = np.load(summary_paths["stability"])
            r = np.load(summary_paths["recon"])
            print(summary_paths['summary'])

            results_pca = {"fractions": d["fractions"], "median": d["p_med"], "low": d["p_lo"], "high": d["p_hi"]}
            results_kgae = {"fractions": d["fractions"], "median": d["k_med"], "low": d["k_lo"], "high": d["k_hi"]}

            stability = {
                "fractions": s["fractions"],
                "pca_pattern_corr": s["pca_pattern_corr"],
                "kgae_alpha_corr_med": s["kgae_alpha_corr_med"],
                "kgae_alpha_corr_lo": s["kgae_alpha_corr_lo"],
                "kgae_alpha_corr_hi": s["kgae_alpha_corr_hi"],
                "kgae_beta_corr_med": s["kgae_beta_corr_med"],
                "kgae_beta_corr_lo": s["kgae_beta_corr_lo"],
                "kgae_beta_corr_hi": s["kgae_beta_corr_hi"],
                "beta_pairs": s["beta_pairs"],
            }

            recon = {
                "fractions": r["fractions"],
                "gap_mse_med": r["gap_mse_med"], "gap_mse_lo": r["gap_mse_lo"], "gap_mse_hi": r["gap_mse_hi"],
                "gap_nmse_within_med": r["gap_nmse_within_med"], "gap_nmse_within_lo": r["gap_nmse_within_lo"], "gap_nmse_within_hi": r["gap_nmse_within_hi"],
                "gap_nmse_refcv_med": r["gap_nmse_refcv_med"], "gap_nmse_refcv_lo": r["gap_nmse_refcv_lo"], "gap_nmse_refcv_hi": r["gap_nmse_refcv_hi"],
            }

            # Backward-compatible: only add PCA recon keys if they exist in recon.npz
            if "pca_gap_nmse_within_med" in r.files:
                recon.update({
                    "pca_gap_mse_med": r["pca_gap_mse_med"], "pca_gap_mse_lo": r["pca_gap_mse_lo"], "pca_gap_mse_hi": r["pca_gap_mse_hi"],
                    "pca_gap_nmse_within_med": r["pca_gap_nmse_within_med"], "pca_gap_nmse_within_lo": r["pca_gap_nmse_within_lo"], "pca_gap_nmse_within_hi": r["pca_gap_nmse_within_hi"],
                    "pca_gap_nmse_refcv_med": r["pca_gap_nmse_refcv_med"], "pca_gap_nmse_refcv_lo": r["pca_gap_nmse_refcv_lo"], "pca_gap_nmse_refcv_hi": r["pca_gap_nmse_refcv_hi"],
                })

            return results_pca, results_kgae, stability, recon

        print("Summary not found — rebuilding from per-fraction cached artifacts (no training).")

        rng = np.random.default_rng(42)
        nF = len(fractions)
        nS = len(seeds)

        ws_pca = np.zeros((nS, n_boot, n_components, nF))
        ws_kgae = np.zeros((nS, n_boot, n_components, nF))

        pca_pat_corr = np.full((n_components, nF), np.nan)

        # Non-boot pattern stability across splits, with seed spread
        alpha_corr_seed = np.full((nS, n_components, nF), np.nan, dtype=np.float32)
        beta_corr_seed = np.full((nS, n_components, n_components, nF), np.nan, dtype=np.float32)  # (seed, tend, pred, frac)

        pca_ref_loadings = None
        kgae_ref_alpha_by_seed = {}
        kgae_ref_beta_by_seed = {}

        # Recon gap bootstrap storage (we still bootstrap the evaluation)
        gap_mse_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
        gap_nmse_within_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
        gap_nmse_refcv_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)

        # PCA recon gaps (same summarizer shape as KGAE; replicate across seeds)
        gap_mse_pca_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
        gap_nmse_within_pca_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
        gap_nmse_refcv_pca_boot  = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)

        for fi, frac_train in enumerate(fractions):
            fparams = _fraction_params(frac_train, deg, n_components, modes_to_examine, seeds, n_epochs, device, n_boot, block_len)
            fpaths = cache_paths(cache_dir, "progressive_split", fparams)

            pca_file = fpaths["pca"]
            pre_file = fpaths["preproc"]
            kgae_dir = fpaths["kgae"]

            if not pca_file.exists():
                raise RuntimeError(
                    f"Missing PCA cache for fraction={frac_train} at {pca_file}. "
                    "Run once with cache_level='preproc+pca' or 'full'."
                )

            # ---- PCA recon degradation needs X_tr/X_te from preproc cache ----
            if not pre_file.exists():
                raise RuntimeError(
                    f"Missing preproc cache for fraction={frac_train} at {pre_file}. "
                    "Run once with cache_level='preproc+pca' or 'full'."
                )
            pre = load_preproc(pre_file)
            X_tr = pre["X_tr"].astype(np.float32)
            X_te = pre["X_te"].astype(np.float32)

            mse_cv_ts_pca, eng_cv_ts_pca, mse_te_ts_pca, eng_te_ts_pca = pca_cv_test_recon_ts(
                X_tr, X_te, n_components=n_components, n_splits=5
            )
            var_ref_cv_pca = float(np.nanmean(eng_cv_ts_pca))

            g_mse_p, g_within_p, g_refcv_p = boot_recon_gaps(
                mse_cv_ts_pca, mse_te_ts_pca,
                eng_cv_ts_pca, eng_te_ts_pca,
                n_boot=n_boot, block_len=block_len, rng=rng,
                var_ref_cv=var_ref_cv_pca
            )

            # replicate PCA bootstrap across seeds so summarize_boot_3d works unchanged
            for si in range(nS):
                gap_mse_pca_boot[si, :, fi] = g_mse_p
                gap_nmse_within_pca_boot[si, :, fi] = g_within_p
                gap_nmse_refcv_pca_boot[si, :, fi] = g_refcv_p

            # ---- PCA loadings + latents from pca cache ----
            pca_d = load_pca(pca_file)
            pcs_tr = pca_d["pcs_tr"]
            pcs_te = pca_d["pcs_te"]
            loadings = pca_d["loadings"]  # (ncomp, nfeat)

            if pca_ref_loadings is None:
                pca_ref_loadings = loadings.copy()

            for m in range(n_components):
                r0 = corrcoef_finite(loadings[m], pca_ref_loadings[m], min_n=2000)
                if np.isfinite(r0) and r0 < 0:
                    loadings[m] *= -1
                    r0 *= -1
                pca_pat_corr[m, fi] = r0

            # ---- KGAE seeds ----
            for si, seed in enumerate(seeds):
                seed_file = kgae_dir / f"seed{seed:03d}.npz"
                if not seed_file.exists():
                    raise RuntimeError(
                        f"Missing cached KGAE seed file {seed_file}. "
                        "Run once with cache_level='full'."
                    )

                kd = load_kgae_seed(kgae_dir, seed)

                ztr = kd["z_tr"]  # (time, mode_sel)
                zte = kd["z_te"]

                alpha_map = canonical_alpha(kd["alpha"], n_components)  # (lat,lon,mode)
                beta_map = canonical_beta(kd["beta"], n_components)     # (lat,lon,tend,pred)

                # set per-seed reference on first fraction
                if fi == 0:
                    kgae_ref_alpha_by_seed[seed] = alpha_map.copy()
                    kgae_ref_beta_by_seed[seed] = beta_map.copy()

                # non-boot alpha corr (abs)
                ref_a = kgae_ref_alpha_by_seed[seed]
                for m in range(n_components):
                    r = corrcoef_finite(alpha_map[..., m], ref_a[..., m], min_n=2000)
                    if np.isfinite(r) and r < 0:
                        r = -r
                    alpha_corr_seed[si, m, fi] = r

                # non-boot beta corr (abs) for each pair
                ref_b = kgae_ref_beta_by_seed[seed]
                for it in range(n_components):
                    for ip in range(n_components):
                        r = corrcoef_finite(beta_map[..., it, ip], ref_b[..., it, ip], min_n=2000)
                        if np.isfinite(r) and r < 0:
                            r = -r
                        beta_corr_seed[si, it, ip, fi] = r

                # --- WS bootstrap ---
                for m in range(n_components):
                    ws_pca[si, :, m, fi] = ws_blockboot_1d(pcs_tr[:, m], pcs_te[:, m], n_boot, block_len, rng)
                    ws_kgae[si, :, m, fi] = ws_blockboot_1d(ztr[:, m], zte[:, m], n_boot, block_len, rng)

                # --- Recon gap bootstrap (KGAE) ---
                mse_cv_ts = kd["mse_cv_ts"].astype(np.float32)
                eng_cv_ts = kd["eng_cv_ts"].astype(np.float32)
                mse_te_ts = kd["mse_te_ts"].astype(np.float32)
                eng_te_ts = kd["eng_te_ts"].astype(np.float32)

                var_ref_cv = float(np.nanmean(eng_cv_ts))  # fixed reference (CV energy)

                g_mse, g_within, g_refcv = boot_recon_gaps(
                    mse_cv_ts, mse_te_ts,
                    eng_cv_ts, eng_te_ts,
                    n_boot=n_boot,
                    block_len=block_len,
                    rng=rng,
                    var_ref_cv=var_ref_cv
                )
                gap_mse_boot[si, :, fi] = g_mse
                gap_nmse_within_boot[si, :, fi] = g_within
                gap_nmse_refcv_boot[si, :, fi] = g_refcv

        # Summaries
        def summarize_ws(ws):
            flat = ws.reshape(-1, ws.shape[2], ws.shape[3])  # (seed*boot, mode, frac)
            med = np.nanmedian(flat, axis=0)
            lo  = np.nanpercentile(flat, 5, axis=0)
            hi  = np.nanpercentile(flat, 95, axis=0)
            return med, lo, hi

        def summarize_boot_3d(arr):
            # arr: (seed, boot, frac)
            flat = arr.reshape(-1, arr.shape[-1])  # (seed*boot, frac)
            med = np.nanmedian(flat, axis=0)
            lo  = np.nanpercentile(flat, 5, axis=0)
            hi  = np.nanpercentile(flat, 95, axis=0)
            return med, lo, hi

        def summarize_seedspread_3d(arr):
            # arr: (seed, mode, frac) -> summarize across seed
            med = np.nanmedian(arr, axis=0)
            lo  = np.nanpercentile(arr, 5, axis=0)
            hi  = np.nanpercentile(arr, 95, axis=0)
            return med, lo, hi

        def summarize_seedspread_4d(arr):
            # arr: (seed, tend, pred, frac) -> summarize across seed
            med = np.nanmedian(arr, axis=0)
            lo  = np.nanpercentile(arr, 5, axis=0)
            hi  = np.nanpercentile(arr, 95, axis=0)
            return med, lo, hi

        p_med, p_lo, p_hi = summarize_ws(ws_pca)
        k_med, k_lo, k_hi = summarize_ws(ws_kgae)

        a_med, a_lo, a_hi = summarize_seedspread_3d(alpha_corr_seed)
        b_med, b_lo, b_hi = summarize_seedspread_4d(beta_corr_seed)

        gm_med, gm_lo, gm_hi = summarize_boot_3d(gap_mse_boot)
        gw_med, gw_lo, gw_hi = summarize_boot_3d(gap_nmse_within_boot)
        gr_med, gr_lo, gr_hi = summarize_boot_3d(gap_nmse_refcv_boot)

        pgm_med, pgm_lo, pgm_hi = summarize_boot_3d(gap_mse_pca_boot)
        pgw_med, pgw_lo, pgw_hi = summarize_boot_3d(gap_nmse_within_pca_boot)
        pgr_med, pgr_lo, pgr_hi = summarize_boot_3d(gap_nmse_refcv_pca_boot)

        results_pca = {"fractions": np.array(fractions, dtype=float), "median": p_med, "low": p_lo, "high": p_hi}
        results_kgae = {"fractions": np.array(fractions, dtype=float), "median": k_med, "low": k_lo, "high": k_hi}

        beta_pairs = np.array([(int(t), int(p)) for t in modes_to_examine for p in modes_to_examine], dtype=np.int32)

        stability = {
            "fractions": np.array(fractions, dtype=float),
            "pca_pattern_corr": pca_pat_corr,
            "kgae_alpha_corr_med": a_med,
            "kgae_alpha_corr_lo": a_lo,
            "kgae_alpha_corr_hi": a_hi,
            "kgae_beta_corr_med": b_med,
            "kgae_beta_corr_lo": b_lo,
            "kgae_beta_corr_hi": b_hi,
            "beta_pairs": beta_pairs,
        }

        recon = {
            "fractions": np.array(fractions, dtype=float),
            "gap_mse_med": gm_med, "gap_mse_lo": gm_lo, "gap_mse_hi": gm_hi,
            "gap_nmse_within_med": gw_med, "gap_nmse_within_lo": gw_lo, "gap_nmse_within_hi": gw_hi,
            "gap_nmse_refcv_med": gr_med, "gap_nmse_refcv_lo": gr_lo, "gap_nmse_refcv_hi": gr_hi,
            "pca_gap_mse_med": pgm_med, "pca_gap_mse_lo": pgm_lo, "pca_gap_mse_hi": pgm_hi,
            "pca_gap_nmse_within_med": pgw_med, "pca_gap_nmse_within_lo": pgw_lo, "pca_gap_nmse_within_hi": pgw_hi,
            "pca_gap_nmse_refcv_med": pgr_med, "pca_gap_nmse_refcv_lo": pgr_lo, "pca_gap_nmse_refcv_hi": pgr_hi,
        }

        # Save summaries
        _npz_save(
            summary_paths["summary"],
            fractions=np.array(fractions, dtype=float),
            p_med=p_med, p_lo=p_lo, p_hi=p_hi,
            k_med=k_med, k_lo=k_lo, k_hi=k_hi,
        )
        _npz_save(
            summary_paths["stability"],
            fractions=np.array(fractions, dtype=float),
            pca_pattern_corr=pca_pat_corr,
            kgae_alpha_corr_med=a_med,
            kgae_alpha_corr_lo=a_lo,
            kgae_alpha_corr_hi=a_hi,
            kgae_beta_corr_med=b_med,
            kgae_beta_corr_lo=b_lo,
            kgae_beta_corr_hi=b_hi,
            beta_pairs=beta_pairs,
        )
        _npz_save(
            summary_paths["recon"],
            fractions=np.array(fractions, dtype=float),
            gap_mse_med=gm_med, gap_mse_lo=gm_lo, gap_mse_hi=gm_hi,
            gap_nmse_within_med=gw_med, gap_nmse_within_lo=gw_lo, gap_nmse_within_hi=gw_hi,
            gap_nmse_refcv_med=gr_med, gap_nmse_refcv_lo=gr_lo, gap_nmse_refcv_hi=gr_hi,
            pca_gap_mse_med=pgm_med, pca_gap_mse_lo=pgm_lo, pca_gap_mse_hi=pgm_hi,
            pca_gap_nmse_within_med=pgw_med, pca_gap_nmse_within_lo=pgw_lo, pca_gap_nmse_within_hi=pgw_hi,
            pca_gap_nmse_refcv_med=pgr_med, pca_gap_nmse_refcv_lo=pgr_lo, pca_gap_nmse_refcv_hi=pgr_hi,
        )
        _json_save(summary_paths["meta"], summary_params)

        return results_pca, results_kgae, stability, recon

    # ---------- Full / preproc+pca: may train ----------
    sst = xr.open_dataset(os.path.expanduser(sst_path)).sst.rename({"latitude": "lat", "longitude": "lon"})
    rng = np.random.default_rng(42)

    nF = len(fractions)
    nS = len(seeds)

    ws_pca = np.zeros((nS, n_boot, n_components, nF))
    ws_kgae = np.zeros((nS, n_boot, n_components, nF))

    pca_pat_corr = np.full((n_components, nF), np.nan)

    # seed-spread alpha/beta corr storage
    alpha_corr_seed = np.full((nS, n_components, nF), np.nan, dtype=np.float32)
    beta_corr_seed = np.full((nS, n_components, n_components, nF), np.nan, dtype=np.float32)

    # recon bootstrap storage
    gap_mse_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
    gap_nmse_within_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
    gap_nmse_refcv_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)

    gap_mse_pca_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
    gap_nmse_within_pca_boot = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)
    gap_nmse_refcv_pca_boot  = np.full((nS, n_boot, nF), np.nan, dtype=np.float32)

    pca_ref_loadings = None
    kgae_ref_alpha_by_seed = {}
    kgae_ref_beta_by_seed = {}

    for fi, frac_train in enumerate(fractions):
        params = _fraction_params(frac_train, deg, n_components, modes_to_examine, seeds, n_epochs, device, n_boot, block_len)
        paths = cache_paths(cache_dir, "progressive_split", params)

        # --- PCA: load or compute ---
        if (cache_level in ("preproc+pca", "full")) and paths["preproc"].exists() and paths["pca"].exists():
            pre = load_preproc(paths["preproc"])
            X_tr = pre["X_tr"]; X_te = pre["X_te"]
            pca_d = load_pca(paths["pca"])
            pcs_tr = pca_d["pcs_tr"]; pcs_te = pca_d["pcs_te"]
            loadings = pca_d["loadings"]
        else:
            train_anom, test_anom = preprocess_train_test_contiguous(sst, frac_train, deg=deg)

            if cache_level in ("preproc+pca", "full"):
                cache_preproc(paths["preproc"], train_anom, test_anom)
                pre = load_preproc(paths["preproc"])
                X_tr = pre["X_tr"]; X_te = pre["X_te"]
            else:
                X_tr, _, _ = stack_time_feature(train_anom)
                X_te, _, _ = stack_time_feature(test_anom)

            pca = PCA(n_components=n_components)
            pca.fit(X_tr)
            pcs_tr = pca.transform(X_tr)
            pcs_te = pca.transform(X_te)

            mu = pcs_tr.mean(axis=0)
            sig = pcs_tr.std(axis=0); sig[sig == 0] = 1.0
            pcs_tr = (pcs_tr - mu) / sig
            pcs_te = (pcs_te - mu) / sig

            loadings = pca.components_.copy()

            if cache_level in ("preproc+pca", "full"):
                save_pca(paths["pca"], pcs_tr, pcs_te, loadings, mu, sig)

        # ---- PCA reconstruction degradation (CV-like) ----
        mse_cv_ts_pca, eng_cv_ts_pca, mse_te_ts_pca, eng_te_ts_pca = pca_cv_test_recon_ts(
            X_tr.astype(np.float32), X_te.astype(np.float32),
            n_components=n_components, n_splits=5
        )

        var_ref_cv_pca = float(np.nanmean(eng_cv_ts_pca))

        g_mse_p, g_within_p, g_refcv_p = boot_recon_gaps(
            mse_cv_ts_pca, mse_te_ts_pca,
            eng_cv_ts_pca, eng_te_ts_pca,
            n_boot=n_boot, block_len=block_len, rng=rng,
            var_ref_cv=var_ref_cv_pca
        )

        # replicate across seeds so summarize_boot_3d works unchanged
        for si in range(nS):
            gap_mse_pca_boot[si, :, fi] = g_mse_p
            gap_nmse_within_pca_boot[si, :, fi] = g_within_p
            gap_nmse_refcv_pca_boot[si, :, fi] = g_refcv_p

        # PCA pattern corr (no bootstrap)
        if pca_ref_loadings is None:
            pca_ref_loadings = loadings.copy()
        for m in range(n_components):
            r0 = corrcoef_finite(loadings[m], pca_ref_loadings[m], min_n=2000)
            if np.isfinite(r0) and r0 < 0:
                loadings[m] *= -1
                r0 *= -1
            pca_pat_corr[m, fi] = r0

        # --- KGAE: require full cache or train ---
        have_all = False
        if cache_level == "full":
            have_all = all((paths["kgae"] / f"seed{seed:03d}.npz").exists() for seed in seeds)

        if have_all:
            for si, seed in enumerate(seeds):
                kd = load_kgae_seed(paths["kgae"], seed)
                ztr = kd["z_tr"]; zte = kd["z_te"]

                alpha_map = canonical_alpha(kd["alpha"], n_components)
                beta_map = canonical_beta(kd["beta"], n_components)

                if fi == 0:
                    kgae_ref_alpha_by_seed[seed] = alpha_map.copy()
                    kgae_ref_beta_by_seed[seed] = beta_map.copy()

                # alpha corr (abs)
                ref_a = kgae_ref_alpha_by_seed[seed]
                for m in range(n_components):
                    r = corrcoef_finite(alpha_map[..., m], ref_a[..., m], min_n=2000)
                    if np.isfinite(r) and r < 0:
                        r = -r
                    alpha_corr_seed[si, m, fi] = r

                # beta corr (abs)
                ref_b = kgae_ref_beta_by_seed[seed]
                for it in range(n_components):
                    for ip in range(n_components):
                        r = corrcoef_finite(beta_map[..., it, ip], ref_b[..., it, ip], min_n=2000)
                        if np.isfinite(r) and r < 0:
                            r = -r
                        beta_corr_seed[si, it, ip, fi] = r

                # WS boot
                for m in range(n_components):
                    ws_pca[si, :, m, fi] = ws_blockboot_1d(pcs_tr[:, m], pcs_te[:, m], n_boot, block_len, rng)
                    ws_kgae[si, :, m, fi] = ws_blockboot_1d(ztr[:, m], zte[:, m], n_boot, block_len, rng)

                # recon boot
                mse_cv_ts = kd["mse_cv_ts"].astype(np.float32)
                eng_cv_ts = kd["eng_cv_ts"].astype(np.float32)
                mse_te_ts = kd["mse_te_ts"].astype(np.float32)
                eng_te_ts = kd["eng_te_ts"].astype(np.float32)
                var_ref_cv = float(np.nanmean(eng_cv_ts))

                g_mse, g_within, g_refcv = boot_recon_gaps(
                    mse_cv_ts, mse_te_ts,
                    eng_cv_ts, eng_te_ts,
                    n_boot=n_boot, block_len=block_len, rng=rng,
                    var_ref_cv=var_ref_cv
                )
                gap_mse_boot[si, :, fi] = g_mse
                gap_nmse_within_boot[si, :, fi] = g_within
                gap_nmse_refcv_boot[si, :, fi] = g_refcv

        else:
            # Train KGAE for this fraction (and cache if full)
            train_anom, test_anom = preprocess_train_test_contiguous(sst, frac_train, deg=deg)
            ref_pat = build_reference_pattern(train_anom)

            for si, seed in enumerate(seeds):
                (
                    z_tr_da,
                    z_te_da,
                    alpha_map,
                    beta_map,
                    mse_cv, eng_cv,
                    mse_te, eng_te
                ) = kgae_cv_latents_alpha_beta_recon(
                    train_anom=train_anom,
                    test_anom=test_anom,
                    reference_pattern=ref_pat,
                    modes_to_examine=modes_to_examine,
                    n_epochs=n_epochs,
                    random_seed=seed,
                    device=device,
                )

                ztr = z_tr_da.values.astype(np.float32)
                zte = z_te_da.values.astype(np.float32)

                # WS
                for m in range(n_components):
                    ws_pca[si, :, m, fi] = ws_blockboot_1d(pcs_tr[:, m], pcs_te[:, m], n_boot, block_len, rng)
                    ws_kgae[si, :, m, fi] = ws_blockboot_1d(ztr[:, m], zte[:, m], n_boot, block_len, rng)

                # set per-seed references on first fraction
                a_np = alpha_map.values.astype(np.float32)  # (lat,lon,mode)
                b_np = beta_map.values.astype(np.float32)   # (lat,lon,tend,pred)
                if fi == 0:
                    kgae_ref_alpha_by_seed[seed] = a_np.copy()
                    kgae_ref_beta_by_seed[seed] = b_np.copy()

                # alpha corr (abs)
                for m in range(n_components):
                    r = corrcoef_finite(a_np[..., m], kgae_ref_alpha_by_seed[seed][..., m], min_n=2000)
                    if np.isfinite(r) and r < 0:
                        r = -r
                    alpha_corr_seed[si, m, fi] = r

                # beta corr (abs)
                for it in range(n_components):
                    for ip in range(n_components):
                        r = corrcoef_finite(b_np[..., it, ip], kgae_ref_beta_by_seed[seed][..., it, ip], min_n=2000)
                        if np.isfinite(r) and r < 0:
                            r = -r
                        beta_corr_seed[si, it, ip, fi] = r

                # recon gaps boot
                var_ref_cv = float(np.nanmean(eng_cv.values))
                g_mse, g_within, g_refcv = boot_recon_gaps(
                    mse_cv.values.astype(np.float32),
                    mse_te.values.astype(np.float32),
                    eng_cv.values.astype(np.float32),
                    eng_te.values.astype(np.float32),
                    n_boot=n_boot, block_len=block_len, rng=rng,
                    var_ref_cv=var_ref_cv
                )
                gap_mse_boot[si, :, fi] = g_mse
                gap_nmse_within_boot[si, :, fi] = g_within
                gap_nmse_refcv_boot[si, :, fi] = g_refcv

                # save seed cache
                if cache_level == "full":
                    save_kgae_seed(
                        paths["kgae"], seed,
                        z_tr_da, z_te_da,
                        alpha_map, beta_map,
                        mse_cv, eng_cv, mse_te, eng_te
                    )

        if not paths["meta"].exists():
            _json_save(paths["meta"], params)

    # -------- summarize all results --------
    def summarize_ws(ws):
        flat = ws.reshape(-1, ws.shape[2], ws.shape[3])
        med = np.nanmedian(flat, axis=0)
        lo  = np.nanpercentile(flat, 5, axis=0)
        hi  = np.nanpercentile(flat, 95, axis=0)
        return med, lo, hi

    def summarize_boot_3d(arr):
        flat = arr.reshape(-1, arr.shape[-1])  # (seed*boot, frac)
        med = np.nanmedian(flat, axis=0)
        lo  = np.nanpercentile(flat, 5, axis=0)
        hi  = np.nanpercentile(flat, 95, axis=0)
        return med, lo, hi

    def summarize_seedspread_3d(arr):
        med = np.nanmedian(arr, axis=0)        # (mode, frac)
        lo  = np.nanpercentile(arr, 5, axis=0)
        hi  = np.nanpercentile(arr, 95, axis=0)
        return med, lo, hi

    def summarize_seedspread_4d(arr):
        med = np.nanmedian(arr, axis=0)        # (tend, pred, frac)
        lo  = np.nanpercentile(arr, 5, axis=0)
        hi  = np.nanpercentile(arr, 95, axis=0)
        return med, lo, hi

    p_med, p_lo, p_hi = summarize_ws(ws_pca)
    k_med, k_lo, k_hi = summarize_ws(ws_kgae)

    a_med, a_lo, a_hi = summarize_seedspread_3d(alpha_corr_seed)
    b_med, b_lo, b_hi = summarize_seedspread_4d(beta_corr_seed)

    gm_med, gm_lo, gm_hi = summarize_boot_3d(gap_mse_boot)
    gw_med, gw_lo, gw_hi = summarize_boot_3d(gap_nmse_within_boot)
    gr_med, gr_lo, gr_hi = summarize_boot_3d(gap_nmse_refcv_boot)

    pgm_med, pgm_lo, pgm_hi = summarize_boot_3d(gap_mse_pca_boot)
    pgw_med, pgw_lo, pgw_hi = summarize_boot_3d(gap_nmse_within_pca_boot)
    pgr_med, pgr_lo, pgr_hi = summarize_boot_3d(gap_nmse_refcv_pca_boot)

    results_pca = {"fractions": np.array(fractions, dtype=float), "median": p_med, "low": p_lo, "high": p_hi}
    results_kgae = {"fractions": np.array(fractions, dtype=float), "median": k_med, "low": k_lo, "high": k_hi}

    beta_pairs = np.array([(int(t), int(p)) for t in modes_to_examine for p in modes_to_examine], dtype=np.int32)

    stability = {
        "fractions": np.array(fractions, dtype=float),
        "pca_pattern_corr": pca_pat_corr,
        "kgae_alpha_corr_med": a_med,
        "kgae_alpha_corr_lo": a_lo,
        "kgae_alpha_corr_hi": a_hi,
        "kgae_beta_corr_med": b_med,
        "kgae_beta_corr_lo": b_lo,
        "kgae_beta_corr_hi": b_hi,
        "beta_pairs": beta_pairs,
    }

    recon = {
        "fractions": np.array(fractions, dtype=float),
        "gap_mse_med": gm_med, "gap_mse_lo": gm_lo, "gap_mse_hi": gm_hi,
        "gap_nmse_within_med": gw_med, "gap_nmse_within_lo": gw_lo, "gap_nmse_within_hi": gw_hi,
        "gap_nmse_refcv_med": gr_med, "gap_nmse_refcv_lo": gr_lo, "gap_nmse_refcv_hi": gr_hi,
        "pca_gap_mse_med": pgm_med, "pca_gap_mse_lo": pgm_lo, "pca_gap_mse_hi": pgm_hi,
        "pca_gap_nmse_within_med": pgw_med, "pca_gap_nmse_within_lo": pgw_lo, "pca_gap_nmse_within_hi": pgw_hi,
        "pca_gap_nmse_refcv_med": pgr_med, "pca_gap_nmse_refcv_lo": pgr_lo, "pca_gap_nmse_refcv_hi": pgr_hi,
    }

    # Save final plotting objects (so future summary mode is instant)
    if cache_level in ("preproc+pca", "full"):
        _npz_save(
            summary_paths["summary"],
            fractions=np.array(fractions, dtype=float),
            p_med=p_med, p_lo=p_lo, p_hi=p_hi,
            k_med=k_med, k_lo=k_lo, k_hi=k_hi,
        )
        _npz_save(
            summary_paths["stability"],
            fractions=np.array(fractions, dtype=float),
            pca_pattern_corr=pca_pat_corr,
            kgae_alpha_corr_med=a_med,
            kgae_alpha_corr_lo=a_lo,
            kgae_alpha_corr_hi=a_hi,
            kgae_beta_corr_med=b_med,
            kgae_beta_corr_lo=b_lo,
            kgae_beta_corr_hi=b_hi,
            beta_pairs=beta_pairs,
        )
        _npz_save(
            summary_paths["recon"],
            fractions=np.array(fractions, dtype=float),
            gap_mse_med=gm_med, gap_mse_lo=gm_lo, gap_mse_hi=gm_hi,
            gap_nmse_within_med=gw_med, gap_nmse_within_lo=gw_lo, gap_nmse_within_hi=gw_hi,
            gap_nmse_refcv_med=gr_med, gap_nmse_refcv_lo=gr_lo, gap_nmse_refcv_hi=gr_hi,
            pca_gap_mse_med=pgm_med, pca_gap_mse_lo=pgm_lo, pca_gap_mse_hi=pgm_hi,
            pca_gap_nmse_within_med=pgw_med, pca_gap_nmse_within_lo=pgw_lo, pca_gap_nmse_within_hi=pgw_hi,
            pca_gap_nmse_refcv_med=pgr_med, pca_gap_nmse_refcv_lo=pgr_lo, pca_gap_nmse_refcv_hi=pgr_hi,
        )
        _json_save(summary_paths["meta"], summary_params)

    return results_pca, results_kgae, stability, recon


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_progressive_split_paper_grayscale_clean_v2(
    wass_pca,
    wass_kgae,
    stability,
    recon,
    ia_mode=0,
    dec_mode=1,
    beta_pred_modes=None,

    # ---- Text customization ----
    titles=None,
    ylabels=None,
    xlabels=None,
    panel_letters=("a", "b", "c", "d", "e"),

    row_labels=("Interannual", "Decadal"),
    row_label_rotation=90,
    row_label_fontsize=11,
    row_label_offset=0.1,   # smaller = closer to axes

    legend_text_top=None,
    legend_text_bottom=None,

    linestyle_pca="--",
    linestyle_kgae="-",
    linestyle_beta_1=":",
    linestyle_beta_2="-.",
    beta_linestyle_map=None,

    # ---- Spread controls (default OFF) ----
    show_spread=False,
    show_spread_components=None,

    # ---- Appearance ----
    median_lw=1.7,
    figsize=(9.6, 9.3),
    sharex=True,
    show_abs_corr=True,
    grid_alpha=0.35,
    letter_fontsize=12,

    # Spacing controls
    wspace=0.25,
    hspace_top=0.5,          # spacing between top row and second row
    hspace_bottom=0.05,       # spacing between legend spacer row and panel (e)

    # Legend y-positions are now relative to dedicated spacer areas
    top_legend_fontsize=9,
    bottom_legend_y=0.02,
    bottom_legend_fontsize=9,
):
    """
    Layout:
      [ a | c ]
      [ b | d ]
      [ legend spacer across both columns ]
      [   e   ]

    Key fixes:
      - Dedicated spacer row for the TOP legend so it never collides with panel (e)
      - Force x tick labels + xlabel to show on the 2nd row (b,d) even with sharex
    """

    # ---------- defaults ----------
    if titles is None:
        titles = {
            "a": "Wasserstein distance",
            "c": "Pattern correlations",
            "e": "Normalized MSE Gap (Test period minus Cross-validation period)",
        }

    if ylabels is None:
        ylabels = {
            "a": "WD (K)",
            "b": "WD (K)",
            "c": "abs. correlation" if show_abs_corr else "pattern corr",
            "d": "abs. correlation" if show_abs_corr else "pattern corr",
            "e": "Δ NMSE"
        }

    # IMPORTANT: default top x label should not be empty
    if xlabels is None:
        xlabels = {
            "top": "Fraction of first contiguous part",
            "bottom": "Fraction of first contiguous part",
        }
    else:
        # if user forgets to set "top", default to bottom
        if "top" not in xlabels or xlabels.get("top", None) is None:
            xlabels["top"] = xlabels.get("bottom", "Fraction of first contiguous part")

    if legend_text_top is None:
        legend_text_top = {
            "pca": "PCA (dashed)",
            "kgae": "KGAE (solid)",
            "beta1": "β (style 1)",
            "beta2": "β (style 2)",
        }

    if legend_text_bottom is None:
        legend_text_bottom = {
            "within": "ΔNMSE_within (solid)",
            "ref": "ΔNMSE_refCV (dashed)",
        }

    # ---------- fractions consistency ----------
    fr = wass_pca["fractions"]
    for fr_other in [wass_kgae["fractions"], stability["fractions"], recon["fractions"]]:
        if not np.allclose(fr, fr_other):
            raise ValueError("Fractions arrays are inconsistent.")

    # ---------- unpack ----------
    Ap = wass_pca["median"]
    Ak = wass_kgae["median"]

    pca_corr = stability["pca_pattern_corr"]
    a_med = stability["kgae_alpha_corr_med"]
    b_med = stability["kgae_beta_corr_med"]

    nmse_w_med = recon["gap_nmse_within_med"]
    nmse_r_med = recon["gap_nmse_refcv_med"]

    pca_nmse_w_med = recon.get("pca_gap_nmse_within_med", None)
    pca_nmse_r_med = recon.get("pca_gap_nmse_refcv_med", None)

    # spreads are OFF by default; keep plumbing for optional future use
    comp = {
        "nmse_within": show_spread,
        "nmse_ref": show_spread,
    }
    if show_spread_components is not None:
        comp.update(show_spread_components)

    # ---------- beta linestyles ----------
    if beta_pred_modes is None:
        beta_pred_modes = [ia_mode, dec_mode]

    if beta_linestyle_map is None:
        beta_linestyle_map = {}

    if len(beta_pred_modes) >= 1:
        beta_linestyle_map.setdefault(beta_pred_modes[0], linestyle_beta_1)
    if len(beta_pred_modes) >= 2:
        beta_linestyle_map.setdefault(beta_pred_modes[1], linestyle_beta_2)
    for p in beta_pred_modes:
        beta_linestyle_map.setdefault(p, ":")

    # ---------- layout ----------
    fig = plt.figure(figsize=figsize)

    # 4 rows: top, second, legend spacer, bottom
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        height_ratios=[1.0, 1.0, 0.03, 0.98],
        hspace=hspace_top,
        wspace=wspace
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1], sharex=ax_a if sharex else None)
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a if sharex else None)
    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_a if sharex else None)

    # spacer axis for top legend (turned off)
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")

    # bottom axis spans both columns
    ax_e = fig.add_subplot(gs[3, :], sharex=ax_a if sharex else None)

    # If you sharex with ax_e, Matplotlib hides tick labels on ax_b/ax_d by default.
    # We'll re-enable them.
    ax_b.tick_params(axis="x", labelbottom=True)
    ax_d.tick_params(axis="x", labelbottom=True)

    # And ensure spacing between legend spacer and bottom panel
    # (a reliable way is adding extra hspace just before bottom)
    # We'll do it via subplot positions after creation:
    # shrink bottom axis slightly downward by adjusting gridspec spacing
    # easiest: use subplots_adjust to add breathing room above ax_e
    fig.subplots_adjust(hspace=hspace_bottom)

    axes = {"a": ax_a, "b": ax_b, "c": ax_c, "d": ax_d, "e": ax_e}

    line_color = "black"

    def label_panel(ax, letter, hgt=1.12):
        ax.text(
            0.01, hgt, f"{letter})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=letter_fontsize,
            fontweight="bold"
        )

    # ---------- (a) IA Wasserstein ----------
    ax_a.plot(fr, Ap[ia_mode], linestyle=linestyle_pca, marker="o",
              color=line_color, linewidth=median_lw)
    ax_a.plot(fr, Ak[ia_mode], linestyle=linestyle_kgae, marker="s",
              color=line_color, linewidth=median_lw)
    fill_gray_pca = "0.90"
    fill_gray_kgae = "0.90"
    fill_alpha = 0.5
    bound_alpha = 0.5
    bound_lw = 0.5

    def add_band(ax, x, lo, hi, linestyle, fill_gray, draw_bounds=True):
        # light fill
        ax.fill_between(x, lo, hi, color=fill_gray, alpha=fill_alpha, linewidth=0)
        # optional boundary lines to make the interval crisp in grayscale
        if draw_bounds:
            ax.plot(x, lo, linestyle=linestyle, color="black", linewidth=bound_lw, alpha=bound_alpha)
            ax.plot(x, hi, linestyle=linestyle, color="black", linewidth=bound_lw, alpha=bound_alpha)


    # PCA band
    add_band(ax_a, fr, wass_pca["low"][ia_mode], wass_pca["high"][ia_mode],
            linestyle_pca, fill_gray_pca, draw_bounds=True)

    # KGAE band (slightly darker)
    add_band(ax_a, fr, wass_kgae["low"][ia_mode], wass_kgae["high"][ia_mode],
            linestyle_kgae, fill_gray_kgae, draw_bounds=True)

    ax_a.set_title(titles.get("a", ""))
    ax_a.set_ylabel(ylabels.get("a", ""))
    ax_a.grid(True, alpha=grid_alpha)
    ax_a.set_ylim(0,0.9)
    label_panel(ax_a, panel_letters[0])

    # ---------- (b) Dec Wasserstein ----------
    ax_b.plot(fr, Ap[dec_mode], linestyle=linestyle_pca, marker="o",
              color=line_color, linewidth=median_lw)
    ax_b.plot(fr, Ak[dec_mode], linestyle=linestyle_kgae, marker="s",
              color=line_color, linewidth=median_lw)
    ax_b.set_ylabel(ylabels.get("b", ""))
    ax_b.grid(True, alpha=grid_alpha)
    label_panel(ax_b, panel_letters[1])
    ax_b.set_ylim(0,0.9)

    # ---------- (c) IA pattern ----------
    ax_c.plot(fr, pca_corr[ia_mode], linestyle=linestyle_pca,
              marker="o", color=line_color, linewidth=median_lw)
    ax_c.plot(fr, a_med[ia_mode], linestyle=linestyle_kgae,
              marker="s", color=line_color, linewidth=median_lw)
    
    add_band(ax_b, fr, wass_pca["low"][dec_mode], wass_pca["high"][dec_mode],
            linestyle_pca, fill_gray_pca, draw_bounds=True)

    add_band(ax_b, fr, wass_kgae["low"][dec_mode], wass_kgae["high"][dec_mode],
            linestyle_kgae, fill_gray_kgae, draw_bounds=True)

    for p in beta_pred_modes:
        ax_c.plot(fr, b_med[ia_mode, p],
                  linestyle=beta_linestyle_map[p],
                  color=line_color, linewidth=1.5)
        
    if comp["beta"]:
        for p in beta_pred_modes:
            add_band(
                ax_c,
                fr,
                stability["kgae_beta_corr_lo"][ia_mode, p],
                stability["kgae_beta_corr_hi"][ia_mode, p],
                beta_linestyle_map[p],
                "0.92",  # very light gray so it doesn’t dominate
                draw_bounds=True
            )

    ax_c.set_title(titles.get("c", ""))
    ax_c.set_ylabel(ylabels.get("c", ""))
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.grid(True, alpha=grid_alpha)
    label_panel(ax_c, panel_letters[2])

    # ---------- (d) Dec pattern ----------
    ax_d.plot(fr, pca_corr[dec_mode], linestyle=linestyle_pca,
              marker="o", color=line_color, linewidth=median_lw)
    ax_d.plot(fr, a_med[dec_mode], linestyle=linestyle_kgae,
              marker="s", color=line_color, linewidth=median_lw)

    for p in beta_pred_modes:
        ax_d.plot(fr, b_med[dec_mode, p],
                  linestyle=beta_linestyle_map[p],
                  color=line_color, linewidth=1.5)
    if comp["beta"]:
        for p in beta_pred_modes:
            add_band(
                ax_d,
                fr,
                stability["kgae_beta_corr_lo"][dec_mode, p],
                stability["kgae_beta_corr_hi"][dec_mode, p],
                beta_linestyle_map[p],
                "0.92",
                draw_bounds=True
            )

    ax_d.set_ylabel(ylabels.get("d", ""))
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.grid(True, alpha=grid_alpha)
    label_panel(ax_d, panel_letters[3])

    # ---------- x-label for top block (put it on second row) ----------
    top_xlabel = xlabels.get("top", "")
    if top_xlabel:
        ax_b.set_xlabel(top_xlabel)
        ax_d.set_xlabel(top_xlabel)

    # ---------- (e) NMSE ----------
    # within solid, ref dashed (per your legend convention)
    ax_e.plot(fr, nmse_w_med, linestyle=linestyle_kgae,
              marker="o", color=line_color, linewidth=median_lw)
    ax_e.plot(fr, nmse_r_med, linestyle=linestyle_pca,
              marker="s", color=line_color, linewidth=median_lw)
    
    # ---- PCA NMSE gaps (same linestyle convention; different markers so it’s distinguishable) ----
    if pca_nmse_w_med is not None:
        ax_e.plot(fr, pca_nmse_w_med, linestyle=linestyle_kgae,
                marker="^", color=line_color, linewidth=1.3, alpha=0.9)

    if pca_nmse_r_med is not None:
        ax_e.plot(fr, pca_nmse_r_med, linestyle=linestyle_pca,
                marker="v", color=line_color, linewidth=1.3, alpha=0.9)
        
    if "pca_gap_nmse_within_lo" in recon and "pca_gap_nmse_within_hi" in recon:
        add_band(ax_e, fr, recon["pca_gap_nmse_within_lo"], recon["pca_gap_nmse_within_hi"],
                linestyle_kgae, fill_gray_pca, draw_bounds=True)

    if "pca_gap_nmse_refcv_lo" in recon and "pca_gap_nmse_refcv_hi" in recon:
        add_band(ax_e, fr, recon["pca_gap_nmse_refcv_lo"], recon["pca_gap_nmse_refcv_hi"],
                linestyle_pca, fill_gray_pca, draw_bounds=True)

    ax_e.set_title(titles.get("e", ""))
    ax_e.set_ylabel(ylabels.get("e", ""))
    ax_e.set_xlabel(xlabels.get("bottom", ""))
    ax_e.grid(True, alpha=grid_alpha)




    add_band(ax_e, fr, recon["gap_nmse_within_lo"], recon["gap_nmse_within_hi"],
         linestyle_kgae, fill_gray_kgae, draw_bounds=True)

    add_band(ax_e, fr, recon["gap_nmse_refcv_lo"], recon["gap_nmse_refcv_hi"],
            linestyle_pca, fill_gray_pca, draw_bounds=True)
    label_panel(ax_e, panel_letters[4], hgt=1.12)

    # ---------- dynamic row-label placement ----------
    def row_center_y(ax_left, ax_right):
        bb1 = ax_left.get_position()
        bb2 = ax_right.get_position()
        y0 = min(bb1.y0, bb2.y0)
        y1 = max(bb1.y1, bb2.y1)
        return 0.5 * (y0 + y1)

    y_top = row_center_y(ax_a, ax_c)
    y_mid = row_center_y(ax_b, ax_d)

    left_edge = ax_a.get_position().x0
    label_x = left_edge - row_label_offset

    fig.text(label_x, y_top, row_labels[0],
             rotation=row_label_rotation, ha="center", va="center",
             fontsize=row_label_fontsize)

    fig.text(label_x, y_mid, row_labels[1],
             rotation=row_label_rotation, ha="center", va="center",
             fontsize=row_label_fontsize)

    # ---------- legends ----------
    # TOP legend goes in the dedicated spacer row; use figure coords at that row’s center
    top_handles = [
        Line2D([0], [0], color="black", linestyle=linestyle_pca, linewidth=1.8, label=legend_text_top["pca"]),
        Line2D([0], [0], color="black", linestyle=linestyle_kgae, linewidth=1.8, label=legend_text_top["kgae"]),
        Line2D([0], [0], color="black", linestyle=linestyle_beta_1, linewidth=1.8, label=legend_text_top["beta1"]),
        Line2D([0], [0], color="black", linestyle=linestyle_beta_2, linewidth=1.8, label=legend_text_top["beta2"]),
    ]

    # Use the spacer axis bounds to place legend nicely in it
    bb_leg = ax_leg.get_position()
    y_leg_center = 0.5 * (bb_leg.y0 + bb_leg.y1)

    fig.legend(
        handles=top_handles,
        loc="center",
        bbox_to_anchor=(0.5, y_leg_center),
        ncol=4,
        frameon=False,
        fontsize=top_legend_fontsize
    )

    bottom_handles = [
        Line2D([0], [0], color="black", linestyle=linestyle_kgae, linewidth=1.8, label=legend_text_bottom["within"]),
        Line2D([0], [0], color="black", linestyle=linestyle_pca, linewidth=1.8, label=legend_text_bottom["ref"]),
    ]

    fig.legend(
        handles=bottom_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_legend_y),
        ncol=2,
        frameon=False,
        fontsize=bottom_legend_fontsize
    )

    # extra bottom space for the bottom legend
    fig.subplots_adjust(bottom=0.11)

    return fig, axes






# ============================================================
# Plot helpers
# ============================================================

def plot_wasserstein_progression(results_a, results_b, mode_labels=None, labels=("PCA", "KGAE"), figsize=(8, 4)):
    fr = results_a["fractions"]
    A, Alo, Ahi = results_a["median"], results_a["low"], results_a["high"]
    B, Blo, Bhi = results_b["median"], results_b["low"], results_b["high"]
    n_modes = A.shape[0]
    if mode_labels is None:
        mode_labels = [f"Mode {i+1}" for i in range(n_modes)]

    fig, axes = plt.subplots(n_modes, 1, figsize=(figsize[0], figsize[1]*n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]
    for m, ax in enumerate(axes):
        ax.plot(fr, A[m], marker="o", label=labels[0])
        ax.fill_between(fr, Alo[m], Ahi[m], alpha=0.2)
        ax.plot(fr, B[m], marker="s", label=labels[1])
        ax.fill_between(fr, Blo[m], Bhi[m], alpha=0.2)
        ax.set_title(mode_labels[m])
        ax.set_ylabel("Wasserstein")
        ax.grid(True)
        if m == 0:
            ax.legend()
    axes[-1].set_xlabel("Fraction of first contiguous part")
    plt.tight_layout()
    plt.show()


def plot_pattern_stability(stability, mode_labels=None, figsize=(9, 3.8)):
    """
    Plots, per KGAE mode:
      - PCA loading corr (deterministic)
      - KGAE alpha corr (seed spread, 5–95 band)
      - KGAE beta corr for that tendency mode with each predictor mode (seed spread bands)
    """
    fr = stability["fractions"]
    pca = stability["pca_pattern_corr"]

    a_med = stability["kgae_alpha_corr_med"]
    a_lo  = stability["kgae_alpha_corr_lo"]
    a_hi  = stability["kgae_alpha_corr_hi"]

    b_med = stability["kgae_beta_corr_med"]  # (tend,pred,frac)
    b_lo  = stability["kgae_beta_corr_lo"]
    b_hi  = stability["kgae_beta_corr_hi"]

    n_modes = pca.shape[0]
    if mode_labels is None:
        mode_labels = [f"Mode {i+1}" for i in range(n_modes)]

    fig, axes = plt.subplots(n_modes, 1, figsize=(figsize[0], figsize[1]*n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]

    for m, ax in enumerate(axes):
        ax.plot(fr, pca[m], marker="o", label="PCA loading corr")

        # alpha
        ax.plot(fr, a_med[m], marker="s", label="KGAE α corr (seed median)")
        ax.fill_between(fr, a_lo[m], a_hi[m], alpha=0.15, label="KGAE α seed spread (5–95%)")

        # beta lines for this tendency mode m with predictors 0/1
        for p in range(n_modes):
            ax.plot(fr, b_med[m, p], marker=None, linestyle="--",
                    label=f"KGAE β corr (tend={m+1}, pred={p+1}) median")
            ax.fill_between(fr, b_lo[m, p], b_hi[m, p], alpha=0.10)

        ax.set_title(mode_labels[m])
        ax.set_ylabel("|pattern corr|")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        if m == 0:
            ax.legend(ncol=1)

    axes[-1].set_xlabel("Fraction of first contiguous part")
    plt.tight_layout()
    plt.show()


def plot_recon_degradation(recon, figsize=(8, 3.2)):
    fr = recon["fractions"]

    fig, axes = plt.subplots(3, 1, figsize=(figsize[0], figsize[1]*3), sharex=True)

    axes[0].plot(fr, recon["gap_mse_med"], marker="o")
    axes[0].fill_between(fr, recon["gap_mse_lo"], recon["gap_mse_hi"], alpha=0.2)
    axes[0].set_title("Reconstruction degradation: MSE gap (test - CV)")
    axes[0].set_ylabel("Δ MSE")
    axes[0].grid(True)

    axes[1].plot(fr, recon["gap_nmse_within_med"], marker="o")
    axes[1].fill_between(fr, recon["gap_nmse_within_lo"], recon["gap_nmse_within_hi"], alpha=0.2)
    axes[1].set_title("Reconstruction degradation: within-period NMSE gap")
    axes[1].set_ylabel("Δ NMSE_within")
    axes[1].grid(True)

    axes[2].plot(fr, recon["gap_nmse_refcv_med"], marker="o")
    axes[2].fill_between(fr, recon["gap_nmse_refcv_lo"], recon["gap_nmse_refcv_hi"], alpha=0.2)
    axes[2].set_title("Reconstruction degradation: ref-CV NMSE gap")
    axes[2].set_ylabel("Δ NMSE_refCV")
    axes[2].grid(True)

    axes[-1].set_xlabel("Fraction of first contiguous part")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results_pca, results_kgae, stability, recon = progressive_split_experiment_cached(
        cache_level="summary",          # FIRST RUN should be "full" to populate new caches (beta + recon metrics)
        fractions=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
        seeds=(0, 1, 2, 3, 4),
        n_boot=1000,
        block_len=12,
        n_epochs=200,
        modes_to_examine=(4, 5),
    )

   # plot_wasserstein_progression(
   #     results_pca, results_kgae,
   #     mode_labels=["Interannual (mode 4)", "Decadal (mode 5)"],
   #     labels=("PCA", "KGAE"),
   # )

   # plot_pattern_stability(
   #     stability,
   #     mode_labels=["Interannual (mode 4)", "Decadal (mode 5)"],
   # )

    fig, axes = plot_progressive_split_paper_grayscale_clean_v2(
        wass_pca=results_pca,
        wass_kgae=results_kgae,
        stability=stability,
        recon=recon,
        ia_mode=0,
        dec_mode=1,
        beta_pred_modes=[0, 1],
        legend_text_top={
            "pca": "EOF",
            "kgae": r"$\alpha$",
            "beta1": "β (interannual modulation)",
            "beta2": "β (decadal modulation)",
        },
        legend_text_bottom={
            "within": "normalized by corresponding period",
            "ref": "normalized by reference period",
        },
        xlabels={
            "top": "Cross-validation Period Fraction",
            "bottom": "Cross-validation Period Fraction",
        },

        # keep spreads off unless you really want them
        show_spread_components={
            "nmse_within": True, 
            "nmse_ref": True,
            "beta": True,
        },
    )

    fig.savefig("progressive_split_grayscale_clean.png", dpi=300, bbox_inches="tight")
    plt.show()