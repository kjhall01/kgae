import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import torch
import pickle

from pathlib import Path
from matplotlib.gridspec import GridSpec

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

import kgae


# =========================================================
# USER SETTINGS
# =========================================================
ERA5_PATH = "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"
TRAINING_PERIOD = (None, pd.Timestamp(2014, 12, 31))
N_SPLITS = 5


# =========================================================
# Bootstrap utilities
# =========================================================
def circular_block_indices(n, block_len, rng):
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block_len) % n) for s in starts])[:n]
    return idx.astype(int)


def _q33_67(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan, np.nan
    return np.nanquantile(x, 0.33), np.nanquantile(x, 0.67)


def _ci_excludes_zero(boot_maps, qlo=0.025, qhi=0.975):
    lo = np.nanquantile(boot_maps, qlo, axis=0)
    hi = np.nanquantile(boot_maps, qhi, axis=0)
    sig = (lo > 0) | (hi < 0)
    med = np.nanquantile(boot_maps, 0.5, axis=0)
    return med, lo, hi, sig


def add_panel_label(ax, label, fontsize=12, dx=0.0, dy=0.02):
    ax.text(
        0.0 + dx, 1.0 + dy, f"{label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
    )


# =========================================================
# Data prep exactly matching CV preprocessing
# =========================================================
def _open_raw_training_sst():
    sst = xr.open_dataset(ERA5_PATH).sst.sel(time=slice(TRAINING_PERIOD[0], TRAINING_PERIOD[1]))
    sst = sst.rename({"latitude": "lat", "longitude": "lon"})
    return sst


def _apply_saved_preprocessing(ds, polyfit_coeffs, monthly_clim):
    """
    Apply split-specific detrend + monthly-climatology removal to a raw SST DataArray.
    Mirrors run_cv_exp.py.
    """
    ds = ds - xr.polyval(ds["time"], polyfit_coeffs)

    dim = "time"
    toconcat = []
    years = sorted(set(pd.Timestamp(t).year for t in ds[dim].values))
    for year in years:
        ds_yearly = (
            ds.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)))
              .groupby(f"{dim}.month")
              .mean()
            - monthly_clim
        )
        ds_yearly = ds_yearly.assign_coords(
            {"month": [pd.Timestamp(year, m, 1) for m in ds_yearly["month"].values]}
        ).rename({"month": dim})
        toconcat.append(ds_yearly)

    ds = xr.concat(toconcat, dim).sortby(dim)
    return ds


def _prepare_train_val_for_split(split):
    """
    Reconstruct the exact train/val preprocessing for one CV split.
    Returns:
      train_stacked : DataArray(time, feature)
      val_stacked   : DataArray(time, feature)
      val_field     : DataArray(time, lat, lon)
    """
    sst = _open_raw_training_sst()

    # original interleaved folds
    splits = [sst.isel(time=slice(i, None, N_SPLITS)) for i in range(N_SPLITS)]

    train = xr.concat([splits[j] for j in range(N_SPLITS) if j != split], "time").sortby("time")
    train, fit, gwm, p = kgae.global_detrend(train, deg=2)
    train, monthly_clim = kgae.remove_climo(train)

    val = splits[split]
    val = _apply_saved_preprocessing(val, p.polyfit_coefficients, monthly_clim)

    train_stacked = (
        train.sortby("time")
             .stack(feature=("lat", "lon"))
             .dropna("feature", how="any")
             .transpose("time", "feature")
    )

    val_stacked = (
        val.sortby("time")
           .stack(feature=("lat", "lon"))
           .dropna("feature", how="any")
           .transpose("time", "feature")
    )

    return train_stacked, val_stacked, val


# =========================================================
# Model loading + pair-only reconstructions
# =========================================================
def _load_model(experiment, seed, split):
    model_path = Path(experiment) / f"seed{seed}" / f"split{split}" / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.eval()
    return model


def _encode_mu(model, X_np):
    with torch.no_grad():
        z = model.encoder(torch.tensor(X_np, dtype=torch.float32))
        if getattr(model, "is_variational", False):
            z = z[:, :model.latent_dim]
    return z.detach().cpu().numpy()


def _decode_np(model, Z_np):
    with torch.no_grad():
        xhat = model.decoder(torch.tensor(Z_np, dtype=torch.float32))
    return xhat.detach().cpu().numpy()


def _raw_from_standardized(z_std, mu, std, flip):
    """
    Convert aligned standardized latents back to raw decoder coordinates.

    aligned_raw = z_std * std + mu
    raw = aligned_raw / flip

    flip is +/- 1 for each mode.
    """
    aligned_raw = z_std * std[None, :] + mu[None, :]
    raw = aligned_raw / flip[None, :]
    return raw

def build_pair_only_reconstructions(
    experiment,
    n_ensemble=100,
    mode_interannual=4,
    mode_decadal=5,
):
    """
    Returns ensemble-mean and per-seed decoded fields for:
      pair  = D(zi, zd)
      ia    = D(zi, 0)
      dec   = D(0, zd)
      zero  = D(0, 0)
      add   = D(zi, 0) + D(0, zd) - D(0, 0)
      resid = D(zi, zd) - add

    All omitted modes are set to zero in the standardized/aligned latent space,
    then mapped back to raw decoder coordinates.
    """
    pair_by_seed = []
    ia_by_seed = []
    dec_by_seed = []
    zero_by_seed = []
    add_by_seed = []
    resid_by_seed = []
    z_by_seed = []

    for seed in range(n_ensemble):
        pair_split_list = []
        ia_split_list = []
        dec_split_list = []
        zero_split_list = []
        add_split_list = []
        resid_split_list = []
        z_split_list = []

        for split in range(N_SPLITS):
            model = _load_model(experiment, seed, split)
            train_stacked, val_stacked, val_field = _prepare_train_val_for_split(split)

            # ---- encode train to get standardization stats
            z_train_raw = _encode_mu(model, train_stacked.values)
            z_train_aligned = z_train_raw * model.flip_signs.values[None, :]
            train_mu = np.nanmean(z_train_aligned, axis=0)
            train_std = np.nanstd(z_train_aligned, axis=0)
            train_std = np.where(train_std == 0, 1.0, train_std)

            # ---- encode validation set
            z_val_raw = _encode_mu(model, val_stacked.values)
            z_val_aligned = z_val_raw * model.flip_signs.values[None, :]
            z_val_std = (z_val_aligned - train_mu[None, :]) / train_std[None, :]

            n_time = z_val_std.shape[0]
            n_mode = z_val_std.shape[1]

            # helper arrays in standardized/aligned latent coords
            z_pair_std = np.zeros_like(z_val_std)
            z_ia_std   = np.zeros_like(z_val_std)
            z_dec_std  = np.zeros_like(z_val_std)
            z_zero_std = np.zeros_like(z_val_std)

            i = mode_interannual - 1
            d = mode_decadal - 1

            z_pair_std[:, i] = z_val_std[:, i]
            z_pair_std[:, d] = z_val_std[:, d]

            z_ia_std[:, i] = z_val_std[:, i]
            z_dec_std[:, d] = z_val_std[:, d]

            # convert back to raw decoder coordinates
            z_pair_raw = _raw_from_standardized(
                z_pair_std, mu=train_mu, std=train_std, flip=model.flip_signs.values
            )
            z_ia_raw = _raw_from_standardized(
                z_ia_std, mu=train_mu, std=train_std, flip=model.flip_signs.values
            )
            z_dec_raw = _raw_from_standardized(
                z_dec_std, mu=train_mu, std=train_std, flip=model.flip_signs.values
            )
            z_zero_raw = _raw_from_standardized(
                z_zero_std, mu=train_mu, std=train_std, flip=model.flip_signs.values
            )

            # decode
            x_pair = _decode_np(model, z_pair_raw)
            x_ia   = _decode_np(model, z_ia_raw)
            x_dec  = _decode_np(model, z_dec_raw)
            x_zero = _decode_np(model, z_zero_raw)

            x_add = x_ia + x_dec - x_zero
            x_resid = x_pair - x_add

            coords = {"time": val_stacked["time"], "feature": val_stacked["feature"]}

            def _to_field(x, name):
                return (
                    xr.DataArray(
                        x,
                        coords=coords,
                        dims=("time", "feature"),
                        name=name,
                    )
                    .unstack("feature")
                    .transpose("time", "lat", "lon")
                    .sortby("time")
                )

            pair_da  = _to_field(x_pair,  "pair")
            ia_da    = _to_field(x_ia,    "ia_only")
            dec_da   = _to_field(x_dec,   "dec_only")
            zero_da  = _to_field(x_zero,  "zero")
            add_da   = _to_field(x_add,   "additive")
            resid_da = _to_field(x_resid, "residual")

            z_da = xr.DataArray(
                z_val_std,
                coords={
                    "time": val_stacked["time"],
                    "mode": np.arange(1, model.latent_dim + 1),
                },
                dims=("time", "mode"),
            ).sortby("time")

            pair_split_list.append(pair_da)
            ia_split_list.append(ia_da)
            dec_split_list.append(dec_da)
            zero_split_list.append(zero_da)
            add_split_list.append(add_da)
            resid_split_list.append(resid_da)
            z_split_list.append(z_da)

        pair_seed  = xr.concat(pair_split_list, dim="time").sortby("time")
        ia_seed    = xr.concat(ia_split_list,   dim="time").sortby("time")
        dec_seed   = xr.concat(dec_split_list,  dim="time").sortby("time")
        zero_seed  = xr.concat(zero_split_list, dim="time").sortby("time")
        add_seed   = xr.concat(add_split_list,  dim="time").sortby("time")
        resid_seed = xr.concat(resid_split_list, dim="time").sortby("time")
        z_seed     = xr.concat(z_split_list, dim="time").sortby("time")

        # de-duplicate times if necessary
        _, unique_idx = np.unique(pair_seed["time"].values, return_index=True)
        unique_idx = np.sort(unique_idx)

        pair_seed  = pair_seed.isel(time=unique_idx)
        ia_seed    = ia_seed.isel(time=unique_idx)
        dec_seed   = dec_seed.isel(time=unique_idx)
        zero_seed  = zero_seed.isel(time=unique_idx)
        add_seed   = add_seed.isel(time=unique_idx)
        resid_seed = resid_seed.isel(time=unique_idx)
        z_seed     = z_seed.isel(time=unique_idx)

        pair_by_seed.append(pair_seed.expand_dims(seed=[seed]))
        ia_by_seed.append(ia_seed.expand_dims(seed=[seed]))
        dec_by_seed.append(dec_seed.expand_dims(seed=[seed]))
        zero_by_seed.append(zero_seed.expand_dims(seed=[seed]))
        add_by_seed.append(add_seed.expand_dims(seed=[seed]))
        resid_by_seed.append(resid_seed.expand_dims(seed=[seed]))
        z_by_seed.append(z_seed.expand_dims(seed=[seed]))

        print(f"finished seed {seed}")

    pair_all  = xr.concat(pair_by_seed, dim="seed")
    ia_all    = xr.concat(ia_by_seed, dim="seed")
    dec_all   = xr.concat(dec_by_seed, dim="seed")
    zero_all  = xr.concat(zero_by_seed, dim="seed")
    add_all   = xr.concat(add_by_seed, dim="seed")
    resid_all = xr.concat(resid_by_seed, dim="seed")
    z_all     = xr.concat(z_by_seed, dim="seed")

    out = {
        "pair_all": pair_all,
        "ia_all": ia_all,
        "dec_all": dec_all,
        "zero_all": zero_all,
        "add_all": add_all,
        "resid_all": resid_all,
        "z_all": z_all,
        "pair_mean": pair_all.mean("seed"),
        "ia_mean": ia_all.mean("seed"),
        "dec_mean": dec_all.mean("seed"),
        "zero_mean": zero_all.mean("seed"),
        "add_mean": add_all.mean("seed"),
        "resid_mean": resid_all.mean("seed"),
        "z_mean": z_all.mean("seed"),
    }
    return out

def composite_once_joint_from_field(
    field, z, i_mode, d_mode, i_sign, d_sign
):
    zi = z.sel(mode=i_mode).values
    zd = z.sel(mode=d_mode).values

    i33, i67 = _q33_67(zi)
    d33, d67 = _q33_67(zd)
    if not (np.isfinite(i33) and np.isfinite(i67) and np.isfinite(d33) and np.isfinite(d67)):
        return None

    mi = zi > i67 if i_sign > 0 else zi < i33
    md = zd > d67 if d_sign > 0 else zd < d33
    gate = mi & md

    idx = np.where(gate)[0]
    if idx.size < 12:
        return None

    return field.isel(time=idx).mean("time", skipna=True)


def bootstrap_joint_composite_from_field(
    field, z, i_mode, d_mode, i_sign, d_sign,
    n_boot=500, block_len=12, rng_seed=0
):
    field2, z2 = xr.align(field, z, join="inner")
    n = field2.sizes["time"]
    rng = np.random.default_rng(rng_seed)

    lat = field2["lat"].values
    lon = field2["lon"].values
    boot_maps = np.full((n_boot, lat.size, lon.size), np.nan, dtype=np.float32)

    for b in range(n_boot):
        idx = circular_block_indices(n, block_len, rng)
        f_b = field2.isel(time=idx)
        z_b = z2.isel(time=idx)

        m = composite_once_joint_from_field(
            f_b, z_b,
            i_mode=i_mode,
            d_mode=d_mode,
            i_sign=i_sign,
            d_sign=d_sign,
        )
        if m is None:
            continue
        boot_maps[b, :, :] = m.values.astype(np.float32)

    med, lo, hi, sig = _ci_excludes_zero(boot_maps)
    med_da = xr.DataArray(med, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    lo_da  = xr.DataArray(lo,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    hi_da  = xr.DataArray(hi,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    sig_da = xr.DataArray(sig, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    return {"med": med_da, "lo": lo_da, "hi": hi_da, "sig": sig_da}

def plot_joint_quadrant_figure(
    field,
    z_all,
    *,
    mode_interannual=4,
    mode_decadal=5,
    n_boot=500,
    block_len=12,
    rng_seed=123,
    central_longitude=180,
    cmap="RdBu_r",
    hist_bins=60,
    smooth_sigma=1.2,
    cbar_label="Composite (K)",
    panel_titles=None,
):
    z = z_all.mean("seed") if "seed" in z_all.dims else z_all
    if "time" not in z.dims:
        z = z.transpose("time", "mode")

    quads = {
        "TL": dict(i_sign=+1, d_sign=-1),
        "BL": dict(i_sign=-1, d_sign=-1),
        "TR": dict(i_sign=+1, d_sign=+1),
        "BR": dict(i_sign=-1, d_sign=+1),
    }

    boot = {}
    for k, q in quads.items():
        boot[k] = bootstrap_joint_composite_from_field(
            field, z,
            i_mode=mode_interannual,
            d_mode=mode_decadal,
            i_sign=q["i_sign"],
            d_sign=q["d_sign"],
            n_boot=n_boot,
            block_len=block_len,
            rng_seed=rng_seed + (0 if k == "TL" else 1 if k == "TR" else 2 if k == "BL" else 3),
        )

    meds = [boot[k]["med"] for k in ["TL", "TR", "BL", "BR"]]
    fill_levels = nice_sym_levels(meds, n=31, robust_q=0.99)
    line_levels = nice_sym_levels(meds, n=9, robust_q=0.99)

    zi = z.sel(mode=mode_interannual).values
    zd = z.sel(mode=mode_decadal).values
    i33, i67 = _q33_67(zi)
    d33, d67 = _q33_67(zd)

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree(central_longitude=central_longitude)

    fig = plt.figure(figsize=(13.0, 6.8))
    gs = GridSpec(
        nrows=3, ncols=4, figure=fig,
        height_ratios=[1.0, 1.0, 0.10],
        width_ratios=[1.0, 1.0, 0.2, 1.6],
        wspace=0.10,
        hspace=0.10,
    )

    ax_TL = fig.add_subplot(gs[0, 0], projection=proj)
    ax_TR = fig.add_subplot(gs[0, 1], projection=proj)
    ax_BL = fig.add_subplot(gs[1, 0], projection=proj)
    ax_BR = fig.add_subplot(gs[1, 1], projection=proj)
    ax_dist = fig.add_subplot(gs[0:2, 3])
    cax_left = fig.add_subplot(gs[2, 0:2])
    cax_right = fig.add_subplot(gs[2, 3])

    add_panel_label(ax_TL, "a")
    add_panel_label(ax_TR, "b")
    add_panel_label(ax_BL, "c")
    add_panel_label(ax_BR, "d")
    add_panel_label(ax_dist, "e")

    if panel_titles is None:
        panel_titles = {
            "TL": "IA+ / D-",
            "TR": "IA+ / D+",
            "BL": "IA- / D-",
            "BR": "IA- / D+",
        }

    mappable = None
    for key, ax in [("TL", ax_TL), ("TR", ax_TR), ("BL", ax_BL), ("BR", ax_BR)]:
        ax.coastlines(linewidth=0.6)

        da = boot[key]["med"]
        sig = boot[key]["sig"]
        da_sig = da.where(sig)

        cf = ax.contourf(
            da["lon"], da["lat"], da_sig,
            levels=fill_levels, cmap=cmap, extend="both",
            transform=data_crs
        )
        if mappable is None:
            mappable = cf

        contour_sign(ax, da, line_levels, transform=data_crs, color="k", lw=0.8)
        ax.set_title(panel_titles[key])

    cb = fig.colorbar(mappable, cax=cax_left, orientation="horizontal")
    cb.set_label(cbar_label)
    cb.ax.tick_params(labelsize=9)

    # density panel
    if "seed" in z_all.dims:
        z_stack = z_all.rename({"time": "time1"}).stack(sample=("seed", "time1"))
    else:
        z_stack = z_all.rename({"time": "sample"})

    zi = z_stack.sel(mode=mode_interannual).values
    zd = z_stack.sel(mode=mode_decadal).values
    ok = np.isfinite(zi) & np.isfinite(zd)
    x = zd[ok]
    y = zi[ok]

    H, xedges, yedges = np.histogram2d(x, y, bins=hist_bins, density=True)
    Hs = H.astype(float)
    if gaussian_filter is not None and smooth_sigma and smooth_sigma > 0:
        Hs = gaussian_filter(Hs, sigma=smooth_sigma)

    # much less blotchy than before
    Hs_masked = np.ma.masked_where(Hs <= 0, Hs)

    cmap_den = plt.get_cmap("magma_r").copy()
    cmap_den.set_bad(color="white")

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax_dist.imshow(
        Hs_masked.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",
        cmap=cmap_den,
    )

    if np.isfinite(d33) and np.isfinite(d67):
        ax_dist.axvline(d33, color="k", lw=1.0)
        ax_dist.axvline(d67, color="k", lw=1.0)
    if np.isfinite(i33) and np.isfinite(i67):
        ax_dist.axhline(i33, color="k", lw=1.0)
        ax_dist.axhline(i67, color="k", lw=1.0)

    ax_dist.set_xlabel("Decadal latent (z_d)")
    ax_dist.set_ylabel("Interannual latent (z_i)")
    try:
        ax_dist.set_box_aspect(1)
    except Exception:
        ax_dist.set_aspect("equal", adjustable="box")

    ax_dist.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.65")
    for sp in ax_dist.spines.values():
        sp.set_visible(False)

    cb_den = fig.colorbar(im, cax=cax_right, orientation="horizontal")
    cb_den.set_label("Joint density")
    cb_den.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.08)
    return fig, boot



# =========================================================
# Compositing core
# =========================================================
def composite_once_joint_paironly(
    rec_pair, z, i_mode, d_mode,
    i_sign, d_sign,
):
    """
    Composite of pair-only reconstructions in a latent quadrant.
    """
    zi = z.sel(mode=i_mode).values
    zd = z.sel(mode=d_mode).values

    i33, i67 = _q33_67(zi)
    d33, d67 = _q33_67(zd)
    if not (np.isfinite(i33) and np.isfinite(i67) and np.isfinite(d33) and np.isfinite(d67)):
        return None

    mi = zi > i67 if i_sign > 0 else zi < i33
    md = zd > d67 if d_sign > 0 else zd < d33
    gate = mi & md

    idx = np.where(gate)[0]
    if idx.size < 12:
        return None

    return rec_pair.isel(time=idx).mean("time", skipna=True)


def bootstrap_joint_composite_paironly(
    rec_pair, z, i_mode, d_mode,
    i_sign, d_sign,
    n_boot=500, block_len=12, rng_seed=0
):
    rec2, z2 = xr.align(rec_pair, z, join="inner")
    n = rec2.sizes["time"]
    rng = np.random.default_rng(rng_seed)

    lat = rec2["lat"].values
    lon = rec2["lon"].values
    boot_maps = np.full((n_boot, lat.size, lon.size), np.nan, dtype=np.float32)

    for b in range(n_boot):
        idx = circular_block_indices(n, block_len, rng)
        rec_b = rec2.isel(time=idx)
        z_b = z2.isel(time=idx)

        m = composite_once_joint_paironly(
            rec_b, z_b,
            i_mode=i_mode, d_mode=d_mode,
            i_sign=i_sign, d_sign=d_sign
        )
        if m is None:
            continue
        boot_maps[b, :, :] = m.values.astype(np.float32)

    med, lo, hi, sig = _ci_excludes_zero(boot_maps)
    med_da = xr.DataArray(med, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    lo_da  = xr.DataArray(lo,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    hi_da  = xr.DataArray(hi,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    sig_da = xr.DataArray(sig, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    return {"med": med_da, "lo": lo_da, "hi": hi_da, "sig": sig_da}


# =========================================================
# Plotting helpers
# =========================================================
def contour_sign(ax, da, levels, *, transform, color="k", lw=0.8):
    pos = [l for l in levels if l > 0]
    neg = [l for l in levels if l < 0]
    if neg:
        ax.contour(
            da["lon"], da["lat"], da,
            levels=neg, transform=transform,
            colors=color, linewidths=lw, linestyles="--"
        )
    if pos:
        ax.contour(
            da["lon"], da["lat"], da,
            levels=pos, transform=transform,
            colors=color, linewidths=lw, linestyles="-"
        )


def nice_sym_levels(data_list, n=17, robust_q=0.99):
    vals = []
    for da in data_list:
        v = da.values
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return np.linspace(-1, 1, n)
    v = np.concatenate(vals)
    vmax = float(np.nanquantile(np.abs(v), robust_q))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(v))) if np.isfinite(np.nanmax(np.abs(v))) else 1.0
        if vmax == 0:
            vmax = 1.0
    return np.linspace(-vmax, vmax, n)


def plot_joint_extremes_paironly_figure(
    rec_pair,
    z_all,
    *,
    mode_interannual=4,
    mode_decadal=5,
    n_boot=500,
    block_len=12,
    rng_seed=123,
    central_longitude=180,
    cmap="RdBu_r",
    hist_bins=80,
    smooth_sigma=1.2,
):
    """
    Same layout as your previous figure, but composites pair-only reconstructions.
    """
    z = z_all
    if "seed" in z.dims:
        z = z.mean("seed")
    if "time" not in z.dims:
        z = z.transpose("time", "mode")

    quads = {
        "TL": dict(i_sign=+1, d_sign=-1),
        "BL": dict(i_sign=-1, d_sign=-1),
        "TR": dict(i_sign=+1, d_sign=+1),
        "BR": dict(i_sign=-1, d_sign=+1),
    }

    boot = {}
    for k, q in quads.items():
        boot[k] = bootstrap_joint_composite_paironly(
            rec_pair, z,
            i_mode=mode_interannual,
            d_mode=mode_decadal,
            i_sign=q["i_sign"],
            d_sign=q["d_sign"],
            n_boot=n_boot,
            block_len=block_len,
            rng_seed=rng_seed + (0 if k == "TL" else 1 if k == "TR" else 2 if k == "BL" else 3),
        )

    meds = [boot[k]["med"] for k in ["TL", "TR", "BL", "BR"]]
    fill_levels = nice_sym_levels(meds, n=31, robust_q=0.99)
    line_levels = nice_sym_levels(meds, n=9, robust_q=0.99)

    zi = z.sel(mode=mode_interannual).values
    zd = z.sel(mode=mode_decadal).values
    i33, i67 = _q33_67(zi)
    d33, d67 = _q33_67(zd)

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree(central_longitude=central_longitude)

    fig = plt.figure(figsize=(13.0, 6.8))
    gs = GridSpec(
        nrows=3, ncols=4, figure=fig,
        height_ratios=[1.0, 1.0, 0.10],
        width_ratios=[1.0, 1.0, 0.2, 1.6],
        wspace=0.10,
        hspace=0.10,
    )

    ax_TL = fig.add_subplot(gs[0, 0], projection=proj)
    ax_TR = fig.add_subplot(gs[0, 1], projection=proj)
    ax_BL = fig.add_subplot(gs[1, 0], projection=proj)
    ax_BR = fig.add_subplot(gs[1, 1], projection=proj)

    ax_dist = fig.add_subplot(gs[0:2, 3])
    cax_left = fig.add_subplot(gs[2, 0:2])
    cax_right = fig.add_subplot(gs[2, 3])

    add_panel_label(ax_TL, "a")
    add_panel_label(ax_TR, "b")
    add_panel_label(ax_BL, "c")
    add_panel_label(ax_BR, "d")
    add_panel_label(ax_dist, "e")

    mappable = None
    order = [("TL", ax_TL), ("TR", ax_TR), ("BL", ax_BL), ("BR", ax_BR)]
    titles = {
        "TL": "IA+ / D-",
        "TR": "IA+ / D+",
        "BL": "IA- / D-",
        "BR": "IA- / D+",
    }

    for key, ax in order:
        ax.coastlines(linewidth=0.6)

        da = boot[key]["med"]
        sig = boot[key]["sig"]
        da_sig = da.where(sig)

        cf = ax.contourf(
            da["lon"], da["lat"], da_sig,
            levels=fill_levels, cmap=cmap, extend="both",
            transform=data_crs
        )
        if mappable is None:
            mappable = cf

        contour_sign(ax, da, line_levels, transform=data_crs, color="k", lw=0.8)
        ax.set_title(titles[key])

    cb = fig.colorbar(mappable, cax=cax_left, orientation="horizontal")
    cb.set_label("Pair-only reconstruction composite (K)")
    cb.ax.tick_params(labelsize=9)

    # right panel: latent density
    if "seed" in z_all.dims:
        z_stack = z_all.rename({"time": "time1"}).stack(sample=("seed", "time1"))
    else:
        z_stack = z_all.rename({"time": "sample"})

    zi = z_stack.sel(mode=mode_interannual).values
    zd = z_stack.sel(mode=mode_decadal).values
    ok = np.isfinite(zi) & np.isfinite(zd)
    x = zd[ok]
    y = zi[ok]

    H, xedges, yedges = np.histogram2d(x, y, bins=hist_bins, density=True)
    Hs = H.astype(float)
    if gaussian_filter is not None and smooth_sigma and smooth_sigma > 0:
        Hs = gaussian_filter(Hs, sigma=smooth_sigma)

    MIN_COUNT = 0.0001
    Hs_masked = np.ma.masked_where(H < MIN_COUNT, Hs)

    cmap_den = plt.get_cmap("magma_r").copy()
    cmap_den.set_bad(color="white")

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax_dist.imshow(
        Hs_masked.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",
        cmap=cmap_den,
    )

    if np.isfinite(d33) and np.isfinite(d67):
        ax_dist.axvline(d33, color="k", lw=1.0)
        ax_dist.axvline(d67, color="k", lw=1.0)
    if np.isfinite(i33) and np.isfinite(i67):
        ax_dist.axhline(i33, color="k", lw=1.0)
        ax_dist.axhline(i67, color="k", lw=1.0)

    ax_dist.set_xlabel("Decadal latent (z_d)")
    ax_dist.set_ylabel("Interannual latent (z_i)")
    try:
        ax_dist.set_box_aspect(1)
    except Exception:
        ax_dist.set_aspect("equal", adjustable="box")

    ax_dist.set_xticks([-2, -1, 0, 1, 2])
    ax_dist.set_yticks([-2, -1, 0, 1, 2])
    ax_dist.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.65")
    for sp in ax_dist.spines.values():
        sp.set_visible(False)

    cb_den = fig.colorbar(im, cax=cax_right, orientation="horizontal")
    cb_den.set_label("Joint density")
    cb_den.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.04, right=0.98, top=0.98, bottom=0.08)
    return fig, boot


# =========================================================
# Optional: additive residual diagnostic
# =========================================================
def compute_superposition_residual(rec_pair, z, i_mode=4, d_mode=5):
    """
    Optional helper if you want the explicit superposition-failure map:
      N = D(i,d) - D(i,0) - D(0,d) + D(0,0)

    Here it is estimated via quadrant composites:
      ( ++ + -- ) - ( +- + -+ )
    """
    comps = {}
    for name, i_sign, d_sign in [
        ("pp", +1, +1),
        ("pm", +1, -1),
        ("mp", -1, +1),
        ("mm", -1, -1),
    ]:
        comps[name] = composite_once_joint_paironly(
            rec_pair, z, i_mode=i_mode, d_mode=d_mode,
            i_sign=i_sign, d_sign=d_sign
        )
    return (comps["pp"] + comps["mm"]) - (comps["pm"] + comps["mp"])

def plot_symmetry_2x3_from_boot(
    boot_add,
    boot_pair,
    boot_resid,
    *,
    central_longitude=180,
    cmap="RdBu_r",
    use_half_difference=True,
    title_row1="0.5 × (TL - BR): IA+/D- vs IA-/D+",
    title_row2="0.5 × (TR - BL): IA+/D+ vs IA-/D-",
):
    """
    Make a 2-row x 3-col summary figure from existing quadrant bootstrap outputs.

    Rows:
      1: TL - BR  (or 0.5*(TL - BR))
      2: TR - BL  (or 0.5*(TR - BL))

    Cols:
      1: additive
      2: pair-only
      3: residual

    Notes
    -----
    This uses the bootstrap medians already stored in boot_*.
    It does NOT propagate significance exactly for the differences.
    If you want significance on the symmetry maps themselves, we should
    bootstrap the differences directly.
    """
    factor = 0.5 if use_half_difference else 1.0

    def sym_maps(boot):
        s1 = factor * (boot["TL"]["med"] - boot["BR"]["med"])
        s2 = factor * (boot["TR"]["med"] - boot["BL"]["med"])
        return s1, s2

    add_s1, add_s2 = sym_maps(boot_add)
    pair_s1, pair_s2 = sym_maps(boot_pair)
    resid_s1, resid_s2 = sym_maps(boot_resid)

    all_maps = [add_s1, pair_s1, resid_s1, add_s2, pair_s2, resid_s2]
    fill_levels = nice_sym_levels(all_maps, n=31, robust_q=0.99)
    line_levels = nice_sym_levels(all_maps, n=9, robust_q=0.99)

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree(central_longitude=central_longitude)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(11.5, 5.8),
        subplot_kw={"projection": proj}
    )
    axes = np.asarray(axes)

    panel_maps = [
        [add_s1, pair_s1, resid_s1],
        [add_s2, pair_s2, resid_s2],
    ]
    col_titles = ["Additive", "Pair-only", "Residual"]
    row_titles = [title_row1, title_row2]

    mappable = None
    label_iter = iter(["a", "b", "c", "d", "e", "f"])

    for r in range(2):
        for c in range(3):
            ax = axes[r, c]
            da = panel_maps[r][c]

            ax.coastlines(linewidth=0.6)

            cf = ax.contourf(
                da["lon"], da["lat"], da,
                levels=fill_levels,
                cmap=cmap,
                extend="both",
                transform=data_crs,
            )
            if mappable is None:
                mappable = cf

            contour_sign(
                ax, da, line_levels,
                transform=data_crs,
                color="k",
                lw=0.8
            )

            add_panel_label(ax, next(label_iter))
            if r == 0:
                ax.set_title(col_titles[c])

    # row labels on left
    axes[0, 0].text(
        -0.10, 0.5, row_titles[0],
        transform=axes[0, 0].transAxes,
        rotation=90, ha="right", va="center", fontsize=10
    )
    axes[1, 0].text(
        -0.10, 0.5, row_titles[1],
        transform=axes[1, 0].transAxes,
        rotation=90, ha="right", va="center", fontsize=10
    )

    cbar = fig.colorbar(
        mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        aspect=40
    )
    cbar.set_label(
        "Symmetry map (K)"
        + ("; half-difference" if use_half_difference else "; raw difference")
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.08, hspace=0.10)
    return fig

from pathlib import Path
import os

def _boot_symmetry_from_field(
    field,
    z,
    *,
    mode_interannual=4,
    mode_decadal=5,
    row="tl_br",   # "tl_br" or "tr_bl"
    n_boot=500,
    block_len=12,
    rng_seed=0,
    half_difference=True,
):
    """
    Bootstrap the symmetry map directly.

    row="tl_br":  factor * (TL - BR) = factor * [(i+,d-) - (i-,d+)]
    row="tr_bl":  factor * (TR - BL) = factor * [(i+,d+) - (i-,d-)]

    Returns dict with med/lo/hi/sig and also the raw boot array.
    """
    factor = 0.5 if half_difference else 1.0

    field2, z2 = xr.align(field, z, join="inner")
    n = field2.sizes["time"]
    lat = field2["lat"].values
    lon = field2["lon"].values

    rng = np.random.default_rng(rng_seed)
    boot_maps = np.full((n_boot, lat.size, lon.size), np.nan, dtype=np.float32)

    if row not in {"tl_br", "tr_bl"}:
        raise ValueError("row must be 'tl_br' or 'tr_bl'")

    for b in range(n_boot):
        idx = circular_block_indices(n, block_len, rng)
        f_b = field2.isel(time=idx)
        z_b = z2.isel(time=idx)

        if row == "tl_br":
            a = composite_once_joint_from_field(
                f_b, z_b,
                i_mode=mode_interannual, d_mode=mode_decadal,
                i_sign=+1, d_sign=-1
            )
            bmap = composite_once_joint_from_field(
                f_b, z_b,
                i_mode=mode_interannual, d_mode=mode_decadal,
                i_sign=-1, d_sign=+1
            )
        else:  # "tr_bl"
            a = composite_once_joint_from_field(
                f_b, z_b,
                i_mode=mode_interannual, d_mode=mode_decadal,
                i_sign=+1, d_sign=+1
            )
            bmap = composite_once_joint_from_field(
                f_b, z_b,
                i_mode=mode_interannual, d_mode=mode_decadal,
                i_sign=-1, d_sign=-1
            )

        if (a is None) or (bmap is None):
            continue

        sym = factor * (a - bmap)
        boot_maps[b, :, :] = sym.values.astype(np.float32)

    med, lo, hi, sig = _ci_excludes_zero(boot_maps)
    return {
        "boot": boot_maps,
        "med": xr.DataArray(med, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")),
        "lo":  xr.DataArray(lo,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon")),
        "hi":  xr.DataArray(hi,  coords={"lat": lat, "lon": lon}, dims=("lat", "lon")),
        "sig": xr.DataArray(sig, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")),
    }

def compute_symmetry_boot_dataset(
    out,
    *,
    mode_interannual=4,
    mode_decadal=5,
    n_boot=500,
    block_len=12,
    rng_seed=123,
):
    """
    Build a cached xarray.Dataset containing bootstrapped symmetry maps
    for add / pair / resid and both row types.

    out is the dict returned by build_pair_only_reconstructions(...).
    """
    z = out["z_all"].mean("seed") if "seed" in out["z_all"].dims else out["z_all"]

    fields = {
        "add": out["add_mean"],
        "pair": out["pair_mean"],
        "resid": out["resid_mean"],
    }
    rows = {
        "tl_br": "IA+/D- minus IA-/D+",
        "tr_bl": "IA+/D+ minus IA-/D-",
    }

    vars_out = {}
    attrs = {
        "mode_interannual": int(mode_interannual),
        "mode_decadal": int(mode_decadal),
        "n_boot": int(n_boot),
        "block_len": int(block_len),
        "rng_seed": int(rng_seed),
        "row_tl_br": rows["tl_br"],
        "row_tr_bl": rows["tr_bl"],
    }

    for ifield, (field_name, field) in enumerate(fields.items()):
        for irow, row_name in enumerate(rows.keys()):
            res = _boot_symmetry_from_field(
                field,
                z,
                mode_interannual=mode_interannual,
                mode_decadal=mode_decadal,
                row=row_name,
                n_boot=n_boot,
                block_len=block_len,
                rng_seed=rng_seed + 100 * ifield + (0 if row_name == "tl_br" else 1),
                half_difference=True,
            )
            for stat in ["med", "lo", "hi", "sig"]:
                vars_out[f"{field_name}_{row_name}_{stat}"] = res[stat]

    ds = xr.Dataset(vars_out, attrs=attrs)
    return ds

def plot_symmetry_2x3_from_dataset(
    ds_sym,
    *,
    central_longitude=180,
    cmap_main="RdBu_r",
    cmap_resid="RdBu_r",
    robust_q_main=0.99,
    robust_q_resid=0.99,
    n_fill=31,
    n_line=9,
    row_titles=None,
):
    """
    Plot 2x3 symmetry figure from cached dataset.

    Columns:
      1 add
      2 pair
      3 resid

    Rows:
      1 tl_br
      2 tr_bl
    """
    if row_titles is None:
        row_titles = {
            "tl_br": "0.5 × [(IA+/D-) − (IA-/D+)]",
            "tr_bl": "0.5 × [(IA+/D+) − (IA-/D-)]",
        }

    # ---- collect maps
    main_maps = [
        ds_sym["add_tl_br_med"], ds_sym["pair_tl_br_med"],
        ds_sym["add_tr_bl_med"], ds_sym["pair_tr_bl_med"],
    ]
    resid_maps = [
        ds_sym["resid_tl_br_med"], ds_sym["resid_tr_bl_med"],
    ]

    fill_levels_main = nice_sym_levels(main_maps, n=n_fill, robust_q=robust_q_main)
    line_levels_main = nice_sym_levels(main_maps, n=n_line, robust_q=robust_q_main)

    fill_levels_resid = nice_sym_levels(resid_maps, n=n_fill, robust_q=robust_q_resid)
    line_levels_resid = nice_sym_levels(resid_maps, n=n_line, robust_q=robust_q_resid)

    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    data_crs = ccrs.PlateCarree(central_longitude=central_longitude)

    fig = plt.figure(figsize=(11.8, 6.2))
    gs = GridSpec(
        nrows=3, ncols=3, figure=fig,
        height_ratios=[1.0, 1.0, 0.10],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.10,
        wspace=0.08,
    )

    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c], projection=proj)

    cax_main = fig.add_subplot(gs[2, 0:2])
    cax_resid = fig.add_subplot(gs[2, 2])

    panel_order = [
        ("add",   "tl_br", axes[0, 0]),
        ("pair",  "tl_br", axes[0, 1]),
        ("resid", "tl_br", axes[0, 2]),
        ("add",   "tr_bl", axes[1, 0]),
        ("pair",  "tr_bl", axes[1, 1]),
        ("resid", "tr_bl", axes[1, 2]),
    ]
    col_titles = ["Additive", "Pair-only", "Residual"]

    mappable_main = None
    mappable_resid = None
    panel_labels = iter(["a", "b", "c", "d", "e", "f"])

    for field_name, row_name, ax in panel_order:
        med = ds_sym[f"{field_name}_{row_name}_med"]
        sig = ds_sym[f"{field_name}_{row_name}_sig"]
        med_sig = med.where(sig)

        ax.coastlines(linewidth=0.6)

        if field_name in ["add", "pair"]:
            cf = ax.contourf(
                med["lon"], med["lat"], med_sig,
                levels=fill_levels_main, cmap=cmap_main, extend="both",
                transform=data_crs
            )
            contour_sign(ax, med, line_levels_main, transform=data_crs, color="k", lw=0.8)
            if mappable_main is None:
                mappable_main = cf
        else:
            cf = ax.contourf(
                med["lon"], med["lat"], med_sig,
                levels=fill_levels_resid, cmap=cmap_resid, extend="both",
                transform=data_crs
            )
            contour_sign(ax, med, line_levels_resid, transform=data_crs, color="k", lw=0.8)
            if mappable_resid is None:
                mappable_resid = cf

        add_panel_label(ax, next(panel_labels))

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title)

    axes[0, 0].text(
        -0.10, 0.5, row_titles["tl_br"],
        transform=axes[0, 0].transAxes,
        rotation=90, ha="right", va="center", fontsize=10
    )
    axes[1, 0].text(
        -0.10, 0.5, row_titles["tr_bl"],
        transform=axes[1, 0].transAxes,
        rotation=90, ha="right", va="center", fontsize=10
    )

    cb_main = fig.colorbar(mappable_main, cax=cax_main, orientation="horizontal")
    cb_main.set_label("Symmetry map (K): additive / pair-only")
    cb_main.ax.tick_params(labelsize=9)

    cb_resid = fig.colorbar(mappable_resid, cax=cax_resid, orientation="horizontal")
    cb_resid.set_label("Symmetry map (K): residual")
    cb_resid.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.10)
    return fig

def load_or_compute_symmetry_boot_dataset(
    cache_path,
    out,
    *,
    mode_interannual=4,
    mode_decadal=5,
    n_boot=500,
    block_len=12,
    rng_seed=123,
    recompute=False,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not recompute:
        return xr.open_dataset(cache_path)

    ds = compute_symmetry_boot_dataset(
        out,
        mode_interannual=mode_interannual,
        mode_decadal=mode_decadal,
        n_boot=n_boot,
        block_len=block_len,
        rng_seed=rng_seed,
    )
    save_symmetry_boot_dataset(ds, cache_path)
    return ds

def save_symmetry_boot_dataset(ds, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = {}
    for v in ds.data_vars:
        if ds[v].dtype.kind == "f":
            enc[v] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        elif ds[v].dtype.kind == "b":
            enc[v] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(path, encoding=enc)
    return path

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_ensemble = 10
    experiment = "large-ensemble-2-1940-2014"

    out = build_pair_only_reconstructions(
        experiment=experiment,
        n_ensemble=n_ensemble,
        mode_interannual=4,
        mode_decadal=5,
    )

    # cache symmetry products
    cache_file = f"cached_symmetry_maps_{experiment}_ia4_dec5.nc"
    ds_sym = load_or_compute_symmetry_boot_dataset(
        cache_file,
        out,
        mode_interannual=4,
        mode_decadal=5,
        n_boot=500,
        block_len=12,
        rng_seed=123,
        recompute=False,   # switch to True if you want to refresh cache
    )

    fig_sym = plot_symmetry_2x3_from_dataset(ds_sym)
    fig_sym.savefig("joint_symmetry_summary_2x3_bootdiff.png", dpi=300, bbox_inches="tight")
    plt.show()
