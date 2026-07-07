import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import kgae
import cartopy.crs as ccrs
from pathlib import Path
from matplotlib.gridspec import GridSpec

# -----------------------------
# Settings
# -----------------------------
experiment = Path("kgvae_cv1940-2014")
n_ensemble = 30
data_set = "xval"

thresh = 0.1                       # frac-of-ensemble sig threshold for alpha/beta rows
cols = [5, 4, 3]                   # (Decadal, Interannual, Quasibiennial)
col_titles = ["Decadal (i=3)", "Interannual (i=2)", "Quasibiennial (i=1)"]

# composite selection + scaling knobs
k_sigma = 1.0                      # event threshold: z_m > +kσ, z_m < -kσ
other_sigma = 1.0                  # restrict *other modes in cols* to +/- other_sigma*sigma
scale_to = "none"                  # "none" | "1sigma" | "2sigma"
eps = 1e-12

# "curvature" composite knobs (amplitude 2nd-difference in |z|)
curv_t1 = 0.5                      # low/mid cutoff in σ units
curv_t2 = 1.0                      # mid/high cutoff in σ units

# bootstrap + plotting knobs
n_boot = 100
boot_alpha = 0.5                  # two-sided
cmap = "RdBu_r"
n_fill_levels = 21
n_contours = 11

proj = ccrs.PlateCarree(central_longitude=180)
data_crs = ccrs.PlateCarree(central_longitude=180)

# -----------------------------
# Load ensemble means
# -----------------------------
alphas = 0
betas = 0
alpha_sigs = 0
beta_sigs = 0
zs = []

for seed in range(n_ensemble):
    alpha = xr.open_dataset(experiment / f"seed{seed}" / f"{data_set}_alpha.nc")["__xarray_dataarray_variable__"]
    alpha_sig = xr.open_dataset(
        experiment / f"seed{seed}" / f'{data_set}_alpha_sig{"s" if data_set != "xval" else ""}.nc'
    )["__xarray_dataarray_variable__"]

    beta = xr.open_dataset(experiment / f"seed{seed}" / f"{data_set}_beta.nc")["__xarray_dataarray_variable__"]
    beta_sig = xr.open_dataset(
        experiment / f"seed{seed}" / f'{data_set}_beta_sig{"s" if data_set != "xval" else ""}.nc'
    )["__xarray_dataarray_variable__"]

    z = xr.open_dataset(experiment / f"seed{seed}" / f"{data_set}_latent.nc")["__xarray_dataarray_variable__"]
    zs.append(z)

    alphas += alpha
    betas += beta
    alpha_sigs += alpha_sig
    beta_sigs += beta_sig

alphas = alphas / n_ensemble
betas = betas / n_ensemble
alpha_sigs = alpha_sigs / n_ensemble
beta_sigs = beta_sigs / n_ensemble

# Latents -> (time, seed, mode)
latents = xr.concat(
    [z.expand_dims(seed=[i]) if "seed" not in z.dims else z for i, z in enumerate(zs)],
    dim="seed",
).transpose("time", "seed", "mode")

# -----------------------------
# SST -> anomalies
# -----------------------------
sst = xr.open_dataset(os.path.expanduser("~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc")).sst
end_year = int(str(experiment)[-4:])
sst = sst.sel(time=slice(None, pd.Timestamp(end_year, 12, 31)))
sst = sst.rename({k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in sst.dims})

ssta, fit, gwm, p = kgae.global_detrend(sst, deg=2)
ssta, mc = kgae.remove_climo(ssta)
ssta = ssta.transpose("time", "lat", "lon")

# -----------------------------
# Helpers
# -----------------------------
def sel_mode_1based(da, dim, m):
    if dim in da.coords and np.any(da[dim].values == m):
        return da.sel({dim: m})
    return da.isel({dim: m - 1})

def symmetric_levels(fields, n=21):
    vmax = max(float(np.nanmax(np.abs(f.values))) for f in fields)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return np.linspace(-vmax, vmax, n)

def contour_levels_from_fill(fill_levels, n_contours=11):
    vmax = float(np.nanmax(np.abs(fill_levels)))
    lev = np.linspace(-vmax, vmax, n_contours)
    return lev[lev != 0]

def overlay_contours(ax, field2d, levels):
    clev = np.asarray(levels)
    clev = clev[clev != 0]
    neg = clev[clev < 0]
    pos = clev[clev > 0]
    if len(neg):
        ax.contour(field2d.lon, field2d.lat, field2d, levels=neg,
                   colors="k", linewidths=0.8, linestyles="dashed",
                   transform=data_crs)
    if len(pos):
        ax.contour(field2d.lon, field2d.lat, field2d, levels=pos,
                   colors="k", linewidths=0.8, linestyles="solid",
                   transform=data_crs)

def bootstrap_sig_mask_pair_scaled(pos, neg, zpos, zneg, sigma_m, op="diff",
                                   n_boot=500, alpha=0.05, scale_to="none", seed=0, eps=1e-12):
    rng = np.random.default_rng(seed)
    sigma_m_val = float(sigma_m.values) if hasattr(sigma_m, "values") else float(sigma_m)

    pos_mean = pos.mean("time").transpose("lat", "lon")
    neg_mean = neg.mean("time").transpose("lat", "lon")
    resp_mean = (pos_mean - neg_mean) if op == "diff" else (pos_mean + neg_mean)

    mu_pos = float(zpos.mean("time").values)
    mu_neg = float(zneg.mean("time").values)
    delta_z = (mu_pos - mu_neg)

    if scale_to == "none":
        base_scale = 1.0
    elif scale_to == "1sigma":
        base_scale = sigma_m_val / (delta_z + eps)
    elif scale_to == "2sigma":
        base_scale = (2.0 * sigma_m_val) / (delta_z + eps)
    else:
        raise ValueError("scale_to must be 'none', '1sigma', or '2sigma'")

    resp_mean_scaled = resp_mean * base_scale

    Tp = pos.sizes["time"]
    Tn = neg.sizes["time"]
    if Tp < 2 or Tn < 2:
        sig = xr.zeros_like(resp_mean_scaled, dtype=bool)
        return resp_mean_scaled, sig

    P = int(pos.sizes["lat"] * pos.sizes["lon"])
    Xp = pos.transpose("time", "lat", "lon").values.reshape(Tp, P)
    Xn = neg.transpose("time", "lat", "lon").values.reshape(Tn, P)
    zp = zpos.values
    zn = zneg.values

    boots = np.empty((n_boot, P), dtype=np.float64)
    for b in range(n_boot):
        ip = rng.integers(0, Tp, size=Tp)
        ineg = rng.integers(0, Tn, size=Tn)

        mp = np.nanmean(Xp[ip, :], axis=0)
        mn = np.nanmean(Xn[ineg, :], axis=0)
        draw = (mp - mn) if op == "diff" else (mp + mn)

        mu_pos_b = np.nanmean(zp[ip])
        mu_neg_b = np.nanmean(zn[ineg])
        delta_z_b = (mu_pos_b - mu_neg_b)

        if scale_to == "none":
            s_b = 1.0
        elif scale_to == "1sigma":
            s_b = sigma_m_val / (delta_z_b + eps)
        else:
            s_b = (2.0 * sigma_m_val) / (delta_z_b + eps)

        boots[b] = draw * s_b

    lo = np.nanquantile(boots, alpha / 2, axis=0)
    hi = np.nanquantile(boots, 1 - alpha / 2, axis=0)
    sig_flat = (lo > 0) | (hi < 0)

    sig = xr.DataArray(
        sig_flat.reshape(pos.sizes["lat"], pos.sizes["lon"]),
        coords={"lat": pos.lat, "lon": pos.lon},
        dims=("lat", "lon"),
    )
    return resp_mean_scaled, sig

def bootstrap_sig_mask_twosample_diff(A, B, n_boot=500, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    A = A.transpose("time", "lat", "lon")
    B = B.transpose("time", "lat", "lon")

    mean_diff = (A.mean("time") - B.mean("time")).transpose("lat", "lon")

    Ta = A.sizes["time"]
    Tb = B.sizes["time"]
    if Ta < 2 or Tb < 2:
        return mean_diff, xr.zeros_like(mean_diff, dtype=bool)

    P = int(A.sizes["lat"] * A.sizes["lon"])
    Xa = A.values.reshape(Ta, P)
    Xb = B.values.reshape(Tb, P)

    boots = np.empty((n_boot, P), dtype=np.float64)
    for b in range(n_boot):
        ia = rng.integers(0, Ta, size=Ta)
        ib = rng.integers(0, Tb, size=Tb)
        ma = np.nanmean(Xa[ia, :], axis=0)
        mb = np.nanmean(Xb[ib, :], axis=0)
        boots[b] = ma - mb

    lo = np.nanquantile(boots, alpha / 2, axis=0)
    hi = np.nanquantile(boots, 1 - alpha / 2, axis=0)
    sig_flat = (lo > 0) | (hi < 0)

    sig = xr.DataArray(
        sig_flat.reshape(A.sizes["lat"], A.sizes["lon"]),
        coords={"lat": A.lat, "lon": A.lon},
        dims=("lat", "lon"),
    )
    return mean_diff, sig

def pattcorr(a, b):
    a = a.transpose("lat", "lon")
    b = b.transpose("lat", "lon")
    w = np.cos(np.deg2rad(a["lat"]))
    w2d = w.broadcast_like(a)

    aa = (a * w2d).values.ravel()
    bb = (b * w2d).values.ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    if m.sum() < 10:
        return np.nan
    aa = aa[m] - aa[m].mean()
    bb = bb[m] - bb[m].mean()
    return float(np.dot(aa, bb) / np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))

def curvature_composite_from_absz(ssta_al, z_m, other_ok, sigma_m,
                                 t1=0.5, t2=1.0, n_boot=500, alpha=0.05, seed=0):
    """
    2nd-difference in amplitude:
      C = mean(|z|>t2σ) - 2*mean(t1σ<|z|<=t2σ) + mean(|z|<=t1σ)
    Returns: C_mean(lat,lon), sig_mask(lat,lon), counts dict
    """
    rng = np.random.default_rng(seed)
    zabs = np.abs(z_m)
    thr1 = t1 * sigma_m
    thr2 = t2 * sigma_m

    low_t = (zabs <= thr1) & other_ok
    mid_t = (zabs > thr1) & (zabs <= thr2) & other_ok
    hi_t  = (zabs > thr2) & other_ok

    low = ssta_al.where(low_t, drop=True)
    mid = ssta_al.where(mid_t, drop=True)
    hi  = ssta_al.where(hi_t,  drop=True)

    C_mean = (hi.mean("time") - 2.0 * mid.mean("time") + low.mean("time")).transpose("lat", "lon")

    Tl, Tm, Th = low.sizes["time"], mid.sizes["time"], hi.sizes["time"]
    counts = {"n_low": int(Tl), "n_mid": int(Tm), "n_hi": int(Th)}
    if min(Tl, Tm, Th) < 2:
        return C_mean, xr.zeros_like(C_mean, dtype=bool), counts

    P = int(ssta_al.sizes["lat"] * ssta_al.sizes["lon"])
    XL = low.transpose("time", "lat", "lon").values.reshape(Tl, P)
    XM = mid.transpose("time", "lat", "lon").values.reshape(Tm, P)
    XH = hi.transpose("time", "lat", "lon").values.reshape(Th, P)

    boots = np.empty((n_boot, P), dtype=np.float64)
    for b in range(n_boot):
        il = rng.integers(0, Tl, size=Tl)
        im = rng.integers(0, Tm, size=Tm)
        ih = rng.integers(0, Th, size=Th)
        ml = np.nanmean(XL[il, :], axis=0)
        mm = np.nanmean(XM[im, :], axis=0)
        mh = np.nanmean(XH[ih, :], axis=0)
        boots[b] = mh - 2.0 * mm + ml

    lo = np.nanquantile(boots, alpha / 2, axis=0)
    hi = np.nanquantile(boots, 1 - alpha / 2, axis=0)
    sig_flat = (lo > 0) | (hi < 0)

    sig = xr.DataArray(
        sig_flat.reshape(ssta_al.sizes["lat"], ssta_al.sizes["lon"]),
        coords={"lat": ssta_al.lat, "lon": ssta_al.lon},
        dims=("lat", "lon"),
    )
    return C_mean, sig, counts

# -----------------------------
# Rows 1–2: alpha / beta_ii
# -----------------------------
alpha_cols = [sel_mode_1based(alphas, "tendency_mode", m).transpose("lat", "lon") for m in cols]
alpha_sig_cols = [sel_mode_1based(alpha_sigs, "tendency_mode", m).transpose("lat", "lon") for m in cols]

beta_ii_cols, beta_sig_ii_cols = [], []
for m in cols:
    b = sel_mode_1based(betas, "tendency_mode", m)
    b = sel_mode_1based(b, "predictor_mode", m).transpose("lat", "lon")
    s = sel_mode_1based(beta_sigs, "tendency_mode", m)
    s = sel_mode_1based(s, "predictor_mode", m).transpose("lat", "lon")
    beta_ii_cols.append(b)
    beta_sig_ii_cols.append(s)

# -----------------------------
# Row 3: beta_sum over predictors in `cols` for each target m
# -----------------------------
beta_sum_cols = []
beta_sum_sig_cols = []
for m in cols:
    b_m = sel_mode_1based(betas, "tendency_mode", m)         # (predictor_mode, lat, lon)
    s_m = sel_mode_1based(beta_sigs, "tendency_mode", m)

    b_list = []
    s_list = []
    for jpred in cols:
        b_ = sel_mode_1based(b_m, "predictor_mode", jpred).transpose("lat", "lon")
        s_ = sel_mode_1based(s_m, "predictor_mode", jpred).transpose("lat", "lon")
        b_list.append(b_)
        s_list.append(s_)

    bsum = xr.concat(b_list, dim="pred_in_cols").sum("pred_in_cols")
    ssig = xr.concat(s_list, dim="pred_in_cols").mean("pred_in_cols")  # fraction of predictors significant

    beta_sum_cols.append(bsum)
    beta_sum_sig_cols.append(ssig)

# -----------------------------
# Rows 4–5: linear (pos-neg) / nonlinear (pos+neg)
# Row 6: amplitude (|z| high - |z| low)
# Row 7: curvature 2nd-difference in |z| (hi - 2*mid + low)
# -----------------------------
zbar = latents.mean("seed").transpose("time", "mode")
sigma_all = zbar.std("time")

lin_resp, nonlin_resp, absdiff_resp, curv_resp = [], [], [], []
lin_mask, nonlin_mask, absdiff_mask, curv_mask = [], [], [], []
n_pos, n_neg, n_hi, n_lo = [], [], [], []
curv_counts = []

for j, m in enumerate(cols):
    z_m = sel_mode_1based(zbar, "mode", m)
    sigma_m = sel_mode_1based(sigma_all, "mode", m)
    z_m, ssta_al, zbar_al = xr.align(z_m, ssta, zbar, join="inner")

    pos_thresh = k_sigma * sigma_m
    neg_thresh = -k_sigma * sigma_m

    other_modes_to_restrict = [mm for mm in cols if mm != m]
    if len(other_modes_to_restrict) == 0:
        other_ok = xr.ones_like(z_m, dtype=bool)
    else:
        z_other = zbar_al.sel(mode=other_modes_to_restrict)
        sig_other = sigma_all.sel(mode=other_modes_to_restrict)
        other_ok = (np.abs(z_other) <= other_sigma * sig_other).all("mode")

    # --- (pos, neg) ---
    pos_time = (z_m > pos_thresh) & other_ok
    neg_time = (z_m < neg_thresh) & other_ok

    pos = ssta_al.where(pos_time, drop=True)
    neg = ssta_al.where(neg_time, drop=True)
    zpos = z_m.where(pos_time, drop=True)
    zneg = z_m.where(neg_time, drop=True)

    lin_mean, lin_sig = bootstrap_sig_mask_pair_scaled(
        pos, neg, zpos, zneg, sigma_m,
        op="diff", n_boot=n_boot, alpha=boot_alpha,
        scale_to=scale_to, seed=100 + j, eps=eps
    )
    nonlin_mean, nonlin_sig = bootstrap_sig_mask_pair_scaled(
        pos, neg, zpos, zneg, sigma_m,
        op="sum", n_boot=n_boot, alpha=boot_alpha,
        scale_to=scale_to, seed=200 + j, eps=eps
    )

    lin_resp.append(lin_mean.transpose("lat", "lon"))
    nonlin_resp.append(nonlin_mean.transpose("lat", "lon"))
    lin_mask.append(lin_sig.transpose("lat", "lon"))
    nonlin_mask.append(nonlin_sig.transpose("lat", "lon"))
    n_pos.append(int(pos.sizes["time"]))
    n_neg.append(int(neg.sizes["time"]))

    # --- absdiff: |z| hi - |z| low ---
    abs_hi_time = (np.abs(z_m) > pos_thresh) & other_ok
    abs_lo_time = (np.abs(z_m) <= pos_thresh) & other_ok
    hi = ssta_al.where(abs_hi_time, drop=True)
    lo = ssta_al.where(abs_lo_time, drop=True)

    abs_mean, abs_sig = bootstrap_sig_mask_twosample_diff(
        hi, lo, n_boot=n_boot, alpha=boot_alpha, seed=300 + j
    )
    absdiff_resp.append(abs_mean.transpose("lat", "lon"))
    absdiff_mask.append(abs_sig.transpose("lat", "lon"))
    n_hi.append(int(hi.sizes["time"]))
    n_lo.append(int(lo.sizes["time"]))

    # --- curvature: 2nd-diff in |z| amplitude ---
    C_mean, C_sig, counts = curvature_composite_from_absz(
        ssta_al, z_m, other_ok, sigma_m,
        t1=curv_t1, t2=curv_t2, n_boot=n_boot, alpha=boot_alpha, seed=400 + j
    )
    curv_resp.append(C_mean.transpose("lat", "lon"))
    curv_mask.append(C_sig.transpose("lat", "lon"))
    curv_counts.append(counts)

# -----------------------------
# Levels per row (for contourf)
# -----------------------------
alpha_levels = symmetric_levels(alpha_cols, n=n_fill_levels)
beta_levels  = symmetric_levels(beta_ii_cols, n=n_fill_levels)
bsum_levels  = symmetric_levels(beta_sum_cols, n=n_fill_levels)
lin_levels   = symmetric_levels(lin_resp, n=n_fill_levels)
non_levels   = symmetric_levels(nonlin_resp, n=n_fill_levels)
abs_levels   = symmetric_levels(absdiff_resp, n=n_fill_levels)
curv_levels  = symmetric_levels(curv_resp, n=n_fill_levels)

alpha_clev = contour_levels_from_fill(alpha_levels, n_contours=n_contours)
beta_clev  = contour_levels_from_fill(beta_levels,  n_contours=n_contours)
bsum_clev  = contour_levels_from_fill(bsum_levels,  n_contours=n_contours)
lin_clev   = contour_levels_from_fill(lin_levels,   n_contours=n_contours)
non_clev   = contour_levels_from_fill(non_levels,   n_contours=n_contours)
abs_clev   = contour_levels_from_fill(abs_levels,   n_contours=n_contours)
curv_clev  = contour_levels_from_fill(curv_levels,  n_contours=n_contours)

# -----------------------------
# Print correlations
# -----------------------------
print("\n==============================")
print("Spatial correlations (area-weighted)")
print("==============================")
for j, name in enumerate(col_titles):
    print(f"\nMode: {name}")
    print(f"  corr(lin(pos-neg), alpha)             = {pattcorr(lin_resp[j], alpha_cols[j]): .3f}")
    print(f"  corr(nonlin(pos+neg), beta_ii)        = {pattcorr(nonlin_resp[j], beta_ii_cols[j]): .3f}")
    print(f"  corr(nonlin(pos+neg), beta_sum(cols)) = {pattcorr(nonlin_resp[j], beta_sum_cols[j]): .3f}")
    print(f"  corr(absdiff(|z|hi-lo), beta_ii)      = {pattcorr(absdiff_resp[j], beta_ii_cols[j]): .3f}")
    print(f"  corr(absdiff(|z|hi-lo), beta_sum)     = {pattcorr(absdiff_resp[j], beta_sum_cols[j]): .3f}")
    print(f"  corr(curv_2diff(|z|), beta_ii)        = {pattcorr(curv_resp[j], beta_ii_cols[j]): .3f}")
    print(f"  corr(curv_2diff(|z|), beta_sum)       = {pattcorr(curv_resp[j], beta_sum_cols[j]): .3f}")
    print(f"  corr(curv_2diff(|z|), nonlin(pos+neg))= {pattcorr(curv_resp[j], nonlin_resp[j]): .3f}")
    cc = curv_counts[j]
    print(f"  curvature counts: n_low={cc['n_low']}, n_mid={cc['n_mid']}, n_hi={cc['n_hi']}")

# -----------------------------
# Plot: 7 rows x (3 maps + 3 colorbars) using GridSpec
# (each panel gets its own colorbar)
# -----------------------------
fig = plt.figure(figsize=(16.5, 19.2))
gs = GridSpec(
    7, 6, figure=fig,
    width_ratios=[1.0, 0.055, 1.0, 0.055, 1.0, 0.055],
    wspace=0.14, hspace=0.22
)

row_labels = [
    "alpha",
    "beta_ii",
    "beta_sum over predictors in cols",
    "Linear / symmetric: (pos − neg)",
    "Nonlinear / asymmetric: (pos + neg)",
    "|z| amplitude: hi − lo",
    f"Curvature (2nd diff in |z|): hi − 2·mid + low (t1={curv_t1}σ, t2={curv_t2}σ)",
]

for r in range(7):
    for j in range(3):
        ax = fig.add_subplot(gs[r, 2*j], projection=proj)
        cax = fig.add_subplot(gs[r, 2*j + 1])

        if j == 0:
            ax.text(-0.10, 0.5, row_labels[r], transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=12)

        if r == 0:
            ax.set_title(col_titles[j])
            mask = alpha_sig_cols[j] >= thresh
            fld = alpha_cols[j].where(mask)
            cf = ax.contourf(fld.lon, fld.lat, fld, levels=alpha_levels, cmap=cmap,
                             extend="both", transform=data_crs)
            overlay_contours(ax, alpha_cols[j], alpha_clev)
            ax.coastlines(linewidth=0.6)
            fig.colorbar(cf, cax=cax, orientation="vertical").ax.tick_params(labelsize=8)

        elif r == 1:
            mask = beta_sig_ii_cols[j] >= thresh
            fld = beta_ii_cols[j].where(mask)
            cf = ax.contourf(fld.lon, fld.lat, fld, levels=beta_levels, cmap=cmap,
                             extend="both", transform=data_crs)
            overlay_contours(ax, beta_ii_cols[j], beta_clev)
            ax.coastlines(linewidth=0.6)
            fig.colorbar(cf, cax=cax, orientation="vertical").ax.tick_params(labelsize=8)

        elif r == 2:
            mask = beta_sum_sig_cols[j] >= thresh
            fld = beta_sum_cols[j].where(mask)
            cf = ax.contourf(fld.lon, fld.lat, fld, levels=bsum_levels, cmap=cmap,
                             extend="both", transform=data_crs)
            overlay_contours(ax, beta_sum_cols[j], bsum_clev)
            ax.coastlines(linewidth=0.6)
            fig.colorbar(cf, cax=cax, orientation="vertical").ax.tick_params(labelsize=8)

        elif r == 3:
            fld = lin_resp[j].where(lin_mask[j])
            cf = ax.contourf(fld.lon, fld.lat, fld, levels=lin_levels, cmap=cmap,
                             extend="both", transform=data_crs)
            overlay_contours(ax, lin_resp[j], lin_clev)
            ax.coastlines(linewidth=0.6)
            ax.text(0.01, 0.01, f"n+={n_pos[j]}, n-={n_neg[j]}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
            fig.colorbar(cf, cax=cax, orientation="vertical").ax.tick_params(labelsize=8)

        elif r == 4:
            fld = nonlin_resp[j].where(nonlin_mask[j])
            cf = ax.contourf(fld.lon, fld.lat, fld, levels=non_levels, cmap=cmap,
                             extend="both", transform=data_crs)
            overlay_contours(ax, nonlin_resp[j], non_clev)
            ax.coastlines(linewidth=0.6)
            ax.text(0.01, 0.01, f"n+={n_pos[j]}, n-={n_neg[j]}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
            fig.colorbar(cf, cax=cax, orientation="vertical").ax.tick_params(labelsize=8)

        elif r == 5:
            fld = absdiff_resp[j].where(absdiff_mask[j])
            cf = ax.contourf(fld.lon, fld.lat, fld, levels=abs_levels, cmap=cmap,
                             extend="both", transform=data_crs)
            overlay_contours(ax, absdiff_resp[j], abs_clev)
            ax.coastlines(linewidth=0.6)
            ax.text(0.01, 0.01, f"n_hi={n_hi[j]}, n_lo={n_lo[j]}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
            fig.colorbar(cf, cax=cax, orientation="vertical").ax.tick_params(labelsize=8)

        else:  # r == 6
            fld = curv_resp[j].where(curv_mask[j])
            cf = ax.contourf(fld.lon, fld.lat, fld, levels=curv_levels, cmap=cmap,
                             extend="both", transform=data_crs)
            overlay_contours(ax, curv_resp[j], curv_clev)
            ax.coastlines(linewidth=0.6)
            cc = curv_counts[j]
            ax.text(0.01, 0.01, f"low={cc['n_low']}, mid={cc['n_mid']}, hi={cc['n_hi']}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
            fig.colorbar(cf, cax=cax, orientation="vertical").ax.tick_params(labelsize=8)

plt.show()
