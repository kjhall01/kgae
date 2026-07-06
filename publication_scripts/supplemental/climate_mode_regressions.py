#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import pandas as pd

# -----------------------------
# Region masks / weighting
# -----------------------------

def _coslat_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()

def region_mask(lat, lon, region: str):
    """
    Returns a boolean mask (lat, lon) for:
      - "full": all points
      - "tropics": |lat| <= 23
      - "north_pacific": lat >= 21
    """
    lat2d, lon2d = xr.broadcast(lat, lon)
    if region == "full":
        return xr.ones_like(lat2d, dtype=bool)
    if region == "tropics":
        return np.abs(lat2d) <= 23
    if region == "north_pacific":
        return lat2d >= 21
    raise ValueError(f"Unknown region: {region}")

def _apply_mask_and_stack(a, mask):
    """
    a: (lat, lon) DataArray
    mask: (lat, lon) boolean DataArray
    returns: stacked 1D arrays of values and weights, with NaNs removed
    """
    aa = a.where(mask)
    vec = aa.stack(point=("lat","lon"))
    ok = np.isfinite(vec)
    vec = vec.where(ok, drop=True)
    return vec

def weighted_pattern_corr(a, b, region="full"):
    """
    Weighted spatial correlation over lat/lon in a region.
    a, b: DataArray(lat, lon)
    """
    a, b = xr.align(a, b, join="inner")
    m = region_mask(a["lat"], a["lon"], region)

    # weights: cos(lat) broadcast to points that survive the mask+nan drop
    wlat = _coslat_weights(a["lat"])
    w2d = xr.broadcast(wlat, a["lon"])[0]  # (lat,lon)

    av = _apply_mask_and_stack(a, m)
    bv = _apply_mask_and_stack(b, m)
    # align points
    av, bv = xr.align(av, bv, join="inner")

    ww = w2d.where(m).stack(point=("lat","lon")).sel(point=av["point"])
    ww = ww.where(np.isfinite(av) & np.isfinite(bv), drop=True)

    av = av.sel(point=ww["point"])
    bv = bv.sel(point=ww["point"])

    if av.size < 10:
        return np.nan

    w = ww.values.astype(float)
    x = av.values.astype(float)
    y = bv.values.astype(float)

    # weighted demean
    wsum = np.sum(w)
    mx = np.sum(w * x) / wsum
    my = np.sum(w * y) / wsum
    x0 = x - mx
    y0 = y - my

    num = np.sum(w * x0 * y0)
    den = np.sqrt(np.sum(w * x0**2) * np.sum(w * y0**2))
    if den == 0:
        return np.nan
    return float(num / den)

def univariate_regression_r2(y, x, region="full"):
    """
    Univariate regression R^2 for y ~ a + b x across gridpoints in a region.
    y, x: DataArray(lat, lon)
    """
    y, x = xr.align(y, x, join="inner")
    m = region_mask(y["lat"], y["lon"], region)

    yv = _apply_mask_and_stack(y, m)
    xv = _apply_mask_and_stack(x, m)
    yv, xv = xr.align(yv, xv, join="inner")

    if yv.size < 10:
        return np.nan

    yy = yv.values.astype(float)
    xx = xv.values.astype(float)

    ok = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[ok]; yy = yy[ok]
    if xx.size < 10:
        return np.nan

    # add intercept
    X = np.vstack([np.ones_like(xx), xx]).T
    # least squares
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((yy - yhat)**2)
    ss_tot = np.sum((yy - yy.mean())**2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)

# -----------------------------
# Temporal correlations (for alpha rows)
# -----------------------------

def temporal_corr(a, b):
    a, b = xr.align(a, b, join="inner")
    aa = a.values.astype(float)
    bb = b.values.astype(float)
    ok = np.isfinite(aa) & np.isfinite(bb)
    if ok.sum() < 10:
        return np.nan
    aa = aa[ok]; bb = bb[ok]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0,1])

# -----------------------------
# Main comparison
# -----------------------------

def compare_kgae_to_climate_modes(
    kgae_pat: xr.DataArray,   # (seed, kgae, lat, lon)
    clim_pat: xr.DataArray,   # (mode, lat, lon)
    clim_ts: xr.DataArray=None,  # (mode, time)
    kgae_z: xr.DataArray=None,   # (seed, mode, time)
    kgae_alpha_to_mode: dict=None,
    regions=("full","tropics","north_pacific"),
    seed_ci=(5,95),
):
    """
    Primary result: correlation/R^2 between the MEAN KGAE pattern (mean over seeds)
    and the climate-mode pattern.

    Spread: 5--95% (or seed_ci) interval across per-seed correlations/R^2.

    Output columns:
      - mean: primary (mean-pattern) score
      - median/low/high: spread of per-seed scores (same as before)
    """

    if set(kgae_pat.dims) != {"seed","kgae","lat","lon"}:
        raise ValueError(f"kgae_pat dims must be ('seed','kgae','lat','lon'), got {kgae_pat.dims}")
    if set(clim_pat.dims) != {"mode","lat","lon"}:
        raise ValueError(f"clim_pat dims must be ('mode','lat','lon'), got {clim_pat.dims}")

    # align grids
    kgae_pat, clim_pat = xr.align(kgae_pat, clim_pat, join="inner")

    seeds = kgae_pat["seed"].values
    kgae_names = kgae_pat["kgae"].values.astype(str)
    idx_names  = clim_pat["mode"].values.astype(str)

    def summarize_seed_spread(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (np.nan, np.nan, np.nan)
        med = np.nanmedian(x)
        lo  = np.nanpercentile(x, seed_ci[0])
        hi  = np.nanpercentile(x, seed_ci[1])
        return med, lo, hi

    rows = []

    for kname in kgae_names:
        for iname in idx_names:
            cp = clim_pat.sel(mode=iname)

            for region in regions:
                # ----------------------------
                # Primary: mean-pattern score
                # ----------------------------
                kp_mean = kgae_pat.sel(kgae=kname).mean("seed")  # (lat,lon)

                c_mean = weighted_pattern_corr(kp_mean, cp, region=region)
                r_mean = univariate_regression_r2(kp_mean, cp, region=region)

                # ----------------------------
                # Spread: per-seed scores
                # ----------------------------
                corr_seed = []
                r2_seed   = []
                for s in seeds:
                    kp = kgae_pat.sel(seed=s, kgae=kname)
                    corr_seed.append(weighted_pattern_corr(kp, cp, region=region))
                    r2_seed.append(univariate_regression_r2(kp, cp, region=region))

                c_med, c_lo, c_hi = summarize_seed_spread(corr_seed)
                r_med, r_lo, r_hi = summarize_seed_spread(r2_seed)

                rows.append(dict(
                    kgae=kname, mode=iname, region=region, metric="pat_corr",
                    mean=c_mean, median=c_med, low=c_lo, high=c_hi
                ))
                rows.append(dict(
                    kgae=kname, mode=iname, region=region, metric="r2",
                    mean=r_mean, median=r_med, low=r_lo, high=r_hi
                ))

    df = pd.DataFrame(rows)

    # --------------------------------------------
    # Optional: temporal correlation for alpha rows
    # Primary: corr(mean_z, ts), spread: seed CI
    # --------------------------------------------
    if clim_ts is not None and kgae_z is not None and kgae_alpha_to_mode is not None:
        if set(clim_ts.dims) != {"mode","time"}:
            raise ValueError(f"clim_ts dims must be ('mode','time'), got {clim_ts.dims}")
        if set(kgae_z.dims) != {"seed","mode","time"}:
            raise ValueError(f"kgae_z dims must be ('seed','mode','time'), got {kgae_z.dims}")

        trows = []
        for kname, z_mode in kgae_alpha_to_mode.items():
            if kname not in kgae_names:
                continue

            for iname in idx_names:
                ts = clim_ts.sel(mode=iname)

                # primary: ensemble-mean latent
                z_mean = kgae_z.sel(mode=z_mode).mean("seed")
                t_mean = temporal_corr(z_mean, ts)

                # spread: per-seed correlations
                tc_seed = []
                for s in seeds:
                    z = kgae_z.sel(seed=s, mode=z_mode)
                    tc_seed.append(temporal_corr(z, ts))

                t_med, t_lo, t_hi = summarize_seed_spread(tc_seed)

                trows.append(dict(
                    kgae=kname, mode=iname, region="(time)", metric="temporal_corr",
                    mean=t_mean, median=t_med, low=t_lo, high=t_hi
                ))

        df = pd.concat([df, pd.DataFrame(trows)], ignore_index=True)

    return df

# -----------------------------
# Formatting helpers
# -----------------------------

def format_cell(mean, lo, hi, fmt="{:+.2f}"):
    if not np.isfinite(mean):
        return ""
    if lo < 0 and hi < 0 or lo > 0 and hi > 0:
        pref = "\\textbf{"
        pst = "}"
    else:
        pref = ""
        pst = ""
    return pref + f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]" + pst

def make_table(df: pd.DataFrame, metric="pat_corr", region="full",
               fmt="{:+.2f}", use_abs=True):
    """
    Returns pivot table with:
      - Primary value = mean
      - CI = [low, high]
      - Bold the maximum SIGNIFICANT mean per row

    use_abs=True → choose maximum by absolute mean
    """

    sub = df[(df["metric"] == metric) & (df["region"] == region)].copy()

    # Determine significance (CI does not cross zero)
    sub["significant"] = (
        ((sub["low"] > 0) & (sub["high"] > 0)) |
        ((sub["low"] < 0) & (sub["high"] < 0))
    )

    # We'll build cells row by row
    rows = []

    for kname, g in sub.groupby("kgae"):

        g = g.copy()

        # Among significant entries only
        g_sig = g[g["significant"] & np.isfinite(g["mean"])]

        if len(g_sig) > 0:
            if use_abs:
                idx_max = g_sig["mean"].abs().idxmax()
            else:
                idx_max = g_sig["mean"].idxmax()
        else:
            idx_max = None

        # Format each cell
        for idx, row in g.iterrows():
            mean, lo, hi = row["mean"], row["low"], row["high"]

            if not np.isfinite(mean):
                cell = ""
            else:
                cell_text = f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"

                if idx == idx_max:
                    cell = r"\textbf{" + cell_text + "}"
                else:
                    cell = cell_text

            rows.append(dict(
                kgae=kname,
                mode=row["mode"],
                cell=cell
            ))

    out = pd.DataFrame(rows)
    tab = out.pivot(index="kgae", columns="mode", values="cell")
    return tab



def df_to_latex_panel(df, title, index_name="", escape=False):
    # df is a pandas DataFrame already formatted as strings in each cell
    tex = df.to_latex(
        index=True,
        escape=escape,
        column_format="l" + "c"*df.shape[1],
    )
    # Remove the outer table environment so we can stack panels neatly
    tex = tex.replace(r"\begin{table}", "").replace(r"\end{table}", "")
    tex = tex.replace(r"\centering", "")
    # Add a panel title line above the tabular
    panel = "\n".join([
        rf"\textbf{{{title}}}\\",
        tex.strip(),
        r"\vspace{0.6em}",
    ])
    return panel

if __name__ == "__main__":
    import kgae 
    import xarray as xr 
    from pathlib import Path 
    
    n_ensemble = 100
    experiment = "large-ensemble-2-1940-2014"
    alpha_modes = [5,4,3]
    beta_ndxs = [(5,5), (5,4), (4,4), (4,5)]

    # Example loading (replace with your paths / variable names):
    clim_pat = xr.open_dataset("climate_mode_patterns.nc")["pattern"]   # (mode,lat,lon)
    clim_ts  = xr.open_dataset("climate_mode_indices.nc")["magnitude"]   # (mode,time)
    
    #latents = kgae.open_latents('xval', experiment=experiment, n_ensemble=n_ensemble)  # (seed, mode, time)
    #alphas, betas = kgae.open_alphabetas('xval', experiment=experiment, n_ensemble=n_ensemble)

#    alphas = [ alphas.sel(tendency_mode=at, drop=True).mean('fold') for at in alpha_modes ]
#    betas = [betas.sel(tendency_mode=bt[0], predictor_mode=bt[1], drop=True).mean('fold') for bt in beta_ndxs]

    kgae_patterns = kgae.compute_foldmean_decoder_probes(
        "~/Desktop/kgae/final_scripts/large-ensemble-2-1940-2014/seed*/split*/model.pkl",
        era5_sst_path="~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc",
        training_period=("1940-01-01", "2014-12-31"),
        latent_dim=5,
        mode_dec=5, mode_ia=4, mode_qb=3,
        a_dec=1.0, a_ia=1.0, a_qb=1.0,
        device="cpu",
    ).rename({'quantity': 'kgae'})



    df = compare_kgae_to_climate_modes(
        kgae_patterns, 
        clim_pat, 
       # clim_ts=clim_ts, 
       # kgae_z=latents,
        kgae_alpha_to_mode={
            r"$\alpha_d$": 5, 
            r"$\alpha_i$": 4, 
            r"$\alpha_q$": 3
        }
    )

    tab_full = make_table(df, metric="pat_corr", region="full")
    tab_trop = make_table(df, metric="pat_corr", region="tropics")
    tab_np   = make_table(df, metric="pat_corr", region="north_pacific")


    # Example usage:
    panelA = df_to_latex_panel(tab_full,   "Panel A: Full basin")
    panelB = df_to_latex_panel(tab_trop,   "Panel B: Tropics (|lat|≤23°)")
    panelC = df_to_latex_panel(tab_np,     "Panel C: North Pacific (lat≥21°)")

    latex = "\n".join([
    r"\begin{table}[t]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",        # tighter columns
    r"\renewcommand{\arraystretch}{1.15}", # taller rows
    panelA,
    panelB,
    panelC,
    r"\caption{Spatial correlations between KGAE patterns and known climate-mode regression patterns. "
    r"Entries show median correlation across seeds with 5--95\% interval in brackets.}",
    r"\label{tab:kgae_climate_panels}",
    r"\end{table}",
    ])

    Path("kgae_climate_panels.tex").write_text(latex)


