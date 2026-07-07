#!/usr/bin/env python3
"""
qb_ia_oni_seasonal_and_lag_figure.py

Makes a compact 2-panel figure:
  (A) Polar "seasonal amplification" (dimensionless) for QB, IA (with seed-based 90% CI), and ONI.
      - Seasonal amplification = centered RMS-by-month, then normalized across months:
          amp(m) = (RMS_m - mean_m RMS_m) / std_m RMS_m
      - Months are arranged Dec at top, clockwise.

  (B) Lead–lag correlation vs ONI for QB and IA (seed-based 90% CI shading)
      - corr(mode(t), ONI(t+lag)) where lag is in months.
      - Negative lag means ONI leads the mode; positive lag means mode leads ONI.

Inputs:
  latents: xr.DataArray with dims ('seed','time','mode') or ('seed','time','mode', ...)
           (we use .sel(mode=...) and then squeeze any leftover dims if needed)
  oni:     xr.DataArray with dim ('time',) (monthly)

Use with preloaded latents and ONI time series.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from scipy.stats import pearsonr 
import xarray as xr 
from pandas.tseries.offsets import DateOffset
import numpy as np 
import pandas as pd 

def month_standardize(da):
    """
    Remove monthly mean and divide by monthly std.
    Returns standardized DataArray with same dims.
    """
    clim_mean = da.groupby("time.month").mean("time")
    clim_std  = da.groupby("time.month").std("time")

    # avoid divide-by-zero
    clim_std = xr.where(clim_std == 0, np.nan, clim_std)

    da_std = (da.groupby("time.month") - clim_mean) / clim_std
    return da_std


def crosscorrelation_by_month(x, y, nlags=36, n_months_per_lag=1):
    # we assume x and y initially have exactly the same time coordinate and it is monthly 
    # we pad y with nlags months of np.nan on either side 
    y_early_pad = xr.ones_like(y.isel(time=slice(None, nlags))).assign_coords({'time': [ pd.Timestamp(i) - DateOffset(months=nlags) for i in y.isel(time=slice(None, nlags)).time.values]}) * np.nan
    y_late_pad = xr.ones_like(y.isel(time=slice(-nlags, None))).assign_coords({'time': [ pd.Timestamp(i) + DateOffset(months=nlags) for i in y.isel(time=slice(-nlags, None)).time.values]}) * np.nan
    y = xr.concat([y_early_pad, y, y_late_pad], 'time')


    corrs, sigs = [], []
    for month in range(1,13): 
        corrs.append(np.full(nlags*2+1, np.nan))
        sigs.append(np.full(nlags*2+1, np.nan))

        x1 = x.isel(time=(x.time.dt.month == month))
        x_data = x1.values 
        xtime = x1.time
        for lag in np.arange(-nlags, nlags+1):
            # negative lag (left side of plot) means that y precedes x 
            y_data = y.sel(time=[pd.Timestamp(i) + DateOffset(months=lag*n_months_per_lag) for i in xtime.values])
            y_data = y_data.values 
            pearson = pearsonr(y_data[~np.isnan(y_data)].squeeze(), x_data[~np.isnan(y_data)].squeeze())
            corrs[-1][lag+nlags] = pearson.statistic
            sigs[-1][lag+nlags] = pearson.pvalue 
    corrs= np.array(corrs)
    sigs = np.array(sigs)

    return corrs, sigs 
# ----------------------------
# Configurable settings
# ----------------------------
MODE_QB = 3
MODE_IA = 4
N_LAGS = 36  # +/- months

CI_Q = (0.25, 0.75)  # 90% CI across seeds
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ORDER_DEC_FIRST = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # for plotting


# ----------------------------
# Helpers
# ----------------------------
def _as_1d_time(da: xr.DataArray) -> xr.DataArray:
    """Force to (time,) by squeezing non-time dims if they are length-1."""
    if "time" not in da.dims:
        raise ValueError(f"Expected 'time' dim. Got dims={da.dims}")
    other = [d for d in da.dims if d != "time"]
    if other:
        da = da.squeeze(other, drop=True)
    return da

def _align_monthly(a: xr.DataArray, b: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Inner-join alignment on time."""
    a2, b2 = xr.align(a, b, join="inner")
    return a2, b2

def centered_rms_seasonal_amp(x: xr.DataArray) -> xr.DataArray:
    """
    x: DataArray(time,) or (seed,time)
    Returns seasonal amplification by month (same leading dims + month):
      RMS_m = sqrt(mean(x^2 | month=m))
      amp_m = (RMS_m - mean(RMS_m over months)) / std(RMS_m over months)
    Dimensionless; focuses purely on shape of seasonal cycle.
    """
    # RMS by month
    rms = np.sqrt((x**2).groupby("time.month").mean("time", skipna=True))
    # Center + scale across months (per seed if seed exists)
    mu = rms.mean("month", skipna=True)
    sd = rms.std("month", skipna=True)
    amp = (rms - mu) / mu
    return amp



def seed_ci(da: xr.DataArray, q=CI_Q) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return (lo, med, hi) across seed for da(seed, ...)."""
    lo = da.quantile(q[0], dim="seed", skipna=True)
    med = da.quantile(0.5, dim="seed", skipna=True)
    hi = da.quantile(q[1], dim="seed", skipna=True)
    return lo, med, hi

def lag_corr_seedwise(mode_seed: xr.DataArray, oni: xr.DataArray, nlags: int) -> xr.DataArray:
    """
    mode_seed: DataArray(seed,time)
    oni:       DataArray(time,)
    Returns:   DataArray(seed, lag) with Pearson r for each seed and lag.
      corr(mode(t), oni(t+lag))
    Convention:
      lag > 0  => oni shifted forward => comparing mode(t) with future oni => mode leads oni by lag
      lag < 0  => oni shifted backward => oni leads mode by |lag|
    """
    mode_seed, oni = _align_monthly(mode_seed, oni)
    lags = np.arange(-nlags, nlags + 1, dtype=int)

    out = np.full((mode_seed.sizes["seed"], lags.size), np.nan, dtype=float)
    # convert to numpy for speed, but keep alignment via shifting in xarray
    for j, lag in enumerate(lags):
        oni_shift = oni.shift(time=-lag)  # so oni_shift(t) = oni(t+lag)
        ms, os = xr.align(mode_seed, oni_shift, join="inner")
        # compute per seed
        for si in range(ms.sizes["seed"]):
            a = ms.isel(seed=si).values
            b = os.values
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 24:  # need at least ~2 years of months
                continue
            out[si, j] = pearsonr(a[ok], b[ok]).statistic
    return xr.DataArray(out, coords={"seed": mode_seed["seed"], "lag": lags}, dims=("seed", "lag"))

def _month_order_reindex(amp_month: xr.DataArray) -> xr.DataArray:
    """Reindex month dimension to Dec..Nov order."""
    return amp_month.sel(month=ORDER_DEC_FIRST)

def _polar_theta_for_12():
    theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    return theta

def _close_loop(theta, y):
    return np.r_[theta, theta[0]], np.r_[y, y[0]]


def make_figure(
    latents: xr.DataArray,
    oni: xr.DataArray,
    mode_qb=MODE_QB,
    mode_ia=MODE_IA,
    nlags=N_LAGS,
    ci_q=CI_Q,
    title_left="Seasonal Amplification",
    title_right="Lead–lag correlation vs ONI",
    title_bottom="Interannual-Quasibiennial Lag correlation",
    sig_level=0.05,
):
    """
    Returns fig, (ax0, ax1, ax2), outdict
      ax0 = polar seasonal amplification
      ax1 = lead–lag vs ONI (lines)
      ax2 = month-by-lag IA–QB correlation (contours)
    """

    # --- Extract QB & IA (seed,time)
    qb = latents.sel(mode=mode_qb).squeeze(drop=True)
    ia = latents.sel(mode=mode_ia).squeeze(drop=True)
    if "seed" not in qb.dims or "seed" not in ia.dims:
        raise ValueError(f"Expected latents to have dim 'seed'. Got qb.dims={qb.dims}")

    # --- ONI to (time,)
    oni = _as_1d_time(oni)
    oni = (oni - oni.mean() ) / oni.std() # we have to normalize for fair comparison to KGAE 

    # --- Align everything on time
    qb0, oni0 = _align_monthly(qb, oni)
    ia0, _    = _align_monthly(ia, oni)

    # ----------------------------
    # (a) Seasonal amplification (per seed)
    # ----------------------------
    qb_amp = centered_rms_seasonal_amp(qb0)     # (seed, month)
    ia_amp = centered_rms_seasonal_amp(ia0)     # (seed, month)
    oni_amp = centered_rms_seasonal_amp(oni0)   # (month,)

    qb_amp = _month_order_reindex(qb_amp)       # Dec..Nov
    ia_amp = _month_order_reindex(ia_amp)
    oni_amp = _month_order_reindex(oni_amp)

    qb_lo, qb_med, qb_hi = seed_ci(qb_amp, q=ci_q)
    ia_lo, ia_med, ia_hi = seed_ci(ia_amp, q=ci_q)

    # ----------------------------
    # (b) Lead–lag correlations vs ONI (per seed)
    # ----------------------------
    qb_r = lag_corr_seedwise(qb0, oni0, nlags=nlags)  # (seed, lag)
    ia_r = lag_corr_seedwise(ia0, oni0, nlags=nlags)

    qb_rlo, qb_rmed, qb_rhi = seed_ci(qb_r, q=ci_q)
    ia_rlo, ia_rmed, ia_rhi = seed_ci(ia_r, q=ci_q)

    # ----------------------------
    # (c) Month-by-lag IA–QB correlation (use ensemble-mean, standardized-by-month)
    # ----------------------------
    ia_mean = ia0.mean("seed", skipna=True)
    qb_mean = qb0.mean("seed", skipna=True)

    ia_std = _as_1d_time(ia_mean)
    qb_std = _as_1d_time(qb_mean)

    corrs, sigs = crosscorrelation_by_month(ia_std, qb_std, nlags=nlags)  # (12, 2*nlags+1)
    corrs_sig = corrs.copy()
    corrs_sig[sigs > sig_level] = np.nan

    corrs2, sigs2 = crosscorrelation_by_month(qb_std, qb_std, nlags=nlags)  # (12, 2*nlags+1)
    corrs_sig2 = corrs2.copy()
    corrs_sig2[sigs2 > sig_level] = np.nan

    corrs3, sigs3 = crosscorrelation_by_month(ia_std, ia_std, nlags=nlags)  # (12, 2*nlags+1)
    corrs_sig3 = corrs3.copy()
    corrs_sig3[sigs3 > sig_level] = np.nan

    # contour levels
    levels = np.linspace(-1, 1, 21)
    levels2 = np.linspace(-1, 1, 11)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ----------------------------
    # Plotting layout: 4 rows
    #   row0: top (a,b)
    #   row1: legend spacer (dedicated)
    #   row2: bottom (c) contour
    #   row3: colorbar spacer (dedicated)
    # ----------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(10.8, 12.6))
    gs = fig.add_gridspec(
        nrows=6, ncols=2,
        height_ratios=[2, 0.20, 1.05, 1.05, 1.05, 0.07],
        width_ratios=[1.05, 1.15],
        hspace=0.4,
        wspace=0.28,
    )

    ax0 = fig.add_subplot(gs[0, 0], projection="polar")
    ax1 = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")
    ax2 = fig.add_subplot(gs[2, :])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4, :])
    cax = fig.add_subplot(gs[5, :])

    # ---- Panel labels
    ax0.text(0.0, 1.08, "a)", transform=ax0.transAxes, ha="left", va="bottom",
             fontweight="bold", clip_on=False)
    ax1.text(0.0, 1.03, "b)", transform=ax1.transAxes, ha="left", va="bottom",
             fontweight="bold", clip_on=False)
    ax2.text(0.0, 1.03, "c)", transform=ax2.transAxes, ha="left", va="bottom",
             fontweight="bold", clip_on=False)
    ax3.text(0.0, 1.03, "d)", transform=ax3.transAxes, ha="left", va="bottom",
            fontweight="bold", clip_on=False)
    ax4.text(0.0, 1.03, "e)", transform=ax4.transAxes, ha="left", va="bottom",
             fontweight="bold", clip_on=False)

    # ----------------------------
    # (a) Polar seasonal amplification
    # ----------------------------
    theta = _polar_theta_for_12()
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.set_xticks(theta)
    ax0.set_xticklabels(["Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov"])
    ax0.grid(True, alpha=0.25)
    ax0.axhline(0, color="0", lw=1.5)
    # Only show radial circles at 0 and 2
    ax0.set_yticks([-0.1, 0, 0.1])
    ax0.set_yticklabels(["", "", ""])
    ax0.set_rlabel_position(90)  # upper-left quadrant often looks best


    # Make them subtle
    ax0.tick_params(axis="y", labelsize=9)

    theta_c, y_qb = _close_loop(theta, qb_med.values)
    _,      y_ia = _close_loop(theta, ia_med.values)
    _,      y_on = _close_loop(theta, oni_amp.values)

    # (no labels here; legend will be custom + shared)
    ax0.plot(theta_c, y_qb, color="red", lw=2.2, ls="--")
    ax0.fill_between(
        theta_c,
        _close_loop(theta, qb_lo.values)[1],
        _close_loop(theta, qb_hi.values)[1],
        color="red", alpha=0.12, linewidth=0
    )

    ax0.plot(theta_c, y_ia, color="blue", lw=2.2, ls="-")
    ax0.fill_between(
        theta_c,
        _close_loop(theta, ia_lo.values)[1],
        _close_loop(theta, ia_hi.values)[1],
        color="blue", alpha=0.10, linewidth=0
    )

    ax0.plot(theta_c, y_on, color="0.35", lw=2.0, ls=":")
    ax0.set_title(title_left, pad=16)

    # ----------------------------
    # (b) Lead–lag vs ONI (lines)
    # ----------------------------
    lags = qb_r["lag"].values
    ax1.axhline(0, color="0.35", lw=0.8, zorder=0)
    ax1.axvline(0, color="0.5", lw=0.8, ls=":", zorder=0)

    ax1.plot(lags, qb_rmed.values, color="red", lw=2.0, ls="--")
    ax1.fill_between(lags, qb_rlo.values, qb_rhi.values, color="red", alpha=0.12, linewidth=0)

    ax1.plot(lags, ia_rmed.values, color="blue", lw=2.0, ls="-")
    ax1.fill_between(lags, ia_rlo.values, ia_rhi.values, color="blue", alpha=0.10, linewidth=0)

    ax1.set_xlim(-nlags, nlags)
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xticks(np.arange(-nlags, nlags + 1, 12))
    ax1.set_xlabel("Lag (months)  [positive = mode leads ONI]")
    ax1.set_ylabel("Pearson Correlation")
    ax1.set_title(title_right)
    ax1.grid(True, alpha=0.25)

    # ----------------------------
    # (c) Month-by-lag IA–QB correlation (contours)
    # ----------------------------
    X = np.arange(-nlags, nlags + 1)
    Y = np.arange(12)

    cp = ax2.contourf(X, Y, corrs_sig, levels=levels, cmap="RdBu_r", extend="both")
    cs = ax2.contour(X, Y, corrs, levels=levels2, colors="k", linewidths=0.8,
                     negative_linestyles="dashed")
    ax2.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax2.set_yticks([0, 3, 6, 9], labels=[months[j] for j in [0, 3, 6, 9]])
    ax2.set_xticks(np.arange(-nlags, nlags + 1, 12))
    ax2.set_xlabel("")
    ax2.set_ylabel("Month (IA)")
    ax2.set_title(title_bottom)

    ax2.text(-nlags+2, 12.5, "QB precedes IA", ha="left", va="top", fontsize=10)
    ax2.text( nlags-2, 12.5, "QB follows IA", ha="right", va="top", fontsize=10)


    ### d) qb lag-correlation
    cp = ax3.contourf(X, Y, corrs_sig2, levels=levels, cmap="RdBu_r", extend="both")
    cs = ax3.contour(X, Y, corrs2, levels=levels2, colors="k", linewidths=0.8,
                     negative_linestyles="dashed")
    ax3.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax3.set_yticks([0, 3, 6, 9], labels=[months[j] for j in [0, 3, 6, 9]])
    ax3.set_xticks(np.arange(-nlags, nlags + 1, 12))
    ax3.set_xlabel("")
    ax3.set_ylabel("Month (QB)")
    ax3.set_title("Quasibiennial Lag Autocorrelation")


    ### d) qb lag-correlation
    cp = ax4.contourf(X, Y, corrs_sig3, levels=levels, cmap="RdBu_r", extend="both")
    cs = ax4.contour(X, Y, corrs3, levels=levels2, colors="k", linewidths=0.8,
                     negative_linestyles="dashed")
    ax4.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax4.set_yticks([0, 3, 6, 9], labels=[months[j] for j in [0, 3, 6, 9]])
    ax4.set_xticks(np.arange(-nlags, nlags + 1, 12))
    ax4.set_xlabel("")
    ax4.set_ylabel("Month (IA)")
    ax4.set_title("Interannual Lag Autocorrelation")

    # --------------------------
    # Dedicated horizontal colorbar row (no overlap)
    # ----------------------------
    cax.cla()
    cbar = fig.colorbar(cp, cax=cax, orientation="horizontal")
    cbar.set_label("Pearson Correlation")
    cbar.set_ticks(np.linspace(-1, 1, 6))

    # ----------------------------
    # One shared legend BETWEEN rows (custom handles; no duplicates)
    # ----------------------------
    handles = [
        Line2D([0],[0], color="red", lw=2.2, ls="--", label="Quasibiennial Mode"),
        Line2D([0],[0], color="blue",   lw=2.2, ls="-",  label="Interannual Mode"),
        Line2D([0],[0], color="0.35",   lw=2.0, ls=":",  label="ONI"),
        Patch(facecolor="k", alpha=0.12, edgecolor="none", label=f"IQR (across seeds)"),
    ]

    ax_leg.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.6,
    )

   # fig.subplots_adjust(bottom=0.12)

    out = {
        "qb_amp": qb_amp, "ia_amp": ia_amp, "oni_amp": oni_amp,
        "qb_r": qb_r, "ia_r": ia_r,
        "corrs_by_month": corrs,
        "pvals_by_month": sigs,
    }

    return fig, (ax0, ax1, ax2), out


# ----------------------------
# Script entry point
# ----------------------------
if __name__ == "__main__":
    import kgae
    from matplotlib.lines import Line2D

    n_ensemble = 100
    experiment = "../final_scripts/large-ensemble-2-1940-2014"

    # latents: (seed,time,mode)
    latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)
    #latents = (latents.groupby('time.month') - latents.groupby('time.month').mean('time'))
    # oni: (time,) computed from references for this analysis.
    refs = kgae.open_references("xval", experiment=experiment)
    oni = refs.sel(lat=slice(-5, 5), lon=slice(-10, 60)).mean(("lat", "lon"))
    oni = _as_1d_time(oni)


    fig, axes, out = make_figure(latents, oni, mode_qb=3, mode_ia=4, nlags=36)
    fig.savefig("figures/figure6_qb-ia.pdf", bbox_inches="tight")
    plt.show()
