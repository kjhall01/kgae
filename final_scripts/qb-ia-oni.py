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


if __name__ == "__main__":
    import kgae 
    import numpy as np 
    import matplotlib.pyplot as plt 

    n_ensemble = 100 
    experiment = 'large-ensemble-2-1940-2014'

    latents = kgae.open_latents('xval', experiment=experiment, n_ensemble=n_ensemble)
    references = kgae.open_references('xval', experiment=experiment)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,5))


    nlags=36
    levels = np.linspace(-1, 1, 21)
    levels2 = np.linspace(-1, 1, 11)
    months = ['Jan', 'Feb', 'Mar', "Apr", 'May', "Jun", 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    ia = latents.sel(mode=4).mean('seed')
    qb = latents.sel(mode=3).mean('seed')
    oni = references.sel(lat=slice(-5,5), lon=slice(-10, 60)).mean(['lat', 'lon'])

   # ax[0].axhline(0, linewidth=0.5, linestyle=':', color='k')

   # ax[0].plot(qb.time, qb, label='QB', linestyle='--', color='k')
   # ax[0].plot(ia.time, ia, label='IA', linestyle='-', color='k')
   # ax[0].plot(oni.time, oni, label='ONI', linestyle=':', alpha=0.8, color='k')
   # ax[0].legend(loc="upper right")
    qbc = np.sqrt(qb**2).groupby('time.month').mean()#.sel(month=[6,7,8,9,10,11,12,1,2,3,4,5])
    iac = np.sqrt(ia**2).groupby('time.month').mean()#.sel(month=[6,7,8,9,10,11,12,1,2,3,4,5])
    onic = np.sqrt(oni**2).groupby('time.month').mean()#.sel(month=[6,7,8,9,10,11,12,1,2,3,4,5])

    # --- Month labels centered on December ---
    months_centered = ["Jul","Aug","Sep","Oct","Nov", "Dec","Jan","Feb","Mar","Apr","May","Jun",]
    month_order = np.array([ 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6,])
    xpos = np.arange(12)

    # --- Helper: per-seed monthly RMS amplitude ---
    def monthly_rms_by_seed(z_seed_time: xr.DataArray) -> xr.DataArray:
        """
        z_seed_time: (seed, time)
        returns: (seed, month) RMS amplitude
        """
        # RMS amplitude = sqrt(mean(z^2)) for each month-of-year, per seed
        rms = np.sqrt((z_seed_time ** 2).groupby("time.month").mean("time", skipna=True))
        return rms

    # --- Build per-seed climatologies for QB & IA ---
    qb_seed = latents.sel(mode=3)  # (seed,time)
    ia_seed = latents.sel(mode=4)  # (seed,time)

    qbc_seed = monthly_rms_by_seed(qb_seed)  # (seed,month)
    iac_seed = monthly_rms_by_seed(ia_seed)  # (seed,month)

    # --- Ensemble summaries (central line + 90% CI) ---
    qbc_med = qbc_seed.quantile(0.50, dim="seed", skipna=True)
    qbc_lo  = qbc_seed.quantile(0.05, dim="seed", skipna=True)
    qbc_hi  = qbc_seed.quantile(0.95, dim="seed", skipna=True)

    iac_med = iac_seed.quantile(0.50, dim="seed", skipna=True)
    iac_lo  = iac_seed.quantile(0.05, dim="seed", skipna=True)
    iac_hi  = iac_seed.quantile(0.95, dim="seed", skipna=True)

    # --- ONI monthly RMS (no seed; just time) ---
    onic = np.sqrt((oni ** 2).groupby("time.month").mean("time", skipna=True))

    # --- Reorder to start at December ---
    def reorder_months(da_month: xr.DataArray) -> xr.DataArray:
        return da_month.sel(month=month_order)

    qbc_med_o = reorder_months(qbc_med)
    qbc_lo_o  = reorder_months(qbc_lo)
    qbc_hi_o  = reorder_months(qbc_hi)

    iac_med_o = reorder_months(iac_med)
    iac_lo_o  = reorder_months(iac_lo)
    iac_hi_o  = reorder_months(iac_hi)

    onic_o    = reorder_months(onic)

    # --- Plot (slight x offsets so errorbars don’t collide) ---
    dx = 0.10
    ax0 = ax[0]

    ax0.grid(True, axis="y", linewidth=0.6, alpha=0.35)
    ax0.set_axisbelow(True)

    # QB with 90% CI
    ax0.errorbar(
        xpos - dx,
        qbc_med_o.values,
        yerr=np.vstack([(qbc_med_o - qbc_lo_o).values, (qbc_hi_o - qbc_med_o).values]),
        fmt="o", ms=4,
        linestyle="--",
        linewidth=1.6,
        capsize=2.5,
        color="k",
        label="QB (mode 3) 90% CI (seeds)",
    )

    # IA with 90% CI
    ax0.errorbar(
        xpos + dx,
        iac_med_o.values,
        yerr=np.vstack([(iac_med_o - iac_lo_o).values, (iac_hi_o - iac_med_o).values]),
        fmt="s", ms=4,
        linestyle="-",
        linewidth=1.8,
        capsize=2.5,
        color="k",
        label="IA (mode 4) 90% CI (seeds)",
    )

    # ONI (no seed CI)
    ax0.plot(
        xpos,
        onic_o.values,
        linestyle=":",
        linewidth=2.0,
        alpha=0.85,
        color="k",
        label="ONI RMS",
    )

    ax0.set_xticks(xpos)
    ax0.set_xticklabels(months_centered)
    ax0.set_xlim(-0.6, 11.6)
    ax0.set_ylabel("RMS amplitude")
    ax0.set_title("Climatological RMS amplitude (centered on Dec)")
    #ax0.legend(loc="upper right", frameon=False, ncol=1)


    ia_std = month_standardize(ia)
    qb_std = month_standardize(qb)

    corrs, sigs = kgae.crosscorrelation_by_month(ia_std, qb_std, nlags=nlags)

    corrsnan = corrs.copy()
    corrsnan[sigs > 0.05] = np.nan
    cp = ax[1].contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
    cp2 = ax[1].contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
    ax[1].clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

    ax[1].set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
    ax[1].set_xticks(np.arange(-nlags, nlags+1, 12))

    ax[1].set_title('Interannual - Quasibiennial')
    ax[1].set_title('a)', loc='left', fontweight='bold')

    # Add bottom left and right text labels
    ax[1].text(-nlags, -3.5, "Quasibiennial precedes Interannual", ha='left', va='top', fontsize=10)
    ax[1].text(nlags, -3.5, "Quasibiennial follows Interannual", ha='right', va='top', fontsize=10)




    cbar_ax = fig.add_axes([0.15, 0.15, .70, 0.02]) 
    fig.colorbar(cp, cax=cbar_ax, **{'label': 'Pearson Correlation', 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5, 'ticks':np.linspace(-1,1, 6), 'extend': 'both'})
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    #plt.savefig("/Users/kylehall/Desktop/hall_molina_2025_final_figures/hall_molina_2025.figure3.png", dpi=1000)
    plt.show()