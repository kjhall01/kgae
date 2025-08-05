import xarray as xr 
import cartopy.crs as ccrs 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import matplotlib.patches as patches
import src 
import numpy as np 
from pathlib import Path 

N=50
run='basin.goodtest'
var = 'sst'
vmax = 1.5
caption = 'SST (K)'
results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

tc3 = []
for init in range(N):
    tc2 = []
    for split in range(5):
        tc = []
        for recursion in range(3):
            encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'val_data.encodings.nc').encodings
            tc.append(encodings)
        tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(3)})
        tc2.append(tc)
    tc2 = xr.concat(tc2, 'time').sortby('time')
    tc3.append(tc2)
tc2 = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).isel(recursion=0)

tc2.to_netcdf('~/Desktop/KGAE-indices.nc')

tc2 = xr.open_dataset('~/Desktop/KGAE-indices.nc').encodings

titles = ['Decadal', "Interannual", "Quasibiennial"]
if True:
    tc2 = xr.concat([src.open_pdo().sel(time=slice("1940-01-01", pd.Timestamp(2014,12,31))).mean('dataset'), src.open_oni().sel(time=slice("1940-01-01", pd.Timestamp(2014,12,31))), src.open_npi().rolling(time=24, center=False).sum().sel(time=slice("1940-01-01", pd.Timestamp(2014,12,31)))], 'mode')
    tc2 = tc2.assign_coords({'mode': ['Decadal', 'Interannual', 'Quasibiennial']}).expand_dims('initialization')

    tc2 = tc2 - tc2.mean('time')
    tc2.mean('initialization').plot.line(x='time', hue='mode')
    plt.show()
    print(tc2)


if var == 'sst':
    sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    sst = sst.rename({'latitude':'lat', 'longitude': 'lon'}) 
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
elif var == 'msl':
    sst = xr.open_dataset('~/Desktop/Data/era5/era5.msl.1x1.1940-2023.nc').msl.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    #sst = sst.rename({'latitude':'lat', 'longitude': 'lon'}) 
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
elif var == 'z500':
    sst = xr.open_dataset('~/Desktop/Data/era5/era5.z500.1x1.1940-2023.nc').z500.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    #sst = sst.rename({'latitude':'lat', 'longitude': 'lon'}) 
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
elif var == 'wind_stress_curl':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.wind_stress_curl.195801-201412.nc').wind_stress_curl
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
    #ssta = ssta.sel(lat=slice(-20, 20))
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
  #  tc2 = tc2.isel(time=[i for i in range(tc2.time.shape[0]) if i % 12 in [11, 0, 1]])
elif var == 'ssh':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.ssh.195801-201412.nc').ssh
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
elif var == 'interior_transport_divergence':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.interior_transport_divergence.195801-201412.nc').interior_transport_divergence
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
   # ssta =ssta.sel(lat=slice(-10, 10))#.isel(time=[i for i in range(sst.time.shape[0]) if i % 12 in [11, 0, 1]])
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
   # tc2 = tc2.isel(time=[i for i in range(tc2.time.shape[0]) if i % 12 in [11, 0, 1]])
elif var == 'upwelling':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.upwelling.195801-201412.nc').upwelling
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
    ssta =ssta.sel(lat=slice(-10, 10))#.isel(time=[i for i in range(sst.time.shape[0]) if i % 12 in [11, 0, 1]])
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
elif var == 'iso20':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.iso20.195801-201412.nc').iso20
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
elif var == 'wind_stress':
    tauy = xr.open_dataset('~/Desktop/Data/oras5/oras5.tauy.195801-201412.nc').tauy 
    tauy, fit, gwm, p  = src.global_detrend(tauy, deg=2)
    tauy , mc = src.remove_climo(tauy)
    taux = xr.open_dataset('~/Desktop/Data/oras5/oras5.taux.195801-201412.nc').taux 
    taux, fit, gwm, p  = src.global_detrend(taux, deg=2)
    taux , mc = src.remove_climo(taux)
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
elif var == 'zonal_wind_stress':
    ssta = xr.open_dataset('~/Desktop/Data/oras5/oras5.taux.195801-201412.nc').taux 
    ssta, fit, gwm, p  = src.global_detrend(ssta, deg=2)
    ssta , mc = src.remove_climo(ssta)
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
elif var == 'u10':
    ssta = xr.open_dataset(f'~/Desktop/Data/era5/era5.{var}.pacific.1x1.1940-2023.nc').u10.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    ssta, fit, gwm, p  = src.global_detrend(ssta, deg=2)
    ssta , mc = src.remove_climo(ssta)
elif var == 'wind': 
    u10 = xr.open_dataset(f'~/Desktop/Data/era5/era5.u10.pacific.1x1.1940-2023.nc').u10.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    u10a, fit, gwm, p  = src.global_detrend(u10, deg=2)
    u10a , mc = src.remove_climo(u10a) 
    v10 = xr.open_dataset(f'~/Desktop/Data/era5/era5.v10.pacific.1x1.1940-2023.nc').v10.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    v10a, fit, gwm, p  = src.global_detrend(v10, deg=2)
    v10a , mc = src.remove_climo(v10a) 
    
    ssta = xr.Dataset({'u': u10a, 'v': v10a})
    ssta = ssta.isel(lat=slice(None, None, 10), lon=slice(None, None, 10)) # subsample to 5 degree resolution


fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(8,9), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
letters0 = ['a)', 'b)', 'c)']
letters1 = ['d)', 'e)', 'f)']
letters2 = ['g)', 'h)', 'i)']
letters3 = ['j)', 'k)', 'l)']
for i, title in enumerate(titles):
    tf = [_ for _ in titles if _ != title]

    # plot positive composite
    pos_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) > tc2.mean('initialization').sel(mode=title).std('time') )  &
        ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
        & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
    n = int(pos_composite_selection.sum() .values)
    print('samples in ', n , np.sum(pos_composite_selection | ~pos_composite_selection))
    pos_composite = ssta.isel(time=pos_composite_selection).mean('time') 

    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
        samp_mean = ssta.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(0.95, 'samps')
    lower = tc1.quantile(0.05, 'samps')
    pos_significance_mask = (pos_composite > upper) | (pos_composite < lower)

    if var != 'wind':
        p = pos_composite.plot(ax=ax[0,i], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
        ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[0,i].coastlines() 
    else:
        pos_composite = pos_composite.where(pos_significance_mask, other=np.nan)
        p = ax[0, i].quiver(
            pos_composite.lon, pos_composite.lat,
            pos_composite.u, pos_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
       # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[0,i].coastlines() 


    # plot negative composite
    neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) < -1*tc2.mean('initialization').sel(mode=title).std('time') )  &
        ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
        & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
    n = int(neg_composite_selection.sum() .values)
    print('samples in ', n , np.sum(neg_composite_selection | ~neg_composite_selection))
    neg_composite = ssta.isel(time=neg_composite_selection).mean('time') 

    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
        samp_mean = ssta.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(0.95, 'samps')
    lower = tc1.quantile(0.05, 'samps')
    neg_significance_mask = (neg_composite > upper) | (neg_composite < lower)

    if var != 'wind':
        p = neg_composite.plot(ax=ax[1,i], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
        ax[1,i].contourf(ssta.lon, ssta.lat, neg_significance_mask, 1, hatches=['', '....'], alpha=0, zorder=2)
        c = ax[1,i].coastlines() 
    else:
        p = ax[1, i].quiver(
            neg_composite.lon, neg_composite.lat,
            neg_composite.u, neg_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
       # ax[1,i].contourf(ssta.lon, ssta.lat, neg_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[1,i].coastlines()
    
    ax[1, i].set_title(None)
    ax[0, i].set_title(None)
    ax[1, i].set_title(letters1[i], fontweight='bold', loc='left')
    ax[0,i].set_title(letters0[i], fontweight='bold', loc='left')
    
    if i ==0:
        decadal_linear_response = (pos_composite - neg_composite)
    
    if var != 'wind':
        p = (pos_composite - neg_composite).plot(ax=ax[2,i], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
        ax[2,i].contourf(ssta.lon, ssta.lat, pos_significance_mask | neg_significance_mask, 1, hatches=['', '....'], alpha=0)
        c = ax[2,i].coastlines()
    else:
        p = ax[2, i].quiver(
            pos_composite.lon, 
            pos_composite.lat,
            pos_composite.u - neg_composite.u, 
            pos_composite.v - neg_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        #ax[2,i].contourf(ssta.lon, ssta.lat, pos_significance_mask | neg_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[2,i].coastlines()


    ax[2, i].set_title(None)
    ax[2, i].set_title(letters2[i], fontweight='bold', loc='left')

    if var != 'wind':
        p = (pos_composite + neg_composite).plot(ax=ax[3,i], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
        c = ax[3,i].coastlines() 
    else:
        p = ax[3, i].quiver(
            pos_composite.lon, 
            pos_composite.lat,
            pos_composite.u + neg_composite.u, 
            pos_composite.v + neg_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        c = ax[3,i].coastlines()

    ax[3, i].set_title(None)
    ax[3, i].set_title(letters3[i], fontweight='bold', loc='left')



ax[0,0].set_title('Decadal Mode')
ax[0,1].set_title('Interannual Mode')
ax[0,2].set_title('Quasibiennial Mode')

ax[0,0].text(-0.03, 0.5, 'Positive Phase', va='center', ha='right', rotation=90, transform=ax[0,0].transAxes)
ax[1,0].text(-0.03, 0.5, 'Negative Phase', va='center', ha='right', rotation=90, transform=ax[1,0].transAxes)
ax[2,0].text(-0.03, 0.5, 'Linear Response', va='center', ha='right', rotation=90, transform=ax[2,0].transAxes)
ax[3,0].text(-0.03, 0.5, 'Nonlinear Response', va='center', ha='right', rotation=90, transform=ax[3,0].transAxes)

cbar_ax = fig.add_axes([0.15, 0.1, .70, 0.02]) 
fig.colorbar(p, cax=cbar_ax, **{'label': caption, 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5, 'extend': 'both'})
plt.tight_layout(pad=0.6, rect=[0, 0.15, 1, 1])
plt.savefig(f'/Users/kylehall/Desktop//hall_molina_2025_final_figures/hall_molina_2025.figure2.{var}.png', dpi=1000)
plt.show()