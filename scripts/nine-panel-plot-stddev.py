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
confidence_level = 0.67
signif_level = 0.05
add_wind = True
wind_vmax= 3


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

titles = ['Decadal', "Interannual", "Quasibiennial"]

if False:
    tc2 = xr.concat([src.open_pdo().sel(time=slice("1940-01-01", pd.Timestamp(2014,12,31))).mean('dataset'), src.open_oni().sel(time=slice("1940-01-01", pd.Timestamp(2014,12,31))), src.open_npi().rolling(time=24, center=False).sum().sel(time=slice("1940-01-01", pd.Timestamp(2014,12,31)))], 'mode')
    tc2 = tc2.assign_coords({'mode': ['Decadal', 'Interannual', z]}).expand_dims('initialization')

    print(tc2)
tc2 = tc2 - tc2.mean('time')
print(tc2.mean('time').values)

if var == 'sst':
    sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    sst = sst.rename({'latitude':'lat', 'longitude': 'lon'}) 
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
elif var == 'msl':
    sst = xr.open_dataset('~/Desktop/Data/era5/era5.msl.1x1.1940-2023.nc').msl.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    #sst = sst.rename({'latitude':'lat', 'longitude': 'lon'}) 
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    #ssta , mc = src.remove_climo(ssta)
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
  #  tc2 = tc2.isel(time=[i for i in range(tc2.time.shape[0]) if i % 12 in [7, 0, 1]])
elif var == 'ssh':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.ssh.195801-201412.nc').ssh
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
elif var == 'interior_transport_divergence':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.interior_transport_divergence.195801-201412.nc').interior_transport_divergence
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
   # ssta =ssta.sel(lat=slice(-10, 10))#.isel(time=[i for i in range(sst.time.shape[0]) if i % 12 in [7, 0, 1]])
    tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))
   # tc2 = tc2.isel(time=[i for i in range(tc2.time.shape[0]) if i % 12 in [7, 0, 1]])
elif var == 'upwelling':
    sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.upwelling.195801-201412.nc').upwelling
    ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
    ssta , mc = src.remove_climo(ssta)
    ssta =ssta.sel(lat=slice(-10, 10))#.isel(time=[i for i in range(sst.time.shape[0]) if i % 12 in [7, 0, 1]])
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

if var == 'wind' or add_wind: 
    u10 = xr.open_dataset(f'~/Desktop/Data/era5/era5.u10.pacific.1x1.1940-2023.nc').u10.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    u10a, fit, gwm, p  = src.global_detrend(u10, deg=2)
    u10a , mc = src.remove_climo(u10a) 
    v10 = xr.open_dataset(f'~/Desktop/Data/era5/era5.v10.pacific.1x1.1940-2023.nc').v10.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    v10a, fit, gwm, p  = src.global_detrend(v10, deg=2)
    v10a , mc = src.remove_climo(v10a) 
    
    wind = xr.Dataset({'u': u10a, 'v': v10a})
    wind = wind.isel(lat=slice(None, None, 5), lon=slice(None, None, 5)) # subsample to 5 degree resolution
    print(wind)
    if var == 'wind':
        ssta = wind
        caption = 'Wind (m/s)'
        vmax = wind_vmax

    




fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(6,6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
letters = [['a)','b)', 'c)'], ['d)', 'e)', 'f)'], ['g)', 'h)', 'i)']]
x = 'Quasibiennial'
y = 'Interannual'
z = 'Decadal'
titles = [ y, z]

neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) < -1*tc2.mean('initialization').sel(mode=y).std('time').values ) & 
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) <  -1*tc2.mean('initialization').sel(mode=x).std('time').values )  )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time')

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')

pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p = neg_neg_composite.plot(ax=ax[2,0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[1,0].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    contours = ax[2,0].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[2, 0].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[2,0].coastlines() 
else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[2, 0].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[1,0].coastlines() 



neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) < -1*tc2.mean('initialization').sel(mode=y).std('time').values ) & 
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) >  tc2.mean('initialization').sel(mode=x).std('time').values )  )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)


if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p = neg_neg_composite.plot(ax=ax[2,2], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[1,1].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    contours = ax[2,2].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')
    c = ax[2,2].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[2, 2].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[2,2].coastlines() 
else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[2, 2].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[2,2].coastlines() 



neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) > tc2.mean('initialization').sel(mode=y).std('time').values ) & 
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) >  tc2.mean('initialization').sel(mode=x).std('time').values )  )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)
if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p = neg_neg_composite.plot(ax=ax[0,2], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[0,1].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    contours = ax[0,2].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')

    c = ax[0,2].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[0, 2].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[0,2].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[0,2].coastlines() 


neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) > tc2.mean('initialization').sel(mode=y).std('time').values ) & 
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) < -1*tc2.mean('initialization').sel(mode=x).std('time').values )  )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p1 = neg_neg_composite.plot(ax=ax[0,0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[0,0].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)

    contours = ax[0,0].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')

    c = ax[0,0].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[0, 0].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[1,0].coastlines() 

else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[0,0].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[0,0].coastlines()


# top middle panel 
neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) > tc2.mean('initialization').sel(mode=y).std('time').values ) & 
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) > -1*tc2.mean('initialization').sel(mode=x).std('time').values ) &
         ( tc2.mean('initialization').sel(mode=x) < tc2.mean('initialization').sel(mode=x).std('time').values ) )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p1 = neg_neg_composite.plot(ax=ax[0,1], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[0,0].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)

    contours = ax[0,1].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')

    c = ax[0,1].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[0, 1].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[0,1].coastlines() 

else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[0,1].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[0,1].coastlines()




# bottom middle panel 
neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) < -1*tc2.mean('initialization').sel(mode=y).std('time').values ) & 
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) > -1*tc2.mean('initialization').sel(mode=x).std('time').values ) &
         ( tc2.mean('initialization').sel(mode=x) < tc2.mean('initialization').sel(mode=x).std('time').values ) )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p1 = neg_neg_composite.plot(ax=ax[2,1], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[0,0].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)

    contours = ax[2,1].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')

    c = ax[2,1].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[2, 1].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[2,1].coastlines() 

else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[2,1].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[2,1].coastlines()



#  middle panel 
neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) < tc2.mean('initialization').sel(mode=y).std('time').values ) & 
                               ( tc2.mean('initialization').sel(mode=y) > -1*tc2.mean('initialization').sel(mode=y).std('time').values ) &
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) > -1*tc2.mean('initialization').sel(mode=x).std('time').values ) &
         ( tc2.mean('initialization').sel(mode=x) < tc2.mean('initialization').sel(mode=x).std('time').values ) )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p1 = neg_neg_composite.plot(ax=ax[1,1], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[0,0].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)

    contours = ax[1,1].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')

    c = ax[1,1].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[1, 1].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[1,1].coastlines() 

else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[1,1].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[2,1].coastlines()



#  middle-left panel 
neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) < tc2.mean('initialization').sel(mode=y).std('time').values ) & 
                               ( tc2.mean('initialization').sel(mode=y) > -1*tc2.mean('initialization').sel(mode=y).std('time').values ) &
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) < -1*tc2.mean('initialization').sel(mode=x).std('time').values )  )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p1 = neg_neg_composite.plot(ax=ax[1,0], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[0,0].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)

    contours = ax[1,0].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')

    c = ax[1,0].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[1, 0].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[1,0].coastlines() 

else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[1,0].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[1,0].coastlines()    

#  middle-right panel 
neg_neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=y) < tc2.mean('initialization').sel(mode=y).std('time').values ) & 
                               ( tc2.mean('initialization').sel(mode=y) > -1*tc2.mean('initialization').sel(mode=y).std('time').values ) &
        ( np.abs(tc2.mean('initialization').sel(mode=z)) <  tc2.mean('initialization').sel(mode=z).std('time') ) &
        ( tc2.mean('initialization').sel(mode=x) > tc2.mean('initialization').sel(mode=x).std('time').values )  )
n = int(neg_neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_neg_composite_selection | ~neg_neg_composite_selection))
neg_neg_composite = ssta.isel(time=neg_neg_composite_selection).mean('time') 

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
    samp_mean = ssta.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(1-signif_level, 'samps')
lower = tc1.quantile(signif_level, 'samps')
pos_significance_mask = (neg_neg_composite > upper) | (neg_neg_composite < lower)

if add_wind:
    neg_neg_wind_composite = wind.isel(time=neg_neg_composite_selection).mean('time')
    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, wind.time.shape[0], size=n)
        samp_mean = wind.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(1-signif_level, 'samps')
    lower = tc1.quantile(signif_level, 'samps')
    pos_significance_mask_wind = (neg_neg_wind_composite > upper) | (neg_neg_wind_composite < lower)

if var != 'wind':
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)

    p1 = neg_neg_composite.plot(ax=ax[1,2], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #ax[0,0].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)

    contours = ax[1,2].contour(ssta.lon, ssta.lat, neg_neg_composite,  levels=np.linspace(-vmax,vmax, 11), colors='black', alpha=1, linewidths=0.5)
    for c in contours.collections:
        level = c.get_paths()[0].vertices[0,1] if c.get_paths() else 0
        if contours.levels[list(contours.collections).index(c)] < 0:
            c.set_linestyle('dashed')

    c = ax[1,2].coastlines() 
    if add_wind:
        neg_neg_wind_composite = neg_neg_wind_composite.where(pos_significance_mask_wind, other=np.nan)
        p = ax[1, 2].quiver(
            neg_neg_wind_composite.lon, neg_neg_wind_composite.lat,
            neg_neg_wind_composite.u, neg_neg_wind_composite.v,
            scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
        )
        # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
        c = ax[1,2].coastlines() 

else:
    neg_neg_composite = neg_neg_composite.where(pos_significance_mask, other=np.nan)
    p = ax[1,2].quiver(
        neg_neg_composite.lon, neg_neg_composite.lat,
        neg_neg_composite.u, neg_neg_composite.v,
        scale=50, width=0.002, headwidth=3, headlength=4, headaxislength=3.5, zorder=1
    )
    # ax[0,i].contourf(ssta.lon, ssta.lat, pos_significance_mask, 1, hatches=['','....'], alpha=0)
    c = ax[1,2].coastlines()    

for i in range(3):
    for j in range(3):
        ax[i,j].set_title(None)
        ax[i,j].set_title(letters[i][j], loc='left', fontweight='bold')


ax[0,0].text(-0.03, 0.5, 'Positive ' + y, va='center', ha='right', rotation=90, transform=ax[0,0].transAxes)
ax[1,0].text(-0.03, 0.5, 'Neutral ' + y, va='center', ha='right', rotation=90, transform=ax[1,0].transAxes)
ax[2,0].text(-0.03, 0.5, 'Negative ' + y, va='center', ha='right', rotation=90, transform=ax[2,0].transAxes)
ax[2,2].text(0.5, -0.1, 'Positive ' + x, va='center', ha='center', transform=ax[2,2].transAxes)
ax[2,1].text(0.5, -0.1, 'Neutral ' + x, va='center', ha='center', transform=ax[2,1].transAxes)
ax[2,0].text(0.5, -0.1, 'Negative ' + x, va='center', ha='center', transform=ax[2,0].transAxes)


cbar_ax = fig.add_axes([0.15, 0.1, .70, 0.02]) 
fig.colorbar(p1, cax=cbar_ax, **{'label': caption, 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5, 'extend': 'both'})
plt.tight_layout( rect=[0, 0.15, 1, 1])
plt.savefig(f'/Users/kylehall/Desktop/hall_molina_2025_final_figures/hall_molina_2025.figure4.{var}.png', dpi=1000)
plt.show()