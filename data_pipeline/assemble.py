import xarray as xr 
from pathlib import Path 
import pandas as pd 
import numpy as np 
import xesmf as xe 

import matplotlib.pyplot as plt 
import cartopy.crs as ccrs 

overwrite = True
approx_desired_res = 1.0
basin = 'pacific'

mask = getattr(xr.open_dataset('/Users/kylehall/Desktop/scratch/data/prod/seamask.nc'), basin) 
mask = mask.assign_coords({'lon': [i - 360 if i >= 180 else i for i in mask.coords['lon'].values ] }).sortby('lon').sortby('lat') 
mask = mask.interp({
    'lon': np.linspace(mask.coords['lon'].values.min(), mask.coords['lon'].values.max(), int(mask.coords['lon'].values.shape[0]*(1/approx_desired_res))),
    'lat': np.linspace(mask.coords['lat'].values.min(), mask.coords['lat'].values.max(), int(mask.coords['lat'].values.shape[0]*(1/approx_desired_res)))
})
mask = mask.rename({'lon': 'longitude', 'lat': 'latitude'})

reg = None


for year in range(1940, 2024):
    print(year)
    ds = ds.assign_coords({'longitude': [i - 360 if i >= 180 else i for i in ds.coords['longitude'].values ] }).sortby('longitude').sortby('latitude') 
    if reg is None:
        reg = xe.Regridder(ds, mask, 'bilinear', ignore_degenerate=True, periodic=True) 

    ds = reg(ds)
    ds = ds.assign_coords({'date': [pd.to_datetime(str(t), format='%Y%m%d') for t in ds.date.values]})
    ds = ds.rename({'date': 'time'})

    ds.sst.to_netcdf(f'era5.{year}/sst.nc')

    #pl = ds.sst.mean('time').plot(subplot_kws={'projection': ccrs.PlateCarree()})
    #pl.axes.coastlines()
    #plt.show()

    ds.t2m.to_netcdf(f'era5.{year}/t2m.nc')
    #pl = ds.t2m.mean('time').plot(subplot_kws={'projection': ccrs.PlateCarree()})
    #pl.axes.coastlines()
    #plt.show()

    ds.tp.to_netcdf(f'era5.{year}/tp.nc')
    pacific = ds.sst.where(mask >0, other=np.nan).dropna('latitude', how='all').dropna('longitude', how='all') 
    pacific = pacific.assign_coords({'longitude': [i + 180 if i <= 0 else i - 180 for i in pacific.coords['longitude'].values ] }).sortby('longitude').sortby('latitude') 
    #pl = pacific.mean('time').plot(subplot_kws={'projection': ccrs.PlateCarree(central_longitude=180)})
    #pl.axes.coastlines()
    #plt.show()

    pacific.to_netcdf(f'era5.{year}/pacific.nc')





