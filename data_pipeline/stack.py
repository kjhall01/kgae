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

tc = []
for year in range(1940, 2024):
    print('sst', year)
    ds = xr.open_dataset(f'era5.{year}/sst.nc')
    tc.append(ds)
tc = xr.concat(tc, 'time')
tc.to_netcdf('era5.sst.1x1.1940-2023.nc')


tc = []
for year in range(1940, 2024):
    print('t2m', year)
    ds = xr.open_dataset(f'era5.{year}/t2m.nc')
    tc.append(ds)
tc = xr.concat(tc, 'time')
tc.to_netcdf('era5.t2m.1x1.1940-2023.nc')

tc = []
for year in range(1940, 2024):
    print('tp', year)
    ds = xr.open_dataset(f'era5.{year}/tp.nc')
    tc.append(ds)
tc = xr.concat(tc, 'time')
tc.to_netcdf('era5.tp.1x1.1940-2023.nc')

tc = []
for year in range(1940, 2024):
    print('pacific', year)
    ds = xr.open_dataset(f'era5.{year}/pacific.nc')
    tc.append(ds)
tc = xr.concat(tc, 'time')
tc.to_netcdf('era5.sst.pacific.1x1.1940-2023.nc')







