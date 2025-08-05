import numpy as np 
import xarray as xr 
import pandas as pd 
from pathlib import Path 
import xesmf as xe 
import matplotlib.pyplot as plt 

overwrite = True
approx_desired_res = 1.0
basin = 'pacific'
sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.rename({'latitude':'lat', 'longitude': 'lon'})
mask = sst.mean('time') / sst.mean('time')
mask.to_netcdf('era5mask.nc')
mask.plot()
plt.show()
ds = xr.open_dataset('../ersstv5/sst.mnmean.nc').sst
ds = ds.assign_coords({'lon': [i - 180 for i in ds.lon]})
ds.mean('time').plot()
plt.show()
ds = ds.sortby('lat').interpolate_na('lat')
reg = xe.Regridder(ds, mask, 'bilinear', ignore_degenerate=True, periodic=True) 
regridded = reg(ds).where(mask >0, other=np.nan)
regridded = regridded.stack(feature=('lat', 'lon')).dropna('feature', how='any').unstack('feature').sortby('lat').sortby('lon') #.dropna('lon', how='all') 
regridded = regridded.assign_coords({'lon': [i-180 for i in regridded.lon]})
regridded.name = "sst"
print(regridded.stack(feature=('lat', 'lon')).dropna('feature', how='any').shape)
print(sst.stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature').shape)
regridded.to_netcdf('../ersstv5/ersstv5.pacific.sst.185401-202501.nc')

regridded.mean('time').plot()
plt.show()