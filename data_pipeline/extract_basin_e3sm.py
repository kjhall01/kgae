import numpy as np 
import xarray as xr 
import pandas as pd 
from pathlib import Path 
import xesmf as xe 

overwrite = True
approx_desired_res = 1.0
basin = 'indian'

mask = getattr(xr.open_dataset('../data/prod/seamask.nc'), basin) 
mask = mask.assign_coords({'lon': [i - 360 if i >= 180 else i for i in mask.coords['lon'].values ] }).sortby('lon').sortby('lat') 
mask = mask.interp({
    'lon': np.linspace(mask.coords['lon'].values.min(), mask.coords['lon'].values.max(), int(mask.coords['lon'].values.shape[0]*(1/approx_desired_res))),
    'lat': np.linspace(mask.coords['lat'].values.min(), mask.coords['lat'].values.max(), int(mask.coords['lat'].values.shape[0]*(1/approx_desired_res)))
})

ds = xr.open_dataset('../data/prod/e3sm.global.preindustrial.sst.500yr.nc')
reg = xe.Regridder(ds.temperature, mask, 'bilinear', ignore_degenerate=True, periodic=True) 
regridded = reg(ds.temperature).where(mask >0, other=np.nan)
regridded = regridded.dropna('lat', how='all').dropna('lon', how='all') 
regridded.name = "sst"
regridded.to_netcdf('../data/prod/e3sm.{}.preindustrial.sst.500yr.nc'.format(basin))