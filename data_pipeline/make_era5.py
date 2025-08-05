import xarray as xr 
import cartopy.crs as ccrs 
import pandas as pd 
from pathlib import Path 
import numpy as np 
import xesmf as xe

basin='pacific'
approx_desired_res = 1.0

mask = getattr(xr.open_dataset('../seamask.nc'), basin)
#mask = mask.assign_coords({'lon': [i - 360 if i >= 180 else i for i in mask.coords['lon'].values ] }).sortby('lon').sortby('lat') 
mask = mask.interp({
    'lon': np.linspace(mask.coords['lon'].values.min(), mask.coords['lon'].values.max(), int(mask.coords['lon'].values.shape[0]*(1/approx_desired_res))),
    'lat': np.linspace(mask.coords['lat'].values.min(), mask.coords['lat'].values.max(), int(mask.coords['lat'].values.shape[0]*(1/approx_desired_res)))
})
print(mask.lon.values)
reg = None
for var in ['u10', 'v10']:
    tc = []
    for year in range(1940, 2024):
        print(var, year)    
        ds =  xr.open_dataset(f'../era5/dev/era5.wind.{year}/data_stream-moda_stepType-avgua.nc')
        da = getattr(ds, var).drop('number').drop('expver').rename({'valid_time': 'time'}) #.mean('pressure_level')
        da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
        #da = da.assign_coords(lon=[i-360 if i > 180 else i for i in da.coords['lon'].values]).sortby('lon')
        print(da.lon.values)
        if reg is None:
            reg = xe.Regridder(da, mask, 'bilinear', ignore_degenerate=True, periodic=True) 
            print(reg)
        regridded = reg(da).where(mask >0, other=np.nan)
        regridded = regridded.dropna('lat', how='all').dropna('lon', how='all')

        #print(regridded.lon.values)
        regridded = regridded.assign_coords({'lon': [ i-180 for i in regridded.lon.values]}) # shift to 0-360
        regridded = regridded.sortby('lon').sortby('lat')


        print(regridded.lon.values)
       # da = da.interp({'lat': np.linspace(90, -90, 181), 'lon': np.linspace(0, 359, 360)})
        da = regridded.assign_coords({'time': [pd.Timestamp(i) for i in da.time.values]})
        tc.append(da)
    tc = xr.concat(tc, 'time')# / 100
    tc.name= var
    tc.to_netcdf(f'../era5/era5.{var}.{basin}.1x1.1940-2023.nc')



#tc = []
#for year in range(1940, 2024):
#    ds =  xr.open_dataset(f'../era5/dev/era5.sea_surface_temperature.{year}/data_0.nc')
#    print(ds)
#    da = ds.sst.drop('number').drop('expver').rename({'valid_time': 'time'}) #.mean('pressure_level')
#    da = da.rename({'latitude': 'lat', 'longitude': 'lon'})
#    da = da.interp({'lat': np.linspace(90, -90, 181), 'lon': np.linspace(0, 359, 360)})
#    da = da.assign_coords({'time': [pd.Timestamp(i) for i in da.time.values]})
#    tc.append(da)
#tc = xr.concat(tc, 'time') / 100
#tc.name= 'sst'
#tc.to_netcdf('../era5/era5.sst.1x1.1940-2023.nc')


