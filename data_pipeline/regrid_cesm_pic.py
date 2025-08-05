import numpy as np 
import xarray as xr 
import pandas as pd 
from pathlib import Path 
import xesmf as xe 
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 

overwrite = True
approx_desired_res = 1.0
basin = 'pacific'

dest_dir = Path('data/dev/cesm-le/') / basin
if not dest_dir.is_dir(): 
    dest_dir.mkdir(exist_ok=True, parents=True) 

mask = getattr(xr.open_dataset('seamask.nc'), basin)
mask = mask.assign_coords({'lon': [i - 360 if i >= 180 else i for i in mask.coords['lon'].values ] }).sortby('lon').sortby('lat') 
mask = mask.interp({
    'lon': np.linspace(mask.coords['lon'].values.min(), mask.coords['lon'].values.max(), int(mask.coords['lon'].values.shape[0]*(1/approx_desired_res))),
    'lat': np.linspace(mask.coords['lat'].values.min(), mask.coords['lat'].values.max(), int(mask.coords['lat'].values.shape[0]*(1/approx_desired_res)))
})


reg = None
root = Path('cesm-pi')
dest_dir = root / 'prod'

for filename in (root / 'orig').glob('*'):
    yrs = str(filename.name).split('.')[-2]
    print('regridding if needed ', filename) 
    if not (dest_dir / filename).is_file() or overwrite:
        ds = xr.open_dataset(filename)
        print(ds.time.max(), ds.time.min(), ds.time.argmax(), ds.time.argmin())
        ds = ds.assign_coords({'time': [pd.Timestamp(1700+iii.year % 100, iii.month, iii.day) for iii in ds.coords['time'].values ]})
        ds = ds.drop('TLONG').drop('TLAT').drop('z_t')
        ds = ds.rename({'ULONG': 'lon', 'ULAT': 'lat'})
        if reg is None:
            reg = xe.Regridder(ds.SST, mask, 'bilinear', ignore_degenerate=True, periodic=True) 
            print(reg)

        regridded = reg(ds.SST).where(mask >0, other=np.nan)
        regreidded = regridded.assign_coords
       # print(regridded.lon.values)
        regridded = regridded.dropna('lat', how='all').dropna('lon', how='all') 
        regridded = regridded.assign_coords(lon=[i+180 for i in regridded.coords['lon'].values]).sortby('lon')
        regridded = regridded.assign_coords(lon=[i-360 if i > 180 else i for i in regridded.coords['lon'].values]).sortby('lon')

       # print(regridded.lon.values)
      #  p =regridded.mean('time').plot(subplot_kws={'projection': ccrs.PlateCarree(central_longitude=180)})
      #  p.axes.coastlines()
      #  plt.show()
        regridded.to_netcdf(dest_dir/f'cesm.piControl.sst.pacific.{yrs}.nc') 

#ds = xr.open_mfdataset(str(dest_dir/'*.nc'))
#ds = ds.rename({'time_counter': 'time', '__xarray_dataarray_variable__': 'sst'})
#ds = ds.assign_coords(lon=[i-180 for i in ds.coords['lon'].values]).sortby('lon')
#ds.to_netcdf('../data/prod/oras5.{}.1958-2023.nc'.format(basin))

# in order to plot centered on Pacific, use 
## ds = ds.assign_coords(lon=[i-180 for i in ds.coords['lon'].values])
## proj = ccrs.PlateCarree(central_longitude=180)
## pl = ds.sst.mean('time').plot(subplot_kws={'projection': proj})