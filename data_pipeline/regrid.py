import numpy as np 
import xarray as xr 
import pandas as pd 
from pathlib import Path 
import xesmf as xe 

overwrite = True
approx_desired_res = 1.0
basin = 'indian'

dest_dir = Path('../data/dev/oras5/') / basin
if not dest_dir.is_dir(): 
    dest_dir.mkdir(exist_ok=True, parents=True) 

mask = getattr(xr.open_dataset('../data/prod/seamask.nc'), basin)
mask = mask.assign_coords({'lon': [i - 360 if i >= 180 else i for i in mask.coords['lon'].values ] }).sortby('lon').sortby('lat') 
mask = mask.interp({
    'lon': np.linspace(mask.coords['lon'].values.min(), mask.coords['lon'].values.max(), int(mask.coords['lon'].values.shape[0]*(1/approx_desired_res))),
    'lat': np.linspace(mask.coords['lat'].values.min(), mask.coords['lat'].values.max(), int(mask.coords['lat'].values.shape[0]*(1/approx_desired_res)))
})

reg = None
dates = pd.date_range(pd.Timestamp(1958,1,1), pd.Timestamp(2023,12,1), freq='MS')
for date in dates:
    if date not in [pd.Timestamp(1960, 2,1 )]:
        filename = 'oras5.{}.{}x{}.{:>04d}{:>02d}.nc'.format(basin, str(approx_desired_res).replace('.', 'p'), str(approx_desired_res).replace('.', 'p'), int(date.year), int(date.month) )
        print('regridding if needed ', filename) 
        if not (dest_dir / filename).is_file() or overwrite:
            if date.year >= 2015:
                ds = xr.open_dataset('../data/dev/oras5/orig/sosstsst_control_monthly_highres_2D_{:>04}{:>02}_OPER_v0.1.nc'.format(date.year, date.month))
            else:
                ds = xr.open_dataset('../data/dev/oras5/orig/sosstsst_control_monthly_highres_2D_{:>04}{:>02}_CONS_v0.1.nc'.format(date.year, date.month))
            if reg is None:
                reg = xe.Regridder(ds.sosstsst, mask, 'bilinear', ignore_degenerate=True, periodic=True) 
                print(reg)
            regridded = reg(ds.sosstsst).where(mask >0, other=np.nan)
            regridded = regridded.dropna('lat', how='all').dropna('lon', how='all') 
            regridded.to_netcdf(dest_dir/filename) 

ds = xr.open_mfdataset(str(dest_dir/'*.nc'))
ds = ds.rename({'time_counter': 'time', '__xarray_dataarray_variable__': 'sst'})
ds = ds.assign_coords(lon=[i-180 for i in ds.coords['lon'].values]).sortby('lon')
ds.to_netcdf('../data/prod/oras5.{}.1958-2023.nc'.format(basin))

# in order to plot centered on pacific, use 
## ds = ds.assign_coords(lon=[i-180 for i in ds.coords['lon'].values])
## proj = ccrs.PlateCarree(central_longitude=180)
## pl = ds.sst.mean('time').plot(subplot_kws={'projection': proj})