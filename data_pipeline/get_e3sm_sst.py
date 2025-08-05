import xarray as xr 
from pathlib import Path 


data_dir = Path('/glade/campaign/cgd/ccr/E3SMv2/FV_regridded/v2.FV1.piControl/ocn/proc/tseries/month_1')
filename_base = 'v2.FV1.piControl.mpaso.hist.am.timeMonthly_avg_activeTracers_temperature.{:>04}01-{:>04}12.nc'

dss = [] 
for i in range(1, 50):
    filename = filename_base.format(i, i+9)
    ds = xr.open_dataset(data_dir / filename)
    da = ds.timeMonthly_avg_activeTracers_temperature
    dss.append(da.isel(nVertLevels=[0,59]))
dss = xr.concat(dss, 'time')
dss.to_netcdf('e3sm.preindustrial.sst.nc')

