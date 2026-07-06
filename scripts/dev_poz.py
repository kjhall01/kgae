import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt 

nnanx=3
nnany=3

data =  np.random.randn(10, 10, 1000)+2

data[np.random.choice(10, nnanx),np.random.randint(10, size=nnany), : ] = np.nan

da = xr.DataArray(
    data = data,
    dims = ('lat', 'lon', 'time'),
    coords = {
        'lat': np.arange(10),
        'lon': np.arange(10), 
        'time': np.arange(1000)
    }
)

ci = 0.95 
pct_lt_zero = (da < 0).mean('time').where(np.isfinite(da.mean('time')), other=np.nan)

up = ci + (1-ci)/ 2
down = (1-ci)/2

is_negative = (pct_lt_zero > up ) 
is_positive = (pct_lt_zero < down)

pct_lt_zero.where(np.isfinite(da.mean('time')), other=np.nan).plot() 
plt.figure()
p = (is_negative | is_positive).where(np.isfinite(da.mean('time')), other=np.nan).plot() 
plt.show() 
