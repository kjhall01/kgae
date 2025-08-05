import xarray as xr 
from pathlib import Path 
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import src 
import pandas as pd 
from sklearn.decomposition import PCA 
from scipy.stats import gaussian_kde

N=50
run='basin.goodtest'

def get_val_mses(recursion1=0):
    results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

    sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
    sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})
    ssta, trend, gwm, p = src.global_detrend(sst,deg=2)
    ssta, monthly_clim = src.remove_climo(ssta)

    mses = []
    for init in range(N):
        tc2 = []
        for split in range(5):
            tc4 = []
            for recursion in [recursion1]:
                dencodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}/val_data.decoded.nc')
                dencodings = getattr(dencodings, '__xarray_dataarray_variable__')
                tc4.append(dencodings)
            tc4 = xr.concat(tc4, 'recursion').assign_coords({'recursion': np.arange(1)})
            tc2.append(tc4)
        tc2 = xr.concat(tc2, 'time').sortby('time')
        mse = ((tc2 - ssta) **2).mean()       #tc3.append(tc2)
        mses.append(mse)
    return np.asarray(mses)

def get_test_mses(recursion1=0):
    results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

    sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(pd.Timestamp(2015,1,1), None))
    sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})
    ssta, trend, gwm, p = src.global_detrend(sst,deg=2)
    ssta, monthly_clim = src.remove_climo(ssta)

    mses = []
    for init in range(N):
        tc2 = []
        mse2 = []
        for split in range(5):
            tc4 = []
            for recursion in [recursion1]:
                dencodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}/test_data.decoded.nc')
                tc4.append(dencodings.sst)
            tc4 = xr.concat(tc4, 'recursion').assign_coords({'recursion': np.arange(1)})
            mse = ((tc4 - ssta) **2).mean() 
            mse2.append(mse)
        mses.append(sum(mse2) / 5 )
    return np.asarray(mses)

val_mses = get_val_mses()
test_mses = get_test_mses()
print('val', val_mses.mean())
print('test', test_mses.mean())
plt.figure(figsize=(8, 5))

# Plot histograms
plt.hist(val_mses.flatten(), bins=30, alpha=0.3, label='Validation MSEs', color='tab:blue', density=True, histtype='stepfilled')
plt.hist(test_mses.flatten(), bins=30, alpha=0.3, label='Test MSEs', color='tab:orange', density=True, histtype='stepfilled')

# Fit and plot KDEs
x_vals = np.linspace(
    min(val_mses.min(), test_mses.min()), 
    max(val_mses.max(), test_mses.max()), 
    200
)
val_kde = gaussian_kde(val_mses.flatten())
test_kde = gaussian_kde(test_mses.flatten())
plt.plot(x_vals, val_kde(x_vals), color='tab:blue', linewidth=2, label='Validation KDE')
plt.plot(x_vals, test_kde(x_vals), color='tab:orange', linewidth=2, label='Test KDE')

plt.xlabel('MSE')
plt.ylabel('Density')
plt.title('Comparison of Validation and Test MSEs')
plt.legend()
plt.tight_layout()
plt.show()
