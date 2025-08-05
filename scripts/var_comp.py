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

def get_val_mses(recursion1=0, mode=''):
    print('getting mode: ', mode)
    results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

    mses = []
    for init in range(N):
        tc2 = []
        for split in range(5):
            tc4 = []
            for recursion in [recursion1]:
                dencodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}/val_data.decoded{mode}.nc')
                dencodings = getattr(dencodings, '__xarray_dataarray_variable__')
                tc4.append(dencodings)
            tc4 = xr.concat(tc4, 'recursion').assign_coords({'recursion': np.arange(1)})
            tc2.append(tc4)
        tc2 = xr.concat(tc2, 'time').sortby('time')
        #mse = ((tc2 - ssta) **2).mean()       #tc3.append(tc2)
        mse = tc2.var('time').sum()
        mses.append(mse)
    return np.asarray(mses)

sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})
ssta, trend, gwm, p = src.global_detrend(sst,deg=2)
ssta, monthly_clim = src.remove_climo(ssta)

modes = [ '.without.Decadal', '.without.Interannual', '.without.Quasibiennial', '.without.HF1', '.without.HF2']
base_mses = get_val_mses(recursion1=0, mode='')
mses = [1 - get_val_mses(recursion1=0, mode=mode) /base_mses for mode in modes]
fig, ax = plt.subplots(figsize=(10, 6))
parts = ax.violinplot([m.flatten() for m in mses], showmeans=True, showmedians=True)

ax.set_xticks(np.arange(1, len(modes) + 1))
ax.set_xticklabels([i.replace('.without.', '') for i in modes])
ax.set_ylabel('Variance Explained (%)')
ax.set_title('')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
