import xarray as xr 
import cartopy.crs as ccrs 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import matplotlib.patches as patches
import src 
import numpy as np 
from pathlib import Path 

N=50
run='basin.goodtest'
var = 'sst'
vmax = 1.5
caption = 'SST Anomaly (K)'
results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

tc3 = []
for init in range(N):
    tc2 = []
    for split in range(5):
        tc = []
        for recursion in range(3):
            encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'val_data.encodings.nc').encodings
            tc.append(encodings)
        tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(3)})
        tc2.append(tc)
    tc2 = xr.concat(tc2, 'time').sortby('time')
    tc3.append(tc2)
encodings= xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).isel(recursion=0)

titles = ['Decadal', "Interannual", "Quasibiennial"]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,5))


nlags=36
levels = np.linspace(-1, 1, 21)
levels2 = np.linspace(-1, 1, 11)
months = ['Jan', 'Feb', 'Mar', "Apr", 'May', "Jun", 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

corrs, sigs = src.crosscorrelation_by_month(encodings.mean('initialization').sel(mode='Interannual'), encodings.mean('initialization').sel(mode='Quasibiennial'), nlags=nlags)
corrsnan = corrs.copy()
corrsnan[sigs > 0.05] = np.nan
cp = ax[0].contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
cp2 = ax[0].contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
ax[0].clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

ax[0].set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
ax[0].set_xticks(np.arange(-nlags, nlags+1, 12))

ax[0].set_title('Interannual - Quasibiennial')
ax[0].set_title('a)', loc='left', fontweight='bold')

# Add bottom left and right text labels
ax[0].text(-nlags, -3.5, "Quasibiennial precedes Interannual", ha='left', va='top', fontsize=10)
ax[0].text(nlags, -3.5, "Quasibiennial follows Interannual", ha='right', va='top', fontsize=10)



corrs, sigs = src.crosscorrelation_by_month(encodings.mean('initialization').sel(mode='Decadal'), src.low_pass(encodings, threshold=7*12*30.4*86400).mean('initialization').sel(mode='Interannual'), nlags=nlags)
corrsnan = corrs.copy()
corrsnan[sigs > 0.05] = np.nan
cp = ax[1].contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
cp2 = ax[1].contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
ax[1].clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

ax[1].set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
ax[1].set_xticks(np.arange(-nlags, nlags+1, 12))

ax[1].set_title('Decadal - Interannual (LPF)')
ax[1].set_title('b)', loc='left', fontweight='bold')

# Add bottom left and right text labels
ax[1].text(-nlags, -3.5, "Interannual precedes Decadal", ha='left', va='top', fontsize=10)
ax[1].text(nlags, -3.5, "Interannual follows Decadal", ha='right', va='top', fontsize=10)


cbar_ax = fig.add_axes([0.15, 0.15, .70, 0.02]) 
fig.colorbar(cp, cax=cbar_ax, **{'label': 'Pearson Correlation', 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5, 'ticks':np.linspace(-1,1, 6), 'extend': 'both'})
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("/Users/kylehall/Desktop/hall_molina_2025_final_figures/hall_molina_2025.figure3.png", dpi=1000)
plt.show()