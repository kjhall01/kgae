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
caption = 'SST (K)'
confidence_level = 0.67

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
tc2 = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).sel(time=slice('1950-01-01', '2014-12-31')).mean('initialization')


run = 'tropical'
results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

tc4 = []
for init in range(N):
    tc5 = []
    for split in range(5):
        tc = []
        for recursion in range(3):
            encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'val_data.encodings.nc').encodings
            tc.append(encodings)
        tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(3)})
        tc5.append(tc)
    tc5 = xr.concat(tc5, 'time').sortby('time')
    tc4.append(tc5)
tc4 = xr.concat(tc4, 'initialization').assign_coords({'initialization': np.arange(N)}).sel(time=slice('1950-01-01', '2014-12-31')).mean('initialization')

oni = src.open_oni().sel(time=tc2.time)
pdo = src.open_pdo().sel(dataset='PSL ERSSTv5 PDO').sel(time=tc2.time)


## correlations between basin-wide primary modes and PDO/ONI

x = [ tc2.isel(recursion=2).sel(mode=['Decadal', 'Interannual', 'Quasibiennial']).values,  tc4.isel(recursion=2).sel(mode=['Decadal', 'Interannual', 'Quasibiennial']).values, ]
x = np.hstack(x) 

correlation_coefficient_matrix = np.corrcoef(x, rowvar=False)
print()

# Define the titles for the columns/rows
titles = ["BW-DM", "BW-IA", "BW-QB", "TR-DM", "TR-IA", "TR-QB"]

# Convert the correlation matrix to a DataFrame for easier formatting
df_corr = pd.DataFrame(correlation_coefficient_matrix, columns=titles, index=titles)

# Print the correlation matrix in LaTeX format
print(df_corr.to_latex(float_format="%.2f"))
