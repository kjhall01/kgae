import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from pathlib import Path 
import kgae

random_seed = 42

run = 'kgvae_cv'

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# open ERA5 training data 
sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))

# split training data into 5-fold crossvalidation splits (taking every fifth value to be one group)
splits = [ sst.isel(time=slice(i, None, 5)) for i in range(5) ]







# for crossvalidation fold in the five-folds:
for split in range(5):
    # make the destination directory for this crossval split / recursion 
    out_dir = Path(run) / f'split{split}'
    out_dir.mkdir(exist_ok=True, parents=True)
    stoutf = sys.stdout#  open(out_dir / 'terminal_output.txt', 'w') 

    # recombine the other four folds to calculate trend and climatology 
    train = xr.concat([splits[j] for j in range(5) if j != split ], 'time').sortby('time')
    train, fit, gwm, p  = kgae.global_detrend(train, deg=2)
    train, monthly_clim = kgae.remove_climo(train)

    # remove trend and climatology (that was calculated on training set) from validation data
    val = splits[split]
    val = val - xr.polyval(val['time'], p.polyfit_coefficients)

    # remove the climatology that was calculated on train- this is what is inside of src.remove_climo, but instead of calculating 
    # the climatology based on the data to be anomalized, we use the one from train
    dim, monthly ='time', val
    toconcat = []
    for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
        ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
        ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
        toconcat.append(ds_yearly)
    val = xr.concat(toconcat, dim).sortby(dim)

    train_splits = [train.isel(time=slice(j, None, 5-1 )) for j in range(5-1)]
    stacked = [ t.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') for t in train_splits ]
    training_data = [ t.values for t in stacked ]

    # extract numpy values of validation set 
    val = val.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') 
    val_template = xr.ones_like(val)
    val_data = val.values 

    # instantiate NN 
    oae = kgae.KGVAE(
        lr=1e-3,
        activation=nn.Tanh,
        verbose=True,
        bottleneck_dim=5,
        hidden_layers=[248,248],
        noise=None,
        m=7,
        scheme='mult',
        fit_power=4,
        beta = 1e-3,
        stout=stoutf,
        dwl_loss=True, 
        variance_calc='salient',
        iou_calc='separate',
        device =  "mps"
    )

    print(f"\n\n\n\n\n\n---------- TRAINING MODEL {split+1} --------------", file=stoutf)
    tracking, val_tracking, mse_tracking, val_mse_tracking, fft_tracking, val_fft_tracking, spread_tracking, val_spread_tracking = oae.fit(
        training_data=training_data,
        val_data=val_data,
        num_epochs=500,
        patience=30,
        outpath=  out_dir / f'model_parameters_split{split}.pth',
        dx=5,
        val_dx=5
    )

    # close NN training output file 
    stoutf.close()
