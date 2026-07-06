import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import sys, os
import pickle
import matplotlib.pyplot as plt
from torch.nn import functional as F
import kgae
import cartopy.crs as ccrs 

run = 'large-ensemble-3-'
n_ensemble = 100 

n_epochs = 200
training_period = (None, pd.Timestamp(2014,12,31))
testing_period = (pd.Timestamp(2015, 1, 1), None)
modes_to_examine = [1, 2, 3, 4, 5]
mode_labels = [  "HF1", "HF2", "Quasibiennial", "Interannual", "Decadal", ]


# open ERA5 training data 
sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(training_period[0], training_period[1]))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))
run = run + f"{pd.Timestamp(sst.isel(time=0).time.values).year}-{pd.Timestamp(sst.isel(time=-1).time.values).year}"

sst_test = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(testing_period[0], testing_period[1]))
sst_test = sst_test.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))

# we can do this because the reference pattern doesn't actually really matter 
# it just aligns the sign of the latent tendencies s.t. positive = positive tropical sst anom
ssta, fit, gwm, p  = kgae.global_detrend(sst, deg=2)
ssta, monthly_clim = kgae.remove_climo(ssta)
oni = ssta.sel(lat=slice(-5, 5), lon=slice(10,60)).mean(['lat', 'lon'])
corr = xr.corr(ssta, oni, dim='time') 
reference_pattern = corr.stack(feature=('lat', 'lon')).dropna('feature', how='any').values 

# split training data into 5-fold crossvalidation splits
splits = [ sst.isel(time=slice(i, None, 5)) for i in range(5) ]

print(f"starting run {run}")
run_dir = Path(run) 
run_dir.mkdir(exist_ok=True, parents=True)

for random_seed in range(n_ensemble):
    print(f". starting seed {random_seed}")
    seed_dir = run_dir / f"seed{random_seed}"


    if seed_dir.is_dir():
        continue 

    seed_dir.mkdir(exist_ok=True, parents=True)

    # for the large-ensemble experiment, we need to save:
    #   - val-set latents (for time-series and distribution and power spectra)
    #   - alpha, beta (calculated on train-set latents and tendencies)
    #   - references ( xval for compositing )

    xval_latents = []  # concat in time, sort  - test latents will come from model fit on full period
    train_alphas = [] # concat along "fold"
    train_betas = [] # concat along "fold"

    if random_seed == 0:
        xval_references = []  # concat in time, sort

    for split in range(5):
        print(f". . starting split {split}")
        out_dir = Path(seed_dir) / f'split{split}'
        out_dir.mkdir(exist_ok=True, parents=True)

        ###### PREPARE DATA ##########
        # TRAINING
        train = xr.concat([splits[j] for j in range(5) if j != split ], 'time').sortby('time')
        train, fit, gwm, p  = kgae.global_detrend(train, deg=2)
        train, monthly_clim = kgae.remove_climo(train)
        
        train_splits = [train.isel(time=slice(j, None, 5-1 )) for j in range(5-1)]
        stacked = [ t.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') for t in train_splits ]
        training_data = [ t.values for t in stacked ]
        
        # VALIDATION
        val = splits[split]
        val = val - xr.polyval(val['time'], p.polyfit_coefficients)

        dim, monthly ='time', val
        toconcat = []
        for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
            ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
            ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
            toconcat.append(ds_yearly)
        val = xr.concat(toconcat, dim).sortby(dim)
        val = val.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') 
        val_data = val.values 

        if random_seed == 0:
            xval_references.append(val)

        # TEST 
        test = sst_test
        test = test - xr.polyval(test['time'], p.polyfit_coefficients)

        dim, monthly ='time', test
        toconcat = []
        for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
            ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
            ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
            toconcat.append(ds_yearly)
        test = xr.concat(toconcat, dim).sortby(dim)
        test = test.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature')
        test_data = test.values 

        ######### TRAIN A KGAE ##########
        oae = kgae.KGAE(
            training_data[0].shape[1],
            5,
            is_variational = True,
            hidden_layers = [248, 248],
            activation = nn.Tanh,
            power_spectrum_smoothing_kernel = 7,
            device = 'cpu',
            seed = random_seed,
            tag = f"fold{split}"
        )

        print(f"\n\n\n\n\n\n---------- TRAINING MODEL {split+1} --------------")
        tracking = oae.fit(
            training_data=training_data,
            val_data=val_data,
            num_epochs=n_epochs,
        )
        oae.eval()

        oae.sort_latents_by_frequency(val_data, reference_pattern=reference_pattern)

        # save model and tracking with pickle
        with open(out_dir / 'model.pkl', 'wb') as f:
            pickle.dump(oae, f)

        with open(out_dir / 'tracking.pkl', 'wb') as f:
            pickle.dump(tracking, f)

        ##### CALCULATE TRAIN-SET Alpha & Beta  
        train_data = np.vstack(training_data)
        train_tendencies = kgae.jacobian(oae, torch.tensor(train_data, dtype=torch.float32), delta=1e-2, device='cpu')
        train_tendencies = xr.DataArray(
            train_tendencies.cpu().detach().numpy(), 
            coords={
                'time': train['time'].values, 
                'mode': np.arange(1, oae.latent_dim+1), 
                'feature': val.feature # this is fine becase train.feature doesn't exist and train.feature should == val.feature
            },
            dims=['time', 'mode', 'feature']
        ).sortby('time')

        with torch.no_grad():
            train_latents = oae.encoder(torch.tensor(train_data, dtype=torch.float32))
            if oae.is_variational:
                train_latents = train_latents[:, :oae.latent_dim]
        train_latents = train_latents.detach().cpu().numpy()

        train_latents = xr.DataArray(
            train_latents,
            coords={'time': train['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
            dims=['time', 'mode']
        ).sortby('time')
        train_latents = train_latents * oae.flip_signs
        train_mu, train_std = train_latents.mean('time'), train_latents.std('time')
        print(train_std)
        train_latents = ( train_latents - train_mu ) / train_std


        # calculate train-set alpha/beta 
        alpha_train, beta_train, _, _, _ = kgae.regress_tendencies(train_latents.sel(mode=modes_to_examine), train_tendencies.unstack('feature').sortby(['time', 'lat', 'lon']).sel(mode=modes_to_examine))
        train_alphas.append(alpha_train)
        train_betas.append(beta_train)


        ##### CALCULATE VAL-SET LATENTS 
        with torch.no_grad():
            split_latents = oae.encoder(torch.tensor(val_data, dtype=torch.float32))
            if oae.is_variational:
                split_latents = split_latents[:, :oae.latent_dim] # mu only

        split_latents = split_latents.detach().cpu().numpy()
        split_latents = xr.DataArray(
            split_latents,
            coords={'time': val['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
            dims=['time', 'mode']
        ).sortby('time')
        split_latents = split_latents * oae.flip_signs
        split_latents = (split_latents - train_mu) / train_std
        xval_latents.append(split_latents)


    ##### SAVE CROSSVALIDATION STATS 
    xval_latents = xr.concat(xval_latents, 'time').sortby('time')
    xval_latents.to_netcdf(seed_dir / 'xval_latent.nc')

    train_alphas = xr.concat(train_alphas, 'fold').assign_coords({'fold': np.arange(5)+1})
    train_betas = xr.concat(train_betas, 'fold').assign_coords({'fold': np.arange(5)+1})
    train_alphas.to_netcdf(seed_dir / 'xval_alpha.nc')
    train_betas.to_netcdf(seed_dir / 'xval_beta.nc')

    if random_seed == 0: 
        xval_references = xr.concat(xval_references, 'time').sortby('time')
        xval_references = xval_references.unstack('feature').sortby(['time', 'lat', 'lon'])
        xval_references.to_netcdf(run_dir / 'xval_references.nc' )



    ###### PREPARE DATA ##########
    # TRAINING (on all splits)
    train = xr.concat([splits[j] for j in range(5)  ], 'time').sortby('time')
    train, fit, gwm, p  = kgae.global_detrend(train, deg=2)
    train, monthly_clim = kgae.remove_climo(train)
    
    train_splits = [train.isel(time=slice(j, None, 5 )) for j in range(5)]
    stacked = [ t.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') for t in train_splits ]
    training_data = [ t.values for t in stacked ]
    
    
    # TEST 
    test = sst_test
    test = test - xr.polyval(test['time'], p.polyfit_coefficients)

    dim, monthly ='time', test
    toconcat = []
    for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
        ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
        ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
        toconcat.append(ds_yearly)
    test = xr.concat(toconcat, dim).sortby(dim)
    test = test.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature')
    test_data = test.values 

    #### NOW TRAIN KGAE ON FULL CV PERIOD: 
    ######### TRAIN A KGAE ##########
    oae = kgae.KGAE(
        training_data[0].shape[1],
        5,
        is_variational = True,
        hidden_layers = [248, 248],
        activation = nn.Tanh,
        power_spectrum_smoothing_kernel = 7,
        device = 'cpu',
        seed = random_seed,
        tag = "full_model"
    )

    print(f"\n\n\n\n\n\n---------- TRAINING MODEL {split+1} --------------")
    tracking = oae.fit_noval(
        training_data=training_data,
        num_epochs=n_epochs,
    )
    oae.eval()

    train_data = np.vstack(training_data)
    oae.sort_latents_by_frequency(train_data, reference_pattern=reference_pattern)

    train_tendencies = kgae.jacobian(oae, torch.tensor(train_data, dtype=torch.float32), delta=1e-2, device='cpu')
    train_tendencies = xr.DataArray(
        train_tendencies.cpu().detach().numpy(), 
        coords={
            'time': train['time'].values, 
            'mode': np.arange(1, oae.latent_dim+1), 
            'feature': val.feature # this is fine becase train.feature doesn't exist and train.feature should == val.feature
        },
        dims=['time', 'mode', 'feature']
    ).sortby('time')

    with torch.no_grad():
        train_latents = oae.encoder(torch.tensor(train_data, dtype=torch.float32))
        if oae.is_variational:
            train_latents = train_latents[:, :oae.latent_dim]
    train_latents = train_latents.detach().cpu().numpy()

    train_latents = xr.DataArray(
        train_latents,
        coords={'time': train['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
        dims=['time', 'mode']
    ).sortby('time')
    train_latents = train_latents * oae.flip_signs
    train_mu, train_std = train_latents.mean('time'), train_latents.std('time')
    train_latents = ( train_latents - train_mu ) / train_std
    

    alpha_full, beta_full, _, _, _ = kgae.regress_tendencies(train_latents.sel(mode=modes_to_examine), train_tendencies.unstack('feature').sortby(['time', 'lat', 'lon']).sel(mode=modes_to_examine))
    alpha_full.to_netcdf(seed_dir / 'full_alpha.nc')
    beta_full.to_netcdf(seed_dir / 'full_beta.nc')

    with torch.no_grad():
        test_latents = oae.encoder(torch.tensor(test_data, dtype=torch.float32))
        if oae.is_variational:
            test_latents = test_latents[:, :oae.latent_dim]
    test_latents = test_latents.detach().cpu().numpy()

    test_latents = xr.DataArray(
        test_latents,
        coords={'time': test['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
        dims=['time', 'mode']
    ).sortby('time')
    test_latents = test_latents * oae.flip_signs
    test_latents = ( test_latents - train_mu ) / train_std

    test_latents.to_netcdf(seed_dir / 'test_latents.nc')
    
    if random_seed == 0:
        xval_test_references = test.unstack('feature').sortby(['time', 'lat', 'lon'])
        xval_test_references.to_netcdf(run_dir / 'test_references.nc' )
