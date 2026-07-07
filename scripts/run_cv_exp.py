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

run = 'kgvae_cv_ema-'
n_bootstrap = 1000
n_epochs = 200
ci_level = 90
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

for random_seed in range(100):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f". starting seed {random_seed}")
    seed_dir = run_dir / f"seed{random_seed}"
    seed_dir.mkdir(exist_ok=True, parents=True)

    xval_latents = [] 
    xval_tendencies = [] 
    xval_reconstructions = [] 

    xval_test_latents = []
    xval_test_reconstructions = []
    xval_test_tendencies = []
    xval_test_alphas = []
    xval_test_betas = []
    xval_test_alpha_sigs = [] 
    xval_test_beta_sigs = [] 

    val_alpha_sigs = []
    val_alphas = []
    val_beta_sigs = []
    val_betas = []

    train_alpha_sigs = []
    train_alphas = []
    train_beta_sigs = []
    train_betas = []

    if random_seed == 0:
        xval_references = [] 
        xval_test_references = []

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
        
        if random_seed == 0:
            xval_test_references.append(test)


        ######### TRAIN A KGAE ##########
        oae = kgae.KGAE(
            training_data[0].shape[1],
            5,
            is_variational = True,
            hidden_layers = [248, 248],
            activation = nn.Tanh,
            power_spectrum_smoothing_kernel = 7,
            device = 'cpu'
        )

        print(f"\n\n\n\n\n\n---------- TRAINING MODEL {split+1} --------------")
        tracking = oae.fit(
            training_data=training_data,
            val_data=val_data,
            num_epochs=n_epochs,
        )
        oae.eval()

        # Align latent modes 
        #if split == 0 and random_seed == 0:
        oae.sort_latents_by_frequency(val_data, reference_pattern=reference_pattern)
        #    reference_model = oae
        #else:
        #    oae.align_latent_dims(reference_model, val_data)

        # save model and tracking with pickle
        with open(out_dir / 'model.pkl', 'wb') as f:
            pickle.dump(oae, f)

        with open(out_dir / 'tracking.pkl', 'wb') as f:
            pickle.dump(tracking, f)


        ######## Save things for Analysis ########
        split_tendencies = kgae.jacobian(oae, torch.tensor(val_data, dtype=torch.float32), delta=1e-2, device='cpu')
        split_tendencies = xr.DataArray(
            split_tendencies.cpu().detach().numpy(), 
            coords={
                'time': val['time'].values, 
                'mode': np.arange(1, oae.latent_dim+1), 
                'feature': val.feature
            },
            dims=['time', 'mode', 'feature']
        ).sortby('time') #* oae.flip_signs # align signs so that positive = positive tropical sst anom
        xval_tendencies.append(split_tendencies)

        # encode train latents to calculate stats for standardizing val latents 
        train_data = np.vstack(training_data)
        
        #train_tendencies = kgae.jacobian(oae, torch.tensor(train_data, dtype=torch.float32), delta=1e-2, device='cpu')
        #train_tendencies = xr.DataArray(
        #    train_tendencies.cpu().detach().numpy(), 
        #    coords={
        #        'time': train['time'].values, 
        #        'mode': np.arange(1, oae.latent_dim+1), 
        #        'feature': val.feature
        #    },
        #    dims=['time', 'mode', 'feature']
        #).sortby('time')


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

        # Encode/Decode Val Latents 
        with torch.no_grad():
            split_latents = oae.encoder(torch.tensor(val_data, dtype=torch.float32))
            if oae.is_variational:
                split_latents = split_latents[:, :oae.latent_dim] # mu only
            split_reconstructions = oae.decoder(split_latents)

        split_latents = split_latents.detach().cpu().numpy()
        split_latents = xr.DataArray(
            split_latents,
            coords={'time': val['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
            dims=['time', 'mode']
        ).sortby('time')
        split_latents = split_latents * oae.flip_signs
        split_latents = (split_latents - train_mu) / train_std
        xval_latents.append(split_latents)

        split_reconstructions = split_reconstructions.detach().cpu().numpy() 
        split_reconstructions= xr.DataArray(
            split_reconstructions,
            coords={
                'time': val['time'].values, 
                'feature': val.feature
            },
            dims=['time', 'feature']
        ).sortby('time')
        xval_reconstructions.append(split_reconstructions)

        ### Encode / Decode Test set ###
        with torch.no_grad():
            test_latents = oae.encoder(torch.tensor(test_data, dtype=torch.float32))
            if oae.is_variational:
                test_latents = test_latents[:, :oae.latent_dim] # mu only
            test_reconstructions = oae.decoder(test_latents)

        test_latents = test_latents.detach().cpu().numpy()
        test_latents = xr.DataArray(
            test_latents,
            coords={'time': test['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
            dims=['time', 'mode']
        ).sortby('time')
        test_latents = test_latents * oae.flip_signs
        test_latents = (test_latents - train_mu) / train_std
        xval_test_latents.append(test_latents)


        # calculate the val-set alpha/beta/sigs 
        #alpha_boot_val, beta_boot_val = kgae.bootstrap_tendency_regression(
        #    split_latents.sel(mode=modes_to_examine), 
        #    split_tendencies.unstack('feature').sortby(['time', 'lat', 'lon']).sel(mode=modes_to_examine),
        #    n_boot=n_bootstrap,
        #    seed=random_seed
        #)

        #alpha_lo_val, alpha_hi_val, alpha_sig_val = kgae.bootstrap_ci(alpha_boot_val)
        #beta_lo_val, beta_hi_val, beta_sig_val = kgae.bootstrap_ci(beta_boot_val)
        #alpha_val, beta_val = kgae.regress_tendencies(split_latents.sel(mode=modes_to_examine), split_tendencies.unstack('feature').sortby(['time', 'lat', 'lon']).sel(mode=modes_to_examine))

        #val_alpha_sigs.append(alpha_sig_val)
        #val_alphas.append(alpha_val)
        #val_beta_sigs.append(beta_sig_val)
        #val_betas.append(beta_val)

        # calculate train-set alpha/beta 
        #alpha_boot_train, beta_boot_train =kgae.bootstrap_tendency_regression(
        #    train_latents.sel(mode=modes_to_examine), 
        #    train_tendencies.unstack('feature').sortby(['time', 'lat', 'lon']).sel(mode=modes_to_examine),
        #    n_boot=n_bootstrap,
        #    seed=random_seed
        #)

        #alpha_lo_train, alpha_hi_train, alpha_sig_train = kgae.bootstrap_ci(alpha_boot_train)
        #beta_lo_train, beta_hi_train, beta_sig_train = kgae.bootstrap_ci(beta_boot_train)
        #alpha_train, beta_train = kgae.regress_tendencies(train_latents.sel(mode=modes_to_examine), train_tendencies.unstack('feature').sortby(['time', 'lat', 'lon']).sel(mode=modes_to_examine))

       # train_alpha_sigs.append(alpha_sig_train)
       # train_alphas.append(alpha_train)
       # train_beta_sigs.append(beta_sig_train)
       # train_betas.append(beta_train)


        # calculate the test-set alpha/beta/sigs 
        test_tendencies = kgae.jacobian(oae, torch.tensor(test_data, dtype=torch.float32), delta=1e-2, device='cpu')
        test_tendencies = xr.DataArray(
            test_tendencies.cpu().detach().numpy(), 
            coords={
                'time': test['time'].values, 
                'mode': np.arange(1, oae.latent_dim+1), 
                'feature': test.feature
            },
            dims=['time', 'mode', 'feature']
        ).unstack('feature').sortby(['time', 'lat', 'lon'])
        xval_test_tendencies.append(test_tendencies)

        test_reconstructions = test_reconstructions.unsqueeze(-1).detach().cpu().numpy() 
        test_reconstructions= xr.DataArray(
            test_reconstructions,
            coords={
                'time': test['time'].values, 
                'feature': test.feature,
                'fold': np.asarray([split])
            },
            dims=['time', 'feature', 'fold']
        ).sortby('time')
        xval_test_reconstructions.append(test_reconstructions)

        #xval_test_alpha_sigs.append(alpha_sig_test)
        #xval_test_alphas.append(alpha_test)
        #xval_test_beta_sigs.append(beta_sig_test)
        #xval_test_betas.append(beta_test)

    ############ compute cross-validated diagnostics 
    xval_latents = xr.concat(xval_latents, 'time').sortby('time')
    xval_latents.to_netcdf(seed_dir / 'xval_latent.nc')

    xval_reconstructions = xr.concat(xval_reconstructions, 'time').sortby('time')
    xval_reconstructions = xval_reconstructions.unstack('feature').sortby(['time', 'lat', 'lon'])
    xval_reconstructions.var('time').to_netcdf(seed_dir / 'xval_reconstruction_variance.nc')
    xval_reconstructions.to_netcdf(seed_dir / 'xval_reconstructions.nc')

    xval_tendencies = xr.concat(xval_tendencies, 'time').sortby('time')
    xval_tendencies = xval_tendencies.unstack('feature').sortby(['time', 'lat', 'lon'])

    ######### save test-set stuff for diagnostics 
    xval_test_latents = xr.concat(xval_test_latents, 'fold').sortby('time')
    xval_test_latents.to_netcdf(seed_dir / 'test_latent.nc')
    xval_test_tendencies = xr.concat(xval_test_tendencies, 'fold').sortby('fold')

    xval_test_reconstructions = xr.concat(xval_test_reconstructions, 'fold').sortby('fold')
    xval_test_reconstructions = xval_test_reconstructions.unstack('feature').sortby(['time','fold', 'lat', 'lon'])
    xval_test_reconstructions.to_netcdf(seed_dir / 'test_reconstructions.nc')

    # r2_train_boot is really from the full cv set- not the bootstrap resampled training set
    alpha_boot, beta_boot, r2_train_boot, r2_test_boot =kgae.bootstrap_tendency_regression(
        xval_latents.sel(mode=modes_to_examine), 
        xval_tendencies.sel(mode=modes_to_examine),
        #ztest_da=xval_test_latents.sel(mode=modes_to_examine),
        #Ttest_da=xval_test_tendencies.sel(mode=modes_to_examine),
        n_boot=n_bootstrap,
        seed=random_seed
    )

    #r2_cv_median = r2_train_boot.quantile(0.5, 'boot')
    #r2_test_median = r2_test_boot.quantile(0.5, 'boot')
    #r2_gen_gaps = r2_test_boot - r2_train_boot
    #r2_gen_gap_median = r2_gen_gaps.quantile(0.5, 'boot')
    #r2_gen_gap_sig = r2_gen_gaps.quantile(0.95, 'boot') < 0     
    #r2_cv_median.to_netcdf(seed_dir / 'xval_tendencyreg_r2_median.nc')
    #r2_test_median.to_netcdf(seed_dir / 'test_tendencyreg_r2_median.nc')
    #r2_gen_gap_median.to_netcdf(seed_dir / 'tendencyreg_gen_gap_median.nc')
    #2_gen_gap_sig.to_netcdf(seed_dir / 'tendencyreg_gen_gap_sig.nc')


    #alpha_lo, alpha_hi, alpha_sig = kgae.bootstrap_ci(alpha_boot)
    #beta_lo, beta_hi, beta_sig = kgae.bootstrap_ci(beta_boot)

    # Save "percent less than zero" and calculate the requested CI.
    alpha_sig = (alpha_boot < 0).mean('boot', skipna=False).where(np.isfinite(alpha_boot.mean('boot')), other=np.nan)
    beta_sig = (beta_boot < 0).mean('boot', skipna=False).where(np.isfinite(beta_boot.mean('boot')), other=np.nan)
    alpha, beta, r2_train_boot, r2_cv_da, r2_test_boot = kgae.regress_tendencies(xval_latents.sel(mode=modes_to_examine), xval_tendencies.sel(mode=modes_to_examine))
    
    alpha.to_netcdf(seed_dir / 'xval_alpha.nc')
    beta.to_netcdf(seed_dir / 'xval_beta.nc')
    alpha_sig.to_netcdf(seed_dir / 'xval_alpha_sig.nc')
    beta_sig.to_netcdf(seed_dir / 'xval_beta_sig.nc')

    if random_seed == 0: 
        xval_references = xr.concat(xval_references, 'time').sortby('time')
        xval_references = xval_references.unstack('feature').sortby(['time', 'lat', 'lon'])
        xval_references.to_netcdf(run_dir / 'xval_references.nc' )


    if random_seed == 0:
        xval_test_references = xr.concat(xval_test_references, 'fold').sortby('fold')
        xval_test_references = xval_test_references.unstack('feature').sortby(['time', 'fold', 'lat', 'lon'])
        xval_test_references.to_netcdf(run_dir / 'test_references.nc' )
