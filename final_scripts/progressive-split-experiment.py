import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import kgae
from sklearn.decomposition import PCA 

def fit_pca(
        train_np,
        test_np,
        reference_pattern, 
        feature,
        train_time = None,
        test_time = None,
        val_np = None ,
        val_time = None, 
        n_components = 2,
    ): 

    pca = PCA(n_components=n_components)
    pca.fit(train_np)
    train_pcs = pca.transform(train_np) 
    train_recon = pca.inverse_transform(train_pcs)
    train_mu, train_std = train_pcs.mean(axis=0).reshape(1,-1), train_pcs.std(axis=0).reshape(1,-1)
    train_pcs = (train_pcs - train_mu) / train_std 

    test_pcs = pca.transform(test_np) 
    test_recon = pca.inverse_transform(test_pcs)
    test_pcs = (test_pcs - train_mu) / train_std



    eofs = xr.DataArray(
        data=pca.components_,
        name='eof',
        dims=('mode', 'feature'),
        coords={
            'mode': np.arange(n_components)+1,
            'feature': feature
        }
    ).unstack('feature').sortby(['lat', 'lon'])
    need_flip = np.sign(xr.corr(eofs, reference_pattern, dim=['lat', 'lon']).values)
    need_flip[need_flip==0] = 1
    eofs = eofs * xr.DataArray(need_flip, dims=('mode',), coords={'mode': np.arange(n_components)+1})

    train_pcs = xr.DataArray(
        train_pcs * need_flip, 
        name='pc',
        dims=('time', 'mode'),
        coords={
            'time': train_time,
            'mode': np.arange(n_components) + 1
        }
    )

    train_recon = xr.DataArray(
        train_recon,
        name='sst',
        dims=('time', 'feature'),
        coords={
            'time': train_time,
            'feature': feature,
        }
    ).unstack('feature').sortby(['lat', 'lon'])

    test_pcs = xr.DataArray(
        test_pcs * need_flip, 
        name='pc',
        dims=('time', 'mode'),
        coords={
            'time': test_time,
            'mode': np.arange(n_components) + 1
        }
    )

    test_recon = xr.DataArray(
        test_recon,
        name='sst',
        dims=('time', 'feature'),
        coords={
            'time': test_time,
            'feature': feature,
        }
    ).unstack('feature').sortby(['lat', 'lon'])

    val_pcs = None 
    val_recon = None 
    if val_np is not None:
        val_pcs = pca.transform(val_np)
        val_recon = pca.inverse_transform(val_pcs)
        val_pcs = (val_pcs - train_mu) / train_std

        val_pcs = xr.DataArray(
            val_pcs * need_flip, 
            name='pc',
            dims=('time', 'mode'),
            coords={
                'time': val_time,
                'mode': np.arange(n_components) + 1
            }
        )

        val_recon = xr.DataArray(
            val_recon,
            name='sst',
            dims=('time', 'feature'),
            coords={
                'time': val_time,
                'feature': feature,
            }
        ).unstack('feature').sortby(['lat', 'lon'])
    return eofs, train_pcs, train_recon, test_pcs, test_recon, val_pcs, val_recon







if __name__ == "__main__":
    run = 'progressive-split'
    n_ensemble = 5 # as determined with large-ensemble 
    splits = np.linspace(0.5, 0.9, 9)
    n_epochs = 200
    modes_to_examine = [1, 2, 3, 4, 5]
    mode_labels = [  "HF1", "HF2", "Quasibiennial", "Interannual", "Decadal", ]
    continue_splits = False 

    dates = pd.date_range("1940-01-01", "2023-12-01", freq="MS") 
    split_indices = [ int(dates.shape[0] * i) for i in splits ]
    split_indices_start_of_test = [_+1 for _ in split_indices]
    split_dates = dates[split_indices]
    split_dates_start_of_test = dates[split_indices_start_of_test]

    exp_dir = Path(run)
    if not exp_dir.is_dir():
        exp_dir.mkdir(exist_ok=True, parents=True)

    for split_date, split_date_start_of_test in zip(split_dates, split_dates_start_of_test):
        split_dir = exp_dir / f"{split_date.year}-{split_date.month}"
        
        if split_dir.is_dir() and not continue_splits:
            continue 
        else:
            split_dir.mkdir(exist_ok=True, parents=True)

        training_period = (None, split_date)
        testing_period = (split_date_start_of_test, None)

        # open ERA5 training data 
        sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(training_period[0], training_period[1]))
        sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))

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
        folds = [ sst.isel(time=slice(i, None, 5)) for i in range(5) ]

        for random_seed in range(n_ensemble):
            print(f". starting seed {random_seed}")
            seed_dir = split_dir / f"seed{random_seed}"

            if seed_dir.is_dir():
                continue 

            seed_dir.mkdir(exist_ok=True, parents=True)

            # for the progressive-split experiment, we need to save:
            #   - (80%-models) train, test, val: reconstructions, latents, references 
            #   - (100%-models) alpha, beta, train/test: latents, reconstructions, references
            #   - (80%-PCA) train, test, val: reconstructions, latents
            #   - (100%-PCA) EOFs, train/test: latents, reconstructions, references 

            for fold in range(5):
                print(f". . starting split {fold}")
                out_dir = Path(seed_dir) / f'fold{fold}'
                out_dir.mkdir(exist_ok=True, parents=True)

                ###### PREPARE DATA ##########
                # TRAINING
                train = xr.concat([folds[j] for j in range(5) if j != fold ], 'time').sortby('time')
                train, fit, gwm, p  = kgae.global_detrend(train, deg=2)
                train, monthly_clim = kgae.remove_climo(train)
                
                train_folds = [train.isel(time=slice(j, None, 5-1 )) for j in range(5-1)]
                stacked = [ t.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') for t in train_folds ]
                training_data = [ t.values for t in stacked ]
                
                # VALIDATION
                val = folds[fold]
                val = val - xr.polyval(val['time'], p.polyfit_coefficients)

                dim, monthly ='time', val
                toconcat = []
                for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
                    ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
                    ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
                    toconcat.append(ds_yearly)
                val = xr.concat(toconcat, dim).sortby(dim)
                val_stacked = val.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature') 
                val_data = val_stacked.values 

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
                test_stacked = test.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature')
                test_data = test_stacked.values 

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
                    tag = f"fold{fold}"
                )

                print(f"\n\n\n\n\n\n---------- TRAINING MODEL {fold+1} --------------")
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

                ### save references, if random_seed == 0
                if random_seed == 0:
                    train.to_netcdf(out_dir / 'train_references.nc')
                    val.to_netcdf(out_dir / 'val_references.nc')
                    test.to_netcdf(out_dir / 'test_references.nc')


                ##### CALCULATE TRAIN-SET Latents, Reconstructions  
                train_data = np.vstack(training_data)
                with torch.no_grad():
                    train_latents = oae.encoder(torch.tensor(train_data, dtype=torch.float32))
                    if oae.is_variational:
                        train_latents = train_latents[:, :oae.latent_dim]
                    train_reconstructions = oae.decoder(train_latents)
                
                train_latents = train_latents.detach().cpu().numpy()
                train_latents = xr.DataArray(
                    train_latents,
                    coords={'time': train['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
                    dims=['time', 'mode']
                ).sortby('time')
                train_latents = train_latents * oae.flip_signs
                train_mu, train_std = train_latents.mean('time'), train_latents.std('time')
                train_latents = ( train_latents - train_mu ) / train_std
                train_latents.to_netcdf(out_dir / 'train_latents.nc')

                train_reconstructions = train_reconstructions.detach().cpu().numpy() 
                train_reconstructions= xr.DataArray(
                    train_reconstructions,
                    coords={
                        'time': train['time'].values, 
                        'feature': train.stack(feature=('lat', 'lon')).dropna('feature', how='any').feature
                    },
                    dims=['time', 'feature']
                ).unstack('feature').sortby(['time', 'lat', 'lon'])
                train_reconstructions.to_netcdf(out_dir / 'train_reconstructions.nc')

                ##### CALCULATE VAL-SET Latents, Reconstructions  
                with torch.no_grad():
                    val_latents = oae.encoder(torch.tensor(val_data, dtype=torch.float32))
                    if oae.is_variational:
                        val_latents = val_latents[:, :oae.latent_dim]
                    val_reconstructions = oae.decoder(val_latents)
                
                val_latents = val_latents.detach().cpu().numpy()
                val_latents = xr.DataArray(
                    val_latents,
                    coords={'time': val['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
                    dims=['time', 'mode']
                ).sortby('time')
                val_latents = val_latents * oae.flip_signs
                val_latents = ( val_latents - train_mu ) / train_std
                val_latents.to_netcdf(out_dir / 'val_latents.nc')

                val_reconstructions = val_reconstructions.detach().cpu().numpy() 
                val_reconstructions= xr.DataArray(
                    val_reconstructions,
                    coords={
                        'time': val['time'].values, 
                        'feature': val.stack(feature=('lat', 'lon')).dropna('feature', how='any').feature
                    },
                    dims=['time', 'feature']
                ).unstack('feature').sortby(['time', 'lat', 'lon'])
                val_reconstructions.to_netcdf(out_dir / 'val_reconstructions.nc')


                ##### CALCULATE TEST-SET Latents, Reconstructions  
                with torch.no_grad():
                    test_latents = oae.encoder(torch.tensor(test_data, dtype=torch.float32))
                    if oae.is_variational:
                        test_latents = test_latents[:, :oae.latent_dim]
                    test_reconstructions = oae.decoder(test_latents)
                
                test_latents = test_latents.detach().cpu().numpy()
                test_latents = xr.DataArray(
                    test_latents,
                    coords={'time': test['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
                    dims=['time', 'mode']
                ).sortby('time')
                test_latents = test_latents * oae.flip_signs
                test_latents = ( test_latents - train_mu ) / train_std
                test_latents.to_netcdf(out_dir / 'test_latents.nc')

                test_reconstructions = test_reconstructions.detach().cpu().numpy() 
                test_reconstructions= xr.DataArray(
                    test_reconstructions,
                    coords={
                        'time': test['time'].values, 
                        'feature': test.stack(feature=('lat', 'lon')).dropna('feature', how='any').feature
                    },
                    dims=['time', 'feature']
                ).unstack('feature').sortby(['time', 'lat', 'lon'])
                test_reconstructions.to_netcdf(out_dir / 'test_reconstructions.nc')

                #### Now that thats over, Fit PCA and save the same stuff (if seed 0): 
                if random_seed == 0:
                    (   
                        eofs, train_pcs, train_recon, 
                        test_pcs, test_recon, val_pcs, 
                        val_recon 
                    ) = fit_pca (
                        train_data,
                        test_data,
                        corr, # this is lat/lon xr.DataArray 
                        val_stacked.feature,
                        train.time,
                        test.time,
                        val_data,
                        val.time, 
                        n_components = 2,
                    )
                    train_pcs.to_netcdf( out_dir / 'pca_train_latents.nc')
                    train_recon.to_netcdf(out_dir / 'pca_train_reconstructions.nc')
                    test_pcs.to_netcdf( out_dir / 'pca_test_latents.nc')
                    test_recon.to_netcdf( out_dir / 'pca_test_reconstructions.nc')
                    val_pcs.to_netcdf( out_dir / 'pca_val_latents.nc')
                    val_recon.to_netcdf( out_dir / 'pca_val_reconstructions.nc')


            ###### Fit on All data (100% model) PREPARE DATA ##########
            # TRAINING (on all splits)
            train = xr.concat([folds[j] for j in range(5)  ], 'time').sortby('time')
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
            test_stacked = test.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature')
            test_data = test_stacked.values 

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

            print(f"\n\n\n\n\n\n---------- TRAINING MODEL {fold+1} --------------")
            tracking = oae.fit_noval(
                training_data=training_data,
                num_epochs=n_epochs,
            )
            oae.eval()

            train_data = np.vstack(training_data)
            oae.sort_latents_by_frequency(train_data, reference_pattern=reference_pattern)

            with torch.no_grad():
                train_latents = oae.encoder(torch.tensor(train_data, dtype=torch.float32))
                if oae.is_variational:
                    train_latents = train_latents[:, :oae.latent_dim]
                train_reconstructions = oae.decoder(train_latents)

            train_tendencies = kgae.jacobian(oae, torch.tensor(train_data, dtype=torch.float32), delta=1e-2, device='cpu')
            train_tendencies = xr.DataArray(
                train_tendencies.cpu().detach().numpy(), 
                coords={
                    'time': train['time'].values, 
                    'mode': np.arange(1, oae.latent_dim+1), 
                    'feature': val_stacked.feature # this is fine becase train.feature doesn't exist and train.feature should == val.feature
                },
                dims=['time', 'mode', 'feature']
            ).unstack('feature').sortby(['time', 'lat', 'lon'])

            train_latents = train_latents.detach().cpu().numpy()
            train_latents = xr.DataArray(
                train_latents,
                coords={'time': train['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
                dims=['time', 'mode']
            ).sortby('time')
            train_latents = train_latents * oae.flip_signs
            train_mu, train_std = train_latents.mean('time'), train_latents.std('time')
            train_latents = ( train_latents - train_mu ) / train_std
            train_latents.to_netcdf(seed_dir / 'full_train_latents.nc')

            train_reconstructions = train_reconstructions.detach().cpu().numpy() 
            train_reconstructions= xr.DataArray(
                train_reconstructions,
                coords={
                    'time': train['time'].values, 
                    'feature': train.stack(feature=('lat', 'lon')).dropna('feature', how='any').feature
                },
                dims=['time', 'feature']
            ).unstack('feature').sortby(['time', 'lat', 'lon'])
            train_reconstructions.to_netcdf(seed_dir / 'full_train_reconstructions.nc')

            alpha_full, beta_full, _, _, _ = kgae.regress_tendencies(train_latents.sel(mode=modes_to_examine), train_tendencies.sel(mode=modes_to_examine))
            alpha_full.to_netcdf(seed_dir / 'full_alpha.nc')
            beta_full.to_netcdf(seed_dir / 'full_beta.nc')

            with torch.no_grad():
                test_latents = oae.encoder(torch.tensor(test_data, dtype=torch.float32))
                if oae.is_variational:
                    test_latents = test_latents[:, :oae.latent_dim]
                test_reconstructions = oae.decoder(test_latents)

            test_latents = test_latents.detach().cpu().numpy()
            test_latents = xr.DataArray(
                test_latents,
                coords={'time': test['time'].values, 'mode': np.arange(1, oae.latent_dim+1)},
                dims=['time', 'mode']
            ).sortby('time')
            test_latents = test_latents * oae.flip_signs
            test_latents = ( test_latents - train_mu ) / train_std
            test_latents.to_netcdf(seed_dir / 'full_test_latents.nc')
            
            test_reconstructions = test_reconstructions.detach().cpu().numpy() 
            test_reconstructions= xr.DataArray(
                test_reconstructions,
                coords={
                    'time': test['time'].values, 
                    'feature': test.stack(feature=('lat', 'lon')).dropna('feature', how='any').feature
                },
                dims=['time', 'feature']
            ).unstack('feature').sortby(['time', 'lat', 'lon'])
            test_reconstructions.to_netcdf(seed_dir / 'full_test_reconstructions.nc')


            #### Now that thats over, Fit PCA and save the same stuff (if seed 0): 
            if random_seed == 0:
                (   
                    eofs, train_pcs, train_recon, 
                    test_pcs, test_recon, val_pcs, 
                    val_recon 
                ) = fit_pca (
                    train_data,
                    test_data,
                    corr, # this is lat/lon xr.DataArray 
                    val_stacked.feature,
                    train.time,
                    test.time,
                    n_components = 2,
                )
                train_pcs.to_netcdf( seed_dir / 'full_pca_train_latents.nc')
                train_recon.to_netcdf(seed_dir / 'full_pca_train_reconstructions.nc')
                test_pcs.to_netcdf( seed_dir / 'full_pca_test_latents.nc')
                test_recon.to_netcdf( seed_dir / 'full_pca_test_reconstructions.nc')
                eofs.to_netcdf(seed_dir / 'full_pca_eofs.nc')

