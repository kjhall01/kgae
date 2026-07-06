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
import cartopy.crs as ccrs 


random_seed = 0
run = 'kgvae_cv'
n_bootstrap = 100
n_epochs = 200
ci_level = 95

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# open ERA5 training data 
sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))
ssta, fit, gwm, p  = kgae.global_detrend(sst, deg=2)
ssta, monthly_clim = kgae.remove_climo(ssta)
oni = ssta.sel(lat=slice(-5, 5), lon=slice(10,60)).mean(['lat', 'lon'])
corr = xr.corr(ssta, oni, dim='time') 
reference_pattern = corr.stack(feature=('lat', 'lon')).dropna('feature', how='any').values 
print('reference shape', reference_pattern.shape)

# split training data into 5-fold crossvalidation splits (taking every fifth value to be one group)
splits = [ sst.isel(time=slice(i, None, 5)) for i in range(5) ]

#for random_seed in range(1):
for random_seed in range(1):
    print(f"starting run {random_seed}")
    run_dir = Path(run) 
    run_dir.mkdir(exist_ok=True, parents=True)

    xval_latents = [] 
    xval_tendencies = [] 
    xval_hessians = []
    for split in range(5):
        # make the destination directory for this crossval split / recursion 
        out_dir = Path(run_dir) / f'split{split}'
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
        oae = kgae.KGAE(
            training_data[0].shape[1],   # input dim
            5,                           # latent dim
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


        if split == 0 and random_seed == 0:
            oae.sort_latents_by_frequency(val_data, reference_pattern=reference_pattern)
            reference_model = oae
        else:
            oae.align_latent_dims(reference_model, val_data)

        # calculate global latent tendencies via finite perturbation
        split_tendencies = kgae.jacobian(oae, torch.tensor(val_data, dtype=torch.float32), delta=1e-2, device='cpu')
        #split_tendencies = kgae.finite_latent_tendency(oae, torch.tensor(val_data, dtype=torch.float32), delta=1e-2, device='cpu')
        split_tendencies = xr.DataArray(
            split_tendencies.cpu().detach().numpy(), 
            coords={
                'time': val_template['time'].values, 
                'mode': np.arange(1, oae.latent_dim+1), 
                'feature': val_template.feature
            },
            dims=['time', 'mode', 'feature']
        ).sortby('time')
        xval_tendencies.append(split_tendencies) # the flip-signs happens inside analytical_tendency now


        # calculate global latent tendencies via finite perturbation
        split_hessians = kgae.hessian1(oae, torch.tensor(val_data, dtype=torch.float32), device='cpu')
        #split_tendencies = kgae.finite_latent_tendency(oae, torch.tensor(val_data, dtype=torch.float32), delta=1e-2, device='cpu')
        split_hessians = xr.DataArray(
            split_hessians.cpu().detach().numpy(), 
            coords={
                'time': val_template['time'].values, 
                'tendency_mode': np.arange(1, oae.latent_dim+1), 
                'predictor_mode': np.arange(1, oae.latent_dim+1), 
                'feature': val_template.feature
            },
            dims=['time', 'tendency_mode', 'predictor_mode', 'feature']
        ).sortby('time')
        xval_hessians.append(split_hessians) # the flip-signs happens inside analytical_tendency now


        train_data = np.vstack(training_data)
        train_latents = oae.encoder( torch.tensor(train_data, dtype=torch.float32) ).detach().cpu().numpy()
        if oae.is_variational:
            train_latents = train_latents[:, :oae.latent_dim]  # take mean only

        train_latents = xr.DataArray(
            train_latents, 
            coords={
                'time': train['time'].values, 
                'mode': np.arange(1, oae.latent_dim+1)
            },
            dims=['time', 'mode']
        ).sortby('time')
        train_latents = train_latents * oae.flip_signs
        train_mu, train_std = train_latents.mean('time'), train_latents.std('time')

        split_latents = oae.encoder( torch.tensor(val_data, dtype=torch.float32) ).detach().cpu().numpy()
        if oae.is_variational:
            split_latents = split_latents[:, :oae.latent_dim]  # take mean only

        split_latents = xr.DataArray(
            split_latents, 
            coords={
                'time': val_template['time'].values, 
                'mode': np.arange(1, oae.latent_dim+1)
            },
            dims=['time', 'mode']
        ).sortby('time')
        split_latents = split_latents * oae.flip_signs  
        split_latents = (split_latents - train_mu) / train_std

        xval_latents.append(split_latents)  # apply sign flips to latents if necessary
    

    # combine crossvalidated latents and tendencies across folds
    xval_latents = xr.concat(xval_latents, 'time').sortby('time')#.rolling(time=5, center=True).mean().dropna('time')
    xval_tendencies = xr.concat(xval_tendencies, 'time').sortby('time')#.sel(time=xval_latents.time)#.rolling(time=5, center=True).mean().dropna('time')
    xval_tendencies = xval_tendencies.unstack('feature').sortby(['time', 'lat', 'lon'])

    xval_hessians = xr.concat(xval_hessians, 'time').sortby('time')
    xval_hessians = xval_hessians.unstack('feature').sortby(['time', 'lat', 'lon'])    
    xval_hessians.to_netcdf('hessians.nc')
    xval_latents.to_netcdf(out_dir / 'latents.nc')
    xval_tendencies.to_netcdf(out_dir / 'tendencies.nc')

if False:
    xval_hessians = xr.open_dataset('hessians.nc')
    xval_hessians = xval_hessians['__xarray_dataarray_variable__']
    
    run_dir = Path(run) 
    out_dir = Path(run_dir) / f'split4'
    out_dir.mkdir(exist_ok=True, parents=True)

    xval_latents = xr.open_dataset(out_dir / 'latents.nc')
    xval_latents = xval_latents['__xarray_dataarray_variable__']

    xval_tendencies = xr.open_dataset(out_dir / 'tendencies.nc')
    xval_tendencies = xval_tendencies['__xarray_dataarray_variable__']


print('variances', xval_latents.std('time'))
print('means', xval_latents.mean('time'))

modes_to_examine = [1, 2, 3, 4, 5]
mode_labels = ["HF1", "HF2", "Quasibiennial (i=1)", "Interannual (i=2)", "Decadal (i=3)", ]
#modes_to_examine = [ 4, 5]
#mode_labels = [ "Interannual (i=2)", "Decadal (i=3)", ]

alpha_composite, beta_composite, alpha_sig_composite, beta_sig_composite = kgae.conditional_composite_alpha_beta(
    D=ssta.sel(time = xval_latents.time),
    Z=xval_latents.sel(mode=modes_to_examine),
    sample_dim='time',
    latent_dim='mode',
    lower_q=0.33,
    upper_q=0.67,
    n_bootstrap=n_bootstrap,
    ci_level=ci_level
)

alpha_composite = kgae.smooth_spatial(alpha_composite, sigma_latlon=1.5)
beta_composite  = kgae.smooth_spatial(beta_composite,  sigma_latlon=1.5)

kgae.plot_alpha_beta_maps(
    alpha_composite, 
    beta_composite, 
    alpha_sig_composite, 
    beta_sig_composite, 
    modes_to_examine, 
    mode_labels, 
    n_contours=11, 
    height = 0.1
)




alpha_boot, beta_boot =kgae.bootstrap_tendency_regression(
    xval_latents.sel(mode=modes_to_examine), 
    xval_tendencies.sel(mode=modes_to_examine),
    n_boot=n_bootstrap,
    seed=0
)

alpha_lo, alpha_hi, alpha_sig = kgae.bootstrap_ci(alpha_boot)
beta_lo, beta_hi, beta_sig = kgae.bootstrap_ci(beta_boot)
alpha, beta = kgae.regress_tendencies(xval_latents.sel(mode=modes_to_examine), xval_tendencies.sel(mode=modes_to_examine))

alpha = kgae.smooth_spatial(alpha, sigma_latlon=1.5)
beta  = kgae.smooth_spatial(beta,  sigma_latlon=1.5)

kgae.plot_alpha_beta_maps(
    alpha, 
    beta, 
    alpha_sig, 
    beta_sig, 
    modes_to_examine, 
    mode_labels, 
    n_contours=11, 
    height = 0.1
)


hess_boot = kgae.bootstrap_hessians(
    xval_latents, 
    xval_hessians,
    n_boot=n_bootstrap,
    seed=0
)

hess_lo, hess_hi, hess_sig = kgae.bootstrap_ci(hess_boot)

kgae.plot_alpha_beta_maps(
    alpha, 
    xval_hessians.mean('time'), 
    alpha_sig, 
    hess_sig, 
    modes_to_examine, 
    mode_labels, 
    n_contours=11, 
    height = 0.1
)



### Plot Power Spectra of Crossvalidated Latents
data = torch.tensor(xval_latents.values, dtype=torch.float32)  # [time, mode]

# unsmoothed power spectra
dx = 1
freqs = torch.fft.rfftfreq(data.shape[0], d=dx)
centered = data - data.mean(dim=0)
coeffs = torch.fft.rfft(centered, dim=0)
unsmoothed_power = (coeffs * torch.conj(coeffs)).real
unsmoothed_power = unsmoothed_power / unsmoothed_power.sum(dim=0)

# smoothed power spectra using the implemented function
smoothed_power, smoothed_freqs = kgae.compute_smoothed_power_spectra(data, dx=dx, kernel_size=oae.power_spectrum_smoothing_kernel*5)

# plot one variable per axis, with unsmoothed and smoothed overlaid
fig, axes = plt.subplots(data.shape[1], 1, figsize=(8, 3 * data.shape[1]), sharex=True)
if data.shape[1] == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    ax.plot(freqs.numpy(), unsmoothed_power[:, i].numpy(), label="Unsmooth", alpha=0.6)
    ax.plot(smoothed_freqs[:, i].numpy(), smoothed_power[:, i].numpy(), label="Smoothed", linewidth=2)
    ax.set_title(f"Dimension {i}")
    ax.set_ylabel("Normalized Power")
    ax.legend()

axes[-1].set_xlabel("Frequency")
plt.tight_layout()
plt.show()

import sys; sys.exit()



corner_composites, corner_composite_sigs = kgae.cross_composites(
    D=ssta,
    Z=xval_latents.sel(mode=modes_to_examine),
    sample_dim='time',
    latent_dim='mode',
    mode1 = 4, 
    mode2 = 5,
    n_bootstrap=n_bootstrap,
    ci_level=ci_level
)

kgae.plot_grid_maps(
    corner_composites,
    corner_composite_sigs,
    'mode1',
    'mode2',
    row_labels=None,
    col_labels=None,
    n_contours=11,
    cmap="coolwarm",
    central_longitude=180,
    figsize=(14, 8),
    cbar_height=0.03,
    cbar_bottom=-0.08,
    title="Cross Composites"
)



corner_composite_diffs, corner_composite_diff_sigs = kgae.cross_composite_diffs(
    D=ssta,
    Z=xval_latents.sel(mode=modes_to_examine),
    sample_dim='time',
    latent_dim='mode',
    mode1 = 4, 
    mode2 = 5,
    n_bootstrap=n_bootstrap,
    ci_level=ci_level
)

kgae.plot_grid_maps(
    corner_composite_diffs,
    corner_composite_diff_sigs,
    'mode1',
    'mode2',
    row_labels=None,
    col_labels=None,
    n_contours=11,
    cmap="coolwarm",
    central_longitude=180,
    figsize=(14, 8),
    cbar_height=0.03,
    cbar_bottom=-0.08,
    title="Cross Composite Diffs"
)