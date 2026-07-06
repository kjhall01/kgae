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
    oae = kgae.KGAE(
        training_data[0].shape[1],   # input dim
        5,                           # latent dim
        is_variational = False, 
        hidden_layers = [248, 248],
        activation = nn.Tanh, 
        power_spectrum_smoothing_kernel = 7,
        device = 'cpu'
    )

    print(f"\n\n\n\n\n\n---------- TRAINING MODEL {split+1} --------------", file=stoutf)
    tracking = oae.fit(
        training_data=training_data,
        val_data=val_data,
        num_epochs=100,
        outpath=  out_dir / f'model_parameters_split{split}.pth',
    )


    for key in tracking['train'].keys():
        plt.figure()
        plt.plot( tracking['train'][key], label='train' )
        plt.plot( tracking['val'][key], label='val' )
        plt.title(key)
        plt.yscale('log')
        plt.legend()
    plt.show()


    # plot modes 
    all_data = sst - xr.polyval(sst['time'], p.polyfit_coefficients)

    # remove the climatology that was calculated on train- this is what is inside of src.remove_climo, but instead of calculating 
    # the climatology based on the data to be anomalized, we use the one from train
    dim, monthly ='time', all_data
    toconcat = []
    for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
        ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_clim
        ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
        toconcat.append(ds_yearly)
    all_data = xr.concat(toconcat, dim).sortby(dim)
    all_data = all_data.sortby('time').stack(feature=('lat', 'lon')).dropna('feature', how='any').transpose('time', 'feature')

    all_data_values = torch.tensor(all_data.values).to(oae.device).float()
    variance_fracs = oae.compute_salient_variance(all_data_values)
    print(variance_fracs)

    modes = oae.decoder( torch.eye(oae.latent_dim).to(oae.device).float() ).cpu().detach().numpy()
    modes = all_data.isel(time=slice(None, oae.latent_dim)).copy(data=modes).rename({'time': 'mode'}).assign_coords({'mode': np.arange(1, oae.latent_dim+1) })
    modes = modes.unstack('feature').sortby(['mode', 'lat', 'lon']) 

    pl = modes.plot(
        col='mode', col_wrap=3, cmap='RdBu_r', 
        vmin=-np.max(np.abs(modes)), vmax=np.max(np.abs(modes)), 
        figsize=(12,8),
        subplot_kws={'projection': ccrs.PlateCarree(central_longitude=180)}
    )

    for ax in pl.axes.flatten():
        ax.coastlines()
    plt.suptitle(f'KGAE Modes Split {split}', fontsize=16)


    modes = oae.decoder( -1*torch.eye(oae.latent_dim).to(oae.device).float() ).cpu().detach().numpy()
    modes = all_data.isel(time=slice(None, oae.latent_dim)).copy(data=modes).rename({'time': 'mode'}).assign_coords({'mode': np.arange(1, oae.latent_dim+1) })
    modes = modes.unstack('feature').sortby(['mode', 'lat', 'lon']) 

    pl = modes.plot(
        col='mode', col_wrap=3, cmap='RdBu_r', 
        vmin=-np.max(np.abs(modes)), vmax=np.max(np.abs(modes)), 
        figsize=(12,8),
        subplot_kws={'projection': ccrs.PlateCarree(central_longitude=180)}
    )

    for ax in pl.axes.flatten():
        ax.coastlines()
    plt.suptitle(f'KGAE Negative Modes Split {split}', fontsize=16)
    plt.show()  


    data = oae.encoder( torch.tensor(all_data_values).to(oae.device).float() ).cpu().detach()

    # unsmoothed power spectra
    dx = 1
    freqs = torch.fft.rfftfreq(data.shape[0], d=dx)
    centered = data - data.mean(dim=0)
    coeffs = torch.fft.rfft(centered, dim=0)
    unsmoothed_power = (coeffs * torch.conj(coeffs)).real
    unsmoothed_power = unsmoothed_power / unsmoothed_power.sum(dim=0)

    # smoothed power spectra using the implemented function
    smoothed_power, smoothed_freqs = kgae.compute_smoothed_power_spectra(data, dx=dx, kernel_size=oae.power_spectrum_smoothing_kernel)

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


    tendencies = kgae.finite_latent_tendency(oae, all_data_values, delta=1e-2, device=oae.device)

    global_tendency_mean = tendencies.mean(dim=0).detach().cpu().numpy()
    global_tendency_mean = all_data.isel(time=slice(None, oae.latent_dim)).copy(data=global_tendency_mean).rename({'time': 'mode'}).assign_coords({'mode': np.arange(1, oae.latent_dim+1) })
    global_tendency_mean = global_tendency_mean.unstack('feature').sortby(['mode', 'lat', 'lon']) 

    global_tendency_var = tendencies.std(dim=0).detach().cpu().numpy() #tendencies.max(dim=0)[0].detach().cpu().numpy() - tendencies.min(dim=0)[0].detach().cpu().numpy()
    global_tendency_var = all_data.isel(time=slice(None, oae.latent_dim)).copy(data=global_tendency_var).rename({'time': 'mode'}).assign_coords({'mode': np.arange(1, oae.latent_dim+1) })
    global_tendency_var = global_tendency_var.unstack('feature').sortby(['mode', 'lat', 'lon']) 

    n_modes = global_tendency_mean.sizes['mode']

    fig, axes = plt.subplots(2, n_modes, figsize=(4 * n_modes, 8),
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    # compute color scales
    maxabs_mean = float(np.max(np.abs(global_tendency_mean.values)))
    vmin_var = float(global_tendency_var.min().values)
    vmax_var = float(global_tendency_var.max().values)

    mean_handles = []
    var_handles = []

    for i in range(n_modes):
        ax_mean = axes[0, i]
        ax_var = axes[1, i]

        im_mean = global_tendency_mean.isel(mode=i).plot(
            ax=ax_mean, cmap='RdBu_r',  add_colorbar=False
        )
        ax_mean.coastlines()
        ax_mean.set_title(f"Mode {i+1} — Mean")

        im_var = global_tendency_var.isel(mode=i).plot(
            ax=ax_var, cmap='viridis', add_colorbar=False
        )
        ax_var.coastlines()
        ax_var.set_title(f"Mode {i+1} — Var")

        mean_handles.append(im_mean)
        var_handles.append(im_var)

    # shared colorbars for each row
    fig.colorbar(mean_handles[0], ax=axes[0, :].tolist(), orientation='vertical', label='Mean')
    fig.colorbar(var_handles[0], ax=axes[1, :].tolist(), orientation='vertical', label='Var')

    plt.suptitle("Global Tendency — Mean (top) and Var (bottom)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

