import xarray as xr
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import kgae
import cartopy.crs as ccrs 
from pathlib import Path 

experiment = Path('kgvae_cv1940-2004')
n_ensemble = 3
thresh = 0.5
data_set = 'xval'
modes_to_examine = [1, 2, 3, 4, 5]
mode_labels = ["HF1", "HF2", "Quasibiennial (i=1)", "Interannual (i=2)", "Decadal (i=3)", ]
need_to_compute_composites = True 

alphas, betas = 0, 0
alpha_sigs, beta_sigs = 0, 0 
for seed in range(n_ensemble):
    alpha = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_alpha.nc')
    alpha = alpha['__xarray_dataarray_variable__']
   
    alpha_sig = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_alpha_sig{"s" if data_set != "xval" else ""}.nc')
    alpha_sig = alpha_sig['__xarray_dataarray_variable__']

    beta = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_beta.nc')
    beta = beta['__xarray_dataarray_variable__']
   
    beta_sig = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_beta_sig{"s" if data_set != "xval" else ""}.nc')
    beta_sig = beta_sig['__xarray_dataarray_variable__']

    alphas += alpha
    betas += beta
    alpha_sigs += alpha_sig
    beta_sigs += beta_sig

alphas = alphas / n_ensemble 
betas = betas / n_ensemble 
alpha_sigs = alpha_sigs / n_ensemble 
beta_sigs = beta_sigs / n_ensemble

alphas = kgae.smooth_spatial(alphas, sigma_latlon=1.5)
betas = kgae.smooth_spatial(betas,  sigma_latlon=1.5)

kgae.plot_alpha_beta_maps(
    alphas, 
    betas,
    xr.ones_like(alpha_sigs).where(alpha_sigs > thresh, other=0),
    xr.ones_like(beta_sigs).where(beta_sigs > thresh, other=0),
    modes_to_examine, 
    mode_labels, 
    n_contours=11, 
    height = 0.1
)



if need_to_compute_composites:
    sst = xr.open_dataset(os.path.expanduser("~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc")).sst
    end_year = int(str(experiment)[-4:])
    sst = sst.sel(time=slice(None, pd.Timestamp(end_year, 12, 31)))
    sst = sst.rename({k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in sst.dims})

    ssta, fit, gwm, p = kgae.global_detrend(sst, deg=2)
    ssta, mc = kgae.remove_climo(ssta)
    ssta = ssta.transpose("time", "lat", "lon")

    
    alpha_composites, beta_composites = 0, 0
    alpha_sigs, beta_sigs = 0, 0 
    for seed in range(n_ensemble):
        print('Doing Composites for seed', seed)
        xval_latents = xr.open_dataset(experiment / f'seed{seed}' / 'xval_latent.nc')
        xval_latents = xval_latents['__xarray_dataarray_variable__']

        modes_to_examine = [1, 2, 3, 4, 5]
        mode_labels = ["HF1", "HF2", "Quasibiennial (i=1)", "Interannual (i=2)", "Decadal (i=3)", ]

        alpha_composite, beta_composite, alpha_sig_composite, beta_sig_composite = kgae.conditional_composite_alpha_beta(
            D=ssta.sel(time = xval_latents.time),
            Z=xval_latents.sel(mode=modes_to_examine),
            sample_dim='time',
            latent_dim='mode',
            lower_q=0.33,
            upper_q=0.67,
            n_bootstrap=100,
            ci_level=95
        )
        alpha_composites += alpha_composite
        beta_composites += beta_composite
        alpha_sigs += alpha_sig_composite
        beta_sigs += beta_sig_composite

    alpha_composites = alpha_composites / n_ensemble 
    beta_composites = beta_composites / n_ensemble 
    alpha_sigs = alpha_sigs / n_ensemble 
    beta_sigs = beta_sigs / n_ensemble

else: 
    alpha_composites = xr.open_dataset('ensemble_bootstrap_composites_alpha.nc')
    alpha_composites = alpha_composites['__xarray_dataarray_variable__']

    alpha_sigs = xr.open_dataset('ensemble_bootstrap_composites_alpha_sig.nc')
    alpha_sigs = alpha_sigs['__xarray_dataarray_variable__']

    beta_composites = xr.open_dataset('ensemble_bootstrap_composites_beta.nc')
    beta_composites = beta_composites['__xarray_dataarray_variable__']
    
    beta_sigs = xr.open_dataset('ensemble_bootstrap_composites_beta_sig.nc')
    beta_sigs = beta_sigs['__xarray_dataarray_variable__']



kgae.plot_alpha_beta_maps(
    alpha_composites, 
    beta_composites,
    xr.ones_like(alpha_sigs).where(alpha_sigs > thresh, other=0),
    xr.ones_like(beta_sigs).where(beta_sigs > thresh, other=0),
    modes_to_examine, 
    mode_labels, 
    n_contours=11, 
    height = 0.1
)







