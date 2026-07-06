import xarray as xr
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import kgae
import cartopy.crs as ccrs 


sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})#.sel(lat=slice(-20,20,None))
ssta, fit, gwm, p  = kgae.global_detrend(sst, deg=2)
ssta, monthly_clim = kgae.remove_climo(ssta)

xval_latents = xr.open_dataset('debug_latents.nc')
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



