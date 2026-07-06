import xarray as xr
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import kgae
import cartopy.crs as ccrs 
from pathlib import Path 

experiment = Path('kgvae_cv1940-2004')
n_ensemble = 10
modes_to_examine = [0,1,2,3,4]
mode_labels = ["HF1", "HF2", "Quasibiennial (i=1)", "Interannual (i=2)", "Decadal (i=3)", ]

def open_latents(data_set, n_ensemble=30):
    latents1 = []
    for seed in range(n_ensemble):
        latents = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_latent.nc')
        latents = latents['__xarray_dataarray_variable__']
        latents1.append(latents)

    return xr.concat(latents1, 'seed')

xval_latents = open_latents('xval', n_ensemble=n_ensemble)
test_latents = open_latents('test', n_ensemble=n_ensemble)


test_latents = (
    test_latents 
    .stack(sample=('seed', 'fold'))
    .transpose('time', 'mode', 'sample')
)

xval_latents = (
    xval_latents 
    .transpose('time', 'mode', 'seed')
)

kgae.plot_mode_histograms(
    xval_latents, test_latents,
    mode_indices=modes_to_examine,
    mode_labels=mode_labels,
    bins=60
)

kgae.plot_joint_histograms(
    xval_latents, test_latents.transpose('time', 'mode', 'sample'),
    mode_indices=(3, 4),
    mode_labels=("Interannual", "Decadal"),
    bins=60
)