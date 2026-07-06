import xarray as xr
import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import kgae
import cartopy.crs as ccrs 
from pathlib import Path 

experiment = Path('kgvae_cv1940-2014')
n_ensemble = 30
modes_to_examine = [1, 2, 3, 4, 5]
mode_labels = ["HF1", "HF2", "Quasibiennial (i=1)", "Interannual (i=2)", "Decadal (i=3)", ]

def open_sensitivities(data_set, n_ensemble=30):
    alphas, betas = [], []
    alpha_sigs, beta_sigs = [], [] 
    for seed in range(n_ensemble):
        alpha = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_alpha.nc')
        alpha = alpha['__xarray_dataarray_variable__']
    
        alpha_sig = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_alpha_sig{"s" if data_set not in ["xval", 'test'] else ""}.nc')
        alpha_sig = alpha_sig['__xarray_dataarray_variable__']

        beta = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_beta.nc')
        beta = beta['__xarray_dataarray_variable__']
    
        beta_sig = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_beta_sig{"s" if data_set not in ["xval", 'test'] else ""}.nc')
        beta_sig = beta_sig['__xarray_dataarray_variable__']

        alphas.append(alpha)
        betas.append(beta)
        alpha_sigs.append(alpha_sig)
        beta_sigs.append(beta_sig)
    

    alphas = xr.concat(alphas, 'seed')
    betas = xr.concat(betas, 'seed')
    alpha_sigs = xr.concat(alpha_sigs, 'seed')
    beta_sigs = xr.concat(beta_sigs, 'seed')

    return alphas, betas, alpha_sigs, beta_sigs

def linregress_with_r2(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 5:
        return np.nan, np.nan, np.nan

    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return m, b, r2

def regress_across_seeds(train, val, sig=None):
    """
    train, val: (point, seed) arrays
    sig: (point, seed) boolean or None
    """
    slopes, intercepts, r2s = [], [], []

    for s in range(train.shape[1]):
        x = train[:, s]
        y = val[:, s]

        if sig is not None:
            m = sig[:, s]
            x = x[m]
            y = y[m]

        m_, b_, r2_ = linregress_with_r2(x, y)
        slopes.append(m_)
        intercepts.append(b_)
        r2s.append(r2_)

    return np.array(slopes), np.array(intercepts), np.array(r2s)


train_a, train_b, train_a_sig, train_b_sig = open_sensitivities('train', n_ensemble=n_ensemble)
val_a, val_b, val_a_sig, val_b_sig = open_sensitivities('val', n_ensemble=n_ensemble)

train_a_stacked = (
    train_a
    .stack(point=('lat', 'lon', 'fold'))
    .transpose('point', 'seed', 'tendency_mode')
)

val_a_stacked = (
    val_a
    .stack(point=('lat', 'lon', 'fold'))
    .transpose('point', 'seed', 'tendency_mode')
)

train_a_sig_stacked = ( 
    train_a_sig
    .stack(point=('lat', 'lon', 'fold'))
    .transpose('point', 'seed', 'tendency_mode')
)

kgae.plot_geometry_cv(
    train_a_stacked, val_a_stacked, sig=train_a_sig_stacked,
    geometry='alpha',
    tendency_modes=modes_to_examine,
    mode_labels=mode_labels
)

train_b_stacked = (
    train_b
    .stack(point=('lat', 'lon', 'fold'))
    .transpose('point', 'seed', 'tendency_mode', 'predictor_mode')
)

val_b_stacked = (
    val_b
    .stack(point=('lat', 'lon', 'fold'))
    .transpose('point', 'seed', 'tendency_mode', 'predictor_mode')
)

train_b_sig_stacked = (
    train_b_sig
    .stack(point=('lat', 'lon', 'fold'))
    .transpose('point', 'seed', 'tendency_mode', 'predictor_mode')
)

kgae.plot_geometry_cv(
    train_b_stacked, val_b_stacked, sig=train_b_sig_stacked,
    geometry='beta',
    tendency_modes=modes_to_examine,
    predictor_modes=train_b.predictor_mode.values,
    mode_labels=mode_labels
)
