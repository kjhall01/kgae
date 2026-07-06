import xarray as xr 
from pathlib import Path


def open_alphabetas(data_set, n_ensemble=30, experiment='kgvae_cv1940-2014'):
    experiment = Path(experiment)
    alphas, betas = [], []
    for seed in range(n_ensemble):
        alpha = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_alpha.nc')
        alpha = alpha['__xarray_dataarray_variable__']

        beta = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_beta.nc')
        beta = beta['__xarray_dataarray_variable__']
    
        alphas.append(alpha)
        betas.append(beta)

    alphas = xr.concat(alphas, 'seed')
    betas = xr.concat(betas, 'seed')

    return alphas, betas

def open_references(data_set, experiment='kgvae_cv1940-2014'):
    experiment = Path(experiment)

    ref = xr.open_dataset(experiment / f'{data_set}_references.nc')
    ref = ref['__xarray_dataarray_variable__']
    return ref 

def open_latents(data_set, n_ensemble=30, experiment='kgvae_cv1940-2014'):
    experiment = Path(experiment)

    latents1 = []
    for seed in range(n_ensemble):
        latents = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_latent{"s" if data_set == "test" else ""}.nc')
        latents = latents['__xarray_dataarray_variable__']
        latents1.append(latents)
    return xr.concat(latents1, 'seed')

def open_reconstructions(data_set, ensemble_member=None, n_ensemble=30, experiment='kgvae_cv1940-2014'):
    experiment = Path(experiment)

    if ensemble_member is None: 
        latents1 = []
        for seed in range(n_ensemble):
            latents = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_reconstructions.nc')
            latents = latents['__xarray_dataarray_variable__']
            latents1.append(latents)
        return xr.concat(latents1, 'seed')
    else: 
        seed = ensemble_member
        latents = xr.open_dataset(experiment / f'seed{seed}' / f'{data_set}_reconstructions.nc')
        latents = latents['__xarray_dataarray_variable__']
        return latents 
    


