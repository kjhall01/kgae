import xarray as xr 
from pathlib import Path 

def open_references(data_set, experiment='kgvae_cv1940-2014'):
    experiment = Path(experiment)

    ref = xr.open_dataset(experiment / f'{data_set}_references.nc')
    ref = ref['__xarray_dataarray_variable__']
    return ref 



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
    
if __name__ == "__main__":

    references = open_references('xval', '../../final_scripts/large-ensemble-2-1940-2014')
    refs_test = open_references('test', '../../final_scripts/large-ensemble-2-1940-2014')
    references = xr.concat([references, refs_test], 'time').sortby('time')

    splits = ['1982-1', '1986-3', '1990-5', '1994-8', '1998-10', '2003-1', '2007-3', '2011-5', '2015-8']

    splt_xval = []
    splt_pca = []
    for split in splits: 
        print(split)
        var_frac = 0
        var_frac_pca = 0
        for seed in range(5):
            for fold in range(5): 

                validation = xr.open_dataset(f'../../final_scripts/progressive-split/{split}/seed{seed}/fold{fold}/val_reconstructions.nc')['__xarray_dataarray_variable__']
                variance_fraction = validation.var('time').sum(['lat', 'lon']) / references.sel(time=validation.time).var('time').sum(['lat', 'lon'])
                var_frac += variance_fraction

                if seed ==0: 
                    validation = xr.open_dataset(f'../../final_scripts/progressive-split/{split}/seed{seed}/fold{fold}/pca_val_reconstructions.nc')['sst']
                    variance_fraction = validation.var('time').sum(['lat', 'lon']) / references.sel(time=validation.time).var('time').sum(['lat', 'lon'])
                    var_frac_pca += variance_fraction

        var_frac_pca /= 5
        var_frac /= 25 
        splt_xval.append(var_frac)
        splt_pca.append(var_frac_pca)

    import matplotlib.pyplot as plt 

    plt.plot(splits, splt_xval, label='KGAE')
    plt.plot(splits, splt_pca, label='PCA')
    ax = plt.gca() 
    ax.set_xlabel('Split date (Training End)')
    ax.set_ylabel('Variance Fraction')
    plt.legend()
    plt.savefig('variance_fraction.pdf')
    plt.show() 

