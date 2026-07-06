import kgae 
import xarray as xr 
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

experiment= 'kgvae_cv1940-2014' 
n_seeds = 5
block_size = 6
nboots = 20

def block_bootstrap_indices(rng, n, block_len):
        """Generate bootstrap indices with contiguous blocks of length block_len."""
        n_blocks = int(np.ceil(n / block_len))
        block_starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        indices = np.hstack([np.arange(start, start + block_len) for start in block_starts])
        return indices[:n]  # trim to exactly n indices

refs_xval = kgae.open_references('xval', experiment=experiment)
refs_test = kgae.open_references('test', experiment='kgvae_cv2-1940-2014')

recon_mse_xval, recon_mse_test = 0, 0  
recon_mse_gap = 0 
recon_mse_gap_sig = 0 
for seed in range(n_seeds):
    print(seed)
    reconstructions_xval = kgae.open_reconstructions('xval', experiment=experiment, ensemble_member=seed)
    reconstructions_test = kgae.open_reconstructions('test', experiment=experiment, ensemble_member=seed)

    recon_mse_test += ( (refs_test - reconstructions_test)**2 ).mean('time')  / refs_test.var('time')
    recon_mse_xval += ( (refs_xval - reconstructions_xval)**2 ).mean('time') / refs_xval.var('time')
    recon_mse_gap += recon_mse_test - recon_mse_xval # avoid division by zero
    
    rng = np.random.default_rng(seed)
    gaps = []
    for boot in range (nboots):
        print(f"  bootstrap {boot}")
        idx = block_bootstrap_indices(rng, refs_xval.time.size, block_len=block_size)
        recon_xval_b = reconstructions_xval.isel(time=idx)
        ref_xval_b = refs_xval.isel(time=idx)
        xval_mse_b = ( (ref_xval_b - recon_xval_b)**2 ).mean('time') / ref_xval_b.var('time')

        idx = block_bootstrap_indices(rng, refs_test.time.size, block_len=block_size)
        recon_test_b = reconstructions_test.isel(time=idx)
        ref_test_b = refs_test.isel(time=idx)
        test_mse_b = ( (ref_test_b - recon_test_b)**2 ).mean('time') / ref_test_b.var('time')

        gap_b = test_mse_b - xval_mse_b # avoid division by zero
        gaps.append(gap_b)
    gaps = xr.concat(gaps, dim='bootstrap')

    recon_mse_gap_sig +=  gaps.quantile(0.75, 'bootstrap') > 0 # 10% decrease 

recon_mse_xval /= float(n_seeds)
recon_mse_test /= float(n_seeds)
recon_mse_gap /= float(n_seeds)
recon_mse_gap_sig = recon_mse_gap_sig / n_seeds

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
recon_mse_xval.plot(ax=ax[0],  vmin=0, vmax=3, transform=ccrs.PlateCarree(central_longitude=180))
ax[0].coastlines() 
ax[0].set_title("CV MSE")
recon_mse_test.mean('fold').plot(ax=ax[1], vmin=0, vmax=3, transform=ccrs.PlateCarree(central_longitude=180))
ax[1].coastlines()
ax[1].set_title("Test MSE")
(recon_mse_gap.mean('fold') / recon_mse_xval.mean()).where(recon_mse_gap_sig.mean('fold') > 0.50).plot(ax=ax[2], vmin=-0.5, vmax=0.5, cmap='RdBu_r', transform=ccrs.PlateCarree(central_longitude=180))
ax[2].coastlines()
ax[2].set_title("Generalization Gap (Test MSE - CV MSE)")
plt.tight_layout()  
plt.show()


print(recon_mse_xval.mean('lat').mean('lon'))
print(recon_mse_test.mean('lat').mean('lon'))

