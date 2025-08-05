import xarray as xr 
from pathlib import Path 
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import src 
import pandas as pd 
from sklearn.decomposition import PCA 

N=50
run='basin.goodtest'
recursion2 = 0 # 0 for first recursion, 1 for second recursion, etc.
nrecursions = 3 
results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

tc3 = []
for init in range(N):
    tc2 = []
    for split in range(5):
        tc = []
        for recursion in range(nrecursions):
            encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'val_data.encodings.nc').encodings
            tc.append(encodings)
        tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(nrecursions)})
        tc2.append(tc)
    tc2 = xr.concat(tc2, 'time').sortby('time')
    tc3.append(tc2)
aencodings = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).isel(recursion=recursion2)

print(aencodings)
tc3 = []
for init in range(N):
    tc2 = 0
    for split in range(5):
        tc = []
        for recursion in range(nrecursions):
            encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'test_data.encodings.nc').encodings
            tc.append(encodings)
        tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(nrecursions)})
        tc2 = tc2 + tc # .append(tc)
   # tc2 = xr.concat(tc2, 'time').sortby('time')
    tc3.append(tc2 / 5)
test_encodings = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).isel(recursion=recursion2)
print(test_encodings)
full_encodings = xr.concat([aencodings, test_encodings], 'time')

print(full_encodings)
titles = [ 'Decadal', "Interannual", "Quasibiennial"] #"HF1", 'HF2', 
# plot AE-modes and uncertainties 
import matplotlib.gridspec as gridspec



nvert, nhori = 100, 9
fig = plt.figure(figsize=(8,10))
gs = gridspec.GridSpec(nvert, nhori)



timeseries_axs = [ fig.add_subplot(gs[35:45, :]), fig.add_subplot(gs[46:56, :]), fig.add_subplot(gs[57:67, :]) ]
letters = ['d)', 'e)', 'f)']
for i in range(3):
    main_line = timeseries_axs[i].plot(full_encodings.time, full_encodings.sel(mode=titles[i]).mean('initialization').values, linewidth=1, color='k')
    top_line = timeseries_axs[i].plot(full_encodings.time, full_encodings.sel(mode=titles[i]).quantile(0.975, 'initialization').values, linewidth=0.5, color='k', alpha=0.5)
    bottom_line = timeseries_axs[i].plot(full_encodings.time, full_encodings.sel(mode=titles[i]).quantile(0.025, 'initialization').values, linewidth=0.5, color='k', alpha=0.5)
    timeseries_axs[i].fill_between(full_encodings.time, full_encodings.sel(mode=titles[i]).quantile(0.025, 'initialization').values, full_encodings.sel(mode=titles[i]).quantile(0.975, 'initialization').values,  alpha=0.3, color='gray')
    timeseries_axs[i].set_ylabel(letters[i], rotation='horizontal', fontweight='bold')
    timeseries_axs[i].set_yticks([-3,0,3])
    timeseries_axs[i].axhline(0, color='k', linestyle='--', linewidth=0.5)

    timeseries_axs[i].spines['right'].set_color('none')
    timeseries_axs[i].spines['top'].set_color('none')   
    if i < 2: 
        timeseries_axs[i].spines['bottom'].set_color('none')
        timeseries_axs[i].set_xticks([])
        #timeseries_axs[i].spines['left'].set_color('none')


powerspectra_ax = fig.add_subplot(gs[75:, :])

sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst#.sel(time=slice(None, pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})
ssta, trend, gwm, p = src.global_detrend(sst,deg=2)
ssta, monthly_clim = src.remove_climo(ssta)
print(ssta)

era5_oni = ssta.sel(lat=slice(-5,5), lon=slice(10,60)).mean('lat').mean('lon').rolling(time=3, center=True).mean().dropna('time')

pcadata = ssta.stack(feature=('lat',"lon")).dropna("feature", how="any").transpose("time", "feature").values
pca = PCA(n_components=3)
pca.fit( pcadata )
raw_pc1 = pca.transform(pcadata)[:,0]


lpf_pcadata = src.low_pass(ssta, threshold=6*12*30.4*86400).load().stack(feature=('lat',"lon")).dropna("feature", how="any").transpose("time", "feature").values
lpf_pca = PCA(n_components=3)
lpf_pca.fit( lpf_pcadata )
lpf_pc1 = lpf_pca.transform(lpf_pcadata)[:,0]
lpfpca_to_raw_pc1 = lpf_pca.transform(pcadata)[:,0]


autoencoder_data = full_encodings.sel(mode=titles).mean('initialization').transpose('time', 'mode').values
titles = ["Decadal", "Interannual", "Quasibiennial", 'UF-PC1', 'LPF-PC1', 'UF-LPF-PC1', 'ERA5 ONI']
data = np.hstack([autoencoder_data, raw_pc1.reshape(-1,1), lpf_pc1.reshape(-1,1), lpfpca_to_raw_pc1.reshape(-1,1)  ])
data = torch.from_numpy(data)
data = data - data.mean(dim=0)
frequencies = torch.fft.rfftfreq(data.shape[0], d=1) #[1:]
fourier_coeffs = torch.fft.rfft(data, dim=0)
fourier_coeffs = (fourier_coeffs*torch.conj(fourier_coeffs)).real#[mask, :]
fc_scale = fourier_coeffs.sum(dim=0) #max(dim=0)[0]
fourier_coeffs = fourier_coeffs / fc_scale
fourier_coeffs, frequencies = src.deniell(fourier_coeffs, frequencies, m_for_deniell_smoothing=7)
fourier_coeffs = fourier_coeffs / fourier_coeffs.sum(dim=0)

ticks= 40, 20, 12, 7, 5, 2, 1 
powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,0].detach().numpy(),  label="Decadal")
powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,1].detach().numpy(),  label="Interannual")
powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,2].detach().numpy(),  label="Quasibiennial")
#powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,3].detach().numpy(), color='gray', linestyle='--', alpha=0.5, label=titles[3])
#powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,4].detach().numpy(), color='gray', linestyle=':', alpha=0.5, label=titles[4])
#powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,5].detach().numpy(), color='black', linestyle='--', alpha=0.6, label=titles[5])



data = era5_oni.values.reshape(-1,1) 
data = torch.from_numpy(data)
data = data - data.mean(dim=0)
frequencies = torch.fft.rfftfreq(data.shape[0], d=1) #[1:]
fourier_coeffs = torch.fft.rfft(data, dim=0)
fourier_coeffs = (fourier_coeffs*torch.conj(fourier_coeffs)).real#[mask, :]
fc_scale = fourier_coeffs.sum(dim=0) #max(dim=0)[0]
fourier_coeffs = fourier_coeffs / fc_scale
fourier_coeffs, frequencies = src.deniell(fourier_coeffs, frequencies, m_for_deniell_smoothing=7)
fourier_coeffs = fourier_coeffs / fourier_coeffs.sum(dim=0)
#powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,0].detach().numpy(), color='gray', linestyle='-.', alpha=0.5, label=titles[6])


powerspectra_ax.set_xscale('log', base=2)
powerspectra_ax.set_xticks([1/(innn*12) for innn in ticks ],labels=ticks)
powerspectra_ax.legend(loc='upper right', fontsize=8, ncols=2)
powerspectra_ax.set_ylabel("Density " + r'$\left(\frac{E_{K}^{2}}{{\Sigma}E_{K}^{2}}\right)$', size=12)
powerspectra_ax.set_xlabel('Period (years)')
powerspectra_ax.set_title('g)',loc='left', fontweight='bold')
powerspectra_ax.grid()
#plt.tight_layout()
powerspectra_ax.spines['right'].set_color('none')
powerspectra_ax.spines['top'].set_color('none') 






map_axs = [ fig.add_subplot(gs[:30, :3], projection=ccrs.PlateCarree(central_longitude=180)), fig.add_subplot(gs[:30, 3:6], projection=ccrs.PlateCarree(central_longitude=180)), fig.add_subplot(gs[:30, 6:], projection=ccrs.PlateCarree(central_longitude=180)) ] 

coeff_da, r2_da, residuals_da, p_values_da = src.multivariate_linear_regression_with_significance(autoencoder_data, ssta.transpose('time', 'lon', 'lat'))


map_axs = [ fig.add_subplot(gs[:30, :3], projection=ccrs.PlateCarree(central_longitude=180)), fig.add_subplot(gs[:30, 3:6], projection=ccrs.PlateCarree(central_longitude=180)), fig.add_subplot(gs[:30, 6:], projection=ccrs.PlateCarree(central_longitude=180)) ] 

titles = ['Decadal', 'Interannual', 'Quasibiennial']
coeff_da = coeff_da.assign_coords({'mode': ['intercept', 'Decadal', 'Interannual', 'Quasibiennial']})
coeff_da_coefs = coeff_da.sel(mode=titles) * autoencoder_data.std(axis=0)
letters = ['a)', 'b)', 'c)']
for i , title in enumerate(titles):

    p = coeff_da_coefs.sel(mode=title).plot(
            ax=map_axs[i],
            cmap='RdBu_r', vmin=-0.5, vmax=0.5, 
            add_colorbar=False
        )
    map_axs[i].set_title(None)

    map_axs[i].set_title(letters[i], loc='left', fontweight='bold')
    map_axs[i].coastlines()
    map_axs[i].contourf(ssta.lon, ssta.lat, p_values_da.isel(mode=i+1).values < 0.05, 1, hatches=['','....'], alpha=0)

cbar_ax = fig.add_subplot(gs[28:29,1:-1])
fig.colorbar(p, cax=cbar_ax, **{'label': 'SST Anomaly (K)', 'orientation': 'horizontal', 'shrink': 0.2, 'extend': 'both'})
plt.savefig('/Users/kylehall/Desktop/hall_molina_2025_final_figures/hall_molina_2025.figure1.png', dpi=1000)
plt.show()