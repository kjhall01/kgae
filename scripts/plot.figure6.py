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
run='tropical'
def get_recursion_decoding(recursion1):
    results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

    tc3 = 0
    for init in range(N):
        tc2 = []
        for split in range(5):
            tc4 = []
            for recursion in [recursion1]:
                dencodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}/val_data.decoded.nc')
                dencodings = getattr(dencodings, '__xarray_dataarray_variable__')
                tc4.append(dencodings)
            tc4 = xr.concat(tc4, 'recursion').assign_coords({'recursion': np.arange(1)})
            tc2.append(tc4)
        tc2 = xr.concat(tc2, 'time').sortby('time')
        tc3 = tc3 + tc2
        #tc3.append(tc2)
    tc3 = tc3 / N #xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).drop('number')
    return tc3.mean('recursion')

results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')
tosel = ['Decadal', 'Interannual', 'Quasibiennial']
tc3 = []
for init in range(N):
    tc2 = []
    for split in range(5):
        tc = []
        for recursion in range(3):
            encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'val_data.encodings.nc').encodings
            tc.append(encodings)
        tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(3)})
        tc2.append(tc)
    tc2 = xr.concat(tc2, 'time').sortby('time')
    tc3.append(tc2)
full_encodings = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).sel(mode=tosel)

tc3 = []
for init in range(N):
    tc2 = 0
    for split in range(5):
        tc = []
        for recursion in range(3):
            encodings = xr.open_dataset(results_base / f'rs{init}' / f'split{split}' / f'recursion{recursion}'/ 'test_data.encodings.nc').encodings
            tc.append(encodings)
        tc = xr.concat(tc, 'recursion').assign_coords({'recursion': np.arange(3)})
        tc2 = tc2 + tc # .append(tc)
   # tc2 = xr.concat(tc2, 'time').sortby('time')
    tc3.append(tc2 / 5)
test_encodings = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).sel(mode=tosel)
#full_encodings = xr.concat([aencodings, test_encodings], 'time')

titles = ['Primary Decadal Mode', "Tertiary Decadal Mode"]
# plot AE-modes and uncertainties 
import matplotlib.gridspec as gridspec



nvert, nhori = 100, 10
fig = plt.figure(figsize=(8,12))
gs = gridspec.GridSpec(nvert, nhori)



timeseries_axs = [ fig.add_subplot(gs[37:45, :]), fig.add_subplot(gs[46:54, :]) ]
letters = ['c)', 'd)']
for i, ndx in enumerate([0,2]):
    main_line = timeseries_axs[i].plot(full_encodings.time, full_encodings.isel(recursion=ndx).sel(mode='Decadal').mean('initialization').values, linewidth=1, color='k')
    top_line = timeseries_axs[i].plot(full_encodings.time, full_encodings.isel(recursion=ndx).sel(mode='Decadal').quantile(0.975, 'initialization').values, linewidth=0.5, color='k', alpha=0.5)
    bottom_line = timeseries_axs[i].plot(full_encodings.time, full_encodings.isel(recursion=ndx).sel(mode='Decadal').quantile(0.025, 'initialization').values, linewidth=0.5, color='k', alpha=0.5)
    timeseries_axs[i].fill_between(full_encodings.time, full_encodings.isel(recursion=ndx).sel(mode='Decadal').quantile(0.025, 'initialization').values, full_encodings.isel(recursion=ndx).sel(mode='Decadal').quantile(0.975, 'initialization').values,  alpha=0.3, color='gray')
    timeseries_axs[i].set_ylabel(letters[i], rotation='horizontal', fontweight='bold')
    timeseries_axs[i].set_yticks([-3,0,3])
    timeseries_axs[i].axhline(0, color='k', linestyle='--', linewidth=0.5)

    timeseries_axs[i].spines['right'].set_color('none')
    timeseries_axs[i].spines['top'].set_color('none')   
    if i < 1: 
        timeseries_axs[i].spines['bottom'].set_color('none')
        timeseries_axs[i].set_xticks([])
        #timeseries_axs[i].spines['left'].set_color('none')


powerspectra_ax = fig.add_subplot(gs[85:, :])

autoencoder_data = full_encodings.isel(recursion=0).mean('initialization').transpose('time', 'mode').values
autoencoder_data2 = full_encodings.isel(recursion=2).mean('initialization').transpose('time', 'mode').values

titles = ["DM1", "DM3", "AEQM",'UF-PC1', 'LPF-PC1', 'UF-LPF-PC1', 'ERA5 ONI']
data = autoencoder_data #np.hstack([autoencoder_data, raw_pc1.reshape(-1,1), lpf_pc1.reshape(-1,1), lpfpca_to_raw_pc1.reshape(-1,1)  ])
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
powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,0].detach().numpy(),  label="Primary")
data = autoencoder_data2 #np.hstack([autoencoder_data, raw_pc1.reshape(-1,1), lpf_pc1.reshape(-1,1), lpfpca_to_raw_pc1.reshape(-1,1)  ])
data = torch.from_numpy(data)
data = data - data.mean(dim=0)
frequencies = torch.fft.rfftfreq(data.shape[0], d=1) #[1:]
fourier_coeffs = torch.fft.rfft(data, dim=0)
fourier_coeffs = (fourier_coeffs*torch.conj(fourier_coeffs)).real#[mask, :]
fc_scale = fourier_coeffs.sum(dim=0) #max(dim=0)[0]
fourier_coeffs = fourier_coeffs / fc_scale
fourier_coeffs, frequencies = src.deniell(fourier_coeffs, frequencies, m_for_deniell_smoothing=7)
fourier_coeffs = fourier_coeffs / fourier_coeffs.sum(dim=0)



powerspectra_ax.plot(frequencies.detach().numpy(), fourier_coeffs[:,0].detach().numpy(),  label="Tertiary")

powerspectra_ax.set_xscale('log', base=2)
powerspectra_ax.set_xticks([1/(innn*12) for innn in ticks ],labels=ticks)
powerspectra_ax.legend(loc='upper right', fontsize=8, ncols=2)
powerspectra_ax.set_ylabel("Density " + r'$\left(\frac{E_{K}^{2}}{{\Sigma}E_{K}^{2}}\right)$', size=12)
powerspectra_ax.set_xlabel('Period (years)')
powerspectra_ax.set_title('f)',loc='left', fontweight='bold')
powerspectra_ax.grid()
#plt.tight_layout()
powerspectra_ax.spines['right'].set_color('none')
powerspectra_ax.spines['top'].set_color('none') 




sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(None, pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'})
ssta, trend, gwm, p = src.global_detrend(sst,deg=2)
ssta, monthly_clim = src.remove_climo(ssta)


map_axs = [ fig.add_subplot(gs[:30, :5], projection=ccrs.PlateCarree(central_longitude=180)), fig.add_subplot(gs[:30, 5:], projection=ccrs.PlateCarree(central_longitude=180))] 
letters1 = ['a)', 'b)']
vmax=1.5
for ndx, recursion in enumerate([0, 2]):
    # plot positive composite
    title ='Decadal'
    tf = ['Interannual', 'Quasibiennial']
    tc2 = full_encodings.isel(recursion=recursion)#.mean('initialization')
    #if recursion == 0:
    #    ssta = ssta
    #elif recursion == 1:
    #    ssta = ssta - get_recursion_decoding(0)
    #elif recursion == 2:
    #    ssta = ssta - get_recursion_decoding(0)
    #    ssta = ssta - get_recursion_decoding(1)

    pos_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) > tc2.mean('initialization').sel(mode=title).std('time') ) &
        ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
        & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
    n = int(pos_composite_selection.sum() .values)
    print('samples in ', n , np.sum(pos_composite_selection | ~pos_composite_selection))
    pos_composite = ssta.isel(time=pos_composite_selection).mean('time') 

    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
        samp_mean = ssta.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(0.95, 'samps')
    lower = tc1.quantile(0.05, 'samps')
    pos_significance_mask = (pos_composite > upper) | (pos_composite < lower)


    #p = pos_composite.plot(ax=map_axs[ndx], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
   # c = map_axs[ndx].coastlines() 


    # plot negative composite
    neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) < -1*tc2.mean('initialization').sel(mode=title).std('time') ) &
        ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
        & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
    n = int(neg_composite_selection.sum() .values)
    print('samples in ', n , np.sum(neg_composite_selection | ~neg_composite_selection))
    neg_composite = ssta.isel(time=neg_composite_selection).mean('time') 

    # take 1000 times the mean of n samples of ssta with replacement 
    tc1 = [] 
    for ii in range(1000):
        if ii %100 == 0:
            print(ii/100)
        random_selection = np.random.randint(0, ssta.time.shape[0], size=n)
        samp_mean = ssta.isel(time=random_selection).mean('time')
        tc1.append(samp_mean)
    tc1 = xr.concat(tc1, 'samps')

    upper = tc1.quantile(0.95, 'samps')
    lower = tc1.quantile(0.05, 'samps')
    neg_significance_mask = (neg_composite > upper) | (neg_composite < lower)


    #p = neg_composite.plot(ax=ax[1,i], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #c = ax[1,i].coastlines() 
    #ax[1, i].set_title(None)

    #ax[0,i].set_title(letters0[i], fontweight='bold', loc='left')
    #if i ==0:
    #    decadal_linear_response = (pos_composite - neg_composite)
   # if recursion == 0: 
   #     p = (pos_composite - neg_composite).sel(lat=slice(-20,20)).plot(ax=map_axs[ndx], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
   # else:
    p = (pos_composite - neg_composite).plot(ax=map_axs[ndx], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #map_axs[ndx].contourf(ssta.lon, ssta.lat, pos_significance_mask & neg_significance_mask, 1, hatches=['','....'], alpha=0)

    # Plot black rectangle from 20S to 20N, 120E to 80W (or full longitude range)
    # Define rectangle from 120E to 80W (i.e., 120 to 280 in 0-360 system), 20S to 20N
    rect_lon_min, rect_lon_max = 101, 289  # 120E to 80W (in 0-360)
    rect_lat_min, rect_lat_max = -20, 20
    rect_lons = [rect_lon_min, rect_lon_max, rect_lon_max, rect_lon_min, rect_lon_min]
    rect_lats = [rect_lat_min, rect_lat_min, rect_lat_max, rect_lat_max, rect_lat_min]
    map_axs[ndx].plot(rect_lons, rect_lats, color='k', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=10)

    # Plot a small 'x' over the Galapagos Islands (~ -90 longitude, 0 latitude)
    galapagos_lon, galapagos_lat = -90, 0
    map_axs[ndx].plot(galapagos_lon, galapagos_lat, marker='x', color='k', markersize=6, markeredgewidth=2, transform=ccrs.PlateCarree(), zorder=11)

   # map_axs[ndx].contourf(ssta.lon, ssta.lat, neg_significance_mask  & pos_significance_mask, 1, hatches=['', '....'], alpha=0, zorder=2)
    map_axs[ndx].set_title(None)
    map_axs[ndx].set_title(letters1[ndx], fontweight='bold', loc='left')
    c = map_axs[ndx].coastlines() 
    #ax[2, i].set_title(None)
    #ax[2, i].set_title(letters2[i], fontweight='bold', loc='left')

    #p = (pos_composite + neg_composite).plot(ax=ax[3,i], cmap='RdBu_r', vmin=-vmax, vmax=vmax, add_colorbar=False)
    #c = ax[3,i].coastlines() 
    #ax[3, i].set_title(None)
    #ax[3, i].set_title(letters3[i], fontweight='bold', loc='left')


cbar_ax = fig.add_subplot(gs[28:29,1:-1])
fig.colorbar(p, cax=cbar_ax, **{'label': 'SST Anomaly (K)', 'orientation': 'horizontal', 'shrink': 0.2, 'extend': 'both'})
#fplt.savefig('hall_molina_2024.figure1.png', dpi=1000)


lagcor_ax = fig.add_subplot(gs[62:78,:])

nlags=120
levels = np.linspace(-1, 1, 21)
levels2 = np.linspace(-1, 1, 11)
months = ['Jan', 'Feb', 'Mar', "Apr", 'May', "Jun", 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#corrs, sigs = src.crosscorrelation_by_month(encodings.mean('initialization').sel(mode='Quasibiennial'), encodings.mean('initialization').sel(mode='Quasibiennial'), nlags=nlags)
corrs, sigs = src.crosscorrelation_by_month(src.low_pass(full_encodings, threshold=12*30.4*86400).isel(recursion=0).mean('initialization').sel(mode='Decadal'), src.low_pass(full_encodings, threshold=12*30.4*86400).isel(recursion=2).mean('initialization').sel(mode='Decadal'), nlags=nlags)
#corrs, sigs = src.crosscorrelation_by_month(full_encodings.isel(recursion=0).mean('initialization').sel(mode='Decadal'), full_encodings.isel(recursion=2).mean('initialization').sel(mode='Decadal'), nlags=nlags)

corrsnan = corrs.copy()
corrsnan[sigs > 0.05] = np.nan
cp = lagcor_ax.contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
cp2 = lagcor_ax.contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
lagcor_ax.clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

lagcor_ax.set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
lagcor_ax.set_xticks(np.arange(-nlags, nlags+1, 12))
# Add bottom left and right text labels
lagcor_ax.text(-nlags, -2, "Tertiary precedes Primary", ha='left', va='top', fontsize=10)
lagcor_ax.text(nlags, -2, "Tertiary follows Primary", ha='right', va='top', fontsize=10)

lagcor_ax.set_title('Primary - Tertiary Lag Cross Correlation')
lagcor_ax.set_title('e)', loc='left', fontweight='bold')
plt.savefig('/Users/kylehall/Desktop/hall_molina_2025_final_figures/hall_molina_2025.figure6.png', dpi=1000)
plt.show()