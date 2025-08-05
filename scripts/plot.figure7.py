import xarray as xr 
import cartopy.crs as ccrs 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import matplotlib.patches as patches
import src 
import numpy as np 
from pathlib import Path 

N=50
run='tropical'
var = 'sst'
vmax = 1.5
caption = 'SST (K)'
confidence_level = 0.67

results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

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
tc2 = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).isel(recursion=0)
tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))

ssh = xr.open_dataset('~/Desktop/Data/oras5/oras5.ssh.195801-201412.nc').ssh.sel(lon=slice(40, 100))
ssha, fit, gwm, p  = src.global_detrend(ssh, deg=2)
ssha , mc = src.remove_climo(ssha)

u = xr.open_dataset('~/Desktop/Data/oras5/oras5.u.195801-201412.nc').u
u = u.sel(lat=slice(-10,10), depthu=slice(None,500))#.sel(lon=slice(40, 100))
ua, fit, gwm, p  = src.global_detrend(u, deg=2)
ua , mc = src.remove_climo(ua)



nvert, nhori = 110, 9
fig = plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(nvert, nhori)

title = 'Decadal'
tf = ['Interannual', 'Quasibiennial']

ssh_profile_ax = fig.add_subplot(gs[:20, :])
# plot positive composite
pos_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) > tc2.mean('initialization').sel(mode=title).std('time') ) &
    ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
    & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
n = int(pos_composite_selection.sum() .values)
print('samples in ', n , np.sum(pos_composite_selection | ~pos_composite_selection))
pos_composite = ssha.isel(time=pos_composite_selection).mean('time').sel(lat=slice(-1.5, 1.5)).mean('lat').sel(lon=slice(40, 100))#.mean('lon')

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssha.time.shape[0], size=n)
    samp_mean = ssha.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(0.95, 'samps')
lower = tc1.quantile(0.05, 'samps')
pos_significance_mask = (pos_composite > upper) | (pos_composite < lower)

ssh_profile_ax.plot(pos_composite.lon + 180, pos_composite, color='r', linewidth=1, label='Positive Phase')
ssh_profile_ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ssh_profile_ax.axvline(89+180, color='k', linewidth=0.5, linestyle='--')

# plot negative composite
neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) < -1*tc2.mean('initialization').sel(mode=title).std('time') ) &
    ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
    & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
n = int(neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_composite_selection | ~neg_composite_selection))
neg_composite = ssha.isel(time=neg_composite_selection).mean('time') .sel(lat=slice(-1.5, 1.5)).mean('lat').sel(lon=slice(40, 100))

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssha.time.shape[0], size=n)
    samp_mean = ssha.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(0.95, 'samps')
lower = tc1.quantile(0.05, 'samps')
neg_significance_mask = (neg_composite > upper) | (neg_composite < lower)

ssh_profile_ax.plot(neg_composite.lon + 180, neg_composite, color='b', linewidth=1, label='Negative Phase')

ssh_profile_ax.set_title(None)
ssh_profile_ax.set_title('a)', fontweight='bold', loc='left')
decadal_linear_response = (pos_composite - neg_composite)

ssh_profile_ax.plot(neg_composite.lon + 180, (pos_composite - neg_composite) / 2, color='k', linestyle='--', linewidth=1, label='Linear Response')
ssh_profile_ax.plot(neg_composite.lon + 180, (pos_composite + neg_composite) / 2, color='k', linestyle=':', linewidth=1, label='Nonlinear Response')


#ssh_profile_ax.legend(loc='upper left', fontsize=8, ncols=1)
ssh_profile_ax.set_ylabel("SSH Anomaly (m)")
ssh_profile_ax.set_xlabel("Longitude")


###### U Composites #######

# plot positive composite
pos_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) > tc2.mean('initialization').sel(mode=title).std('time') ) &
    ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
    & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
n = int(pos_composite_selection.sum() .values)
print('samples in ', n , np.sum(pos_composite_selection | ~pos_composite_selection))
pos_composite = ua.isel(time=pos_composite_selection).mean('time').sel(lat=slice(-1.5, 1.5)).mean('lat').sel(lon=slice(40, 100))

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssha.time.shape[0], size=n)
    samp_mean = ssha.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(0.95, 'samps')
lower = tc1.quantile(0.05, 'samps')
pos_significance_mask = (pos_composite > upper) | (pos_composite < lower)
#pos_composite = pos_composite * pos_significance_mask



# plot negative composite
neg_composite_selection = ( ( tc2.mean('initialization').sel(mode=title) < -1*tc2.mean('initialization').sel(mode=title).std('time') ) &
    ( np.abs(tc2.mean('initialization').sel(mode=tf[0])) <  tc2.mean('initialization').sel(mode=tf[0]).std('time') ) 
    & ( np.abs(tc2.mean('initialization').sel(mode=tf[1])) < tc2.mean('initialization').sel(mode=tf[1]).std('time') )  )
n = int(neg_composite_selection.sum() .values)
print('samples in ', n , np.sum(neg_composite_selection | ~neg_composite_selection))
neg_composite = ua.isel(time=neg_composite_selection).mean('time') .sel(lat=slice(-1.5, 1.5)).mean('lat').sel(lon=slice(40, 100))

# take 1000 times the mean of n samples of ssta with replacement 
tc1 = [] 
for ii in range(1000):
    if ii %100 == 0:
        print(ii/100)
    random_selection = np.random.randint(0, ssha.time.shape[0], size=n)
    samp_mean = ssha.isel(time=random_selection).mean('time')
    tc1.append(samp_mean)
tc1 = xr.concat(tc1, 'samps')

upper = tc1.quantile(0.95, 'samps')
lower = tc1.quantile(0.05, 'samps')
neg_significance_mask = (neg_composite > upper) | (neg_composite < lower)
#neg_composite = neg_composite * neg_significance_mask


u_ax_deep = fig.add_subplot(gs[30:50, :])
u_ax_deep.axvline(89+180, color='k', linewidth=0.5, linestyle='--')
u_ax_deep.axhline(0, color='k', linewidth=1, linestyle='-')
u_ax_deep.plot(pos_composite.lon+180, pos_composite.sel(depthu=slice(50,100)).mean('depthu'), color='r', linewidth=1, label='Positive Phase') 
u_ax_deep.plot(pos_composite.lon+180, neg_composite.sel(depthu=slice(50,100)).mean('depthu'), color='b', linewidth=1, label='Negative Phase') 

u_ax_deep.set_title(None)
u_ax_deep.set_title('b)', fontweight='bold', loc='left')
u_ax_deep.set_xlabel('Longitude')
u_ax_deep.set_ylabel('EUC Anomaly (m/s)')

u_ax_deep.plot(neg_composite.lon+180, (pos_composite - neg_composite).sel(depthu=slice(50,100)).mean('depthu') , color='k', linestyle='--', linewidth=1, label='Linear Response')
u_ax_deep.plot(neg_composite.lon+180, (pos_composite + neg_composite).sel(depthu=slice(50,100)).mean('depthu') , color='k', linestyle=':', linewidth=1, label='Nonlinear Response')

u_ax_deep.legend(loc='lower center', bbox_to_anchor=(0.5, -0.85), fontsize=8, ncols=2, frameon=True)










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
tc2 = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).isel(recursion=2)
tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))


sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst.sel(time=slice(pd.Timestamp(1958,1,1), pd.Timestamp(2014,12,31)))
sst = sst.rename({'latitude':'lat', 'longitude': 'lon'}) 
ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
ssta , mc = src.remove_climo(ssta)


lat1, lat2 = -2, 2
lon1, lon2 = 88, 90

#ssta_times_ua = ssta.sel(lat=slice(lat1,lat2)).mean('lat').sel(lon=slice(lon1, lon2)).differentiate('lon').mean('lon') / 111000* ua.sel(depthu=slice(None, 10)).mean('depthu').sel(lat=slice(lat1,lat2)).mean('lat').sel(lon=slice(lon1,lon2)).mean('lon')
#ssta_times_ua = ssta.sel(lat=slice(lat1,lat2)).mean('lat').sel(lon=slice(lon1, lon2)).differentiate('lon').mean('lon') / 111000* ua.sel(depthu=slice(None, 50)).mean('depthu').sel(lat=slice(lat1,lat2)).mean('lat').sel(lon=slice(lon1,lon2)).mean('lon')
ssta_times_ua = ssta.sel(lat=slice(lat1,lat2)).mean('lat').sel(lon=slice(lon1, lon2)).mean('lon') 

#### Advection precedes TPDV
lagcor_ax= fig.add_subplot(gs[70:90, :])

nlags=24
levels = np.linspace(-1, 1, 21)
levels2 = np.linspace(-1, 1, 11)
months = ['Jan', 'Feb', 'Mar', "Apr", 'May', "Jun", 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#corrs, sigs = src.crosscorrelation_by_month(encodings.mean('initialization').sel(mode='Quasibiennial'), encodings.mean('initialization').sel(mode='Quasibiennial'), nlags=nlags)
#corrs, sigs = src.crosscorrelation_by_month(src.low_pass(ssta_times_ua, threshold=7*12*30.4*86400), src.low_pass(tc2.mean('initialization').sel(mode='Decadal'), threshold=7*12*30.4*86400), nlags=nlags)
int_ssta_times_ua = ssta_times_ua.rolling(time=24, center=False).sum().dropna('time')
corrs, sigs = src.crosscorrelation_by_month(int_ssta_times_ua,  tc2.mean('initialization').sel(mode='Decadal').sel(time=int_ssta_times_ua.time), nlags=nlags)
#corrs, sigs = src.crosscorrelation_by_month(ssta_times_ua, tc2.mean('initialization').sel(mode='Decadal'), nlags=nlags)

#corrs, sigs = src.crosscorrelation_by_month(full_encodings.isel(recursion=0).mean('initialization').sel(mode='Decadal'), full_encodings.isel(recursion=2).mean('initialization').sel(mode='Decadal'), nlags=nlags)

corrsnan = corrs.copy()
corrsnan[sigs > 0.05] = np.nan
cp = lagcor_ax.contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
cp2 = lagcor_ax.contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
lagcor_ax.clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

lagcor_ax.set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
lagcor_ax.set_xticks(np.arange(-nlags, nlags+1, 12))

lagcor_ax.set_title(None)
lagcor_ax.set_title('c)', loc='left', fontweight='bold')

# Add bottom left and right text labels
lagcor_ax.text(-nlags, -2.8, "Tertiary precedes GCP", ha='left', va='top', fontsize=10)
lagcor_ax.text(nlags, -2.8, "Tertiary follows GCP", ha='right', va='top', fontsize=10)


cbar_ax =fig.add_subplot(gs[100:102, :])

fig.colorbar(cp, cax=cbar_ax, **{'label': 'Pearson Correlation', 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5, 'ticks':np.linspace(-1,1, 6), 'extend': 'both'})
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig(f'/Users/kylehall/Desktop/hall_molina_2025_final_figures/hall_molina_2025.figure7.{var}.png', dpi=1000)
plt.show()