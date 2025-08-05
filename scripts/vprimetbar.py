import xarray as xr 
import pandas as pd 
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import src 
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import statsmodels.api as sm

results_base = Path('/Users/kylehall/Desktop/spectral_modes/final.crossvalidated.1940-2014')

# plot crossvalidtated timeseries 
N=100
tc2 = []
for rs in range(N):
    tc = []
    for split in range(5):
        ds = xr.open_dataset(results_base /  f'rs{rs}/split{split}/val_data.encodings.nc')
        tc.append(ds.encodings)
    tc = xr.concat(tc, 'time').sortby('time')
    tc2.append(tc)
tc2 = xr.concat(tc2, 'initialization').assign_coords({'initialization': np.arange(N)})#.sel(time=slice(pd.Timestamp(1958, 1,1), pd.Timestamp(2014,12,31)))#.rolling(time=5, center=True).mean().dropna('time', how='all').isel(time=slice(9, None,12)) #.rolling(time=5).mean().isel(time=slice(11, None, 12))
titles = ['Decadal', 'Interannual', 'Quasibiennial']

sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.iso20.195801-201412.nc').iso20.sel(lat=slice(-16,16))
ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
ssta , mc = src.remove_climo(ssta)
tc2 = tc2.sel(time=slice('1958-01-01', '2014-12-31'))

yaxis = "Interannual"
zaxis = 'Quasibiennial'
xaxis = 'Decadal'

tosel = 'Quasibiennial' if yaxis == 'Quasibienniel' else yaxis 

names = {
    'Decadal': 'TPDV',
    'Interannual': 'ENSO',
    'Quasibienniel': 'QB-KW'
}

labels = [f'{names[yaxis]}+ {names[xaxis]}-', f'{names[yaxis]}+ {names[xaxis]}.', f'{names[yaxis]}+ {names[xaxis]}+', f'{names[yaxis]}. {names[xaxis]}-', f'{names[yaxis]}. {names[xaxis]}.', f'{names[yaxis]}. {names[xaxis]}+', f'{names[yaxis]}- {names[xaxis]}-', f'{names[yaxis]}- {names[xaxis]}.', f'{names[yaxis]}- {names[xaxis]}+']

confidence_level = 0.05
confidence_level2 = 0.05/2
xupper, xlower = tc2.sel(mode=xaxis).quantile(1 - confidence_level/2, 'initialization').values, tc2.sel(mode=xaxis).quantile(confidence_level/2, 'initialization').values
yupper, ylower = tc2.sel(mode=tosel).quantile(1 - confidence_level/2, 'initialization').values, tc2.sel(mode=tosel).quantile(confidence_level/2, 'initialization').values
zupper, zlower = tc2.sel(mode=zaxis).quantile(1 - confidence_level2, 'initialization').values, tc2.sel(mode=zaxis).quantile(confidence_level2, 'initialization').values



xmag1, xmag2 = 1.5, -1.5 # tc2.sel(mode=xaxis).mean('initialization').quantile(1 - q, 'time').values, tc2.sel(mode=xaxis).mean('initialization').quantile(1 - q, 'time').values
ymag1, ymag2 = 1.5, -1.5 #tc2.sel(mode=tosel).mean('initialization').quantile(1 - q, 'time').values, tc2.sel(mode=tosel).mean('initialization').quantile(1 - q, 'time').values
zmag1, zmag2 = 0.5, -0.5# tc2.sel(mode=zaxis).mean('initialization').quantile(1 - q, 'time').values, tc2.sel(mode=zaxis).mean('initialization').quantile(1 - q, 'time').values


print('Positive ENSO Boundary: ', ymag1)
print('Negative ENSO Boundary: ', ymag2)
print('Positive TPDV Boundary: ', xmag1)
print('Negative TPDV Boundary: ', xmag2)
print('Positive QB-KW Boundary: ', zmag1)
print('Negative QB-KW Boundary: ', zmag2)

#zrec=True
zrec = ~(zupper < zmag2) & ~(zlower > zmag1) # neutral z 
#zrec = (zlower > zmag1) # positive z 
#zrec = (zupper < zmag2) # negative z 


print('percent allowed past qb-kw restriction: ', np.sum(zrec).sum() )

composite_selections = [
    (ylower > ymag1 ) & ( xupper < xmag2 ) & zrec, # y positive, x negative
    (ylower > ymag1 ) & ~( xupper < xmag2 ) & ~(  xlower > xmag1 ) & zrec, # y positive, x neutral
    (ylower > ymag1 ) & (  xlower > xmag1 ) & zrec, # y positive, x positive
    
    ~(ylower > ymag1 ) & ~(yupper < ymag2 ) & (xupper < xmag2 ) & zrec, # y neutral x negative 
    ~(ylower > ymag1 ) & ~(yupper < ymag2 ) & ~( xupper < xmag2 ) & ~(  xlower > xmag1 ) & zrec, # neutral neutral
    ~(ylower > ymag1 ) & ~(yupper < ymag2 ) & (  xlower > xmag1 ) & zrec, # y neutral x positive

    (yupper < ymag2) & ( xupper < xmag2 ) & zrec, # negative negative
    (yupper < ymag2) & ~( xupper < xmag2 ) & ~(  xlower > xmag1 ) & zrec, # negative neutral
    (yupper < ymag2) & (  xlower > xmag1 ) & zrec # negative positive 
]

cmpsts, masks = [], [] 
ns = [] 
for i in range(9):
    print('doing composite number ', i+1)
    composite_selection = composite_selections[i]
    n = composite_selection.sum() 
    ns.append(n)
    print('samples in ', n / np.sum(composite_selection | ~composite_selection))
    composite = ssta.isel(time=composite_selection).mean('time') 
    cmpsts.append(composite)

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
    significance_mask = (composite > upper) | (composite < lower)
    masks.append(significance_mask)

print(sum(ns) / np.sum(composite_selection | ~composite_selection))

sst2 = xr.open_dataset('~/Desktop/Data/oras5/oras5.interior_transport.195801-201412.nc').interior_transport * 4.26
ssta2, fit, gwm, p  = src.global_detrend(sst2, deg=2)
ssta2 , mc = src.remove_climo(ssta2)

fig, ax = plt.subplots(3, 3, figsize=(10, 8))
cmpsts = xr.concat(cmpsts, 'type').rolling(lon=3, center=True).mean()
for i in range(3):
    for j in range(3):
        ax[j][i].set_title(labels[j*3 + i] + f" (N: {ns[j*3+i]})")
        ax[j][i].axhline(0, color='k', linestyle='--')
        ax[j][i].plot(cmpsts.isel(type=j*3+i).lon, -1*cmpsts.isel(type=j*3+i).sel(lat=slice(-1, 1)).mean('lat'), color='r', label='20C Isotherm Depth Anomaly (m)')
        ax[j][i].set_ylim(-22.5,22.5)
        ax[j][i].set_ylabel('m')


sst = xr.open_dataset('~/Desktop/Data/oras5/oras5.interior_transport.195801-201412.nc').interior_transport.sel(lat=slice(-16,16)) * 4.26 / (111000**2) * 86400 * 365
ssta, fit, gwm, p  = src.global_detrend(sst, deg=2)
ssta , mc = src.remove_climo(ssta)

cmpsts, masks = [], [] 
ns = [] 
for i in range(9):
    print('doing composite number ', i+1)
    composite_selection = composite_selections[i]
    n = composite_selection.sum() 
    ns.append(n)
    print('samples in ', n / np.sum(composite_selection | ~composite_selection))
    composite = ssta.isel(time=composite_selection).mean('time') 
    cmpsts.append(composite)

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
    significance_mask = (composite > upper) | (composite < lower)
    masks.append(significance_mask)

print(sum(ns) / np.sum(composite_selection | ~composite_selection))

cmpsts = xr.concat(cmpsts, 'type').rolling(lon=3, center=True).mean()
for i in range(3):
    for j in range(3):
        x2 = ax[j][i].twinx()
        x2.set_ylim(-1, 1)
        x2.set_ylabel('m/yr')
        x2.plot(cmpsts.isel(type=j*3+i).lon, cmpsts.isel(type=j*3+i).sel(lat=slice(-10, -5)).mean('lat') - cmpsts.isel(type=j*3+i).sel(lat=slice(5, 10)).mean('lat') , color='b', label='Pycnocline Velocity')

ax[j][i].plot([],[], color='b', label='Pycnocline Velocity (m/yr)')
ax[j][i].legend(loc='lower right')
plt.tight_layout()
#plt.savefig(f'composites.era5.{var}.{xaxis}-{yaxis}.png', dpi=300)
plt.show()
