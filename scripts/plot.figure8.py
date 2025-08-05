import xarray as xr 
import cartopy.crs as ccrs 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import matplotlib.patches as patches
import src 
import numpy as np 
from pathlib import Path 



nvert, nhori = 100, 110
fig = plt.figure(figsize=(8,10))
gs = gridspec.GridSpec(nvert, nhori)


N=50
M=21
mtall=25
run='basin.goodtest'
vmax = 0.003
results_base = Path(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014')

hs  = []
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
    plt.figure()
    h, xe, ye, im = plt.hist2d(tc2.sel(mode='Decadal', recursion=0).values, tc2.sel(mode='Interannual', recursion=0).values, cmap="magma_r", bins=45, range=[[-5,5], [-5,5]], density=True)
    h = h / np.nansum(h)
    plt.close()
    hs.append(h)
    tc3.append(tc2)
tc2 = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N)}).isel(recursion=0)
encodings = tc2


cbar_ax =fig.add_subplot(gs[mtall+6:mtall+8, :])


lagcc_ax2 = fig.add_subplot(gs[8+mtall+10+15+7+15+7:8+mtall+10+15+7+15+7+115, :])
lagcc_ax0 = fig.add_subplot(gs[8+mtall+10:8+mtall+10+15, :])
lagcc_ax1 = fig.add_subplot(gs[8+mtall+10+15+7:8+mtall+10+15+7+15, :])


nlags=36
levels = np.linspace(-1, 1, 21)
levels2 = np.linspace(-1, 1, 11)
months = ['Jan', 'Feb', 'Mar', "Apr", 'May', "Jun", 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

corrs, sigs = src.crosscorrelation_by_month(encodings.mean('initialization').sel(mode='Interannual'), encodings.mean('initialization').sel(mode='Quasibiennial'), nlags=nlags)
corrsnan = corrs.copy()
corrsnan[sigs > 0.05] = np.nan
cp = lagcc_ax0.contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
cp2 = lagcc_ax0.contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
lagcc_ax0.clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

lagcc_ax0.set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
lagcc_ax0.set_xticks(np.arange(-nlags, nlags+1, 12))

#lagcc_ax2.set_title('Interannual - Quasibiennial')
lagcc_ax0.set_title('d)', loc='left', fontweight='bold')

# Add bottom left and right text labels
#lagcc_ax0.text(-nlags, -3.5, "Quasibiennial precedes Interannual", ha='left', va='top', fontsize=10)
#lagcc_ax0.text(nlags, -3.5, "Quasibiennial follows Interannual", ha='right', va='top', fontsize=10)


circle_axis1 = fig.add_subplot(gs[:mtall, :30])
circle_axis1.set_title('a)',loc='left', fontweight='bold')
#circle_axis1.set_title('ERA5', fontsize=10)

hs = np.dstack(hs)
mean_h = hs.mean(axis=-1)
mean_h = mean_h / np.nansum(mean_h)
mean_h[mean_h < 0.000001] = np.nan

std_h = hs.std(axis=-1)
std_h[mean_h < 0.001] = np.nan 
low = mean_h *0.75
high = mean_h*1.25
hatches = np.nanmean( (hs > low.reshape(45,45,1) ) & (hs < high.reshape(45,45,1)), axis=-1) 
mask = hatches > 0.9
p = circle_axis1.pcolormesh(xe, ye, mean_h, cmap='magma_r', vmax=vmax)
circle_axis1.tick_params(axis='both', which='major', labelsize=8)

fig.colorbar(p, cax=cbar_ax, **{'label': 'density', 'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.5, 'extend': 'max'})


circle_axis1.set_xticks([-4,-2, 0, 2, 4])
circle_axis1.set_yticks([-4,-2, 0, 2, 4])

circle_axis1.spines['right'].set_color('none')
circle_axis1.spines['top'].set_color('none')
circle_axis1.spines['bottom'].set_color('none')
circle_axis1.spines['left'].set_color('none')

r= 2
phases = [np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8, 9*np.pi/8, 11*np.pi/8, 13*np.pi/8, 15*np.pi/8]
for phase in phases: 
    xx, yy = np.cos(phase), np.sin(phase)
    circle_axis1.plot([r*xx,5*xx], [r*yy,5*yy], color='k', alpha=0.3)


x_min, x_max = circle_axis1.get_xlim()
x_position = (x_min + x_max) / 2
y_position = circle_axis1.get_ylim()[0] - 0.15 * (circle_axis1.get_ylim()[1] - circle_axis1.get_ylim()[0])
circle_axis1.text(x_position, y_position,  'Decadal Mode', ha='center', va='top')

y_min, y_max = circle_axis1.get_ylim()
y_position = (y_min + y_max*1.8) / 2 
x_position = circle_axis1.get_xlim()[0] - 0.15 * (circle_axis1.get_xlim()[1] - circle_axis1.get_xlim()[0])
circle_axis1.text(x_position, y_position, 'Interannual Mode', ha='right', va='top', rotation=90)

circle = patches.Ellipse((0,0),width=2*r, height=2*r,  linewidth=1.5, edgecolor='k', alpha=0.3, facecolor='none')
circle_axis1.add_patch(circle)
circle_axis1.grid(linestyle='--', alpha=0.3)
circle_axis1.contourf(0.5*xe[1:] + 0.5*xe[:-1], 0.5*ye[1:] + 0.5*ye[:-1], mask, 1, hatches=['','/'], alpha=0)



print('e3sm')

hs  = []
tc3 = []
for rs in range(N):
    tc = []
    for split in range(5):
        for member in range(M):
            print(f'rs{rs} split{split} member{member}')
            ds = xr.open_dataset(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014/rs{rs}/split{split}/recursion0/e3sm/e3sm.mem{member}.encodings{split}.nc')
            plt.figure()
            h, xe, ye, im = plt.hist2d(ds.encodings.sel(mode='Decadal').values, ds.encodings.sel(mode='Interannual').values, cmap="magma_r", bins=45, range=[[-5,5], [-5,5]], density=True)
            h = h / np.nansum(h)
            plt.close()
            hs.append(h)
            tc3.append(ds.encodings)

tc3 = xr.concat(tc3, 'initialization').assign_coords({'initialization': np.arange(N*M*5)})
encodings = tc3 



nlags=36
levels = np.linspace(-1, 1, 21)
levels2 = np.linspace(-1, 1, 11)
months = ['Jan', 'Feb', 'Mar', "Apr", 'May', "Jun", 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

corrs, sigs = src.crosscorrelation_by_month(encodings.mean('initialization').sel(mode='Interannual'), encodings.mean('initialization').sel(mode='Quasibiennial'), nlags=nlags)
corrsnan = corrs.copy()
corrsnan[sigs > 0.05] = np.nan
cp = lagcc_ax1.contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
cp2 = lagcc_ax1.contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
lagcc_ax1.clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

lagcc_ax1.set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
lagcc_ax1.set_xticks(np.arange(-nlags, nlags+1, 12))

#lagcc_ax1.set_title('Interannual - Quasibiennial')
lagcc_ax1.set_title('e) ', loc='left', fontweight='bold')


# Add bottom left and right text labels
#lagcc_ax1.text(-nlags, -3.5, "Quasibiennial precedes Interannual", ha='left', va='top', fontsize=10)
#lagcc_ax1.text(nlags, -3.5, "Quasibiennial follows Interannual", ha='right', va='top', fontsize=10)


circle_axis2 = fig.add_subplot(gs[:mtall, 40:70])
#circle_axis2.set_title('E3SM', fontsize=10)
circle_axis2.set_title('b)',loc='left', fontweight='bold')

hs = np.dstack(hs)

mean_h = hs.mean(axis=-1)
mean_h = mean_h / np.nansum(mean_h)
mean_h[mean_h < 0.000001] = np.nan

std_h = hs.std(axis=-1)
std_h[mean_h < 0.001] = np.nan 
low = mean_h *0.75
high = mean_h*1.25
hatches = np.nanmean( (hs > low.reshape(45,45,1) ) & (hs < high.reshape(45,45,1)), axis=-1) 
mask = hatches > 0.9
circle_axis2.pcolormesh(xe, ye, mean_h, cmap='magma_r', vmax=vmax)
circle_axis2.tick_params(axis='both', which='major', labelsize=8)

circle_axis2.set_xticks([-4,-2, 0, 2, 4])
circle_axis2.set_yticks([-4,-2, 0, 2, 4])

circle_axis2.spines['right'].set_color('none')
circle_axis2.spines['top'].set_color('none')
circle_axis2.spines['bottom'].set_color('none')
circle_axis2.spines['left'].set_color('none')

r= 2
phases = [np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8, 9*np.pi/8, 11*np.pi/8, 13*np.pi/8, 15*np.pi/8]
for phase in phases: 
    xx, yy = np.cos(phase), np.sin(phase)
    circle_axis2.plot([r*xx,5*xx], [r*yy,5*yy], color='k', alpha=0.3)


x_min, x_max = circle_axis2.get_xlim()
x_position = (x_min + x_max) / 2
y_position = circle_axis2.get_ylim()[0] - 0.15 * (circle_axis2.get_ylim()[1] - circle_axis2.get_ylim()[0])
circle_axis2.text(x_position, y_position, 'Decadal Mode', ha='center', va='top')

y_min, y_max = circle_axis2.get_ylim()
y_position = (y_min + y_max*1.8) / 2 
x_position = circle_axis2.get_xlim()[0] - 0.15 * (circle_axis2.get_xlim()[1] - circle_axis2.get_xlim()[0])
circle_axis2.text(x_position, y_position,  'Interannual Mode', ha='right', va='top', rotation=90)

circle = patches.Ellipse((0,0),width=2*r, height=2*r,  linewidth=1.5, edgecolor='k', alpha=0.3, facecolor='none')
circle_axis2.add_patch(circle)
circle_axis2.grid(linestyle='--', alpha=0.3)
circle_axis2.contourf(0.5*xe[1:] + 0.5*xe[:-1], 0.5*ye[1:] + 0.5*ye[:-1], mask, 1, hatches=['','/'], alpha=0)


print('cesm')

hs  = []
tc2 = []
for rs in range(N):
    tc = []
    for split in range(5):
        for member in range(M):
            print(f'rs{rs} split{split} member{member}')
            ds = xr.open_dataset(f'/Users/kylehall/Desktop/spectral_modes/{run}.recursive.crossvalidated.1940-2014/rs{rs}/split{split}/recursion0/cesm/cesm.mem{member}.encodings{split}.nc')
            plt.figure()
            h, xe, ye, im = plt.hist2d(ds.encodings.sel(mode='Decadal').values, ds.encodings.sel(mode='Interannual').values, cmap="magma_r", bins=45, range=[[-5,5], [-5,5]], density=True)
            h = h / np.nansum(h)
            plt.close()
            hs.append(h)
            tc2.append(ds.encodings)

tc3 = xr.concat(tc2, 'initialization').assign_coords({'initialization': np.arange(N*M*5)})
encodings = tc3 



nlags=36
levels = np.linspace(-1, 1, 21)
levels2 = np.linspace(-1, 1, 11)
months = ['Jan', 'Feb', 'Mar', "Apr", 'May', "Jun", 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

corrs, sigs = src.crosscorrelation_by_month(encodings.mean('initialization').sel(mode='Interannual'), encodings.mean('initialization').sel(mode='Quasibiennial'), nlags=nlags)
corrsnan = corrs.copy()
corrsnan[sigs > 0.05] = np.nan
cp = lagcc_ax2.contourf(  np.arange(-nlags, nlags+1), np.arange(12), corrsnan, levels=levels, cmap='RdBu_r') 
cp2 = lagcc_ax2.contour(  np.arange(-nlags, nlags+1), np.arange(12), corrs, levels=levels2, colors='k', negative_linestyles='dashed')
lagcc_ax2.clabel(cp2, inline=True, fontsize=8, fmt="%.2f")

lagcc_ax2.set_yticks([0,3,6,9], labels=[months[j] for j in [0,3,6,9]])
lagcc_ax2.set_xticks(np.arange(-nlags, nlags+1, 12))

#lagcc_ax2.set_title('Interannual - Quasibiennial')
lagcc_ax2.set_title('f)', loc='left', fontweight='bold')

# Add bottom left and right text labels
lagcc_ax2.text(-nlags, -3.5, "Quasibiennial precedes Interannual", ha='left', va='top', fontsize=10)
lagcc_ax2.text(nlags, -3.5, "Quasibiennial follows Interannual", ha='right', va='top', fontsize=10)



circle_axis3 = fig.add_subplot(gs[:mtall, 80:])
#circle_axis3.set_title('CESM', fontsize=10)
circle_axis3.set_title('c)',loc='left', fontweight='bold')

hs = np.dstack(hs)
mean_h = hs.mean(axis=-1)
mean_h = mean_h / np.nansum(mean_h)

mean_h[mean_h < 0.000001] = np.nan

std_h = hs.std(axis=-1)
std_h[mean_h < 0.000001] = np.nan 
low = mean_h *0.75
high = mean_h*1.25
hatches = np.nanmean( (hs > low.reshape(45,45,1) ) & (hs < high.reshape(45,45,1)), axis=-1) 
mask = hatches > 0.9
circle_axis3.pcolormesh(xe, ye, mean_h, cmap='magma_r', vmax=vmax)
circle_axis3.tick_params(axis='both', which='major', labelsize=8)

circle_axis3.set_xticks([-4,-2, 0, 2, 4])
circle_axis3.set_yticks([-4,-2, 0, 2, 4])

circle_axis3.spines['right'].set_color('none')
circle_axis3.spines['top'].set_color('none')
circle_axis3.spines['bottom'].set_color('none')
circle_axis3.spines['left'].set_color('none')

r= 2
phases = [np.pi/8, 3*np.pi/8, 5*np.pi/8, 7*np.pi/8, 9*np.pi/8, 11*np.pi/8, 13*np.pi/8, 15*np.pi/8]
for phase in phases: 
    xx, yy = np.cos(phase), np.sin(phase)
    circle_axis3.plot([r*xx,5*xx], [r*yy,5*yy], color='k', alpha=0.3)


x_min, x_max = circle_axis3.get_xlim()
x_position = (x_min + x_max) / 2
y_position = circle_axis3.get_ylim()[0] - 0.15 * (circle_axis3.get_ylim()[1] - circle_axis3.get_ylim()[0])
circle_axis3.text(x_position, y_position, 'Decadal Mode', ha='center', va='top')

y_min, y_max = circle_axis3.get_ylim()
y_position = (y_min + y_max*1.8) / 2 
x_position = circle_axis3.get_xlim()[0] - 0.15 * (circle_axis3.get_xlim()[1] - circle_axis3.get_xlim()[0])
circle_axis3.text(x_position, y_position, 'Interannual Mode', ha='right', va='top', rotation=90)

circle = patches.Ellipse((0,0),width=2*r, height=2*r,  linewidth=1.5, edgecolor='k', alpha=0.3, facecolor='none')
circle_axis3.add_patch(circle)
circle_axis3.grid(linestyle='--', alpha=0.3)

circle_axis3.contourf(0.5*xe[1:] + 0.5*xe[:-1], 0.5*ye[1:] + 0.5*ye[:-1], mask, 1, hatches=['','/'], alpha=0)




plt.tight_layout()
plt.savefig('/Users/kylehall/Desktop/hall_molina_2025.figure8.png', dpi=300, bbox_inches='tight')
plt.show()

