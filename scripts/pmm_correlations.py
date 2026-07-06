import numpy as np
import xarray as xr
import kgae
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def regress_out(y, x, dim='time'):
    """
    Remove the linear regression of y on x (gridpoint-wise if y has lat/lon).
    y: DataArray with dim including 'time'
    x: DataArray with dim 'time'
    returns: residual y_resid = y - b*x
    """
    y, x = xr.align(y, x, join="inner")
    x_anom = x - x.mean(dim)
    y_anom = y - y.mean(dim)
    b = xr.cov(y_anom, x_anom, dim=dim) / x_anom.var(dim)
    return y - b * x  # keep y's mean; or use y_anom - b*x_anom if you prefer anomalies

def pattern_corr(a, b):
    """
    Area-weighted spatial pattern correlation over lat/lon.
    a, b: DataArray with dims lat, lon
    """
    a, b = xr.align(a, b, join="inner")
    w = np.cos(np.deg2rad(a['lat']))
    w2d = w / w.mean()  # normalize weights
    # weighted demean
    am = (a * w2d).mean(('lat','lon'))
    bm = (b * w2d).mean(('lat','lon'))
    aa = a - am
    bb = b - bm
    num = (aa * bb * w2d).sum(('lat','lon'))
    den = np.sqrt((aa**2 * w2d).sum(('lat','lon')) * (bb**2 * w2d).sum(('lat','lon')))
    return num / den

# --- your data ---
pmm = xr.open_dataset('pmm.nc').value  # DataArray(time)
sst = xr.open_dataset('~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc').sst
sst = sst.rename({'latitude':'lat', 'longitude':'lon'})

ssta, fit, gwm, p  = kgae.global_detrend(sst, deg=2)
ssta, mc = kgae.remove_climo(ssta)

beta_means = xr.open_dataset("cache_betas_composites_regression.nc").beta_mean
beta_id = beta_means.sel(tendency_mode=4, predictor_mode=5)

# Niño boxes (in degrees East but expressed in [-180,180] convention)
# Niño4: 5S–5N, 160E–150W  -> lon 160..210E => [-20..30] in [-180,180]
nino4 = ssta.sel(lat=slice(-5, 5), lon=slice(-20, 30)).mean(["lat", "lon"])

# Niño1+2: 0–10S, 90W–80W -> lon 270..280E => [-90..-80] in [-180,180]
nino12 = ssta.sel(lat=slice(-10, 0), lon=slice(90, 100)).mean(["lat", "lon"])

# (Optional) Niño3 if you still want to plot it / compare
# Niño3: 5S–5N, 150W–90W -> lon 210..270E => [30..-90] crosses dateline in [-180,180]
# easiest is to do it in 0..360 instead; leaving out unless you need it

# Takahashi-style simplified CP index
C_index = 1.7 * nino4 - 0.1 * nino12
C_index = (C_index - C_index.mean("time")) / C_index.std("time")
C_index.name = "C_index_approx"

# Spatial pattern: correlation of SST anomalies with C-index
C_pattern = xr.corr(ssta, C_index, dim="time")

# Plot
p = C_pattern.plot(
    subplot_kws={"projection": ccrs.PlateCarree(central_longitude=180)},
    cmap="RdBu_r"
)
p.axes.coastlines()
plt.title("C-index (approx) pattern: corr(SSTa, C)")
plt.show()

npgo_index = xr.open_dataset('npgo.nc').value 
npgo_pattern = xr.corr(ssta, npgo_index, dim="time")
p = npgo_pattern.plot(
    subplot_kws={"projection": ccrs.PlateCarree(central_longitude=180)},
    cmap="RdBu_r"
)
p.axes.coastlines()
plt.title("NPGO pattern: corr(SSTa, NPGO)")
plt.show()


spatial_corr_C = pattern_corr(
    beta_id.sel(lat=slice(-23, 23)),
    C_pattern.sel(lat=slice(-23, 23))
)
print("C-index (approx) vs Beta I-D (|lat|<=23):", float(spatial_corr_C.values))

oni_index = kgae.open_oni()  # DataArray(time)
CTI_LAT = slice(-6, 6)
CTI_LON = slice(180-180, 270-180)

cti = ssta.sel(lat=CTI_LAT, lon=CTI_LON).mean(["lat","lon"])
# Ensure consistent lon convention for domain slicing
#ssta = to_0360(ssta)

# --- PMM MCA-style domain ---
# Common PMM/NPMM domain used in meridional-mode (MCA) constructions:
# lat: 20S–30N, lon: 175E–95W (i.e., 175–265 in 0..360)
PMM_LAT = slice(-21, 32)
PMM_LON = slice(175-180, 265-180)

ssta_pmm_dom = ssta.sel(lat=PMM_LAT, lon=PMM_LON)

# --- remove ENSO from SST before forming PMM pattern (recommended) ---
# This is the typical “ENSO-removed” step used in many meridional-mode constructions.
ssta_pmm_noenso = regress_out(ssta_pmm_dom, cti, dim='time')

# Align time for correlation with PMM index
ssta_pmm_noenso, pmm = xr.align(ssta_pmm_noenso, pmm, join="inner")

# PMM spatial pattern: correlation (or regression) of SST onto PMM index
pmm_pattern = xr.corr(ssta_pmm_noenso, pmm, dim='time')

# Plot PMM pattern
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
pmm_pattern.plot(ax=ax, transform=ccrs.PlateCarree(central_longitude=180), cmap='RdBu_r')
ax.coastlines()
plt.title("PMM pattern (SST with ENSO regressed out; 20S–30N, 175E–95W)")
plt.show()

# --- compare to your beta_di ---
beta_di = beta_means.sel(
    tendency_mode=5, predictor_mode=4
)
#beta_di = to_0360(beta_di)

beta_pmm_dom = beta_di.sel(lat=PMM_LAT, lon=PMM_LON)

# Area-weighted pattern correlation (recommended)
spatial_corr = pattern_corr(beta_pmm_dom, pmm_pattern)
print("PMM vs Beta D-I (PMM MCA domain, ENSO removed from SST):", float(spatial_corr.values))

# --- PDO stays basically as you had, but area-weight weights is nicer ---
pdo_index = kgae.open_pdo().sel(dataset='PSL ERSSTv5 PDO')

# (Optional) match lon convention
#ssta_for_pdo = to_0360(ssta)

pdo_pattern = xr.corr(ssta, pdo_index, dim='time')

ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
pdo_pattern.plot(ax=ax, transform=ccrs.PlateCarree(central_longitude=180), cmap='RdBu_r')
ax.coastlines()
plt.title("PDO pattern (corr SST with PDO index)")
plt.show()

# NOAA PDO definition is usually North Pacific 20N–70N (often excluding tropics)
PDO_LAT = slice(20, 70)
beta_pdo_dom = beta_di.sel(lat=PDO_LAT)
pdo_dom = pdo_pattern.sel(lat=PDO_LAT)

spatial_corr_pdo = pattern_corr(beta_pdo_dom, pdo_dom)
print("PDO vs Beta D-I (20N–70N):", float(spatial_corr_pdo.values))
