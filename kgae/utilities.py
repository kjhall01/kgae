import xarray as xr 
import pandas as pd 
import torch 
from torch.nn import functional as F
from scipy.stats import norm 
import os
import shutil
import numpy as np 
import matplotlib.patches as patches
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset




def deniell(fourier_coeffs, frequencies, m_for_deniell_smoothing=15):
  initial_shape = frequencies.shape[0]
  if m_for_deniell_smoothing == 0:
    return fourier_coeffs, frequencies
  padded = F.pad(fourier_coeffs.t(), (m_for_deniell_smoothing//2 , m_for_deniell_smoothing//2), mode='reflect').t()
  freqs = torch.vstack([frequencies.reshape(1,-1), frequencies.reshape(1,-1)])
  padded_freqs = F.pad(freqs, (m_for_deniell_smoothing//2, m_for_deniell_smoothing//2), mode='reflect')
  tc = []
  tc2 = []
  smoothing_weights = norm.pdf(np.linspace(-2,2, m_for_deniell_smoothing))
  smoothing_weights = smoothing_weights / smoothing_weights.sum()
  for i in range(m_for_deniell_smoothing):
      tcc =  padded[i: i+fourier_coeffs.shape[0], : ]
      tcc2 = padded_freqs[0, i:i+fourier_coeffs.shape[0]].squeeze()
      tc.append(tcc * smoothing_weights[i])
      tc2.append(tcc2 * smoothing_weights[i])

  fourier_coeffs = torch.dstack(tc).sum(dim=-1)
  frequencies = torch.dstack(tc2).sum(dim=-1).squeeze()
  assert frequencies.shape[0] == initial_shape, 'initial shape changes - {} vs {}'.format(initial_shape, frequencies.shape[0])
  return fourier_coeffs, frequencies

def calc_wnl(dsdt, dx=5, dx_todiv=1):
  dsdt = dsdt - dsdt.mean(dim=0)
  x, y = [], []
  widths = []
  ii = dsdt.shape[0] // dx_todiv // 2
  while ii > 10:
    widths.append(ii)
    ii = ii // 2

  for step in widths:
    for window in range(0, dsdt.shape[0]-step-1, step):
      sample = dsdt[window:window+step, :]
      freqs = np.fft.rfftfreq(sample.shape[0], d=dx)
      ps = torch.fft.rfft(sample, dim=0).detach().numpy()
      ps = (ps * np.conj(ps)).real
      x.append(freqs)
      y.append(ps)
    sample = dsdt[dsdt.shape[0]-step-1:, :]
    freqs = np.fft.rfftfreq(sample.shape[0], d=dx)
    ps = torch.fft.rfft(sample, dim=0).detach().numpy()
    ps = (ps * np.conj(ps)).real
    x.append(freqs)
    y.append(ps)

  ys = np.zeros((np.fft.rfftfreq(dsdt.shape[0],d=dx).shape[0], dsdt.shape[1]), dtype=float)
  for i in range(len(x)):
    temp = np.ones_like(ys, dtype=float)*0.0
    for jj in range(dsdt.shape[1]):
      newy = np.interp(np.fft.rfftfreq(dsdt.shape[0], d=dx), x[i].squeeze(), y[i][:,jj].squeeze())
      temp[:, jj] = newy
    temp = temp / np.asarray(temp).max(axis=0)
    ys = ys + temp
  ys = ys / len(x)
  return torch.from_numpy(ys.mean(axis=0))



def detrend(da, dim='time', deg=2):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit, p.polyfit_coefficients.sel(degree=1), p


def remove_climo(monthly, dim='time', sub=None ):
    if sub is not None: 
        monthly_climatology = monthly.sel({dim:sub}).groupby(f'{dim}.month').mean()
    else:
        monthly_climatology = monthly.groupby(f'{dim}.month').mean()
    toconcat = []
    for year in sorted(list(set( [ pd.Timestamp(i).year for i in monthly.coords[dim].values] ))):
        ds_yearly = monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12,31))).groupby(f'{dim}.month').mean() - monthly_climatology
        ds_yearly = ds_yearly.assign_coords({'month': [ pd.Timestamp(year, j, 1) for j in ds_yearly.coords['month'].values ] } ).rename({'month': dim})
        toconcat.append(ds_yearly)
    monthly_anom = xr.concat(toconcat, dim).sortby(dim)
    return monthly_anom, monthly_climatology

def global_detrend(da, time_dim='time', spatial_dims=['lat', 'lon'], deg=2):
    # detrend along a single dimension
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"

    # Apply the weights and compute the mean across lat/lon
    da_weighted = da.weighted(weights)
    weighted_mean = da_weighted.mean(dim=spatial_dims)
    p = weighted_mean.polyfit(dim=time_dim, deg=deg)
    fit = xr.polyval(da[time_dim], p.polyfit_coefficients)
    return da - fit, fit, weighted_mean, p #.polyfit_coefficients


def remove_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    else:
        print(f"Directory does not exist or is not a directory: {path}")

def open_oni():
    return xr.open_dataset('/Users/kylehall/Desktop/Data/PSL/psl.oni.195001-202407.nc').ONI


def open_npi1():
    web = "https://psl.noaa.gov/data/timeseries/month/data/np.long.data"
    ds = pd.read_csv(web, skiprows=1, delim_whitespace=True, names=["YEAR", 'J', 'F', 'M', 'April', 'May', 'June', 'July', 'A', 'S', 'O', "N", 'D'])
    data = ds.values[:126, :]
    values, coords = [], []
    for i in range(126):
        yr = int(data[i, 0])
        for j in range(1, 13):
            values.append(float(data[i, j]))
            coords.append(pd.Timestamp(yr, j, 1))
    values = np.asarray(values)
    values[values == -999] = np.nan
    npi = xr.DataArray(name='npi', data=values, dims=('time'), coords={'time': coords}, attrs={'source': web})
    return npi

def open_npi():
    return xr.open_dataset('/Users/kylehall/Desktop/Data/PSL/npi.nc').npi

def open_pdo1():
    labels = ['PSL Ensemble PDO', 'PSL ERSSTv5 PDO', "PSL HadISST 1.1 PDO", "PSL COBE2 PDO"]
    Ns = [155, 171, 155, 175]
    webs = ['https://psl.noaa.gov/pdo/data/pdo.timeseries.sstens.data', 'https://psl.noaa.gov/pdo/data/pdo.timeseries.ersstv5.data', 'https://psl.noaa.gov/pdo/data/pdo.timeseries.hadisst1-1.data', 'https://psl.noaa.gov/pdo/data/pdo.timeseries.cobe2-sst.data' ]  

    times, datas = [], []

    tc = []
    for ii, web in enumerate(webs):
        ds = pd.read_csv(web, skiprows=1, delim_whitespace=True, names=["YEAR", 'J', 'F', 'M', 'April', 'May', 'June', 'July', 'A', 'S', 'O', "N", 'D'])
        data = ds.values[:Ns[ii], :]
        values, coords = [], []
        for i in range(Ns[ii]):
            yr = int(data[i, 0])
            for j in range(1, 13):
                values.append(float(data[i, j]))
                coords.append(pd.Timestamp(yr, j, 1))
        #values, coords = np.asarray(values), np.asarray(coords)
        values = np.asarray(values)
        values[values == -999] = np.nan
        pdo = xr.DataArray(name='pdo', data=values, dims=('time'), coords={'time': coords})
        tc.append(pdo)
    return xr.concat(tc, 'dataset').assign_coords({'dataset': labels})

def open_pdo():
    return xr.open_dataset('/Users/kylehall/Desktop/Data/PSL/pdo.nc').pdo

  

def multivariate_linear_regression_with_significance(predictors, data_array):
    # Get the dimensions of the data array
    N, X, Y = data_array.shape
    M = predictors.shape[1]
    
    modes = [f'mode{i}' for i in range(predictors.shape[1]+1)]
    modes[0] = 'intercept'

    # Initialize arrays to store results
    coefficients = np.zeros((X, Y, M+1)) * np.nan
    r_squared = np.zeros((X, Y)) * np.nan
    residuals = np.zeros((N, X, Y)) * np.nan
    p_values = np.zeros((X, Y, M+1)) * np.nan
    count, total = 0, X * Y
    print('[', end='')
    # Iterate over each grid point (x, y)
    for i in range(X):
        for j in range(Y):
            count += 1

            if count % (total // 10) == 0:
                print('-', end='', flush=True)
            # skip if all values are NaN
            if np.all(np.isnan(data_array[:, i, j].values)):
                #print(f'Skipping grid point {i}, {j} due to NaN values')
                continue
            # Extract the time series data for this grid point
            y = data_array[:, i, j].values
            
            # Add a constant term for intercept
            X_with_intercept = sm.add_constant(predictors)
            # Fit multivariate linear regression using statsmodels
            model = sm.OLS(y, X_with_intercept).fit()
            # Store coefficients
            coefficients[i, j, :] = model.params  # Exclude intercept
            
            # Calculate R-squared for this grid point
            r_squared[i, j] = model.rsquared
            
            # Store residuals
            residuals[:, i, j] = model.resid
            
            # Store p-values
            p_values[i, j, :] = model.pvalues  # Exclude intercept
    
    # Create xarray DataArrays for the results
    coeff_da = xr.DataArray(coefficients, dims=['lon', 'lat', 'mode'], 
                            coords={'lon': data_array.coords['lon'], 
                                    'lat': data_array.coords['lat'], 
                                    'mode': modes},
                            name='coefficients').transpose('lat', 'lon', 'mode')
    
    r2_da = xr.DataArray(r_squared, dims=['lon', 'lat'], 
                         coords={'lon': data_array.coords['lon'], 
                                 'lat': data_array.coords['lat']},
                         name='r_squared').transpose('lat', 'lon')
    
    residuals_da = xr.DataArray(residuals, dims=['time', 'lon', 'lat'], 
                                coords={'time': data_array.coords['time'], 
                                        'lon': data_array.coords['lon'], 
                                        'lat': data_array.coords['lat']},
                                name='residuals').transpose('lat', 'lon', 'time')
    
    p_values_da = xr.DataArray(p_values, dims=['lon', 'lat', 'mode'], 
                               coords={'lon': data_array.coords['lon'], 
                                       'lat': data_array.coords['lat'], 
                                        'mode': modes},
                               name='p_values').transpose('lat', 'lon', 'mode')
    print(']')
    return coeff_da, r2_da, residuals_da, p_values_da

from scipy.stats import pearsonr 

def crosscorrelation_by_month(x, y, nlags=36, n_months_per_lag=1):
    # we assume x and y initially have exactly the same time coordinate and it is monthly 
    # we pad y with nlags months of np.nan on either side 
    y_early_pad = xr.ones_like(y.isel(time=slice(None, nlags))).assign_coords({'time': [ pd.Timestamp(i) - DateOffset(months=nlags) for i in y.isel(time=slice(None, nlags)).time.values]}) * np.nan
    y_late_pad = xr.ones_like(y.isel(time=slice(-nlags, None))).assign_coords({'time': [ pd.Timestamp(i) + DateOffset(months=nlags) for i in y.isel(time=slice(-nlags, None)).time.values]}) * np.nan
    y = xr.concat([y_early_pad, y, y_late_pad], 'time')


    corrs, sigs = [], []
    for month in range(1,13): 
        corrs.append(np.full(nlags*2+1, np.nan))
        sigs.append(np.full(nlags*2+1, np.nan))

        x1 = x.isel(time=(x.time.dt.month == month))
        x_data = x1.values 
        xtime = x1.time
        for lag in np.arange(-nlags, nlags+1):
            # negative lag (left side of plot) means that y precedes x 
            y_data = y.sel(time=[pd.Timestamp(i) + DateOffset(months=lag*n_months_per_lag) for i in xtime.values])
            y_data = y_data.values 
            pearson = pearsonr(y_data[~np.isnan(y_data)].squeeze(), x_data[~np.isnan(y_data)].squeeze())
            corrs[-1][lag+nlags] = pearson.statistic
            sigs[-1][lag+nlags] = pearson.pvalue 
    corrs= np.array(corrs)
    sigs = np.array(sigs)

    return corrs, sigs 

def crosscorrelation_all(x, y, nlags=36, n_months_per_lag=1):
    # we assume x and y initially have exactly the same time coordinate and it is monthly 
    # we pad y with nlags months of np.nan on either side 
    y_early_pad = xr.ones_like(y.isel(time=slice(None, nlags))).assign_coords({'time': [ pd.Timestamp(i) - DateOffset(months=nlags) for i in y.isel(time=slice(None, nlags)).time.values]}) * np.nan
    y_late_pad = xr.ones_like(y.isel(time=slice(-nlags, None))).assign_coords({'time': [ pd.Timestamp(i) + DateOffset(months=nlags) for i in y.isel(time=slice(-nlags, None)).time.values]}) * np.nan
    y = xr.concat([y_early_pad, y, y_late_pad], 'time')


    corrs, sigs = [], []

    x_data = x.values 
    xtime = x.time
    for lag in np.arange(-nlags, nlags+1):
        # negative lag (left side of plot) means that y precedes x 
        y_data = y.sel(time=[pd.Timestamp(i) + DateOffset(months=lag*n_months_per_lag) for i in xtime.values])
        y_data = y_data.values 
        pearson = pearsonr(y_data[~np.isnan(y_data)].squeeze(), x_data[~np.isnan(y_data)].squeeze())
        corrs.append(pearson.statistic ) 
        sigs.append( pearson.pvalue )
    corrs= np.array(corrs)
    sigs = np.array(sigs)

    return corrs, sigs 