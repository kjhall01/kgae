import numpy as np
import xarray as xr
import time as systime
import sys
import pandas as pd
import dask.array as da
from scipy.ndimage import gaussian_filter1d
from scipy import signal

pi = np.pi
re = 6.371008e6  # Earth's radius in meters
g = 9.80665  # Gravitational acceleration [m s^{-2}]
omega = 7.292e-05  # Angular speed of rotation of Earth [rad s^{-1}]
beta = 2. * omega / re  # beta parameter at the equator


filters = {
    'kelvin': [2.5, 20, 1, 14, 8, 90],
    'mrg': [2, 6, -250, 250, -9999, -9999],
    'ig1': [1.2, 2.6, -15, -1, 12, 90],
    'er': [10, 40, -10, -1, 8, 90 ],
    'mjo': [30, 96, 0, 250, -9999, -9999]
}

def kf_filter_mask(fftIn, obsPerDay=None, tMin=None, tMax=None, kMin=None, kMax=None, hMin=None, hMax=None, waveName=None, filter_size=None):
    """
    Maria Gehne - Tropical Diagnostics
    """
    fftData = np.copy(fftIn)
    fftData = np.transpose(fftData)
    nf, nk = fftData.shape  # frequency, wavenumber array
    fftData = fftData[:, ::-1]

    nt = (nf - 1) * 2
    jMin = int(round(nt / (tMax * obsPerDay)))
    jMax = int(round(nt / (tMin * obsPerDay)))
    jMax = np.array([jMax, nf]).min()

    if kMin < 0:
        iMin = int(round(nk + kMin))
        iMin = np.array([iMin, nk // 2]).max()
    else:
        iMin = int(round(kMin))
        iMin = np.array([iMin, nk // 2]).min()

    if kMax < 0:
        iMax = int(round(nk + kMax))
        iMax = np.array([iMax, nk // 2]).max()
    else:
        iMax = int(round(kMax))
        iMax = np.array([iMax, nk // 2]).min()

    # set the appropriate coefficients outside the frequency range to zero
    # print(fftData[:, 0])
    if jMin > 0:
        fftData[0:jMin, :] = 0
    if jMax < nf:
        fftData[jMax + 1:nf, :] = 0
    if iMin < iMax:
        # Set things outside the wavenumber range to zero, this is more normal
        if iMin > 0:
            fftData[:, 0:iMin] = 0
        if iMax < nk:
            fftData[:, iMax + 1:nk] = 0
    else:
        # Set things inside the wavenumber range to zero, this should be somewhat unusual
        fftData[:, iMax + 1:iMin] = 0

    c = np.empty([2])
    if hMin == -9999:
        c[0] = np.nan
        if hMax == -9999:
            c[1] = np.nan
    else:
        if hMax == -9999:
            c[1] = np.nan
        else:
            c = np.sqrt(g * np.array([hMin, hMax]))

    spc = 24 * 3600. / (2 * pi * obsPerDay)  # seconds per cycle

    # Now set things to zero that are outside the wave dispersion. Loop through wavenumbers
    # and find the limits for each one.
    for i in range(nk):
        if i < (nk / 2):
            # k is positive
            k = i / re
        else:
            # k is negative
            k = -(nk - i) / re

        freq = np.array([0, nf]) / spc
        jMinWave = 0
        jMaxWave = nf
        if (waveName == "Kelvin") or (waveName == "kelvin") or (waveName == "KELVIN"):
            ftmp = k * c
            freq = np.array(ftmp)
        if (waveName == "ER") or (waveName == "er"):
            ftmp = -beta * k / (k ** 2 + 3 * beta / c)
            freq = np.array(ftmp)
        if (waveName == "MRG") or (waveName == "IG0") or (waveName == "mrg") or (waveName == "ig0"):
            if k == 0:
                ftmp = np.sqrt(beta * c)
                freq = np.array(ftmp)
            else:
                if k > 0:
                    ftmp = k * c * (0.5 + 0.5 * np.sqrt(1 + 4 * beta / (k ** 2 * c)))
                    freq = np.array(ftmp)
                else:
                    ftmp = k * c * (0.5 - 0.5 * np.sqrt(1 + 4 * beta / (k ** 2 * c)))
                    freq = np.array(ftmp)
        if (waveName == "IG1") or (waveName == "ig1"):
            ftmp = np.sqrt(3 * beta * c + k ** 2 * c ** 2)
            freq = np.array(ftmp)
        if (waveName == "IG2") or (waveName == "ig2"):
            ftmp = np.sqrt(5 * beta * c + k ** 2 * c ** 2)
            freq = np.array(ftmp)

        if hMin == -9999:
            jMinWave = 0
        else:
            jMinWave = int(np.floor(freq[0] * spc * nt))
        if hMax == -9999:
            jMaxWave = nf
        else:
            jMaxWave = int(np.ceil(freq[1] * spc * nt))
        jMaxWave = np.array([jMaxWave, 0]).max()
        jMinWave = np.array([jMinWave, nf]).min()

        # set appropriate coefficients to zero
        if jMinWave > 0:
            fftData[0:jMinWave, i] = 0
        if jMaxWave < nf:
            fftData[jMaxWave + 1:nf, i] = 0

    fftData = fftData[:, ::-1]
    fftData = np.transpose(fftData)
    return fftData

def kf_filter(data, kwargs={}, transpose_data=False):
    """
    Maria Gehne - Tropical Diagnostics package
    """
    original_shape = data.shape
    data = data.squeeze()
    assert len(list(data.shape)) == 2, 'data needs to be 2-dimensional after squeeze - no "one point at a time" stuff here'
    # data should be in (lon x time) to be able to use rfft on the time dimension
    if transpose_data:
        data = data.T
    if kwargs['filter_size'] is not None:
        data = gaussian_filter1d(data, kwargs['filter_size'], axis=0)  # wave filter abs
    fftdata = np.fft.rfft2(data, axes=(0, 1))
    fftfilt = kf_filter_mask(fftdata, **kwargs)
    datafilt = np.fft.irfft2(fftfilt, s=data.shape, axes=(0, 1))
    semifinal_shape = datafilt.shape
#    datafilt = np.transpose(datafilt, axes=[1, 0])
    nearfinal_shape = datafilt.shape
    datafilt = datafilt.T if transpose_data else datafilt
    #print('original_shape: ', original_shape, 'processed shape: ', data.shape, 'semifinal_shape', semifinal_shape, 'nearfinal_shape',nearfinal_shape, 'final_shape', datafilt.shape)
    return datafilt.reshape(*original_shape)

def wave_filter(xda, time_dim='step', lon_dim='longitude', waveName='Kelvin', obsPerDay=1, threshold_percentile=None, threshold_value=None, filter_size=None):
    """
    Kyle Hall - packaging in xarray
    """
    assert time_dim in xda.dims, '{} not present on data'.format(time_dim)
    assert lon_dim in xda.dims, '{} not present on data'.format(lon_dim)
    dim_sizes = list(xda.shape) # this is the sizes of the dimensions
    dim_names = list(xda.dims) # this is the names of the dimensions
    potential_dims = 'abcdefghijkmnopqrsuvwxyz' # alphabet without l - longitude and t - time
    assert len(dim_sizes) < 24, 'how literally dare you use 24-dimensional data- are you a monster?'
    dim_str = potential_dims[:len(dim_sizes)] # this is an einstein summation index notation representing the data
    timedim_ndx = dim_names.index(time_dim)
    londim_ndx = dim_names.index(lon_dim)
    chunksizes = [1 for i in range(len(dim_sizes))]
    chunksizes[timedim_ndx] = dim_sizes[timedim_ndx]
    chunksizes[londim_ndx] = dim_sizes[londim_ndx]
    data = da.from_array(xda.values, chunks=chunksizes)
    tMin, tMax, kMin, kMax, hMin, hMax = filters[waveName.lower()]
    kwargs = {'tMin': tMin, 'tMax': tMax, 'kMin': kMin, 'kMax': kMax, 'hMin': hMin, 'hMax': hMax, 'obsPerDay': obsPerDay, 'filter_size': filter_size}
    transpose_data = londim_ndx > timedim_ndx
    results = da.blockwise(kf_filter, dim_str, data, dim_str, dtype=float, concatenate=True, transpose_data=transpose_data, kwargs=kwargs).persist()
    ret = xr.ones_like(xda, dtype=float) * results
    if threshold_percentile is not None or threshold_value is not None:
        ret = np.abs(ret)
        if threshold_percentile is not None:
            assert threshold_percentile >0 and threshold_percentile < 1, 'threshold_percentile must be on (0, 1)'
        ret = ret.compute()
        thresh = ret.where(ret >0).quantile(threshold_percentile, dim=[time_dim, lon_dim]) if threshold_value is None else threshold_value
        return thresh, xr.ones_like(ret).where(ret > thresh, other=0)
    else:
        return ret


import torch

def low_pass_filter(data, kwargs={}, transpose_data=False):
    """
    Kyle Hall - packaging in xarray
    """
    #data = torch.from_numpy(data)
    need_fix = False

    if len(list(data.shape)) == 1:
        data = data.reshape(-1, 1)
        need_fix = True 

    if False:
        data = data - data.mean(dim=0)
        frequencies = torch.fft.rfftfreq(data.shape[0], d=1/kwargs['obs_per_second']) #[1:]
        fourier_coeffs = torch.fft.rfft(data, dim=0)
        fourier_coeffs[frequencies > (1 / kwargs['threshold'])] = 0
        rec = torch.fft.irfft(fourier_coeffs, dim=0, n=data.shape[0])
    else: 
        fc = 1/kwargs['threshold']  # Cut-off frequency of the filter
        w = fc / (kwargs['obs_per_second']/2) # Normalize the frequency
        b, a = signal.butter(4, w, btype='lowpass')
        rec = signal.filtfilt(b, a, data, axis=kwargs['axis'])
    if need_fix:
        return rec.reshape(-1)#.detach().numpy().reshape(-1)
    else:
        return rec#.detach().numpy()




def low_pass(xda, time_dim='time', threshold=7*12*30.4*86400, obs_per_second=1/(86400*30.4) ):
    """
    Kyle Hall - packaging in xarray
    """
    assert time_dim in xda.dims, '{} not present on data'.format(time_dim)
    dim_sizes = list(xda.shape) # this is the sizes of the dimensions
    dim_names = list(xda.dims) # this is the names of the dimensions
    potential_dims = 'abcdefghijkmnopqrsuvwxyz' # alphabet without l - longitude and t - time
    assert len(dim_sizes) < 24, 'how literally dare you use 24-dimensional data- are you a monster?'
    dim_str = potential_dims[:len(dim_sizes)] # this is an einstein summation index notation representing the data
    timedim_ndx = dim_names.index(time_dim)
    chunksizes = [1 for i in range(len(dim_sizes))]
    chunksizes[timedim_ndx] = dim_sizes[timedim_ndx]
    data = da.from_array(xda.values, chunks=chunksizes)
    kwargs = {'obs_per_second': obs_per_second, 'threshold': threshold, 'axis': timedim_ndx} 
    results = da.blockwise(low_pass_filter, dim_str, data, dim_str, dtype=float, concatenate=True,  kwargs=kwargs).persist()
    return xr.ones_like(xda, dtype=float) * results


def high_pass_filter(data, kwargs={}, transpose_data=False):
    """
    Kyle Hall - packaging in xarray
    """
    #data = torch.from_numpy(data)
    need_fix = False

    if len(list(data.shape)) == 1:
        data = data.reshape(-1, 1)
        need_fix = True 

    if False:
        data = data - data.mean(dim=0)
        frequencies = torch.fft.rfftfreq(data.shape[0], d=1/kwargs['obs_per_second']) #[1:]
        fourier_coeffs = torch.fft.rfft(data, dim=0)
        fourier_coeffs[frequencies > (1 / kwargs['threshold'])] = 0
        rec = torch.fft.irfft(fourier_coeffs, dim=0, n=data.shape[0])
    else: 
        fc = 1/kwargs['threshold']  # Cut-off frequency of the filter
        w = fc / (kwargs['obs_per_second']/2) # Normalize the frequency
        b, a = signal.butter(4, w, btype='highpass')
        rec = signal.filtfilt(b, a, data, axis=kwargs['axis'])
    if need_fix:
        return rec.reshape(-1)#.detach().numpy()
    else:
        return rec#.detach().numpy()

def high_pass(xda, time_dim='time', threshold=3*12*30.4*86400, obs_per_second=1/(86400*30.4) ):
    """
    Kyle Hall - packaging in xarray
    """
    assert time_dim in xda.dims, '{} not present on data'.format(time_dim)
    dim_sizes = list(xda.shape) # this is the sizes of the dimensions
    dim_names = list(xda.dims) # this is the names of the dimensions
    potential_dims = 'abcdefghijkmnopqrsuvwxyz' # alphabet without l - longitude and t - time
    assert len(dim_sizes) < 24, 'how literally dare you use 24-dimensional data- are you a monster?'
    dim_str = potential_dims[:len(dim_sizes)] # this is an einstein summation index notation representing the data
    timedim_ndx = dim_names.index(time_dim)
    chunksizes = [1 for i in range(len(dim_sizes))]
    chunksizes[timedim_ndx] = dim_sizes[timedim_ndx]
    data = da.from_array(xda.values, chunks=chunksizes)
    kwargs = {'obs_per_second': obs_per_second, 'threshold': threshold, 'axis': timedim_ndx} 
    results = da.blockwise(high_pass_filter, dim_str, data, dim_str, dtype=float, concatenate=True,  kwargs=kwargs).persist()
    return xr.ones_like(xda, dtype=float) * results


def band_pass_filter(data, kwargs={}, transpose_data=False):
    """
    Kyle Hall - packaging in xarray
    """
    data = torch.from_numpy(data)
    need_fix = False
    if len(list(data.shape)) == 1:
        data = data.reshape(-1, 1)
        need_fix=True
    if False:
        data = data - data.mean(dim=0)
        frequencies = torch.fft.rfftfreq(data.shape[0], d=1/kwargs['obs_per_second']) #[1:]
        fourier_coeffs = torch.fft.rfft(data, dim=0)
        fourier_coeffs[(frequencies > (1 / kwargs['high_threshold'])) | (frequencies < (1 / kwargs['low_threshold']))] = 0
        rec = torch.fft.irfft(fourier_coeffs, dim=0, n=data.shape[0])
    else:
        fc = 1/kwargs['low_threshold']  # Cut-off frequency of the filter
        w = fc / (kwargs['obs_per_second']/2) # Normalize the frequency
        fc2 = 1/kwargs['high_threshold']  # Cut-off frequency of the filter
        w2 = fc2 / (kwargs['obs_per_second']/2) # Normalize the frequency
        
        b, a = signal.butter(4, [w, w2], btype='bandpass')
        rec = signal.filtfilt(b, a, data, axis=0)
    if need_fix:
        return rec.reshape(-1)#.detach().numpy()
    else:
        return rec#.detach().numpy()
    
def band_pass(xda, time_dim='time', high_threshold=3*12*30.4*86400, low_threshold=7*12*30.4*86400, obs_per_second=1/(86400*30.4) ):
    """
    Kyle Hall - packaging in xarray
    """
    assert time_dim in xda.dims, '{} not present on data'.format(time_dim)
    dim_sizes = list(xda.shape) # this is the sizes of the dimensions
    dim_names = list(xda.dims) # this is the names of the dimensions
    potential_dims = 'abcdefghijkmnopqrsuvwxyz' # alphabet without l - longitude and t - time
    assert len(dim_sizes) < 24, 'how literally dare you use 24-dimensional data- are you a monster?'
    dim_str = potential_dims[:len(dim_sizes)] # this is an einstein summation index notation representing the data
    timedim_ndx = dim_names.index(time_dim)
    chunksizes = [1 for i in range(len(dim_sizes))]
    chunksizes[timedim_ndx] = dim_sizes[timedim_ndx]
    data = da.from_array(xda.values, chunks=chunksizes)
    kwargs = {'obs_per_second': obs_per_second, 'high_threshold': high_threshold, 'low_threshold': low_threshold} 
    results = da.blockwise(band_pass_filter, dim_str, data, dim_str, dtype=float, concatenate=True,  kwargs=kwargs).persist()
    return xr.ones_like(xda, dtype=float) * results
