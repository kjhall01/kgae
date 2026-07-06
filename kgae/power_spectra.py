import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np

def compute_smoothed_power_spectra(data, dx = 5, kernel_size = 7):
    N, M = data.shape
    # compute frequencies on which power spectra are evaluated
    frequencies = torch.fft.rfftfreq(N, d=dx).reshape(-1,1).to(data.device)

    # center data 
    data = data - data.mean(dim=0)

    #compute power spectra 
    fourier_coeffs = torch.fft.rfft(data, dim=0)
    fourier_coeffs = (fourier_coeffs*torch.conj(fourier_coeffs)).real

    # normalize power spectra - this makes the assumption that the total variance 
    # attributable to each latent dimension Z_i is equal -- this is then re-done
    total_variance = fourier_coeffs.sum(dim=0) 
    fourier_coeffs = fourier_coeffs / total_variance

    # smooth power spectra with a Gaussian kernel
    t = torch.linspace(-2, 2, steps=kernel_size).to(data.device)  # 5 points from -2 to 2
    kernel = torch.exp(-0.5 * (t[:, None])**2)  # shape (5, 1)
    kernel = kernel / kernel.sum(dim=0, keepdim=True)   # normalize
    kernel = kernel.repeat(1, M)  # shape (Z, M)

    # apply reflected padding to the power spectra and associated frequencies 
    padded_normed_power_spectra = F.pad(fourier_coeffs.t(), (kernel_size//2 , kernel_size//2), mode='reflect').t()
    padded_normed_power_spectra_unsqueezed = padded_normed_power_spectra.T.unsqueeze(0)  # shape (1, M, N)

    freqs = frequencies.repeat(1, M).t()
    padded_freqs = F.pad(freqs, (kernel_size//2, kernel_size//2), mode='reflect')
    padded_freqs_unsqueezed = padded_freqs.unsqueeze(0)  # shape (1, M, N)
    
    kernel_unsqueezed = kernel.T.unsqueeze(1)  # shape (M, 1, kernel_size)

    # convolve power spectra and frequencies with kernel
    smoothed_power_spectra = F.conv1d(padded_normed_power_spectra_unsqueezed, kernel_unsqueezed, padding=kernel_size//2, groups=M)
    smoothed_power_spectra = smoothed_power_spectra.squeeze(0).T  # back to shape (N, M)

    smoothed_frequencies = F.conv1d(padded_freqs_unsqueezed, kernel_unsqueezed, padding=kernel_size//2, groups=M)
    smoothed_frequencies = smoothed_frequencies.squeeze(0).T  # back to shape (N, M)


    if kernel_size %2 == 1:
        ret, retf = smoothed_power_spectra[kernel_size//2:-kernel_size//2+1], freqs.t()
    else:
        ret, retf = smoothed_power_spectra[kernel_size//2:-kernel_size//2-1], freqs.t()

    return ret / ret.sum(dim=0), retf