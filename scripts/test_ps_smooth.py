import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np

def compute_smoothed_power_spectra(data, dx = 5, kernel_size = 7):
    N, M = data.shape
    # compute frequencies on which power spectra are evaluated
    frequencies = torch.fft.rfftfreq(N, d=dx).reshape(-1,1)

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
    t = torch.linspace(-2, 2, steps=kernel_size)  # 5 points from -2 to 2
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

    # renormalize smoothed power spectra
    sums = smoothed_frequencies.sum(dim=0)
    smoothed_frequencies = smoothed_frequencies / sums

    if kernel_size %2 == 1:
        return smoothed_power_spectra[kernel_size//2:-kernel_size//2+1], freqs.t()
    else:
        return smoothed_power_spectra[kernel_size//2:-kernel_size//2-1], freqs.t()

  #  ndxs = smoothed_frequencies.argsort(dim=0)[:, 0]
  #  return smoothed_power_spectra[ndxs], smoothed_frequencies[ndxs]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generate random data
    N, M = 512, 3
    dx = 5
    kernel_size = 8
    torch.manual_seed(0)
    data = torch.randn(N, M)

    # unsmoothed power spectra
    freqs = torch.fft.rfftfreq(N, d=dx)
    centered = data - data.mean(dim=0)
    coeffs = torch.fft.rfft(centered, dim=0)
    unsmoothed_power = (coeffs * torch.conj(coeffs)).real
    unsmoothed_power = unsmoothed_power / unsmoothed_power.sum(dim=0)

    # smoothed power spectra using the implemented function
    smoothed_power, smoothed_freqs = compute_smoothed_power_spectra(data, dx=dx, kernel_size=kernel_size)

    # plot one variable per axis, with unsmoothed and smoothed overlaid
    fig, axes = plt.subplots(M, 1, figsize=(8, 3 * M), sharex=True)
    if M == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(freqs.numpy(), unsmoothed_power[:, i].numpy(), label="Unsmooth", alpha=0.6)
        ax.plot(smoothed_freqs[:, i].numpy(), smoothed_power[:, i].numpy(), label="Smoothed", linewidth=2)
        ax.set_title(f"Dimension {i}")
        ax.set_ylabel("Normalized Power")
        ax.legend()
    axes[-1].set_xlabel("Frequency")
    plt.tight_layout()
    plt.show()
   