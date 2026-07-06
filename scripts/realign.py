import kgae 
import xarray as xr 
import matplotlib.pyplot as plt
import torch 

l = kgae.open_latents('xval', experiment='kgvae_cv1940-2014')
l2 = xr.concat([ l.isel(time=slice(i, None, 5)).drop('time') for i in range(5) ], 'fold')
modes = [1,2,3,4,5]
ks = 15
# plot one variable per axis, with unsmoothed and smoothed overlaid
fig, ax = plt.subplots(ncols=1, nrows=len(modes), figsize=(8, 3 * len(modes)), sharex=True)
for _, mode in enumerate(modes): 
    all_latents = l2.sel(mode=mode).stack(sample=('seed', 'fold'))
    all_latents = all_latents.transpose( 'time', 'sample',)
    torch_latents = torch.from_numpy(all_latents.values).float()
    psd, freqs = kgae.compute_smoothed_power_spectra(torch_latents, kernel_size=7, dx=5)

    for i in range(psd.shape[1]):
        ax[_].plot(1/freqs[:,i].numpy()/12, psd[:,i].numpy(), alpha=0.6)
        ax[_].set_title(f"Dimension {i}")
        ax[_].set_ylabel("Normalized Power")
        ax[_].legend()
    ax[_].set_xlabel(f"Frequency (Mode {_})")

    ticks = [40, 20, 12, 7, 5, 2, 1]
    ax[_].set_xscale('log', base=2)
    ax[_].set_xticks([t for t in ticks], labels=ticks)
plt.tight_layout()
plt.suptitle("Pre Realignment")



import numpy as np

def psd_based_mode_order(
    psd_fm: np.ndarray,
    freqs: np.ndarray | None = None,
    mode_axis: int = 1,
    peak_weight: float = 1.0,
    centroid_weight: float = 0.5,
    bandwidth_weight: float = 0.25,
    peak_to_total_weight: float = 0.25,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build robust PSD summary features per mode and return indices that sort modes
    from lowest-frequency to highest-frequency (by a composite score).

    Parameters
    ----------
    psd_fm : np.ndarray
        Power spectral density array with shape (freq, mode) (your stated convention).
        Values should be nonnegative.
    freqs : np.ndarray, optional
        Frequency coordinate with shape (freq,). If None, uses np.arange(nfreq),
        which still yields a consistent ordering but not physical units.
    mode_axis : int
        Axis corresponding to mode (default 1 for (freq, mode)).
    peak_weight, centroid_weight, bandwidth_weight, peak_to_total_weight : float
        Weights in the composite "frequency score". Higher score => higher-frequency mode.
        If you want the simplest behavior, set only peak_weight=1 and others=0.
    eps : float
        Numerical stability.

    Returns
    -------
    order : np.ndarray
        Integer indices such that psd_fm[:, order] sorts modes from low to high frequency.

    Notes
    -----
    Composite score uses:
      - peak frequency f_peak (argmax)
      - spectral centroid f_centroid
      - bandwidth (std around centroid)
      - peak-to-total power ratio (sharpness)
    The last term helps separate "spiky" high-freq modes from broad low-freq ones when peaks tie.
    """
    psd = np.asarray(psd_fm)
    if psd.ndim != 2:
        raise ValueError(f"psd_fm must be 2D (freq, mode); got shape {psd.shape}")

    # Move to (freq, mode)
    if mode_axis != 1:
        psd = np.moveaxis(psd, mode_axis, 1)

    nfreq, nmode = psd.shape
    if freqs is None:
        f = np.arange(nfreq, dtype=float)
    else:
        f = np.asarray(freqs, dtype=float)
        if f.shape != (nfreq,):
            raise ValueError(f"freqs must have shape ({nfreq},); got {f.shape}")

    # Ensure nonnegative (clip tiny negatives from numerical noise)
    psd = np.maximum(psd, 0.0)

    # Normalize each mode to a probability distribution over frequency
    P = psd / (np.sum(psd, axis=0, keepdims=True) + eps)  # (freq, mode)

    # Features
    f_centroid = np.sum(f[:, None] * P, axis=0)  # (mode,)
    f2_centroid = np.sum((f[:, None] ** 2) * P, axis=0)
    bandwidth = np.sqrt(np.maximum(f2_centroid - f_centroid**2, 0.0))  # (mode,)

    peak_idx = np.argmax(psd, axis=0)  # (mode,)
    f_peak = f[peak_idx]

    peak_power = psd[peak_idx, np.arange(nmode)]
    total_power = np.sum(psd, axis=0) + eps
    peak_to_total = peak_power / total_power  # (mode,)

    # Composite frequency score (higher => "higher-frequency")
    score = (
        peak_weight * f_peak
        + centroid_weight * f_centroid
        + bandwidth_weight * bandwidth
        + peak_to_total_weight * peak_to_total
    )

    # Sort from low to high frequency score
    order = np.argsort(score)

    return order



new_modes = {1: [], 2: [], 3: [], 4: [], 5: []}

for seed in range(30):
    for fold in range(5): 
        # get latents: 
        data = l.isel(time=slice(fold, None, 5), seed=seed)
        data = data.transpose('time', 'mode')
        torch_latents = torch.from_numpy(data.values).float()
        psd, freqs = kgae.compute_smoothed_power_spectra(torch_latents, kernel_size=ks, dx=5)
        col_ind = psd_based_mode_order(psd.numpy(), freqs[:, 0].numpy(), mode_axis=1 )
        #peak_ndxs = psd.argmax(dim=0)
        #frequencies_of_max_power = freqs[peak_ndxs, :].diag() 
        #col_ind = torch.argsort(frequencies_of_max_power).flip(0).cpu().detach().numpy() # sort highest-freq to lowest-freq
        for newmode in range(5):
            new_modes[newmode+1].append(torch_latents[:, col_ind[newmode]].reshape(-1,1))
new_modes = {k: torch.hstack(new_modes[k]) for k in new_modes.keys()}

# plot one variable per axis, with unsmoothed and smoothed overlaid
fig, ax = plt.subplots(ncols=1, nrows=len(modes), figsize=(8, 3 * len(modes)), sharex=True)
for _, mode in enumerate(modes): 
    torch_latents = new_modes[mode]
    psd, freqs = kgae.compute_smoothed_power_spectra(torch_latents, kernel_size=ks, dx=5)

    for i in range(psd.shape[1]):
        ax[_].plot(1/freqs[:,i].numpy()/12, psd[:,i].numpy(), alpha=0.6)
        ax[_].set_title(f"Dimension {i}")
        ax[_].set_ylabel("Normalized Power")
        ax[_].legend()
    ax[_].set_xlabel(f"Frequency (Mode {_})")

    ticks = [40, 20, 12, 7, 5, 2, 1]
    ax[_].set_xscale('log', base=2)
    ax[_].set_xticks([t for t in ticks], labels=ticks)
plt.tight_layout()
plt.suptitle("Post Realignment")
plt.show()
