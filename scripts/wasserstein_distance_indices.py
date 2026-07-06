import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import xarray as xr
import kgae

def wasserstein_time_progression(da, fractions=np.linspace(0.5, 0.9, 9),
                                 n_boot=200, seed=42):
    """
    Compute Wasserstein distance between contiguous splits of a time series
    for multiple modes.

    Parameters
    ----------
    da : xarray.DataArray
        Shape (time, mode). Each mode is treated independently.
    fractions : array-like
        Fractions of the time series to assign to the first contiguous part.
    n_boot : int
        Number of bootstrap resamples to estimate sampling uncertainty.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        'fractions': fractions
        'median': array of shape (n_modes, n_fractions)
        'low': 5th percentile (CI)
        'high': 95th percentile (CI)
    """
    np.random.seed(seed)
    N, M = da.shape
    fractions = np.asarray(fractions)

    median = np.zeros((M, len(fractions)))
    low = np.zeros_like(median)
    high = np.zeros_like(median)

    for i, f in enumerate(fractions):
        split = int(f * N)
        if split == 0 or split >= N:
            raise ValueError(f"Fraction {f} leads to invalid split {split} for N={N}")
        for m in range(M):
            part1 = da[:, m][:split].values
            part2 = da[:, m][split:].values
            part1 = part1[np.isfinite(part1)]
            part2 = part2[np.isfinite(part2)]

            ws_boot = []
            for _ in range(n_boot):
                sample1 = np.random.choice(part1, size=len(part1), replace=True)
                sample2 = np.random.choice(part2, size=len(part2), replace=True)
                ws_boot.append(wasserstein_distance(sample1, sample2))

            ws_boot = np.array(ws_boot)
            median[m, i] = np.median(ws_boot)
            low[m, i] = np.percentile(ws_boot, 5)
            high[m, i] = np.percentile(ws_boot, 95)

    results = {
        'fractions': fractions,
        'median': median,
        'low': low,
        'high': high
    }
    return results


def plot_wasserstein_progression(results, mode_labels=None, figsize=(8,5)):
    """
    Plot Wasserstein distance progression for multiple modes on the same axis.
    """
    fractions = results['fractions']
    n_modes = results['median'].shape[0]

    # Ensure mode_labels is consistent
    if mode_labels is None:
        mode_labels = [f"Mode {i}" for i in range(n_modes)]
    elif len(mode_labels) != n_modes:
        raise ValueError(f"mode_labels has length {len(mode_labels)} but number of modes is {n_modes}")

    plt.figure(figsize=figsize)
    colors = plt.cm.tab10(np.arange(n_modes))

    for m in range(n_modes):
        plt.plot(fractions, results['median'][m], marker='o', color=colors[m], label=mode_labels[m])
        plt.fill_between(fractions,
                         results['low'][m],
                         results['high'][m],
                         color=colors[m],
                         alpha=0.2)

    plt.xlabel("Fraction of first contiguous part")
    plt.ylabel("Wasserstein distance")
    plt.title("Contiguous-time Wasserstein distance progression")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    # Load indices
    enso = kgae.open_oni()              # xarray.DataArray with 'time' dimension
    pdo = kgae.open_pdo().mean('dataset').sel(time=enso.time)

    # Combine into a single DataArray with a 'mode' dimension
    combined = xr.concat([pdo, enso], dim='mode')
    combined = combined.assign_coords(mode=['PDO', 'ENSO']).transpose('time', 'mode')  # give meaningful names

    # Compute Wasserstein distance progression
    results = wasserstein_time_progression(combined, fractions=np.linspace(0.5, 0.9, 21), n_boot=200)

    # Plot results
    plot_wasserstein_progression(results, mode_labels=['PDO', 'ENSO'])

    plt.figure()
    combined.plot.line(x='time')
    plt.show() 