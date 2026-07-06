import numpy as np
import matplotlib.pyplot as plt

def plot_wasserstein_progression(
    results_a,
    results_b,
    mode_labels=None,
    labels=("A", "B"),
    figsize=(8, 5)
):
    """
    Plot Wasserstein distance progression for multiple modes on separate axes,
    overlaying two results objects for comparison.

    Parameters
    ----------
    results_a, results_b : dict
        Output of wasserstein_time_progression / wasserstein_dist_on_pcs
    mode_labels : list of str, optional
        Labels for each mode
    labels : tuple of str
        Legend labels for results_a and results_b
    figsize : tuple
        Base figure size (height scaled by n_modes)
    """

    # --- unpack ---
    fractions_a = results_a['fractions']
    fractions_b = results_b['fractions']

    if not np.allclose(fractions_a, fractions_b):
        raise ValueError("Fractions in results_a and results_b must match")

    fractions = fractions_a

    median_a, low_a, high_a = (
        results_a['median'],
        results_a['low'],
        results_a['high'],
    )
    median_b, low_b, high_b = (
        results_b['median'],
        results_b['low'],
        results_b['high'],
    )

    n_modes = median_a.shape[0]

    if median_b.shape[0] != n_modes:
        raise ValueError("results_a and results_b must have the same number of modes")

    # --- mode labels ---
    if mode_labels is None:
        mode_labels = [f"Mode {i}" for i in range(n_modes)]
    elif len(mode_labels) != n_modes:
        raise ValueError(
            f"mode_labels has length {len(mode_labels)} but number of modes is {n_modes}"
        )

    # --- plotting ---
    fig, axes = plt.subplots(
        n_modes, 1,
        figsize=(figsize[0], figsize[1] * n_modes),
        sharex=True
    )

    if n_modes == 1:
        axes = [axes]

    color_a = "tab:blue"
    color_b = "tab:orange"

    for m, ax in enumerate(axes):
        # results A
        ax.plot(
            fractions,
            median_a[m],
            marker='o',
            color=color_a,
            label=labels[0]
        )
        ax.fill_between(
            fractions,
            low_a[m],
            high_a[m],
            color=color_a,
            alpha=0.25
        )

        # results B
        ax.plot(
            fractions,
            median_b[m],
            marker='s',
            color=color_b,
            label=labels[1]
        )
        ax.fill_between(
            fractions,
            low_b[m],
            high_b[m],
            color=color_b,
            alpha=0.25
        )

        ax.set_ylabel("Wasserstein dist.")
        ax.set_title(mode_labels[m])
        ax.grid(True)

        # Only need legend once
        if m == 0:
            ax.legend()

    axes[-1].set_xlabel("Fraction of first contiguous part")

    plt.tight_layout()
    plt.show()
