import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import kgae  # needed so pickled KGAE objects can be deserialized


"""
Estimate how many ensemble members are needed for a subset mean to approximate
the full-ensemble mean of explicit decoder-probe patterns.

This version uses:

    linear(target i) = D(+e_i) - D(-e_i)

and the state-dependence objects from the supplied figure3.py / figure5.py
logic, namely

    S_{i<-j} = [N_i(+e_j) - N_i(-e_j)] / (2 a_j)

where

    N_i(z_j) = [D(z_j, +e_i) + D(z_j, -e_i) - 2 D(z_j, 0)] / a_i^2.

Rows of the final 4x3 figure are:
    1) linear(target)
    2) Decadal mode impact on {target}
    3) Interannual mode impact on {target}
    4) Quasibiennial mode impact on {target}

Per the user's request, each seed estimate is formed by averaging these probe
patterns across the saved cross-validation fold models for that seed.
"""


MODES_TO_EXAMINE = [5, 4, 3]
MODE_LABELS = {
    5: "Decadal Mode",
    4: "Interannual Mode",
    3: "Quasibiennial Mode",
}


def plot_vs_ensemble_size(
    da: xr.DataArray,
    ax=None,
    dim: str = "ensemble_size",
    label: str | None = None,
    lw: float = 2.0,
    marker: str | None = None,
):
    if dim not in da.dims:
        raise ValueError(f"DataArray must contain dimension {dim!r}. Got dims {da.dims}")
    if da.ndim != 1:
        raise ValueError(f"DataArray must be 1D. Got shape {da.shape}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    x = da[dim].values
    y = da.values

    ax.plot(x, y, lw=lw, marker=marker, label=label)
    ax.set_xlabel(dim)
    ax.set_ylabel(da.name if da.name is not None else "value")
    ax.set_ylim(0.0, 1.1)
    ax.axhline(0.95, linewidth=2, linestyle=":")
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend(frameon=False)

    return fig, ax


def moving_block_bootstrap_indices(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    if block_len <= 1:
        return rng.integers(0, n, size=n)

    if n <= block_len:
        start = int(rng.integers(0, n))
        return (np.arange(n) + start) % n

    n_starts = n - block_len + 1
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n_starts, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)
    return idx[:n]


def bootstrap_rel_l2_dist(
    da_full,
    n_resamples=1000,
    dim="seed",
    block_length=1,
    rng_seed=-1,
    cl=0.05,
    ensemble_size=4,
):
    rng = np.random.default_rng(rng_seed)
    full_ensemble_size = da_full.sizes[dim]
    full_mean = da_full.mean(dim)

    ests = []
    for boot_idx in range(n_resamples):
        print(boot_idx)
        _ = moving_block_bootstrap_indices  # keep helper available / silence lint
        idx = rng.choice(full_ensemble_size, size=ensemble_size, replace=False)
        print(idx)
        boot_mean = da_full.isel(**{dim: idx}).mean(dim)
        corr = xr.corr(boot_mean, full_mean, dim=["lat", "lon"])
        ests.append(corr)

    return xr.concat(ests, "boot").quantile(cl, "boot")


def search_ensemble_sizes(
    da_full,
    n_resamples=1000,
    dim="seed",
    block_length=1,
    rng_seed=-1,
    ensemble_sizes=(3, 5, 7, 10, 15, 20, 25, 30, 40, 50),
    cl=0.05,
):
    out = []
    for es in ensemble_sizes:
        print("STARTING ES:", es)
        metric = bootstrap_rel_l2_dist(
            da_full,
            ensemble_size=es,
            n_resamples=n_resamples,
            dim=dim,
            block_length=block_length,
            rng_seed=rng_seed,
            cl=cl,
        )
        out.append(metric)

    out = xr.concat(out, "ensemble_size")
    return out.assign_coords({"ensemble_size": list(ensemble_sizes)})


def _seed_sort_key(path: Path):
    stem = path.name
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else stem


def _get_feature_template(experiment_dir: Path) -> xr.DataArray:
    root_reference = experiment_dir / "xval_references.nc"
    if root_reference.exists():
        ref = xr.open_dataarray(root_reference)
        feature = ref.isel(time=0).stack(feature=("lat", "lon")).dropna("feature", how="any").feature
        ref.close()
        return feature

    fallback = experiment_dir / "seed0" / "xval_reconstructions.nc"
    if fallback.exists():
        ref = xr.open_dataarray(fallback)
        feature = ref.isel(time=0).stack(feature=("lat", "lon")).dropna("feature", how="any").feature
        ref.close()
        return feature

    raise FileNotFoundError(
        "Could not locate xval_references.nc or seed0/xval_reconstructions.nc "
        f"inside {experiment_dir}."
    )


def _load_model(model_path: Path, device: str = "cpu"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.device = torch.device(device)
    model.to(model.device)
    model.eval()
    return model


@torch.no_grad()
def decode_model(model, z_np: np.ndarray, device: str = "cpu") -> np.ndarray:
    latent_dim = model.latent_dim
    z_np = np.asarray(z_np, dtype=np.float32)
    if z_np.shape != (latent_dim,):
        raise ValueError(f"Expected latent vector shape {(latent_dim,)}, got {z_np.shape}")

    if hasattr(model, "flip_signs") and model.flip_signs is not None:
        signs = np.asarray(model.flip_signs.values, dtype=np.float32)
    else:
        signs = np.ones(latent_dim, dtype=np.float32)

    z = torch.tensor((z_np * signs)[None, :], dtype=torch.float32, device=device)

    if hasattr(model, "decode") and callable(getattr(model, "decode")):
        x = model.decode(z)
    elif hasattr(model, "decoder") and callable(getattr(model, "decoder")):
        x = model.decoder(z)
    else:
        raise AttributeError("Model has neither callable .decode nor callable .decoder")

    return x.detach().cpu().numpy().squeeze()


def linear_probe(model, target_mode_zero_based: int, amplitude: float = 1.0, device: str = "cpu") -> np.ndarray:
    z_plus = np.zeros(model.latent_dim, dtype=np.float32)
    z_minus = np.zeros(model.latent_dim, dtype=np.float32)
    z_plus[target_mode_zero_based] = +amplitude
    z_minus[target_mode_zero_based] = -amplitude
    return decode_model(model, z_plus, device=device) - decode_model(model, z_minus, device=device)


def nonlinear_response_target_given_bg(
    model,
    *,
    i_target_zero_based: int,
    j_bg_zero_based: int,
    z_bg: float,
    a_target: float,
    device: str = "cpu",
) -> np.ndarray:
    z0 = np.zeros(model.latent_dim, dtype=np.float32)
    zp = np.zeros(model.latent_dim, dtype=np.float32)
    zm = np.zeros(model.latent_dim, dtype=np.float32)

    z0[j_bg_zero_based] = z_bg
    zp[j_bg_zero_based] = z_bg
    zm[j_bg_zero_based] = z_bg

    zp[i_target_zero_based] = +a_target
    zm[i_target_zero_based] = -a_target

    return (
        decode_model(model, zp, device=device)
        + decode_model(model, zm, device=device)
        - 2.0 * decode_model(model, z0, device=device)
    ) / (a_target * a_target)


def state_dependence_probe(
    model,
    *,
    target_mode_zero_based: int,
    background_mode_zero_based: int,
    a_target: float = 1.0,
    a_bg: float = 1.0,
    device: str = "cpu",
) -> np.ndarray:
    n_plus = nonlinear_response_target_given_bg(
        model,
        i_target_zero_based=target_mode_zero_based,
        j_bg_zero_based=background_mode_zero_based,
        z_bg=+a_bg,
        a_target=a_target,
        device=device,
    )
    n_minus = nonlinear_response_target_given_bg(
        model,
        i_target_zero_based=target_mode_zero_based,
        j_bg_zero_based=background_mode_zero_based,
        z_bg=-a_bg,
        a_target=a_target,
        device=device,
    )
    return (n_plus - n_minus) / (2.0 * a_bg)


def compute_seed_probe_patterns(
    experiment_dir: str | Path,
    n_ensemble: int | None = None,
    linear_amplitude: float = 1.0,
    state_target_amplitude: float = 1.0,
    state_background_amplitude: float = 1.0,
    device: str = "cpu",
    cache_path: str | Path | None = None,
    overwrite: bool = False,
    modes_to_examine: list[int] | tuple[int, ...] = tuple(MODES_TO_EXAMINE),
):
    experiment_dir = Path(experiment_dir)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite:
            return xr.open_dataset(cache_path)

    feature = _get_feature_template(experiment_dir)
    target_mode_coord = np.asarray(modes_to_examine, dtype=int)
    background_mode_coord = np.asarray(modes_to_examine, dtype=int)

    seed_dirs = sorted([p for p in experiment_dir.glob("seed*") if p.is_dir()], key=_seed_sort_key)
    if n_ensemble is not None:
        seed_dirs = seed_dirs[:n_ensemble]

    linear_by_seed = []
    state_dependence_by_seed = []

    for seed_idx, seed_dir in enumerate(seed_dirs):
        print(f"Loading seed {seed_dir.name} ({seed_idx + 1}/{len(seed_dirs)})")
        split_dirs = sorted([p for p in seed_dir.glob("split*") if p.is_dir()], key=_seed_sort_key)
        if not split_dirs:
            raise FileNotFoundError(f"No split directories found inside {seed_dir}")

        seed_linear_folds = []
        seed_state_folds = []
        for split_dir in split_dirs:
            model_path = split_dir / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

            model = _load_model(model_path, device=device)

            linear_maps = []
            for target_mode in target_mode_coord:
                linear_maps.append(
                    linear_probe(
                        model,
                        target_mode_zero_based=int(target_mode) - 1,
                        amplitude=linear_amplitude,
                        device=device,
                    )
                )
            linear_np = np.stack(linear_maps, axis=0)

            state_maps_by_bg = []
            for background_mode in background_mode_coord:
                row = []
                for target_mode in target_mode_coord:
                    row.append(
                        state_dependence_probe(
                            model,
                            target_mode_zero_based=int(target_mode) - 1,
                            background_mode_zero_based=int(background_mode) - 1,
                            a_target=state_target_amplitude,
                            a_bg=state_background_amplitude,
                            device=device,
                        )
                    )
                state_maps_by_bg.append(np.stack(row, axis=0))
            state_np = np.stack(state_maps_by_bg, axis=0)

            linear_da = xr.DataArray(
                linear_np,
                coords={"target_mode": target_mode_coord, "feature": feature},
                dims=["target_mode", "feature"],
            ).unstack("feature")
            state_da = xr.DataArray(
                state_np,
                coords={
                    "background_mode": background_mode_coord,
                    "target_mode": target_mode_coord,
                    "feature": feature,
                },
                dims=["background_mode", "target_mode", "feature"],
            ).unstack("feature")

            seed_linear_folds.append(linear_da.expand_dims({"fold": [split_dir.name]}))
            seed_state_folds.append(state_da.expand_dims({"fold": [split_dir.name]}))

        seed_linear = xr.concat(seed_linear_folds, "fold").mean("fold")
        seed_state = xr.concat(seed_state_folds, "fold").mean("fold")

        seed_label = seed_dir.name
        linear_by_seed.append(seed_linear.expand_dims({"seed": [seed_label]}))
        state_dependence_by_seed.append(seed_state.expand_dims({"seed": [seed_label]}))

    ds = xr.Dataset(
        {
            "linear": xr.concat(linear_by_seed, "seed"),
            "state_dependence": xr.concat(state_dependence_by_seed, "seed"),
        }
    )
    ds.attrs["linear_amplitude"] = linear_amplitude
    ds.attrs["state_target_amplitude"] = state_target_amplitude
    ds.attrs["state_background_amplitude"] = state_background_amplitude
    ds.attrs["notes"] = (
        "Per-seed decoder probe patterns averaged across fold models. "
        "linear(target) = D(+) - D(-). "
        "state_dependence(background,target) = [N_target(+background) - N_target(-background)]/(2 a_bg), "
        "with N_target(z_bg) = [D(z_bg,+target)+D(z_bg,-target)-2D(z_bg,0)]/a_target^2."
    )

    if cache_path is not None:
        ds.to_netcdf(cache_path)

    return ds


def _row_ylabel(background_mode: int | None) -> str:
    if background_mode is None:
        return "Pattern Correlation\n(linear)"
    return f"Pattern Correlation\n({MODE_LABELS[int(background_mode)].replace(' Mode', '')})"


if __name__ == "__main__":
    experiment = "../../final_scripts/large-ensemble-2-1940-2014"
    n_ensemble = 100
    n_bootstraps = 1000
    overwrite = False
    cl = 0.05
    ensemble_sizes_to_search = range(1, 20)

    linear_amplitude = 1.0
    state_target_amplitude = 1.0
    state_background_amplitude = 1.0

    modes_to_examine = list(MODES_TO_EXAMINE)
    mode_labels = [MODE_LABELS[m] for m in modes_to_examine]

    cache_file = Path(experiment) / (
        "direct_decoder_linear_state_dependence_foldavg"
        f"_lin{linear_amplitude:g}_tgt{state_target_amplitude:g}_bg{state_background_amplitude:g}.nc"
    )
    probe_ds = compute_seed_probe_patterns(
        experiment,
        n_ensemble=n_ensemble,
        linear_amplitude=linear_amplitude,
        state_target_amplitude=state_target_amplitude,
        state_background_amplitude=state_background_amplitude,
        device="cpu",
        cache_path=cache_file,
        overwrite=overwrite,
        modes_to_examine=modes_to_examine,
    )

    linear_ensemble_sizes = search_ensemble_sizes(
        probe_ds["linear"].sel(target_mode=modes_to_examine),
        n_resamples=n_bootstraps,
        block_length=1,
        rng_seed=0,
        dim="seed",
        ensemble_sizes=ensemble_sizes_to_search,
        cl=cl,
    )

    state_ensemble_sizes = search_ensemble_sizes(
        probe_ds["state_dependence"].sel(
            background_mode=modes_to_examine,
            target_mode=modes_to_examine,
        ),
        n_resamples=n_bootstraps,
        block_length=1,
        rng_seed=0,
        dim="seed",
        ensemble_sizes=ensemble_sizes_to_search,
        cl=cl,
    )

    n_modes = len(modes_to_examine)
    n_rows = n_modes + 1
    n_cols = n_modes
    pidx = 0
    panel_letters = [f"{chr(ord('a') + i)})" for i in range(n_rows * n_cols)]

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.3 * n_cols, 2.6 * n_rows),
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)

    for ci, target_mode in enumerate(modes_to_examine):
        ax = axes[0, ci]
        ax.text(
            0.0,
            1.07,
            panel_letters[pidx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )
        pidx += 1
        plot_vs_ensemble_size(linear_ensemble_sizes.sel(target_mode=target_mode), ax)
        ax.set_title(mode_labels[ci])
        ax.set_xlabel("")
        ax.set_ylabel("Pattern Correlation\n(to full mean, linear)" if ci == 0 else "")

        for ri, background_mode in enumerate(modes_to_examine, start=1):
            ax = axes[ri, ci]
            ax.text(
                0.0,
                1.07,
                panel_letters[pidx],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
            )
            pidx += 1
            plot_vs_ensemble_size(
                state_ensemble_sizes.sel(background_mode=background_mode, target_mode=target_mode),
                ax,
            )
            ax.set_title("")
            ax.set_ylabel(_row_ylabel(background_mode) if ci == 0 else "")
            ax.set_xlabel("Ensemble Size" if ri == n_rows - 1 else "")

    plt.savefig("large-ens-approximation-direct-decoder-probes.pdf")
    plt.show()
