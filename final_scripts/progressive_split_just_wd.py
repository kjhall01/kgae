#!/usr/bin/env python3
"""
progressive_split_figure_wd_only.py

Same as progressive_split_figure_aies.py but retains only panels:
  a) Wasserstein distance (Interannual / PC1)
  b) Wasserstein distance (Decadal / PC2)

Everything else (pattern corr + gaps) removed.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance


# -----------------------
# CONFIG
# -----------------------
ROOT = Path("progressive-split")

MODE_INTERANNUAL = 4
MODE_DECADAL = 5
PC_INTERANNUAL = 1
PC_DECADAL = 2

N_BOOT = 1000
BLOCK = 12
RNG_SEED = 123

COLOR_KGAE = "#1f77b4"   # blue
COLOR_PCA  = "#ff7f0e"   # orange

STYLE = {
    "wd_within": dict(linestyle="-",  linewidth=2.2),
    "wd_shift":  dict(linestyle="--", linewidth=2.2),
}

FILE_TRAIN_LAT = "train_latents.nc"
FILE_VAL_LAT   = "val_latents.nc"
FILE_TEST_LAT  = "test_latents.nc"

FILE_PCA_TRAIN_LAT = "pca_train_latents.nc"
FILE_PCA_VAL_LAT   = "pca_val_latents.nc"
FILE_PCA_TEST_LAT  = "pca_test_latents.nc"


# -----------------------
# Utilities
# -----------------------
def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def parse_split_dirname(name: str) -> Optional[pd.Timestamp]:
    m = re.match(r"^(\d{4})-(\d{1,2})$", name)
    if not m:
        return None
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)


def try_open_dataarray_any(path: Path) -> Optional[xr.DataArray]:
    if not path.exists():
        return None
    try:
        ds = xr.open_dataset(path)
        if len(ds.data_vars) == 1:
            return next(iter(ds.data_vars.values()))
        for key in ["sst", "pc", "eof", "alpha", "beta"]:
            if key in ds:
                return ds[key]
        return next(iter(ds.data_vars.values()))
    except Exception as e:
        warn(f"Failed opening {path}: {e}")
        return None


def load_latent(sd: Path, fold: int, fname: str, mode: int) -> Optional[xr.DataArray]:
    fp = sd / f"fold{fold}" / fname
    da = try_open_dataarray_any(fp)
    if da is None:
        return None
    try:
        return da.sel(mode=mode)
    except Exception:
        return None


def ensemble_mean_over_seeds(arrs: List[xr.DataArray], dim: str = "seed") -> Optional[xr.DataArray]:
    if len(arrs) == 0:
        return None
    try:
        cat = xr.concat(arrs, dim=dim)
        return cat.mean(dim, skipna=True)
    except Exception:
        return None


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=k)
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
    return idx


def summarise_boot(b: Optional[np.ndarray]) -> np.ndarray:
    if b is None or len(b) == 0 or not np.isfinite(b).any():
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    b = b[np.isfinite(b)]
    if len(b) == 1:
        return np.array([b[0], b[0], b[0]], dtype=float)
    return np.array([np.nanmedian(b), np.nanquantile(b, 0.05), np.nanquantile(b, 0.95)], dtype=float)


def bootstrap_wasserstein_1d(
    x: np.ndarray, y: np.ndarray, n_boot: int, block: int, rng: np.random.Generator
) -> Optional[np.ndarray]:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return None
    if len(x) < block or len(y) < block:
        return np.array([wasserstein_distance(x, y)], dtype=float)

    out = np.empty(n_boot, dtype=float)
    nx, ny = len(x), len(y)
    for i in range(n_boot):
        ix = block_bootstrap_indices(nx, block, rng)
        iy = block_bootstrap_indices(ny, block, rng)
        out[i] = wasserstein_distance(x[ix], y[iy])
    return out


@dataclass
class SplitPaths:
    split_date: pd.Timestamp
    split_dir: Path
    seed_dirs: List[Path]


def discover_splits(root: Path) -> List[SplitPaths]:
    splits: List[SplitPaths] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        ts = parse_split_dirname(d.name)
        if ts is None:
            continue
        seed_dirs = sorted([p for p in d.iterdir() if p.is_dir() and p.name.startswith("seed")])
        splits.append(SplitPaths(ts, d, seed_dirs))
    splits.sort(key=lambda s: s.split_date)
    return splits


def compute_wasserstein_panels(splits: List[SplitPaths], mode: int, pc: int) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(RNG_SEED)

    out = {"kgae": {"train_val": [], "val_test": []},
           "pca":  {"train_val": [], "val_test": []}}

    for sp in splits:
        seed0 = sp.split_dir / "seed0"

        k_tv_draws, k_vt_draws = [], []
        p_tv_draws, p_vt_draws = [], []

        for fold in range(5):
            # ---- KGAE: ensemble mean over seeds FIRST (per fold) ----
            kt_list, kv_list, kq_list = [], [], []
            for sd in sp.seed_dirs:
                kt = load_latent(sd, fold, FILE_TRAIN_LAT, mode)
                kv = load_latent(sd, fold, FILE_VAL_LAT, mode)
                kq = load_latent(sd, fold, FILE_TEST_LAT, mode)
                if kt is None or kv is None or kq is None:
                    continue
                kt_list.append(kt); kv_list.append(kv); kq_list.append(kq)

            ktm = ensemble_mean_over_seeds(kt_list)
            kvm = ensemble_mean_over_seeds(kv_list)
            kqm = ensemble_mean_over_seeds(kq_list)

            if ktm is not None and kvm is not None:
                b = bootstrap_wasserstein_1d(ktm.values, kvm.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    k_tv_draws.append(b)

            if kvm is not None and kqm is not None:
                b = bootstrap_wasserstein_1d(kvm.values, kqm.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    k_vt_draws.append(b)

            # ---- PCA: seed0 only (per fold) ----
            pt = load_latent(seed0, fold, FILE_PCA_TRAIN_LAT, pc)
            pv = load_latent(seed0, fold, FILE_PCA_VAL_LAT, pc)
            pq = load_latent(seed0, fold, FILE_PCA_TEST_LAT, pc)

            if pt is not None and pv is not None:
                b = bootstrap_wasserstein_1d(pt.values, pv.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    p_tv_draws.append(b)

            if pv is not None and pq is not None:
                b = bootstrap_wasserstein_1d(pv.values, pq.values, N_BOOT, BLOCK, rng)
                if b is not None and len(b) == N_BOOT:
                    p_vt_draws.append(b)

        def foldmean_summary(draws: List[np.ndarray]) -> np.ndarray:
            if len(draws) == 0:
                return np.array([np.nan, np.nan, np.nan], dtype=float)
            D = np.vstack(draws)            # (n_folds, N_BOOT)
            m = np.nanmean(D, axis=0)       # (N_BOOT,)
            return summarise_boot(m)

        out["kgae"]["train_val"].append(foldmean_summary(k_tv_draws))
        out["kgae"]["val_test"].append(foldmean_summary(k_vt_draws))
        out["pca"]["train_val"].append(foldmean_summary(p_tv_draws))
        out["pca"]["val_test"].append(foldmean_summary(p_vt_draws))

    for model in out:
        for key in out[model]:
            out[model][key] = np.vstack(out[model][key])
    return out


# -----------------------
# Plot helper
# -----------------------
def plot_ci(ax, x, summary, style, label=None, zorder=3, alpha_fill=0.14):
    med, lo, hi = summary[:, 0], summary[:, 1], summary[:, 2]
    ax.plot(x, med, label=label, zorder=zorder, **style)
    ax.fill_between(x, lo, hi, color=style.get("color", "0.2"), alpha=alpha_fill, linewidth=0, zorder=zorder-1)


# -----------------------
# Main
# -----------------------
def main():
    if not ROOT.exists():
        raise SystemExit(f"Root directory not found: {ROOT.resolve()}")

    splits = discover_splits(ROOT)
    if len(splits) == 0:
        raise SystemExit("No split directories found.")

    x_dates = [sp.split_date for sp in splits]
    x = np.arange(len(splits))

    w_inter = compute_wasserstein_panels(splits, MODE_INTERANNUAL, PC_INTERANNUAL)
    w_deca  = compute_wasserstein_panels(splits, MODE_DECADAL,     PC_DECADAL)

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
    })

    fig = plt.figure(figsize=(7.8, 5.1))
    gs = GridSpec(nrows=2, ncols=1, hspace=0.28, figure=fig)

    ax_w_i = fig.add_subplot(gs[0, 0])
    ax_w_d = fig.add_subplot(gs[1, 0])

    def add_panel_letter(ax, letter):
        ax.text(
            -0.02, 1.05, f"{letter})",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=11, fontweight="bold",
            clip_on=False
        )

    # Panel a: WD Interannual / PC1
    plot_ci(ax_w_i, x, w_inter["kgae"]["train_val"], dict(color=COLOR_KGAE, **STYLE["wd_within"]))
    plot_ci(ax_w_i, x, w_inter["kgae"]["val_test"],  dict(color=COLOR_KGAE, **STYLE["wd_shift"]))
    plot_ci(ax_w_i, x, w_inter["pca"]["train_val"],  dict(color=COLOR_PCA,  **STYLE["wd_within"]))
    plot_ci(ax_w_i, x, w_inter["pca"]["val_test"],   dict(color=COLOR_PCA,  **STYLE["wd_shift"]))

    ax_w_i.set_ylabel("Wasserstein distance")
    ax_w_i.set_xlim(x.min(), x.max())
    add_panel_letter(ax_w_i, "a")

    # Panel b: WD Decadal / PC2
    plot_ci(ax_w_d, x, w_deca["kgae"]["train_val"], dict(color=COLOR_KGAE, **STYLE["wd_within"]))
    plot_ci(ax_w_d, x, w_deca["kgae"]["val_test"],  dict(color=COLOR_KGAE, **STYLE["wd_shift"]))
    plot_ci(ax_w_d, x, w_deca["pca"]["train_val"],  dict(color=COLOR_PCA,  **STYLE["wd_within"]))
    plot_ci(ax_w_d, x, w_deca["pca"]["val_test"],   dict(color=COLOR_PCA,  **STYLE["wd_shift"]))

    ax_w_d.set_ylabel("Wasserstein distance")
    ax_w_d.set_xlim(x.min(), x.max())
    ax_w_d.set_xlabel("Split date (training end)")
    add_panel_letter(ax_w_d, "b")

    # X ticks: equally spaced labels
    n = len(x_dates)
    tick_step = max(1, int(np.ceil(n / 8)))
    tick_idx = np.arange(0, n, tick_step)
    tick_labels = [x_dates[i].strftime("%Y-%m") for i in tick_idx]

    for ax in (ax_w_i, ax_w_d):
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")

    # Legend (compact)
    handles = [
        Line2D([0], [0], color=COLOR_KGAE, linewidth=3, label="KGAE"),
        Line2D([0], [0], color=COLOR_PCA,  linewidth=3, label="PCA"),
        Line2D([0], [0], color="k", **STYLE["wd_within"], label="Train–Val WD"),
        Line2D([0], [0], color="k", **STYLE["wd_shift"],  label="Val–Test WD"),
    ]

    fig.subplots_adjust(left=0.09, right=0.995, top=0.96, bottom=0.18)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=3.0,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.02),
    )

    outpng = Path("progressive_split_wd_only.png")
    outpdf = Path("progressive_split_wd_only.pdf")
    fig.savefig(outpng, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outpdf, bbox_inches="tight", pad_inches=0.04)
    print(f"Wrote: {outpng.resolve()}")
    print(f"Wrote: {outpdf.resolve()}")


if __name__ == "__main__":
    main()