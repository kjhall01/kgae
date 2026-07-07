#!/usr/bin/env python3
"""
Fix time-coordinate mislabeling for train_latents.nc / train_reconstructions.nc
(and full_train_* equivalents) produced by vstack over interleaved train_splits.

This script:
- Walks progressive-split/<split>/seed*/fold*/ for fold-level train files (step=4)
- Walks progressive-split/<split>/seed*/ for full-model train files (step=5)
- Uses train_references.nc (or full_train_references if available; else fold0 refs) to get chronological time T
- Computes vstack_time = concat(T[j::step] for j in range(step))
- Reassigns the correct time coords to the misordered arrays
- Sorts by time
- Writes a backup *.bak.nc before overwriting unless --no-backup is passed
- Skips gracefully if files are missing or already-fixed (heuristic)
"""

from __future__ import annotations
from pathlib import Path
import argparse
import shutil
import xarray as xr
import numpy as np

ROOT = Path("progressive-split")

FOLD_TRAIN_REF = "train_references.nc"
FOLD_TRAIN_LAT = "train_latents.nc"
FOLD_TRAIN_REC = "pca_train_reconstructions.nc"

FULL_TRAIN_LAT = "full_train_latents.nc"
FULL_TRAIN_REC = "full_train_reconstructions.nc"

def vstack_time_from_T(T: np.ndarray, step: int) -> np.ndarray:
    # T is chronological time vector
    parts = [T[j::step] for j in range(step)]
    return np.concatenate(parts, axis=0)

def backup_file(fp: Path):
    bak = fp.with_suffix(fp.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(fp, bak)

def open_da(fp: Path) -> xr.DataArray:
    ds = xr.open_dataset(fp)
    if len(ds.data_vars) != 1:
        # pick first
        v = list(ds.data_vars)[0]
        return ds[v]
    return next(iter(ds.data_vars.values()))

def fix_one(train_ref_fp: Path, target_fp: Path, step: int, do_backup: bool = True) -> bool:
    if (not train_ref_fp.exists()) or (not target_fp.exists()):
        return False

    ref = open_da(train_ref_fp)
    if "time" not in ref.dims:
        print(f"[SKIP] No time dim in ref: {train_ref_fp}")
        return False
    T = ref["time"].values
    nT = T.shape[0]

    da = open_da(target_fp)
    if "time" not in da.dims:
        print(f"[SKIP] No time dim in target: {target_fp}")
        return False
    if da.sizes["time"] != nT:
        print(f"[SKIP] time length mismatch: {target_fp} has {da.sizes['time']} but ref has {nT}")
        return False

    # Heuristic: if already fixed, then the existing time coordinate should be a permutation
    # that when sorted matches T. But in the broken case it's exactly T already.
    # We'll detect the broken case by checking whether da.time equals T exactly.
    # If it already differs from T, we assume it's already fixed.
    same_as_T = np.array_equal(da["time"].values, T)
    if not same_as_T:
        print(f"[OK?] {target_fp} time != ref time; assuming already fixed, skipping.")
        return False

    true_time = vstack_time_from_T(T, step=step)

    # Assign corrected time, then sort chronologically.
    da2 = da.copy(deep=False)
    da2 = da2.assign_coords(time=true_time).sortby("time")

    # Preserve attrs/name
    da2.name = da.name
    da2.attrs.update(da.attrs)

    if do_backup:
        backup_file(target_fp)

    # Overwrite (write as dataset to keep var name)
    ds_out = da2.to_dataset(name=da2.name or "var")
    fixed_fp = target_fp.with_name(target_fp.stem + "_fixed.nc")
    ds_out.to_netcdf(fixed_fp)
    print(f"[FIXED] {target_fp} (step={step})")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(ROOT))
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    do_backup = not args.no_backup

    if not root.exists():
        raise SystemExit(f"Root not found: {root.resolve()}")

    n_fixed = 0

    # -------- fold-level fixes (step = 4) --------
    for split_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for seed_dir in sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("seed")]):
            for fold_dir in sorted([p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("fold")]):
                ref_fp = fold_dir / FOLD_TRAIN_REF
                for target_name in [FOLD_TRAIN_LAT, FOLD_TRAIN_REC]:
                    target_fp = fold_dir / target_name
                    if fix_one(ref_fp, target_fp, step=4, do_backup=do_backup):
                        n_fixed += 1

    # -------- full-model fixes (step = 5) --------
    # full_train_references.nc is unavailable, so use a reference time vector.
    # Best available proxy: use fold0/train_references.nc from seed0 (same split),
    # because "train" for the full model is concat of folds 0..4 and should have same time coords
    # as the split's training period. If that file is missing, we skip.
    for split_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        # try to get a reference time for this split
        candidate_ref = split_dir / "seed0" / "fold0" / FOLD_TRAIN_REF
        if not candidate_ref.exists():
            continue

        for seed_dir in sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("seed")]):
            for target_name in [FULL_TRAIN_LAT, FULL_TRAIN_REC]:
                target_fp = seed_dir / target_name
                if fix_one(candidate_ref, target_fp, step=5, do_backup=do_backup):
                    n_fixed += 1

    print(f"Done. Fixed {n_fixed} files.")

if __name__ == "__main__":
    main()
