#!/usr/bin/env python3
"""
Center-of-action SST index check for KGAE latent-coordinate bimodality.

This is a reviewer-response diagnostic, not a new pattern-index method. It
compares learned latent-coordinate distributions against simple physical SST
box averages aligned with the dominant positive and negative centers of action.
The SST index is always a box mean difference:

    index = mean(SSTA in positive box) - mean(SSTA in negative box)

No full spatial pattern projection is computed.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "kgae_center_action_cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
(Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)
(Path(os.environ["XDG_CACHE_HOME"])).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, kurtosis, skew

try:
    from sklearn.mixture import GaussianMixture
    GMM_IMPORT_ERROR = None
except Exception:  # pragma: no cover - optional dependency
    GaussianMixture = None
    GMM_IMPORT_ERROR = sys.exc_info()[1]

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover - optional dependency
    ccrs = None


MODE_ORDER = ["Decadal", "Interannual", "Quasibiennial"]
MODE_TO_TENDENCY_ID = {"Decadal": 5, "Interannual": 4, "Quasibiennial": 3}
MODE_ALIASES = {
    "decadal": "Decadal",
    "interannual": "Interannual",
    "quasibiennial": "Quasibiennial",
    "quasibiennialmode": "Quasibiennial",
    "quasi_biennial": "Quasibiennial",
    "quasi-biennial": "Quasibiennial",
}


@dataclass(frozen=True)
class Box:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_mapping(cls, item: dict[str, Any]) -> "Box":
        if {"lat_min", "lat_max", "lon_min", "lon_max"}.issubset(item):
            lat_min = item["lat_min"]
            lat_max = item["lat_max"]
            lon_min = item["lon_min"]
            lon_max = item["lon_max"]
        elif "lat" in item and "lon" in item:
            lat_min, lat_max = item["lat"]
            lon_min, lon_max = item["lon"]
        else:
            raise ValueError(
                "Box must contain lat_min/lat_max/lon_min/lon_max or lat/lon ranges"
            )
        return cls(
            lat_min=float(min(lat_min, lat_max)),
            lat_max=float(max(lat_min, lat_max)),
            lon_min=float(lon_min),
            lon_max=float(lon_max),
        )

    def label(self) -> str:
        return (
            f"{self.lat_min:g} to {self.lat_max:g} lat, "
            f"{self.lon_min:g} to {self.lon_max:g} lon"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


@dataclass
class ModeBoxConfig:
    mode: str
    positive: list[Box]
    negative: list[Box]
    latent_mode: str
    tendency_mode: int
    source: str


@dataclass
class Discovery:
    search_roots: list[Path]
    sst_candidates: list[Path]
    latent_candidates: list[Path]
    pattern_candidates: list[Path]
    figure_candidates: list[Path]


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).with_name("center_action_boxes.yaml")
    parser = argparse.ArgumentParser(
        description=(
            "Compare KGAE latent-coordinate PDFs with simple center-of-action "
            "SST box-index PDFs."
        )
    )
    parser.add_argument("--sst", type=Path, default=None, help="SST/SSTA NetCDF path")
    parser.add_argument(
        "--latents", type=Path, default=None, help="KGAE latent time-series NetCDF path"
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=None,
        help="Optional KGAE pattern/linear-response NetCDF path for auto boxes",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis/center_action_bimodality_outputs"),
        help="Directory for figures, tables, and reports",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config if default_config.exists() else None,
        help="Optional YAML file with manually specified positive/negative boxes",
    )
    parser.add_argument("--sst-var", default=None, help="SST/SSTA variable name")
    parser.add_argument("--latent-var", default=None, help="Latent variable name")
    parser.add_argument("--pattern-var", default=None, help="Pattern variable name")
    parser.add_argument(
        "--sst-is-anomaly",
        action="store_true",
        help="Treat SST variable as an anomaly field and skip detrending/climatology removal",
    )
    parser.add_argument(
        "--box-half-lat",
        type=float,
        default=10.0,
        help="Half-width in degrees latitude for auto-selected boxes",
    )
    parser.add_argument(
        "--box-half-lon",
        type=float,
        default=20.0,
        help="Half-width in degrees longitude for auto-selected boxes",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of fixed histogram bins over -4 to 4 standard deviations",
    )
    return parser.parse_args()


def normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def repo_root() -> Path:
    return Path.cwd().resolve()


def display_path(path: Path, root: Path | None = None) -> str:
    try:
        if root is not None:
            return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        pass
    return str(path)


def default_search_roots(root: Path) -> list[Path]:
    roots = [
        root,
        root / "results",
        root / "final_scripts",
        root / "publication_scripts",
        root / "scripts",
        root / "data_pipeline",
        root.parent / "results",
    ]
    try:
        for item in root.parent.iterdir():
            name = item.name.lower()
            if item.is_dir() and ("kgae" in name or "result" in name):
                roots.append(item)
    except OSError:
        pass

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in roots:
        if path.exists() and path.is_dir():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
    return unique


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def discover_artifacts(root: Path) -> Discovery:
    roots = default_search_roots(root)
    nc_files: list[Path] = []
    figure_files: list[Path] = []
    for search_root in roots:
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".nc":
                nc_files.append(path.resolve())
            elif suffix in {".png", ".pdf", ".svg"}:
                lower = path.name.lower()
                if any(key in lower for key in ("alpha", "decoder", "climate", "figure")):
                    figure_files.append(path.resolve())

    nc_files = sorted(set(nc_files))
    figure_files = sorted(set(figure_files))

    def by_name(keys: tuple[str, ...]) -> list[Path]:
        return [p for p in nc_files if any(k in p.name.lower() for k in keys)]

    sst = [
        p
        for p in by_name(("sst", "ssta", "tos"))
        if not any(k in p.name.lower() for k in ("mask", "climate_mode"))
    ]
    latents = by_name(("latent", "latents", "encoding", "encodings"))
    patterns = by_name(
        (
            "alpha",
            "pattern",
            "linear",
            "decoder",
            "composite",
            "beta",
        )
    )
    return Discovery(roots, sorted(sst), sorted(latents), sorted(patterns), figure_files)


def score_candidate(path: Path, kind: str) -> int:
    text = str(path).lower()
    score = 0
    if kind == "sst":
        if "era5.sst.pacific" in text:
            score += 100
        if "sst" in path.name.lower():
            score += 20
        if "pacific" in text:
            score += 10
        if "mask" in text:
            score -= 100
    elif kind == "latents":
        if "final_scripts/era5.latents.v1.nc" in text:
            score += 100
        if "publication_scripts/era5.latents.v1.nc" in text:
            score += 90
        if path.name.lower() == "kgae_encodings.nc":
            score += 50
        if "latent" in path.name.lower():
            score += 20
    elif kind == "patterns":
        if "cache_alpha_and_linear_composites.nc" in text:
            score += 100
        if "alpha_seed.nc" in text:
            score += 80
        if "decoder_patterns" in text:
            score += 50
        if "climate_mode_patterns" in text:
            score -= 20
        if "alpha" in path.name.lower():
            score += 20
    return score


def choose_existing_path(
    explicit: Path | None, candidates: list[Path], kind: str
) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: score_candidate(p, kind), reverse=True)[0]


def write_discovery_report(
    outdir: Path,
    discovery: Discovery,
    selected: dict[str, Path | None],
    notes: list[str],
    missing: list[str] | None = None,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    root = repo_root()
    report = outdir / "center_action_bimodality_discovery_report.md"
    lines = [
        "# Center-Action Bimodality Artifact Discovery",
        "",
        "## Search Roots",
    ]
    lines.extend(f"- `{display_path(p, root)}`" for p in discovery.search_roots)
    lines.extend(["", "## Selected Inputs"])
    for key, path in selected.items():
        value = "`not selected`" if path is None else f"`{display_path(path, root)}`"
        lines.append(f"- {key}: {value}")
    if missing:
        lines.extend(["", "## Missing Required Inputs"])
        lines.extend(f"- {item}" for item in missing)
    if notes:
        lines.extend(["", "## Processing Notes"])
        lines.extend(f"- {note}" for note in notes)

    def add_candidates(title: str, paths: list[Path]) -> None:
        lines.extend(["", f"## {title}"])
        if not paths:
            lines.append("- None found")
            return
        for path in paths[:30]:
            lines.append(f"- `{display_path(path, root)}`")
        if len(paths) > 30:
            lines.append(f"- ... {len(paths) - 30} more omitted")

    add_candidates("SST Candidates", discovery.sst_candidates)
    add_candidates("Latent Candidates", discovery.latent_candidates)
    add_candidates("Pattern Candidates", discovery.pattern_candidates)
    add_candidates("Relevant Figure Candidates", discovery.figure_candidates)
    report.write_text("\n".join(lines) + "\n")
    return report


def rename_lat_lon(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    rename: dict[str, str] = {}
    for old, new in (("latitude", "lat"), ("longitude", "lon")):
        if old in obj.dims or old in obj.coords:
            rename[old] = new
    return obj.rename(rename) if rename else obj


def open_dataset_checked(path: Path, label: str) -> xr.Dataset:
    if path is None or not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    return xr.open_dataset(path)


def choose_variable(
    ds: xr.Dataset, requested: str | None, kind: str, notes: list[str]
) -> xr.DataArray:
    if requested is not None:
        if requested not in ds.data_vars:
            raise ValueError(
                f"Requested {kind} variable `{requested}` not found. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return ds[requested]

    candidates: list[tuple[int, str]] = []
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        score = 0
        lname = name.lower()
        if kind == "sst":
            if "time" in dims and {"lat", "lon"}.issubset(dims):
                score += 10
            if lname in {"sst", "ssta", "tos", "sst_anom", "sst_anomaly"}:
                score += 20
        elif kind == "latents":
            if "time" in dims and ("mode" in dims or "tendency_mode" in dims):
                score += 10
            if lname in {"encodings", "latents", "__xarray_dataarray_variable__"}:
                score += 20
        elif kind == "patterns":
            has_space = {"lat", "lon"}.issubset(dims)
            has_mode = "mode" in dims or "tendency_mode" in dims
            if has_space and has_mode:
                score += 10
            if lname in {"alpha_mean", "linear_response", "pattern"}:
                score += 30
            elif lname in {"comp_mean", "__xarray_dataarray_variable__"}:
                score += 15
        if score > 0:
            candidates.append((score, name))

    if not candidates:
        raise ValueError(
            f"Could not infer {kind} variable. Available variables: {list(ds.data_vars)}"
        )

    _, name = sorted(candidates, reverse=True)[0]
    notes.append(f"Selected `{name}` as the {kind} variable.")
    return ds[name]


def is_datetime_coord(da: xr.DataArray, dim: str = "time") -> bool:
    if dim not in da.coords:
        return False
    try:
        pd.to_datetime(da[dim].values)
        return True
    except Exception:
        return False


def subset_to_time_span(da: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims or "time" not in reference.dims:
        return da
    if not (is_datetime_coord(da) and is_datetime_coord(reference)):
        return da
    ref_times = pd.to_datetime(reference["time"].values)
    if len(ref_times) == 0:
        return da
    return da.sel(time=slice(ref_times.min(), ref_times.max()))


def looks_like_anomaly(name: str, da: xr.DataArray) -> bool:
    lname = name.lower()
    long_name = str(da.attrs.get("long_name", "")).lower()
    standard_name = str(da.attrs.get("standard_name", "")).lower()
    return any(
        key in text
        for text in (lname, long_name, standard_name)
        for key in ("anom", "anomaly", "ssta")
    )


def prepare_ssta(
    sst: xr.DataArray, sst_name: str, sst_is_anomaly: bool, notes: list[str]
) -> xr.DataArray:
    sst = rename_lat_lon(sst)
    required = {"time", "lat", "lon"}
    if not required.issubset(sst.dims):
        raise ValueError(f"SST variable must have dims including {required}; got {sst.dims}")
    sst = sst.transpose("time", "lat", "lon")

    if sst_is_anomaly or looks_like_anomaly(sst_name, sst):
        notes.append("Using SST variable as an anomaly field.")
        return sst

    if not is_datetime_coord(sst):
        notes.append(
            "SST time coordinate is not datetime-like; subtracting only the time mean."
        )
        return sst - sst.mean("time")

    detrended = global_detrend_like_repo(sst)
    ssta = remove_climo_like_repo(detrended)
    ssta.name = f"{sst_name}_anomaly"
    notes.append(
        "Prepared SSTA with the same steps as kgae.global_detrend(..., deg=2) "
        "followed by kgae.remove_climo(...): subtract the cosine-latitude-weighted "
        "Pacific mean quadratic trend, then subtract the monthly climatology over "
        "the loaded/common period."
    )
    return ssta


def global_detrend_like_repo(sst: xr.DataArray) -> xr.DataArray:
    """Mirror kgae.utilities.global_detrend for this standalone script."""
    weights = xr.DataArray(
        np.cos(np.deg2rad(sst["lat"])), dims=("lat",), coords={"lat": sst["lat"]}
    )
    weighted_mean = sst.weighted(weights).mean(("lat", "lon"))
    poly = weighted_mean.polyfit(dim="time", deg=2)
    fit = xr.polyval(sst["time"], poly.polyfit_coefficients)
    return sst - fit


def remove_climo_like_repo(monthly: xr.DataArray) -> xr.DataArray:
    """Mirror kgae.utilities.remove_climo for monthly data."""
    monthly_climatology = monthly.groupby("time.month").mean()
    pieces = []
    years = sorted({pd.Timestamp(t).year for t in monthly["time"].values})
    for year in years:
        yearly = (
            monthly.sel(time=slice(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)))
            .groupby("time.month")
            .mean()
            - monthly_climatology
        )
        yearly = yearly.assign_coords(
            {"month": [pd.Timestamp(year, int(month), 1) for month in yearly["month"].values]}
        ).rename({"month": "time"})
        pieces.append(yearly)
    return xr.concat(pieces, "time").sortby("time")


def reduce_latent_array(da: xr.DataArray, notes: list[str]) -> xr.DataArray:
    da = da.rename({"tendency_mode": "mode"}) if "tendency_mode" in da.dims else da
    if "time" not in da.dims or "mode" not in da.dims:
        raise ValueError(f"Latent array must include time and mode dims; got {da.dims}")

    for dim in list(da.dims):
        if dim in {"time", "mode"}:
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
            notes.append(f"Dropped singleton latent dimension `{dim}`.")
        elif dim in {"seed", "initialization", "member", "ensemble"}:
            da = da.mean(dim, skipna=True)
            notes.append(f"Averaged latent dimension `{dim}` to form an ensemble mean.")
        elif dim == "run":
            values = list(da[dim].values)
            selected = "basin.goodtest" if "basin.goodtest" in values else values[0]
            da = da.sel({dim: selected})
            notes.append(f"Selected latent `{dim}` value `{selected}`.")
        elif dim == "recursion":
            values = list(da[dim].values)
            selected = 0 if 0 in values else values[0]
            da = da.sel({dim: selected})
            notes.append(f"Selected latent `{dim}` value `{selected}`.")
        else:
            raise ValueError(
                f"Latent variable has unsupported extra dimension `{dim}` with "
                f"size {da.sizes[dim]}. Select/reduce this before running."
            )
    return da.transpose("time", "mode")


def select_latent_mode(da: xr.DataArray, mode: str, tendency_id: int) -> xr.DataArray:
    labels = da["mode"].values if "mode" in da.coords else np.arange(da.sizes["mode"])
    normalized = [normalize_label(v) for v in labels]
    target = normalize_label(mode)
    if target in normalized:
        return da.isel(mode=normalized.index(target), drop=True)

    alias_targets = [k for k, v in MODE_ALIASES.items() if v == mode]
    for alias in alias_targets:
        if alias in normalized:
            return da.isel(mode=normalized.index(alias), drop=True)

    numeric_labels = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy()
    if np.any(np.isfinite(numeric_labels) & (numeric_labels == tendency_id)):
        idx = int(np.where(numeric_labels == tendency_id)[0][0])
        return da.isel(mode=idx, drop=True)

    zero_based_idx = tendency_id - 1
    if 0 <= zero_based_idx < da.sizes["mode"]:
        return da.isel(mode=zero_based_idx, drop=True)

    raise ValueError(
        f"Could not select latent mode `{mode}` from labels {list(labels)}."
    )


def load_latents(
    path: Path, var_name: str | None, mode_configs: list[ModeBoxConfig], notes: list[str]
) -> dict[str, xr.DataArray]:
    ds = open_dataset_checked(path, "latents")
    ds = rename_lat_lon(ds)
    da = choose_variable(ds, var_name, "latents", notes)
    da = reduce_latent_array(da, notes)
    latents: dict[str, xr.DataArray] = {}
    for cfg in mode_configs:
        selected = select_latent_mode(da, cfg.latent_mode, cfg.tendency_mode)
        selected.name = f"{cfg.mode}_latent"
        latents[cfg.mode] = selected
    return latents


def load_manual_boxes(config_path: Path | None, notes: list[str]) -> list[ModeBoxConfig] | None:
    if config_path is None:
        return None
    if not config_path.exists():
        raise FileNotFoundError(f"Box config does not exist: {config_path}")
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    modes = config.get("modes", {})
    configs: list[ModeBoxConfig] = []
    for mode in MODE_ORDER:
        if mode not in modes:
            raise ValueError(f"Config is missing mode `{mode}`.")
        item = modes[mode]
        configs.append(
            ModeBoxConfig(
                mode=mode,
                positive=parse_box_list(item, "positive"),
                negative=parse_box_list(item, "negative"),
                latent_mode=str(item.get("latent_mode", mode)),
                tendency_mode=int(item.get("tendency_mode", MODE_TO_TENDENCY_ID[mode])),
                source=f"manual config: {config_path}",
            )
        )
    notes.append(f"Loaded manual center-of-action boxes from `{config_path}`.")
    return configs


def parse_box_list(item: dict[str, Any], sign: str) -> list[Box]:
    plural_key = f"{sign}_boxes"
    singular_key = f"{sign}_box"
    if plural_key in item:
        boxes = item[plural_key]
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"`{plural_key}` must be a non-empty list.")
        return [Box.from_mapping(box) for box in boxes]
    if singular_key in item:
        return [Box.from_mapping(item[singular_key])]
    raise ValueError(f"Config item is missing `{singular_key}` or `{plural_key}`.")


def select_pattern_mode(da: xr.DataArray, mode: str, tendency_id: int) -> xr.DataArray:
    da = rename_lat_lon(da)
    mode_dim = "tendency_mode" if "tendency_mode" in da.dims else "mode"
    if mode_dim not in da.dims:
        raise ValueError(f"Pattern array has no mode dimension: {da.dims}")
    labels = da[mode_dim].values if mode_dim in da.coords else np.arange(da.sizes[mode_dim])
    normalized = [normalize_label(v) for v in labels]
    target = normalize_label(mode)
    if target in normalized:
        return da.isel({mode_dim: normalized.index(target)}, drop=True)
    numeric_labels = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy()
    if np.any(np.isfinite(numeric_labels) & (numeric_labels == tendency_id)):
        idx = int(np.where(numeric_labels == tendency_id)[0][0])
        return da.isel({mode_dim: idx}, drop=True)
    zero_based_idx = tendency_id - 1
    if mode_dim == "mode" and 0 <= zero_based_idx < da.sizes[mode_dim]:
        return da.isel({mode_dim: zero_based_idx}, drop=True)
    raise ValueError(f"Could not select pattern mode `{mode}` from labels {list(labels)}.")


def find_smoothed_extrema(field: xr.DataArray) -> tuple[tuple[float, float], tuple[float, float]]:
    field = rename_lat_lon(field).transpose("lat", "lon")
    arr = np.asarray(field.values, dtype=float)
    mask = np.isfinite(arr)
    if not np.any(mask):
        raise ValueError("Pattern field is all NaN.")
    filled = np.where(mask, arr, 0.0)
    weights = gaussian_filter(mask.astype(float), sigma=3.0)
    smooth = gaussian_filter(filled, sigma=3.0) / np.maximum(weights, 1e-12)
    smooth[~mask] = np.nan
    pos_idx = np.unravel_index(int(np.nanargmax(smooth)), smooth.shape)
    neg_idx = np.unravel_index(int(np.nanargmin(smooth)), smooth.shape)
    pos = (float(field["lat"].values[pos_idx[0]]), float(field["lon"].values[pos_idx[1]]))
    neg = (float(field["lat"].values[neg_idx[0]]), float(field["lon"].values[neg_idx[1]]))
    return pos, neg


def clipped_box(
    center: tuple[float, float],
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    half_lat: float,
    half_lon: float,
) -> Box:
    lat, lon = center
    return Box(
        lat_min=max(float(np.nanmin(lat_values)), lat - half_lat),
        lat_max=min(float(np.nanmax(lat_values)), lat + half_lat),
        lon_min=max(float(np.nanmin(lon_values)), lon - half_lon),
        lon_max=min(float(np.nanmax(lon_values)), lon + half_lon),
    )


def auto_boxes_from_patterns(
    path: Path,
    var_name: str | None,
    half_lat: float,
    half_lon: float,
    notes: list[str],
) -> list[ModeBoxConfig]:
    ds = open_dataset_checked(path, "patterns")
    ds = rename_lat_lon(ds)
    da = choose_variable(ds, var_name, "patterns", notes)
    da = rename_lat_lon(da)
    configs: list[ModeBoxConfig] = []
    for mode in MODE_ORDER:
        tendency_id = MODE_TO_TENDENCY_ID[mode]
        field = select_pattern_mode(da, mode, tendency_id).transpose("lat", "lon")
        pos_center, neg_center = find_smoothed_extrema(field)
        pos = clipped_box(pos_center, field["lat"].values, field["lon"].values, half_lat, half_lon)
        neg = clipped_box(neg_center, field["lat"].values, field["lon"].values, half_lat, half_lon)
        configs.append(
            ModeBoxConfig(
                mode=mode,
                positive=[pos],
                negative=[neg],
                latent_mode=mode,
                tendency_mode=tendency_id,
                source=f"auto from {path}, centers pos={pos_center}, neg={neg_center}",
            )
        )
    notes.append(
        f"Auto-selected boxes from smoothed pattern extrema in `{path}` "
        f"with half-widths +/-{half_lat:g} lat and +/-{half_lon:g} lon."
    )
    return configs


def reduce_pattern_array(da: xr.DataArray, notes: list[str]) -> xr.DataArray:
    da = rename_lat_lon(da)
    if not {"lat", "lon"}.issubset(da.dims):
        raise ValueError(f"Pattern array must include lat/lon dims; got {da.dims}")
    for dim in list(da.dims):
        if dim in {"lat", "lon", "mode", "tendency_mode"}:
            continue
        if da.sizes[dim] == 1:
            da = da.isel({dim: 0}, drop=True)
            notes.append(f"Dropped singleton pattern dimension `{dim}`.")
        elif dim in {"seed", "initialization", "member", "ensemble", "fold"}:
            da = da.mean(dim, skipna=True)
            notes.append(f"Averaged pattern dimension `{dim}` for plotting.")
        else:
            raise ValueError(
                f"Pattern variable has unsupported extra dimension `{dim}` with "
                f"size {da.sizes[dim]}."
            )
    return da


def load_pattern_fields(
    path: Path | None,
    var_name: str | None,
    mode_configs: list[ModeBoxConfig],
    notes: list[str],
) -> dict[str, xr.DataArray] | None:
    if path is None or not path.exists():
        notes.append("No pattern file available; top-row pattern panels omitted.")
        return None
    try:
        ds = rename_lat_lon(open_dataset_checked(path, "patterns"))
        da = choose_variable(ds, var_name, "patterns", notes)
        da = reduce_pattern_array(da, notes)
        fields: dict[str, xr.DataArray] = {}
        for cfg in mode_configs:
            field = select_pattern_mode(da, cfg.mode, cfg.tendency_mode)
            fields[cfg.mode] = rename_lat_lon(field).transpose("lat", "lon")
        notes.append(f"Loaded top-row mode patterns from `{path}`.")
        return fields
    except Exception as exc:
        notes.append(f"Could not load pattern panels from `{path}`: {exc}")
        return None


def box_subset(da: xr.DataArray, box: Box) -> xr.DataArray:
    da = rename_lat_lon(da)
    lat_values = da["lat"].values
    lon_values = da["lon"].values

    lat_slice = slice(box.lat_min, box.lat_max)
    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(box.lat_max, box.lat_min)
    sub = da.sel(lat=lat_slice)

    if box.lon_min <= box.lon_max:
        lon_slice = slice(box.lon_min, box.lon_max)
        if lon_values[0] > lon_values[-1]:
            lon_slice = slice(box.lon_max, box.lon_min)
        sub = sub.sel(lon=lon_slice)
    else:
        left = sub.sel(lon=slice(box.lon_min, float(np.nanmax(lon_values))))
        right = sub.sel(lon=slice(float(np.nanmin(lon_values)), box.lon_max))
        sub = xr.concat([left, right], dim="lon")

    if sub.sizes.get("lat", 0) == 0 or sub.sizes.get("lon", 0) == 0:
        raise ValueError(f"Box has no grid cells in data domain: {box}")
    return sub


def area_weighted_box_mean(ssta: xr.DataArray, box: Box) -> xr.DataArray:
    sub = box_subset(ssta, box)
    weights = xr.DataArray(
        np.cos(np.deg2rad(sub["lat"])), dims=("lat",), coords={"lat": sub["lat"]}
    )
    return sub.weighted(weights).mean(("lat", "lon"), skipna=True)


def box_mask(ssta: xr.DataArray, box: Box) -> xr.DataArray:
    lat = ssta["lat"]
    lon = ssta["lon"]
    lat_cond = (lat >= box.lat_min) & (lat <= box.lat_max)
    if box.lon_min <= box.lon_max:
        lon_cond = (lon >= box.lon_min) & (lon <= box.lon_max)
    else:
        lon_cond = (lon >= box.lon_min) | (lon <= box.lon_max)
    mask = lat_cond & lon_cond
    if int(mask.sum()) == 0:
        raise ValueError(f"Box has no grid cells in data domain: {box}")
    return mask


def area_weighted_boxes_mean(ssta: xr.DataArray, boxes: list[Box]) -> xr.DataArray:
    if not boxes:
        raise ValueError("At least one box is required.")
    mask = xr.zeros_like(ssta.isel(time=0, drop=True), dtype=bool)
    for box in boxes:
        mask = mask | box_mask(ssta, box)
    weights = xr.DataArray(
        np.cos(np.deg2rad(ssta["lat"])), dims=("lat",), coords={"lat": ssta["lat"]}
    )
    return ssta.where(mask).weighted(weights).mean(("lat", "lon"), skipna=True)


def compute_box_indices(
    ssta: xr.DataArray, mode_configs: list[ModeBoxConfig]
) -> dict[str, xr.DataArray]:
    indices: dict[str, xr.DataArray] = {}
    for cfg in mode_configs:
        pos = area_weighted_boxes_mean(ssta, cfg.positive)
        neg = area_weighted_boxes_mean(ssta, cfg.negative)
        idx = pos - neg
        idx.name = f"{cfg.mode}_box_index"
        indices[cfg.mode] = idx
    return indices


def monthly_period_index(da: xr.DataArray) -> pd.PeriodIndex | None:
    if "time" not in da.coords:
        return None
    try:
        return pd.to_datetime(da["time"].values).to_period("M")
    except Exception:
        return None


def align_pair(
    latent: xr.DataArray, box_index: xr.DataArray, mode: str, notes: list[str]
) -> pd.DataFrame:
    if "time" not in latent.dims or "time" not in box_index.dims:
        n = min(latent.size, box_index.size)
        notes.append(
            f"{mode}: no usable time dimension on both series; aligned first {n} ordered samples."
        )
        return pd.DataFrame(
            {
                "latent": np.asarray(latent.values).reshape(-1)[:n],
                "box": np.asarray(box_index.values).reshape(-1)[:n],
            }
        )

    if is_datetime_coord(latent) and is_datetime_coord(box_index):
        latent_series = pd.Series(
            np.asarray(latent.values).reshape(-1),
            index=pd.to_datetime(latent["time"].values),
        )
        box_series = pd.Series(
            np.asarray(box_index.values).reshape(-1),
            index=pd.to_datetime(box_index["time"].values),
        )
        common = latent_series.index.intersection(box_series.index)
        if len(common) > 0:
            df = pd.DataFrame(
                {"latent": latent_series.loc[common], "box": box_series.loc[common]}
            )
            notes.append(
                f"{mode}: aligned by exact datetime coordinates over {len(df)} samples "
                f"({common.min().date()} to {common.max().date()})."
            )
            return df

        latent_periods = latent_series.index.to_period("M")
        box_periods = box_series.index.to_period("M")
        common_periods = latent_periods.intersection(box_periods)
        if len(common_periods) > 0:
            ldf = pd.DataFrame({"period": latent_periods, "latent": latent_series.values})
            bdf = pd.DataFrame({"period": box_periods, "box": box_series.values})
            df = pd.merge(ldf, bdf, on="period", how="inner").set_index("period")
            notes.append(
                f"{mode}: exact datetimes did not overlap; aligned by common ordered "
                f"monthly periods over {len(df)} samples."
            )
            return df

    n = min(latent.sizes["time"], box_index.sizes["time"])
    notes.append(
        f"{mode}: datetime alignment unavailable; aligned first {n} ordered monthly samples."
    )
    return pd.DataFrame(
        {
            "latent": np.asarray(latent.values).reshape(-1)[:n],
            "box": np.asarray(box_index.values).reshape(-1)[:n],
        }
    )


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sigma = np.nanstd(values)
    if not np.isfinite(sigma) or sigma == 0:
        raise ValueError("Cannot standardize a constant or all-NaN series.")
    return (values - mu) / sigma


def bimodality_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {
            "skewness": np.nan,
            "kurtosis_pearson": np.nan,
            "bimodality_coefficient": np.nan,
            "gmm_bic_1_minus_2": np.nan,
        }
    sk = float(skew(x, bias=False))
    ku = float(kurtosis(x, fisher=False, bias=False))
    bc = float((sk * sk + 1.0) / ku) if ku != 0 else np.nan
    bic_delta = np.nan
    if GaussianMixture is not None and x.size >= 20:
        data = x.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=10, reg_covar=1e-6)
        gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=20, reg_covar=1e-6)
        bic1 = gmm1.fit(data).bic(data)
        bic2 = gmm2.fit(data).bic(data)
        bic_delta = float(bic1 - bic2)
    return {
        "skewness": sk,
        "kurtosis_pearson": ku,
        "bimodality_coefficient": bc,
        "gmm_bic_1_minus_2": bic_delta,
    }


def kde_line(values: np.ndarray, xgrid: np.ndarray) -> np.ndarray | None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4 or np.nanstd(x) == 0:
        return None
    try:
        return gaussian_kde(x, bw_method=0.25)(xgrid)
    except np.linalg.LinAlgError:
        return None


def shifted_lon_label(x: float) -> str:
    true_lon = (float(x) + 180.0) % 360.0
    if math.isclose(true_lon, 180.0, abs_tol=1e-6):
        return "180"
    if math.isclose(true_lon, 0.0, abs_tol=1e-6):
        return "0"
    if true_lon < 180.0:
        return f"{true_lon:g}E"
    return f"{360.0 - true_lon:g}W"


def add_real_longitude_ticks(ax: plt.Axes, data_crs: Any | None) -> None:
    xticks = [-60, -30, 0, 30, 60, 90]
    yticks = [-40, -20, 0, 20, 40, 60]
    if data_crs is not None and hasattr(ax, "set_xticks"):
        ax.set_xticks(xticks, crs=data_crs)
        ax.set_yticks(yticks, crs=data_crs)
    else:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
    ax.set_xticklabels([shifted_lon_label(x) for x in xticks])
    ax.set_yticklabels([f"{abs(y):g}{'S' if y < 0 else ('N' if y > 0 else '')}" for y in yticks])


def add_box_to_axis(
    ax: plt.Axes,
    box: Box,
    color: str,
    label: str,
    transform: Any | None = None,
) -> None:
    rectangles: list[tuple[float, float]] = []
    if box.lon_min <= box.lon_max:
        rectangles.append((box.lon_min, box.lon_max))
    else:
        x0, x1 = ax.get_xlim()
        rectangles.append((box.lon_min, x1))
        rectangles.append((x0, box.lon_max))

    for lon_min, lon_max in rectangles:
        rect = Rectangle(
            (lon_min, box.lat_min),
            lon_max - lon_min,
            box.lat_max - box.lat_min,
            fill=False,
            edgecolor=color,
            linewidth=1.8,
            transform=transform if transform is not None else ax.transData,
            zorder=6,
        )
        ax.add_patch(rect)
    x_text = (box.lon_min + box.lon_max) / 2.0
    if box.lon_min > box.lon_max:
        x_text = rectangles[0][0] + 0.5 * (rectangles[0][1] - rectangles[0][0])
    y_text = box.lat_min + 0.5 * (box.lat_max - box.lat_min)
    ax.text(
        x_text,
        y_text,
        label,
        color=color,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        transform=transform if transform is not None else ax.transData,
        zorder=7,
    )


def plot_pattern_and_distributions(
    standardized: dict[str, pd.DataFrame],
    mode_configs: list[ModeBoxConfig],
    pattern_fields: dict[str, xr.DataArray] | None,
    outdir: Path,
    bins: int,
) -> None:
    fig = plt.figure(figsize=(12.2, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.9])
    map_crs = ccrs.PlateCarree(central_longitude=180) if ccrs is not None else None
    data_crs = map_crs
    top_axes = [
        fig.add_subplot(gs[0, i], projection=map_crs) if map_crs is not None else fig.add_subplot(gs[0, i])
        for i in range(3)
    ]
    pdf_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    edges = np.linspace(-4, 4, bins + 1)
    xgrid = np.linspace(-4, 4, 500)
    colors = {"latent": "#222222", "box": "#0072B2"}
    cfg_by_mode = {cfg.mode: cfg for cfg in mode_configs}

    pattern_vmax = None
    if pattern_fields:
        values = [
            np.asarray(pattern_fields[mode].values, dtype=float)
            for mode in MODE_ORDER
            if mode in pattern_fields
        ]
        finite_abs = [
            float(np.nanmax(np.abs(v))) for v in values if np.any(np.isfinite(v))
        ]
        if finite_abs:
            pattern_vmax = max(finite_abs)

    mappable = None
    for ax, mode in zip(top_axes, MODE_ORDER):
        cfg = cfg_by_mode[mode]
        field = None if pattern_fields is None else pattern_fields.get(mode)
        if field is None:
            ax.text(
                0.5,
                0.5,
                "Pattern unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )
            ax.set_xlim(-80.5, 109.5)
            ax.set_ylim(-49.5, 65.5)
        else:
            vmax = pattern_vmax
            if not vmax or not np.isfinite(vmax) or vmax == 0:
                vmax = float(np.nanmax(np.abs(field.values))) or 1.0
            mappable = ax.pcolormesh(
                field["lon"],
                field["lat"],
                field,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="auto",
                transform=data_crs,
            )
            try:
                ax.contour(
                    field["lon"],
                    field["lat"],
                    field,
                    levels=[0],
                    colors="0.2",
                    linewidths=0.7,
                    transform=data_crs,
                )
            except Exception:
                pass
            extent = [
                float(field["lon"].min()),
                float(field["lon"].max()),
                float(field["lat"].min()),
                float(field["lat"].max()),
            ]
            if data_crs is not None and hasattr(ax, "set_extent"):
                ax.set_extent(extent, crs=data_crs)
                ax.coastlines(resolution="110m", linewidth=0.7, color="0.25", zorder=5)
            else:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
        for box in cfg.positive:
            add_box_to_axis(ax, box, "#D55E00", "+", transform=data_crs)
        for box in cfg.negative:
            add_box_to_axis(ax, box, "#0072B2", "-", transform=data_crs)
        ax.set_title(mode, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        add_real_longitude_ticks(ax, data_crs)
        ax.tick_params(labelsize=8)

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=list(top_axes),
            orientation="vertical",
            shrink=0.82,
            pad=0.01,
            label="Mode pattern value",
        )

    for ax, mode in zip(pdf_axes, MODE_ORDER):
        df = standardized[mode]
        latent = df["latent_z"].to_numpy()
        box = df["box_z"].to_numpy()
        ax.hist(
            latent,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.4,
            linestyle="--",
            alpha=0.6,
            color=colors["latent"],
            label="Latent coordinate",
        )
        ax.hist(
            box,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.4,
            linestyle="--",
            alpha=0.6,
            color=colors["box"],
            label="SST box index",
        )
        latent_kde = kde_line(latent, xgrid)
        box_kde = kde_line(box, xgrid)
        if latent_kde is not None:
            ax.plot(xgrid, latent_kde, color=colors["latent"], linewidth=1.6)
        if box_kde is not None:
            ax.plot(xgrid, box_kde, color=colors["box"], linewidth=1.6)
        ax.axvline(0, color="0.35", linewidth=0.7, linestyle="--")
        ax.set_ylabel("Density")
        ax.set_xlim(-4, 4)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Standardized value")
    legend_handles = [
        Line2D([0], [0], color=colors["latent"], linestyle="--", linewidth=1.8, alpha=0.6),
        Line2D([0], [0], color=colors["box"], linestyle="--", linewidth=1.8, alpha=0.6),
    ]
    fig.legend(
        legend_handles,
        ["Latent coordinate", "SST box index"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncols=2,
        frameon=False,
        fontsize=9,
    )
    for ext in ("pdf", "svg", "png"):
        dpi = 220 if ext == "png" else None
        fig.savefig(outdir / f"center_action_bimodality_check.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def format_box_for_table(box: Box) -> str:
    return (
        f"lat[{box.lat_min:g},{box.lat_max:g}], "
        f"lon[{box.lon_min:g},{box.lon_max:g}]"
    )


def format_boxes_for_table(boxes: list[Box]) -> str:
    return "; ".join(format_box_for_table(box) for box in boxes)


def write_summary(
    standardized: dict[str, pd.DataFrame],
    mode_configs: list[ModeBoxConfig],
    outdir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cfg_by_mode = {cfg.mode: cfg for cfg in mode_configs}
    for mode in MODE_ORDER:
        df = standardized[mode]
        latent_stats = bimodality_stats(df["latent_z"].to_numpy())
        box_stats = bimodality_stats(df["box_z"].to_numpy())
        cfg = cfg_by_mode[mode]
        rows.append(
            {
                "mode": mode,
                "positive_box": format_boxes_for_table(cfg.positive),
                "negative_box": format_boxes_for_table(cfg.negative),
                "n_valid_samples": int(len(df)),
                "latent_skewness": latent_stats["skewness"],
                "latent_kurtosis_pearson": latent_stats["kurtosis_pearson"],
                "latent_bimodality_coefficient": latent_stats[
                    "bimodality_coefficient"
                ],
                "latent_gmm_bic_1_minus_2": latent_stats["gmm_bic_1_minus_2"],
                "box_skewness": box_stats["skewness"],
                "box_kurtosis_pearson": box_stats["kurtosis_pearson"],
                "box_bimodality_coefficient": box_stats["bimodality_coefficient"],
                "box_gmm_bic_1_minus_2": box_stats["gmm_bic_1_minus_2"],
                "box_source": cfg.source,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "center_action_bimodality_summary.csv", index=False)
    return summary


def interpretation_from_summary(summary: pd.DataFrame) -> str:
    latent_bc = summary["latent_bimodality_coefficient"].to_numpy(dtype=float)
    box_bc = summary["box_bimodality_coefficient"].to_numpy(dtype=float)
    less_bimodal = np.isfinite(latent_bc) & np.isfinite(box_bc) & (box_bc < latent_bc - 0.05)
    box_support = np.isfinite(box_bc) & (box_bc >= 5.0 / 9.0)
    box_gmm = summary["box_gmm_bic_1_minus_2"].to_numpy(dtype=float)
    if np.any(np.isfinite(box_gmm) & (box_gmm > 10)):
        box_support = box_support | (np.isfinite(box_gmm) & (box_gmm > 10))

    if int(np.sum(box_support)) >= 2:
        return (
            "The SST box indices also show bimodality or two-component support for "
            "multiple modes, which supports a physical contribution to the latent "
            "bimodality. This does not by itself establish discrete SST regimes."
        )
    if int(np.sum(less_bimodal)) >= 2:
        return (
            "The SST box indices are less bimodal than the learned latent-coordinate "
            "PDFs for most modes, so the latent bimodality is likely partly "
            "representational. The latent coordinates remain useful learned indices "
            "for composites and conditional analysis."
        )
    return (
        "The box-index and latent-coordinate distribution metrics are mixed. The "
        "diagnostic should be interpreted as a sanity check rather than evidence "
        "for or against discrete physical regimes."
    )


def write_response_note(
    outdir: Path,
    selected: dict[str, Path | None],
    summary: pd.DataFrame,
    notes: list[str],
) -> None:
    root = repo_root()
    interp = interpretation_from_summary(summary)
    lines = [
        "# Center-of-Action Bimodality Check Response Note",
        "",
        "We added a simple center-of-action diagnostic to address the reviewer concern "
        "that bimodality in learned latent-coordinate PDFs may be induced partly by "
        "the KGAE representation geometry rather than by directly bimodal SST.",
        "",
        "For each learned mode, the diagnostic computes an area-weighted SST anomaly "
        "index from fixed positive and negative center-of-action boxes aligned with "
        "the paper maps. When a mode has multiple same-sign boxes, the script first "
        "forms the union of those grid cells. The index is positive-box-union mean "
        "SSTA minus negative-box-union mean SSTA. This is intentionally a "
        "box-average sanity check, not a full spatial projection onto the KGAE "
        "pattern.",
        "",
        "The figure is arranged with the KGAE mode/linear-response pattern in the "
        "top row and the corresponding latent-coordinate versus SST-box-index PDFs "
        "in the bottom row. The boxes shown on the top-row maps are the same boxes "
        "used to compute the bottom-row SST indices.",
        "",
        "The SST anomaly preprocessing follows the same sequence used in the other "
        "analysis scripts: `kgae.global_detrend(..., deg=2)` style removal of the "
        "cosine-latitude-weighted Pacific mean quadratic trend, followed by "
        "`kgae.remove_climo(...)` style monthly climatology removal over the loaded "
        "common period.",
        "",
        "## Inputs",
    ]
    for key in ("sst", "latents", "patterns", "config"):
        path = selected.get(key)
        value = "not used" if path is None else f"`{display_path(path, root)}`"
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Box Definitions and Distribution Metrics", ""])
    table = summary.copy()
    numeric_cols = [
        c for c in table.columns if c not in {"mode", "positive_box", "negative_box", "box_source"}
    ]
    for col in numeric_cols:
        table[col] = table[col].map(
            lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else v)
        )
    lines.append(table.to_markdown(index=False))
    lines.extend(["", "## Interpretation", "", interp, ""])
    lines.append(
        "In all cases, the latent-coordinate PDFs should be described as PDFs of "
        "learned coordinates. They are not direct PDFs of physical SST variables, "
        "even when the corresponding box-average SST indices show similar "
        "distributional structure."
    )
    if notes:
        lines.extend(["", "## Reproducibility Notes"])
        lines.extend(f"- {note}" for note in notes)
    (outdir / "center_action_bimodality_response_note.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> int:
    args = parse_args()
    root = repo_root()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    if GaussianMixture is None:
        detail = "" if GMM_IMPORT_ERROR is None else f" ({str(GMM_IMPORT_ERROR).splitlines()[0]})"
        notes.append(f"Optional sklearn GaussianMixture unavailable; GMM BIC metrics omitted{detail}.")

    discovery = discover_artifacts(root)
    sst_path = choose_existing_path(args.sst, discovery.sst_candidates, "sst")
    latent_path = choose_existing_path(args.latents, discovery.latent_candidates, "latents")
    pattern_path = choose_existing_path(args.patterns, discovery.pattern_candidates, "patterns")

    selected = {
        "sst": sst_path,
        "latents": latent_path,
        "patterns": pattern_path,
        "config": args.config.expanduser().resolve() if args.config else None,
    }

    missing: list[str] = []
    if sst_path is None or not sst_path.exists():
        missing.append("SST/SSTA NetCDF with time/lat/lon coordinates")
    if latent_path is None or not latent_path.exists():
        missing.append("KGAE latent-coordinate NetCDF with time and mode dimensions")

    try:
        mode_configs = load_manual_boxes(args.config, notes)
    except Exception as exc:
        missing.append(f"usable manual box config ({exc})")
        mode_configs = None

    if mode_configs is None:
        if pattern_path is None or not pattern_path.exists():
            missing.append(
                "KGAE pattern/linear-response NetCDF for objective box selection"
            )
        else:
            try:
                mode_configs = auto_boxes_from_patterns(
                    pattern_path,
                    args.pattern_var,
                    args.box_half_lat,
                    args.box_half_lon,
                    notes,
                )
            except Exception as exc:
                missing.append(f"usable pattern maps for auto boxes ({exc})")

    if missing:
        report = write_discovery_report(outdir, discovery, selected, notes, missing)
        print("Missing required inputs; no fabricated data were used.", file=sys.stderr)
        print(f"Missing: {missing}", file=sys.stderr)
        print(f"Discovery report: {report}", file=sys.stderr)
        return 2

    assert mode_configs is not None
    try:
        latents = load_latents(latent_path, args.latent_var, mode_configs, notes)
        first_latent = next(iter(latents.values()))
        sst_ds = rename_lat_lon(open_dataset_checked(sst_path, "sst"))
        sst_da = choose_variable(sst_ds, args.sst_var, "sst", notes)
        sst_name = sst_da.name or args.sst_var or "sst"
        sst_da = subset_to_time_span(sst_da, first_latent)
        ssta = prepare_ssta(sst_da, sst_name, args.sst_is_anomaly, notes)
        box_indices = compute_box_indices(ssta, mode_configs)
        pattern_fields = load_pattern_fields(
            pattern_path, args.pattern_var, mode_configs, notes
        )

        standardized: dict[str, pd.DataFrame] = {}
        for mode in MODE_ORDER:
            df = align_pair(latents[mode], box_indices[mode], mode, notes)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < 12:
                raise ValueError(f"{mode}: fewer than 12 valid aligned samples.")
            df["latent_z"] = standardize(df["latent"].to_numpy())
            df["box_z"] = standardize(df["box"].to_numpy())
            standardized[mode] = df

        plot_pattern_and_distributions(
            standardized, mode_configs, pattern_fields, outdir, args.bins
        )
        summary = write_summary(standardized, mode_configs, outdir)
        write_response_note(outdir, selected, summary, notes)
        write_discovery_report(outdir, discovery, selected, notes)
    except Exception as exc:
        notes.append(f"Run failed after input discovery: {exc}")
        report = write_discovery_report(outdir, discovery, selected, notes, [str(exc)])
        print(f"Analysis failed: {exc}", file=sys.stderr)
        print(f"Discovery report: {report}", file=sys.stderr)
        return 1

    print(f"Wrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
