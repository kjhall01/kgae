import numpy as np
import xarray as xr
import kgae

# ----------------------------
# user settings
# ----------------------------
experiment = "kgvae_cv1940-2014"
n_ensemble = 95

# which mode to diagnose (QB)
qb_mode = 5

# Optional time-range limit
time_slice = (None, None)  # e.g. ("1950-01-01","2014-12-31")

# period band (years) to search for a peak (QB-ish)
band_years = (7, 40)

# ----------------------------
# helpers
# ----------------------------
def _rename_mode_dim(latents: xr.DataArray, mode_coord_name="tendency_mode") -> xr.DataArray:
    if "mode" in latents.dims and mode_coord_name not in latents.dims:
        latents = latents.rename({"mode": mode_coord_name})
    return latents


def _infer_ens_dim(da: xr.DataArray):
    for cand in ["seed", "ensemble", "member"]:
        if cand in da.dims:
            return cand
    return None


def pairwise_corr_matrix(Z: xr.DataArray, ens_dim: str, time_dim: str = "time") -> xr.DataArray:
    """
    Z: (ens, time) DataArray
    returns C: (ens, ens) correlation matrix
    """
    # ensure dims order
    Z = Z.transpose(ens_dim, time_dim)
    A = Z.values
    # remove time mean per member
    A = A - np.nanmean(A, axis=1, keepdims=True)
    # std per member
    std = np.nanstd(A, axis=1, ddof=0, keepdims=True)
    A = A / np.where(std == 0, np.nan, std)
    # corr = (A A^T) / T
    # use nan-safe via mask
    mask = np.isfinite(A)
    # compute with pairwise valid counts
    C = np.full((A.shape[0], A.shape[0]), np.nan, dtype=np.float64)
    for i in range(A.shape[0]):
        ai = A[i]
        mi = mask[i]
        for j in range(i, A.shape[0]):
            aj = A[j]
            mj = mask[j]
            m = mi & mj
            if m.sum() < 5:
                continue
            cij = np.dot(ai[m], aj[m]) / m.sum()
            C[i, j] = cij
            C[j, i] = cij

    return xr.DataArray(
        C,
        coords={ens_dim: Z[ens_dim].values, f"{ens_dim}_2": Z[ens_dim].values},
        dims=(ens_dim, f"{ens_dim}_2"),
    )


def coherence_to_mean(Z: xr.DataArray, ens_dim: str, time_dim: str = "time") -> xr.DataArray:
    """
    Returns corr(member, ensemble-mean) for each member.
    Z: (ens, time)
    """
    Z = Z.transpose(ens_dim, time_dim)
    zbar = Z.mean(ens_dim, skipna=True)
    out = []
    for e in Z[ens_dim].values:
        r = xr.corr(Z.sel({ens_dim: e}), zbar, dim=time_dim)
        out.append(r)
    return xr.concat(out, dim=ens_dim).assign_coords({ens_dim: Z[ens_dim].values})


def peak_period_years_fft(z: np.ndarray, dt_years: float, band_years=(1.5, 3.5)):
    """
    z: (time,) numpy
    dt_years: sampling interval in years (monthly -> 1/12)
    returns peak_period (years) within band, peak_power, and band_fractional_power
    """
    z = z.astype(np.float64)
    m = np.isfinite(z)
    if m.sum() < 24:
        return np.nan, np.nan, np.nan

    z = z[m]
    z = z - z.mean()
    # FFT
    n = z.size
    fft = np.fft.rfft(z)
    pxx = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=dt_years)  # cycles per year

    # drop zero freq
    freqs = freqs[1:]
    pxx = pxx[1:]
    if freqs.size < 5:
        return np.nan, np.nan, np.nan

    periods = 1.0 / freqs  # years

    # band mask
    lo, hi = band_years
    band = (periods >= lo) & (periods <= hi)
    if not np.any(band):
        return np.nan, np.nan, np.nan

    # peak in band
    idx = np.nanargmax(pxx[band])
    peak_power = pxx[band][idx]
    peak_period = periods[band][idx]

    # fraction of total power in band (excluding 0-freq already)
    total_power = np.nansum(pxx)
    band_power = np.nansum(pxx[band])
    frac = band_power / total_power if total_power > 0 else np.nan

    return float(peak_period), float(peak_power), float(frac)


# ----------------------------
# load latents
# ----------------------------
latents = kgae.open_latents("xval", experiment=experiment, n_ensemble=n_ensemble)

# normalize dim names
latents = _rename_mode_dim(latents, mode_coord_name="tendency_mode")

ens_dim = _infer_ens_dim(latents)
if ens_dim is None:
    raise ValueError(f"Could not infer ensemble dim; latents.dims={latents.dims}")

# enforce expected dims
needed = {ens_dim, "time", "tendency_mode"}
if not needed.issubset(set(latents.dims)):
    raise ValueError(f"Expected dims to include {needed}, got {latents.dims}")

# subset time + QB mode
Z = latents.sel(tendency_mode=qb_mode)
if time_slice != (None, None):
    Z = Z.sel(time=slice(time_slice[0], time_slice[1]))

# Z: (ens, time) or (time, ens) -> standardize
Z = Z.transpose(ens_dim, "time")

print(f"Loaded latents: dims={latents.dims}")
print(f"Using ensemble dim='{ens_dim}', QB mode={qb_mode}, shape={Z.shape}")

# ----------------------------
# metric 1: pairwise corr matrix
# ----------------------------
C = pairwise_corr_matrix(Z, ens_dim=ens_dim, time_dim="time")

# summarize correlation structure
# off-diagonal distribution
C_np = C.values
off = C_np[~np.eye(C_np.shape[0], dtype=bool)]
off = off[np.isfinite(off)]

print("\n--- Pairwise correlation (member vs member) ---")
print(f"off-diag: mean={np.nanmean(off):.3f}, median={np.nanmedian(off):.3f}, "
      f"p10={np.nanpercentile(off,10):.3f}, p90={np.nanpercentile(off,90):.3f}")
print(f"fraction |corr|<0.2 : {np.mean(np.abs(off) < 0.2):.3f}")
print(f"fraction corr<0      : {np.mean(off < 0):.3f}")

# ----------------------------
# metric 2: coherence to ensemble mean
# ----------------------------
r_to_mean = coherence_to_mean(Z, ens_dim=ens_dim, time_dim="time")
r_np = r_to_mean.values
r_np = r_np[np.isfinite(r_np)]

print("\n--- Correlation to ensemble mean ---")
print(f"mean={np.nanmean(r_np):.3f}, median={np.nanmedian(r_np):.3f}, "
      f"min={np.nanmin(r_np):.3f}, p10={np.nanpercentile(r_np,10):.3f}, "
      f"p90={np.nanpercentile(r_np,90):.3f}, max={np.nanmax(r_np):.3f}")

# ----------------------------
# metric 3: peak-period scatter in QB band
# ----------------------------
# infer dt in years from time coordinate (assumes regular sampling)
t = Z["time"].values
if t.size < 3:
    raise ValueError("Not enough time points to infer dt.")

# monthly data: dt ~ 1 month
# use median spacing in days -> years
dt_days = np.median(np.diff(t).astype("timedelta64[D]").astype(float))
dt_years = dt_days / 365.25

peak_periods = []
peak_powers = []
band_fracs = []

for e in Z[ens_dim].values:
    z_e = Z.sel({ens_dim: e}).values
    pp, pw, frac = peak_period_years_fft(z_e, dt_years=dt_years, band_years=band_years)
    peak_periods.append(pp)
    peak_powers.append(pw)
    band_fracs.append(frac)

peak_periods = np.array(peak_periods, dtype=float)
peak_powers = np.array(peak_powers, dtype=float)
band_fracs = np.array(band_fracs, dtype=float)

m = np.isfinite(peak_periods)
print("\n--- Peak period in band (FFT) ---")
print(f"band_years={band_years}, dt_years≈{dt_years:.4f}")
if m.sum() == 0:
    print("No valid peak periods found (too much missing data or insufficient length).")
else:
    print(f"valid members: {m.sum()}/{peak_periods.size}")
    print(f"peak period: mean={np.nanmean(peak_periods[m]):.3f}y, "
          f"median={np.nanmedian(peak_periods[m]):.3f}y, "
          f"p10={np.nanpercentile(peak_periods[m],10):.3f}y, "
          f"p90={np.nanpercentile(peak_periods[m],90):.3f}y")
    print(f"band power fraction: mean={np.nanmean(band_fracs):.3f}, "
          f"median={np.nanmedian(band_fracs):.3f}, "
          f"p10={np.nanpercentile(band_fracs,10):.3f}, "
          f"p90={np.nanpercentile(band_fracs,90):.3f}")

# ----------------------------
# optional: quick flags for "mis-sorting"
# ----------------------------
print("\n--- Simple instability flags (heuristics) ---")
flag_low_coherence = np.nanmean(r_np) < 0.6
flag_many_low_pair = np.mean(np.abs(off) < 0.2) > 0.2
flag_wide_peak = (np.nanpercentile(peak_periods[m], 90) - np.nanpercentile(peak_periods[m], 10)) > 0.8 if m.sum() > 10 else False

print(f"low mean corr-to-mean (<0.6): {flag_low_coherence}")
print(f"many weak pairwise correlations (|r|<0.2 in >20% pairs): {flag_many_low_pair}")
print(f"wide peak-period spread (p90-p10 > 0.8y): {flag_wide_peak}")

# Optional array export:
# C.to_netcdf("qb_pairwise_corr.nc")
# xr.Dataset({"r_to_mean": r_to_mean}).to_netcdf("qb_corr_to_mean.nc")
# xr.Dataset({"peak_period_years": (ens_dim, peak_periods),
#             "band_power_frac": (ens_dim, band_fracs)}).to_netcdf("qb_peak_periods.nc")
