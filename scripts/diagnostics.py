import numpy as np
import xarray as xr
import kgae 

def diagnostic1_delta_z(
    z: xr.DataArray,
    mode_dim: str = "mode",
    time_dim: str = "time",
    extreme_sigma: float = 1.0,
    neutral_sigma: float = 0.25,
) -> xr.Dataset:
    """
    Diagnostic 1: For each mode i, compute the actual latent displacement vector

        Δz(i, k) = E[z_k | z_i > +extreme_sigma*std_i, others neutral]
                - E[z_k | z_i < -extreme_sigma*std_i, others neutral]

    where 'others neutral' means: for all m != i, |z_m| < neutral_sigma*std_m.

    Inputs
    ------
    z : xr.DataArray
        dims must include (mode_dim, time_dim) in any order.
    mode_dim, time_dim : str
        Dimension names for mode and time.
    extreme_sigma : float
        Threshold in std units for defining i+ / i-.
    neutral_sigma : float
        Threshold in std units for defining neutrality on the other modes.

    Returns
    -------
    xr.Dataset with variables:
      - delta_z: (mode_i, mode_k)  [Δz vector for each i]
      - mu_pos : (mode_i, mode_k)  [E[z_k | i+, neutral]]
      - mu_neg : (mode_i, mode_k)  [E[z_k | i-, neutral]]
      - n_pos  : (mode_i,)         [# of time points in i+ group]
      - n_neg  : (mode_i,)         [# of time points in i- group]
    """
    if mode_dim not in z.dims or time_dim not in z.dims:
        raise ValueError(f"z must have dims '{mode_dim}' and '{time_dim}'. Got {z.dims}")

    z = z.transpose(time_dim, mode_dim)  # (time, mode) for convenience
    modes = z[mode_dim].values

    # std per mode (scalar thresholds)
    std = z.std(time_dim, ddof=0)  # (mode,)

    # output containers
    mu_pos_rows = []
    mu_neg_rows = []
    n_pos_list = []
    n_neg_list = []

    for i in modes:
        zi = z.sel({mode_dim: i})

        # neutrality on all other modes
        other_modes = [m for m in modes if m != i]
        if len(other_modes) > 0:
            z_other = z.sel({mode_dim: other_modes})
            std_other = std.sel({mode_dim: other_modes})
            neutral = (np.abs(z_other) < (neutral_sigma * std_other)).all(mode_dim)
        else:
            neutral = xr.ones_like(zi, dtype=bool)

        # i+ and i-
        std_i = float(std.sel({mode_dim: i}).values)
        pos = (zi > (extreme_sigma * std_i)) & neutral
        neg = (zi < (-extreme_sigma * std_i)) & neutral

        # conditional means across time -> (mode,)
        mu_pos = z.where(pos).mean(time_dim)
        mu_neg = z.where(neg).mean(time_dim)

        # store with explicit "mode_i" axis
        mu_pos_rows.append(mu_pos.expand_dims({"mode_i": [i]}))
        mu_neg_rows.append(mu_neg.expand_dims({"mode_i": [i]}))
        n_pos_list.append(int(pos.sum().values))
        n_neg_list.append(int(neg.sum().values))

    mu_pos = xr.concat(mu_pos_rows, dim="mode_i").rename({mode_dim: "mode_k"})
    mu_neg = xr.concat(mu_neg_rows, dim="mode_i").rename({mode_dim: "mode_k"})
    delta = mu_pos - mu_neg

    out = xr.Dataset(
        data_vars=dict(
            delta_z=delta,
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            n_pos=xr.DataArray(n_pos_list, coords={"mode_i": modes}, dims=("mode_i",)),
            n_neg=xr.DataArray(n_neg_list, coords={"mode_i": modes}, dims=("mode_i",)),
        )
    )
    return out


# ---- script entry point ----
# z: xr.DataArray with dims ('time','mode') or ('mode','time')
z =  kgae.open_latents("xval", experiment="kgvae_cv1940-2014", n_ensemble=100).mean('seed')
ds = diagnostic1_delta_z(z, mode_dim="mode", time_dim="time", extreme_sigma=1.0, neutral_sigma=1.0)
print(ds["delta_z"])
print(ds["n_pos"].to_pandas(), ds["n_neg"].to_pandas())


def latent_leakage_metric(delta_z: xr.DataArray,
                          mode_i_dim: str = "mode_i",
                          mode_k_dim: str = "mode_k") -> xr.DataArray:
    """
    Compute leakage metric for each conditioning mode i:

        leakage_i = sum_{k != i} |Δz_{ik}| / |Δz_{ii}|

    Values:
      - ~0   : very "pure" composite displacement
      - ~0.2 : mild leakage
      - >0.5 : substantial leakage

    Returns
    -------
    xr.DataArray with dim mode_i, named "leakage"
    """
    # absolute values
    abs_dz = np.abs(delta_z)

    # diagonal term |Δz_ii|
    diag = abs_dz.sel({mode_k_dim: abs_dz[mode_i_dim]})

    # sum of off-diagonals
    off = abs_dz.where(abs_dz[mode_i_dim] != abs_dz[mode_k_dim], drop=True)\
                .sum(dim=mode_k_dim)

    leakage = (off / diag).rename("leakage")

    return leakage


# ---- script entry point ----
leak = latent_leakage_metric(ds["delta_z"])
print(leak)
