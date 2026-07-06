import glob
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import xarray as xr
import torch


def build_feature_template(
    era5_sst_path: str,
    training_period: Tuple[str, str],
    *,
    var: str = "sst",
) -> xr.DataArray:
    """
    Build a (feature,) template with a MultiIndex feature=(lat,lon) matching training domain mask.
    """
    ds = xr.open_dataset(era5_sst_path)
    if var not in ds:
        raise KeyError(f"Variable {var!r} not found in {era5_sst_path}. Available: {list(ds.data_vars)}")
    sst = ds[var].sel(time=slice(training_period[0], training_period[1]))

    # normalize coord names
    if "latitude" in sst.dims:
        sst = sst.rename({"latitude": "lat"})
    if "longitude" in sst.dims:
        sst = sst.rename({"longitude": "lon"})

    sst_feat = (
        sst.stack(feature=("lat", "lon"))
           .dropna("feature", how="any")  # match training mask
           .transpose("time", "feature")
    )
    feat = sst_feat["feature"]
    return xr.DataArray(
        np.zeros(feat.size, dtype=np.float32),
        dims=("feature",),
        coords={"feature": feat},
        name="template",
    )


def parse_seed_split(path: str) -> Tuple[int, int]:
    """
    Expects .../seed{N}/split{K}/model.pkl
    """
    parts = str(path).split("/")
    seed = int([s for s in parts if s.startswith("seed")][0].replace("seed", ""))
    split = int([s for s in parts if s.startswith("split")][0].replace("split", ""))
    return seed, split


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


@torch.no_grad()
def decode_model(model, z_np: np.ndarray) -> np.ndarray:
    """
    Decode latent vector z_np (LATENT_DIM,) -> flat numpy (n_feature,).
    Applies model.flip_signs internally (assumes z_np is in reference orientation).
    """
    device = next(model.parameters()).device
    sgn = np.asarray(model.flip_signs.values, dtype=np.float32)  # (L,)
    z = _to_tensor((z_np * sgn)[None, :], device=device)

    if hasattr(model, "decode") and callable(getattr(model, "decode")):
        x = model.decode(z)
    elif hasattr(model, "decoder") and callable(getattr(model, "decoder")):
        x = model.decoder(z)
    else:
        raise AttributeError("Model has neither callable .decode nor callable .decoder")

    return x.detach().cpu().numpy().squeeze()


def linear_response_single_mode(
    model,
    *,
    latent_dim: int,
    i_mode0: int,
    a: float,
) -> np.ndarray:
    """
    [D(+a e_i) - D(-a e_i)] / (2a)
    """
    zp = np.zeros(latent_dim, dtype=np.float32); zp[i_mode0] = +a
    zm = np.zeros(latent_dim, dtype=np.float32); zm[i_mode0] = -a
    return (decode_model(model, zp) - decode_model(model, zm)) / (2.0 * a)


def nonlinear_even_response_single_mode(
    model,
    *,
    latent_dim: int,
    i_mode0: int,
    a: float,
) -> np.ndarray:
    """
    [D(+a e_i) + D(-a e_i) - 2D(0)] / a^2
    """
    z0 = np.zeros(latent_dim, dtype=np.float32)
    zp = np.zeros(latent_dim, dtype=np.float32); zp[i_mode0] = +a
    zm = np.zeros(latent_dim, dtype=np.float32); zm[i_mode0] = -a
    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a * a)

def cubic_self_dependence_single_mode(
    model,
    *,
    latent_dim: int,
    i_mode0: int,
    A: float,
    delta: float,
) -> np.ndarray:
    r"""
    "Self-dependence cubic" = linear response (in z) of the local even-curvature (in z)
    along the *same* latent coordinate i.

    Define local curvature around basepoint z0:
        N(z0) = [ D(z0+δ) + D(z0-δ) - 2 D(z0) ] / δ^2      (units: K / z^2)

    Then the cubic self-dependence diagnostic is:
        S = [ N(+A) - N(-A) ] / (2A)                      (units: K / z^3)

    Parameters
    ----------
    A : float
        Basepoint amplitude for the "background state" (typically 1.0 in latent units).
    delta : float
        Small step used to estimate curvature around the basepoint (e.g., 0.1–0.3).
        Should be smaller than A.

    Returns
    -------
    np.ndarray
        Flat decoded field (n_feature,) of the cubic self-dependence term.
    """
    # --- curvature around +A ---
    z0p = np.zeros(latent_dim, dtype=np.float32); z0p[i_mode0] = +A
    zpp = np.zeros(latent_dim, dtype=np.float32); zpp[i_mode0] = +A + delta
    zpm = np.zeros(latent_dim, dtype=np.float32); zpm[i_mode0] = +A - delta
    Np = (decode_model(model, zpp) + decode_model(model, zpm) - 2.0 * decode_model(model, z0p)) / (delta * delta)

    # --- curvature around -A ---
    z0m = np.zeros(latent_dim, dtype=np.float32); z0m[i_mode0] = -A
    zmp = np.zeros(latent_dim, dtype=np.float32); zmp[i_mode0] = -A + delta
    zmm = np.zeros(latent_dim, dtype=np.float32); zmm[i_mode0] = -A - delta
    Nm = (decode_model(model, zmp) + decode_model(model, zmm) - 2.0 * decode_model(model, z0m)) / (delta * delta)

    # --- linear response of curvature wrt z at ±A ---
    return (Np - Nm) / (2.0 * A)


def nonlinear_response_target_given_bg(
    model,
    *,
    latent_dim: int,
    i_target0: int,
    j_bg0: int,
    z_bg: float,
    a_target: float,
) -> np.ndarray:
    """
    N_i(z_bg) = [D(z_bg, +a_i) + D(z_bg, -a_i) - 2 D(z_bg, 0)] / a_i^2
    All other latents set to 0.
    """
    z0 = np.zeros(latent_dim, dtype=np.float32)
    z0[j_bg0] = z_bg
    z0[i_target0] = 0.0

    zp = np.zeros(latent_dim, dtype=np.float32)
    zp[j_bg0] = z_bg
    zp[i_target0] = +a_target

    zm = np.zeros(latent_dim, dtype=np.float32)
    zm[j_bg0] = z_bg
    zm[i_target0] = -a_target

    return (decode_model(model, zp) + decode_model(model, zm) - 2.0 * decode_model(model, z0)) / (a_target * a_target)


def linear_response_of_nonlinear(
    model,
    *,
    latent_dim: int,
    i_target0: int,
    j_bg0: int,
    a_target: float,
    a_bg: float,
) -> np.ndarray:
    """
    S_{i<-j} = [N_i(+a_bg) - N_i(-a_bg)] / (2 a_bg)
    """
    Np = nonlinear_response_target_given_bg(
        model, latent_dim=latent_dim, i_target0=i_target0, j_bg0=j_bg0, z_bg=+a_bg, a_target=a_target
    )
    Nm = nonlinear_response_target_given_bg(
        model, latent_dim=latent_dim, i_target0=i_target0, j_bg0=j_bg0, z_bg=-a_bg, a_target=a_target
    )
    return (Np - Nm) / (2.0 * a_bg)


def compute_foldmean_decoder_probes(
    model_glob: str,
    *,
    era5_sst_path: str,
    training_period: Tuple[str, str],
    latent_dim: int = 5,
    # mode labels are 1-based (as in your scripts)
    mode_dec: int = 5,
    mode_ia: int = 4,
    mode_qb: int = 3,
    # step sizes
    a_dec: float = 1.0,
    a_ia: float = 1.0,
    a_qb: float = 1.0,
    # template var name
    template_var: str = "sst",
    device: Optional[str] = "cpu",
) -> xr.DataArray:
    """
    Iterate all models seed*/split*/model.pkl, compute per-model maps, then average across splits
    within each seed (fold-mean). Returns:

      xr.DataArray with dims ('seed','lat','lon','quantity')

    Quantities (in this order):
      - lin_qb, lin_ia, lin_dec
      - nonlin_ia, nonlin_dec
      - mod_dec_to_ia, mod_ia_to_dec, mod_qb_to_ia

    Notes:
      - "nonlin" here is the even curvature: [D(+)+D(-)-2D(0)]/a^2
      - "mod" is linear response (wrt background) of the nonlinear response of target.
    """
    # Build template for unstacking feature->(lat,lon)
    template = build_feature_template(era5_sst_path, training_period, var=template_var)
    n_feat = template.sizes["feature"]

    # Resolve indices (0-based)
    i_dec0 = int(mode_dec) - 1
    i_ia0  = int(mode_ia)  - 1
    i_qb0  = int(mode_qb)  - 1
    if max(i_dec0, i_ia0, i_qb0) >= latent_dim:
        raise ValueError("Mode index exceeds latent_dim. Check mode_* and latent_dim.")

    quantities = [
        r"$QB_{lin}$",
        r"$IA_{lin}$",
        r"$DEC_{lin}$",
        r"$IA_{self}$",
        r"$DEC_{self}$",
        r"$DEC_{IA}$",  # DEC modulates IA nonlinearity
        r"$IA_{DEC}$",  # IA  modulates DEC nonlinearity
       # r"$QB_{IA}$",   # QB  modulates IA nonlinearity
    ]

    # per_seed[seed][quantity] = list of (n_feat,) arrays over splits
    per_seed: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    paths = sorted(glob.glob(model_glob))
    if not paths:
        raise FileNotFoundError(f"No models found for model_glob={model_glob}")

    for p in paths:
        seed_idx, split_idx = parse_seed_split(p)

        with open(p, "rb") as f:
            model = pickle.load(f)

        model.eval()
        if device is not None:
            try:
                model.to(torch.device(device))
            except Exception:
                pass

        # feature-length sanity check
        test0 = decode_model(model, np.zeros(latent_dim, dtype=np.float32))
        if test0.shape[0] != n_feat:
            raise ValueError(
                f"Decoded length {test0.shape[0]} != template n_feature {n_feat}. "
                "ERA5 mask/domain mismatch with this model."
            )

        # ---- linear responses ----
        per_seed[seed_idx][r"$QB_{lin}$"].append(linear_response_single_mode(model, latent_dim=latent_dim, i_mode0=i_qb0, a=a_qb))
        per_seed[seed_idx][r"$IA_{lin}$"].append(linear_response_single_mode(model, latent_dim=latent_dim, i_mode0=i_ia0, a=a_ia))
        per_seed[seed_idx][r"$DEC_{lin}$"].append(linear_response_single_mode(model, latent_dim=latent_dim, i_mode0=i_dec0, a=a_dec))

        # ---- nonlinear (even) responses ----
        per_seed[seed_idx][r"$IA_{self}$"].append(cubic_self_dependence_single_mode(model, latent_dim=latent_dim, i_mode0=i_ia0, A=a_ia, delta=a_ia/2))
        per_seed[seed_idx][r"$DEC_{self}$"].append(cubic_self_dependence_single_mode(model, latent_dim=latent_dim,  i_mode0=i_dec0, A=a_dec, delta=a_ia/2))

        # ---- cross-mode modulation of nonlinearity ----
        per_seed[seed_idx][r"$DEC_{IA}$"].append(
            linear_response_of_nonlinear(model, latent_dim=latent_dim, i_target0=i_ia0, j_bg0=i_dec0, a_target=a_ia, a_bg=a_dec)
        )
        per_seed[seed_idx][r"$IA_{DEC}$"].append(
            linear_response_of_nonlinear(model, latent_dim=latent_dim, i_target0=i_dec0, j_bg0=i_ia0, a_target=a_dec, a_bg=a_ia)
        )


    # ---- fold-mean within seed (mean over splits) and stack into DataArray ----
    seed_list = sorted(per_seed.keys())
    if not seed_list:
        raise RuntimeError("No seeds parsed from model_glob paths.")

    # build per-seed per-quantity lat/lon maps
    seed_maps = []
    for s in seed_list:
        q_maps = []
        for q in quantities:
            arrs = per_seed[s].get(q, [])
            if len(arrs) == 0:
                raise RuntimeError(f"Seed {s} is missing quantity {q}. Check your model_glob and mode settings.")
            q_maps.append(np.stack(arrs, axis=0).mean(axis=0))  # (n_feat,)

        q_stack = np.stack(q_maps, axis=0)  # (quantity, n_feat)

        da = xr.DataArray(
            q_stack,
            dims=("quantity", "feature"),
            coords={"quantity": quantities, "feature": template["feature"]},
        ).unstack("feature")  # -> (quantity, lat, lon)

        # ensure consistent dim order
        da = da.transpose("quantity", "lat", "lon")
        seed_maps.append(da)

    out = xr.concat(seed_maps, dim="seed").assign_coords(seed=seed_list)
    # final dim order requested
    out = out.transpose("seed", "lat", "lon", "quantity")
    out.name = "decoder_probes_foldmean"
    return out