import torch
from scipy.ndimage import gaussian_filter
import numpy as np 
import xarray as xr


def jacobian(model, x_batch, device='cpu', delta=1e-2):
    model.eval()
    x_batch = x_batch.to(device)

    # ---- NO GRAD for encoder + setup ----
    with torch.no_grad():
        z = model.encoder(x_batch)
        if model.is_variational:
            z, _ = torch.chunk(z, 2, dim=1)

    # ---- ENABLE grad only for z ----
    z = z.detach().requires_grad_(True)

    B, L = z.shape
    X = x_batch.shape[1]

    tendencies = torch.zeros(B, L, X, device=z.device)

    # ---- JVP requires autograd ----
    for j in range(L):
        v = torch.zeros_like(z)
        v[:, j] = 1.0

        _, jvp = torch.autograd.functional.jvp(
            model.decoder,
            z,
            v,
            create_graph=False
        )

        tendencies[:, j, :] = jvp

    # ---- NO GRAD again ----
    with torch.no_grad():
        tendencies *= (
            torch.tensor(model.flip_signs.values, device=z.device)
            .view(1, L, 1)
        )

    return tendencies


def _eval_r2(X_eval, T_eval_stack, coef, eps=1e-12):
    # T_eval_stack: (time, space) numpy
    Y = T_eval_stack
    Yhat = X_eval @ coef
    resid = Y - Yhat
    ss_res = np.nansum(resid**2, axis=0)
    ybar = np.nanmean(Y, axis=0)
    ss_tot = np.nansum((Y - ybar)**2, axis=0)
    return 1.0 - (ss_res / np.maximum(ss_tot, eps))



def regress_tendencies(
    z,
    T
):
    """
    Full multiple linear regression (fit on z,T):
        T_j(t, x) = alpha_j(x) + sum_k beta_jk(x) z_k(t)

    Parameters
    ----------
    z : xr.DataArray
        dims: (time, mode)   (this is the FIT set; e.g., a bootstrap resample)
    T : xr.DataArray
        dims: (time, mode, lat, lon)

    Returns
    -------
    alpha : xr.DataArray
        dims: (tendency_mode, lat, lon)
    beta : xr.DataArray
        dims: (tendency_mode, predictor_mode, lat, lon)
    """

    # --- sanity checks ---
    assert z.dims == ("time", "mode")
    assert T.dims[:2] == ("time", "mode")

    # Align fit data
    z, T = xr.align(z, T, join="inner")
    time = z.sizes["time"]
    L = z.sizes["mode"]

    # stack space for fit target
    T_stack = T.stack(space=("lat", "lon"))  # (time, mode, space)
    space = T_stack["space"]

    # design matrix for fit: [1, z1, ..., zL]
    X = np.concatenate([np.ones((time, 1)), z.values], axis=1)  # (time, L+1)
    Xt = X.T
    XtX_inv_Xt = np.linalg.pinv(Xt @ X) @ Xt  # (L+1, time)


    alphas = []
    betas = []

    for j in range(L):
        # ----- Fit coefficients for tendency mode j -----
        Y = T_stack[:, j, :].values  # (time, space)
        coef = XtX_inv_Xt @ Y        # (L+1, space)

        alpha_j = coef[0]            # (space,)
        beta_j = coef[1:]            # (L, space)

        alphas.append(alpha_j)
        betas.append(beta_j)

    # stack results
    alpha = np.stack(alphas, axis=0)              # (mode, space)
    beta = np.stack(betas, axis=0)                # (mode, predictor_mode, space)

    # unstack to xarray
    alpha_da = xr.DataArray(
        alpha,
        dims=("tendency_mode", "space"),
        coords={"tendency_mode": T_stack["mode"].values, "space": space},
    ).unstack("space")

    beta_da = xr.DataArray(
        beta,
        dims=("tendency_mode", "predictor_mode", "space"),
        coords={
            "tendency_mode": T_stack["mode"].values,
            "predictor_mode": z["mode"].values,
            "space": space,
        },
    ).unstack("space")


    return alpha_da, beta_da, None, None, None # for signature match 

def regress_tendencies_r2(
    z,
    T,
    ztest=None,
    Ttest=None,
    *,
    zfull=None,
    Tfull=None,
    eps=1e-12,
):
    """
    Full multiple linear regression (fit on z,T):
        T_j(t, x) = alpha_j(x) + sum_k beta_jk(x) z_k(t)

    Computes gridpoint-wise R^2 fields:
      - r2_train: evaluated on the same (z,T) used for fitting (in-sample for regression)
      - r2_cv   : evaluated on (zfull,Tfull) if provided (fixed evaluation set; recommended for bootstrap)
      - r2_test : evaluated on (ztest,Ttest) if provided (out-of-sample for regression)
                 If ztest/Ttest have an extra 'fold' dim, computes R^2 per fold and then averages across folds.

    Parameters
    ----------
    z : xr.DataArray
        dims: (time, mode)   (this is the FIT set; e.g., a bootstrap resample)
    T : xr.DataArray
        dims: (time, mode, lat, lon)
    ztest, Ttest : xr.DataArray, optional
        Either dims:
          - ztest: (time, mode), Ttest: (time, mode, lat, lon)
        or (fold-ensemble):
          - ztest: (fold, time, mode), Ttest: (fold, time, mode, lat, lon)
    zfull, Tfull : xr.DataArray, optional (keyword-only)
        "CV evaluation" set; dims as above. Used to compute r2_cv on a fixed dataset
        even when (z,T) are resampled for fitting.
    eps : float
        small number to avoid divide-by-zero when variance is ~0

    Returns
    -------
    alpha : xr.DataArray
        dims: (tendency_mode, lat, lon)
    beta : xr.DataArray
        dims: (tendency_mode, predictor_mode, lat, lon)
    r2_train : xr.DataArray
        dims: (tendency_mode, lat, lon)
    r2_cv : xr.DataArray or None
        dims: (tendency_mode, lat, lon) if zfull/Tfull provided, else None
    r2_test : xr.DataArray or None
        dims: (tendency_mode, lat, lon) if ztest/Ttest provided, else None
    """

    # --- sanity checks ---
    assert z.dims == ("time", "mode")
    assert T.dims[:2] == ("time", "mode")

    if (ztest is None) ^ (Ttest is None):
        raise ValueError("Provide both ztest and Ttest, or neither.")
    if (zfull is None) ^ (Tfull is None):
        raise ValueError("Provide both zfull and Tfull, or neither.")

    # Align fit data
   # z, T = xr.align(z, T, join="inner")
    time = z.sizes["time"]
    L = T.sizes["mode"]

    # stack space for fit target
    T_stack = T.stack(space=("lat", "lon"))  # (time, mode, space)
    space = T_stack["space"]

    # design matrix for fit: [1, z1, ..., zL]
    X = np.concatenate([np.ones((time, 1)), z.values], axis=1)  # (time, L+1)
    Xt = X.T
    XtX_inv_Xt = np.linalg.pinv(Xt @ X) @ Xt  # (L+1, time)

    # (Optional) build CV-eval design/targets
    if zfull is not None:
        assert zfull.dims == ("time", "mode")
        assert Tfull.dims[:2] == ("time", "mode")
        zfull, Tfull = xr.align(zfull, Tfull, join="inner")

        if not np.array_equal(zfull["mode"].values, z["mode"].values):
            raise ValueError("zfull.mode must match z.mode (same labels/order).")

        Xfull = np.concatenate([np.ones((zfull.sizes["time"], 1)), zfull.values], axis=1)  # (time_full, L+1)
        Tfull_stack = Tfull.stack(space=("lat", "lon"))  # (time_full, mode, space)

        if not np.array_equal(Tfull_stack["mode"].values, T_stack["mode"].values):
            raise ValueError("Tfull.mode must match T.mode (same labels/order).")

        if not np.array_equal(Tfull_stack["space"].values, space.values):
            Tfull_stack = Tfull_stack.sel(space=space)

        print(zfull.time.identical(Tfull.time))
        print(zfull.time[0].values, zfull.time[-1].values)
        print(Tfull.time[0].values, Tfull.time[-1].values)

    # (Optional) prepare test evaluation (supports optional 'fold' dim)
    test_has_fold = False
    fold_vals = None

    if ztest is not None:
        # Accept either (time,mode) or (fold,time,mode)
        if ztest.dims == ("time", "mode"):
            assert Ttest.dims[:2] == ("time", "mode")
            ztest, Ttest = xr.align(ztest, Ttest, join="inner")

            if not np.array_equal(ztest["mode"].values, z["mode"].values):
                raise ValueError("ztest.mode must match z.mode (same labels/order).")

            Xtest = np.concatenate([np.ones((ztest.sizes["time"], 1)), ztest.values], axis=1)  # (time_test, L+1)
            Ttest_stack = Ttest.stack(space=("lat", "lon"))  # (time_test, mode, space)

            if not np.array_equal(Ttest_stack["mode"].values, T_stack["mode"].values):
                raise ValueError("Ttest.mode must match T.mode (same labels/order).")

            if not np.array_equal(Ttest_stack["space"].values, space.values):
                Ttest_stack = Ttest_stack.sel(space=space)

        elif ztest.dims == ("fold", "time", "mode"):
            test_has_fold = True
            assert Ttest.dims[:3] == ("fold", "time", "mode")
            ztest, Ttest = xr.align(ztest, Ttest, join="inner")

            if not np.array_equal(ztest["mode"].values, z["mode"].values):
                raise ValueError("ztest.mode must match z.mode (same labels/order).")

            fold_vals = ztest["fold"].values

            # We'll build Xtest and Ttest_stack per-fold inside the loop,
            # because each fold has its own (time, mode) slice.

            # Also ensure target modes align
            if not np.array_equal(Ttest["mode"].values, T["mode"].values):
                raise ValueError("Ttest.mode must match T.mode (same labels/order).")

        else:
            raise ValueError(
                f"Unsupported ztest dims {ztest.dims}. Expected ('time','mode') or ('fold','time','mode')."
            )

    alphas = []
    betas = []
    r2_train_list = []
    r2_cv_list = [] if zfull is not None else None
    r2_test_list = [] if ztest is not None else None

    for j in range(L):
        # ----- Fit coefficients for tendency mode j -----
        Y = T_stack[:, j, :].values  # (time, space)
        coef = XtX_inv_Xt @ Y        # (L+1, space)

        alpha_j = coef[0]            # (space,)
        beta_j = coef[1:]            # (L, space)

        alphas.append(alpha_j)
        betas.append(beta_j)

        # ----- In-sample (fit-set) R^2 for mode j -----
        r2_j = _eval_r2(X, Y, coef, eps=eps)
        r2_train_list.append(r2_j)

        # ----- CV-eval R^2 on fixed (zfull,Tfull) -----
        if zfull is not None:
            Yf = Tfull_stack[:, j, :].values
            r2_f = _eval_r2(Xfull, Yf, coef, eps=eps)
            r2_cv_list.append(r2_f)

            # pick one gridpoint index s0
            valid = np.isfinite(Yf).any(axis=0)          # space points with at least one finite value
            valid &= np.isfinite(Yf).sum(axis=0) > 50    # require enough finite months (tune threshold)

            idx_valid = np.where(valid)[0]
            print("Valid space points:", idx_valid.size)

            s0 = int(idx_valid[len(idx_valid)//2])  # pick a middle one deterministically
            print("Chosen s0:", s0, "finite count:", np.isfinite(Yf[:, s0]).sum())


            Yf = Tfull_stack[:, j, :].values          # (time, space)
            Yhat_f = Xfull @ coef                     # (time, space)

            # anomalies (remove mean)
            ya = Yf[:, s0] - np.nanmean(Yf[:, s0])
            pa = Yhat_f[:, s0] - np.nanmean(Yhat_f[:, s0])

            print("N:", ya.size)
            print("SS_tot:", float(np.sum(ya**2)))
            print("SS_res:", float(np.sum((ya - pa)**2)))
            print("R2_point:", 1.0 - float(np.sum((ya - pa)**2)) / max(float(np.sum(ya**2)), 1e-12))
            print("var(Y):", float(np.var(ya)), "var(pred):", float(np.var(pa)))
            print("corr:", float(np.corrcoef(ya, pa)[0,1]))

        # ----- Test R^2 (optional), supports fold-ensemble -----
        if ztest is not None:
            if not test_has_fold:
                Yt = Ttest_stack[:, j, :].values
                r2_t = _eval_r2(Xtest, Yt, coef, eps=eps)
                r2_test_list.append(r2_t)
            else:
                # Compute per-fold R^2, then average across folds
                r2_fold = []
                for fv in fold_vals:
                    z_m = ztest.sel(fold=fv)   # (time, mode)
                    T_m = Ttest.sel(fold=fv)   # (time, mode, lat, lon)

                    # align per-fold time axis (just in case)
                    z_m, T_m = xr.align(z_m, T_m, join="inner")

                    X_m = np.concatenate(
                        [np.ones((z_m.sizes["time"], 1)), z_m.values],
                        axis=1
                    )  # (time_m, L+1)

                    Tm_stack = T_m.stack(space=("lat", "lon"))  # (time_m, mode, space)
                    if not np.array_equal(Tm_stack["space"].values, space.values):
                        Tm_stack = Tm_stack.sel(space=space)

                    Y_m = Tm_stack[:, j, :].values  # (time_m, space)
                    r2_fold.append(_eval_r2(X_m, Y_m, coef, eps=eps))

                r2_fold = np.stack(r2_fold, axis=0)  # (fold, space)
                r2_t = np.nanmean(r2_fold, axis=0)   # (space,)
                r2_test_list.append(r2_t)

    # stack results
    alpha = np.stack(alphas, axis=0)              # (mode, space)
    beta = np.stack(betas, axis=0)                # (mode, predictor_mode, space)
    r2_train = np.stack(r2_train_list, axis=0)    # (mode, space)
    r2_cv = np.stack(r2_cv_list, axis=0) if zfull is not None else None
    r2_test = np.stack(r2_test_list, axis=0) if ztest is not None else None

    # unstack to xarray
    alpha_da = xr.DataArray(
        alpha,
        dims=("tendency_mode", "space"),
        coords={"tendency_mode": T_stack["mode"].values, "space": space},
    ).unstack("space")

    beta_da = xr.DataArray(
        beta,
        dims=("tendency_mode", "predictor_mode", "space"),
        coords={
            "tendency_mode": T_stack["mode"].values,
            "predictor_mode": z["mode"].values,
            "space": space,
        },
    ).unstack("space")

    r2_train_da = xr.DataArray(
        r2_train,
        dims=("tendency_mode", "space"),
        coords={"tendency_mode": T_stack["mode"].values, "space": space},
    ).unstack("space")

    r2_cv_da = None
    if r2_cv is not None:
        r2_cv_da = xr.DataArray(
            r2_cv,
            dims=("tendency_mode", "space"),
            coords={"tendency_mode": T_stack["mode"].values, "space": space},
        ).unstack("space")

    r2_test_da = None
    if r2_test is not None:
        r2_test_da = xr.DataArray(
            r2_test,
            dims=("tendency_mode", "space"),
            coords={"tendency_mode": T_stack["mode"].values, "space": space},
        ).unstack("space")

    return alpha_da, beta_da, r2_train_da, r2_cv_da, r2_test_da


BLOCK_LEN_MONTHS = 12

def moving_block_bootstrap_indices(rng: np.random.Generator, n: int, block_len: int) -> np.ndarray:
    """
    Moving-block bootstrap indices for a length-n time series.

    Draw contiguous blocks of length block_len with replacement, concatenate until length n,
    then truncate to n.

    This preserves autocorrelation structure up to ~block_len lags.
    """
    if block_len <= 1:
        # degenerate -> IID bootstrap
        return rng.integers(0, n, size=n)

    if n <= block_len:
        # if record shorter than one block, just sample cyclic shifts of the full record
        start = int(rng.integers(0, n))
        return (np.arange(n) + start) % n

    n_starts = n - block_len + 1
    n_blocks = int(np.ceil(n / block_len))

    starts = rng.integers(0, n_starts, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts], axis=0)
    return idx[:n]

def bootstrap_tendency_regression(
    z_da,
    T_da,
    ztest_da=None,
    Ttest_da=None,
    n_boot=100,
    seed=0
):
    rng = np.random.default_rng(seed)

    alpha_list = []
    beta_list = []
    r2_train_list = []
    r2_test_list = []

    time = z_da.sizes["time"]

    for _ in range(n_boot):
        print(_)
        #idx = rng.integers(0, time, size=time) # keep in case we switch to IID bootstrap
        idx = moving_block_bootstrap_indices(rng, time, block_len=BLOCK_LEN_MONTHS)

        n_unique = np.unique(idx).size
        print( f"bootstrap is sampling {n_unique / time*100}% of data points")
        alpha_b, beta_b, r2_train_da, r2_cv_da, r2_test_da = regress_tendencies(
            z_da.isel(time=idx),
            T_da.isel(time=idx),
           # ztest=ztest_da,
           # Ttest=Ttest_da,
           # zfull=z_da,
           # Tfull=T_da,
        )

        alpha_list.append(alpha_b)
        beta_list.append(beta_b)
        r2_train_list.append(r2_cv_da)
        if r2_test_da is not None:
            r2_test_list.append(r2_test_da)

    alpha_boot = xr.concat(alpha_list, dim="boot")
    beta_boot  = xr.concat(beta_list, dim="boot")

    r2_train_boot = xr.concat(r2_train_list, dim="boot") if None not in r2_train_list and len(r2_train_list) > 0 else None
    r2_test_boot = xr.concat(r2_test_list, dim="boot") if None not in r2_test_list and len(r2_test_list) > 0 else None
    return alpha_boot, beta_boot, r2_train_boot, r2_test_boot



def bootstrap_ci(pct_lt_zero, ci=0.95):
    up = ci + (1-ci)/ 2
    down = (1-ci)/2
    print(up, down)
    is_negative = (pct_lt_zero > up ) 
    is_positive = (pct_lt_zero < down)
    return (is_negative | is_positive).where(np.isfinite(pct_lt_zero), other=np.nan)

def smooth_spatial(da, sigma_latlon=1.5):
    """
    Apply Gaussian smoothing over lat/lon while preserving original NaNs.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with last two dimensions being lat/lon.
    sigma_latlon : float
        Gaussian smoothing sigma for lat/lon.

    Returns
    -------
    xarray.DataArray
        Smoothed array with original NaNs preserved.
    """
    data = da.values
    # Save original NaN mask
    orig_nan_mask = ~np.isfinite(data)

    # Prepare masked arrays for smoothing
    filled = np.where(orig_nan_mask, 0.0, data)
    weights = (~orig_nan_mask).astype(float)

    nd = data.ndim
    sigma = [0.0] * nd
    sigma[-2] = sigma_latlon
    sigma[-1] = sigma_latlon

    # Smooth data and weights
    smooth_data = gaussian_filter(filled, sigma=sigma)
    smooth_weights = gaussian_filter(weights, sigma=sigma)

    # Normalize
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smooth_data / smooth_weights

    # Set any originally NaN points back to NaN
    out[orig_nan_mask] = np.nan

    return da.copy(data=out)
