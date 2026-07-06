import numpy as np
import xarray as xr


def conditional_composite_alpha_beta(
        D: xr.DataArray,
        Z: xr.DataArray,
        sample_dim: str = 'time',
        latent_dim: str = 'mode',
        lower_q: float = 0.33,
        upper_q: float = 0.67,
        n_bootstrap: int = 100,
        ci_level: float = 95
    ):
    """
    Memory-safe conditional composite alpha/beta for D with arbitrary spatial dims.
    Uses NumPy for bootstrapping to avoid xarray overhead.
    """
    # Dimensions
    tendency_dim = f"tendency_{latent_dim}"
    predictor_dim = f"predictor_{latent_dim}"
    latent_vals = Z[latent_dim].values
    spatial_dims = [d for d in D.dims if d != sample_dim]
    D_flat = D.stack(feature=spatial_dims).values.astype(np.float32)
    n_features = D_flat.shape[1]

    # Quantiles
    #lower_thresh = Z.mean(sample_dim).values -Z.std(sample_dim).values 
    #upper_thresh = Z.mean(sample_dim).values +Z.std(sample_dim).values 
    
    upper_thresh = Z.quantile(upper_q, dim=sample_dim).values
    lower_thresh = Z.quantile(lower_q, dim=sample_dim).values
    
    alpha_maps, alpha_sig_maps = [], []
    alpha_dict = {}

    # --- Alpha ---
    for j_idx, j in enumerate(latent_vals):
        z_j = Z.sel({latent_dim: j}).values.astype(np.float32)
        pos_idx = np.where(z_j > upper_thresh[j_idx])[0]
        neg_idx = np.where(z_j < lower_thresh[j_idx])[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            # Not enough samples
            alpha_j = np.full(n_features, np.nan, dtype=np.float32)
            alpha_sig = np.zeros(n_features, dtype=bool)
        else:
            dlatent = z_j[pos_idx].mean() - z_j[neg_idx].mean()
            alpha_j = (D_flat[pos_idx].mean(axis=0) - D_flat[neg_idx].mean(axis=0)) #/ dlatent

            # Bootstrap
            boot = np.empty((n_bootstrap, n_features), dtype=np.float32)
            for b in range(n_bootstrap):
                s_pos = np.random.choice(pos_idx, len(pos_idx), replace=True)
                s_neg = np.random.choice(neg_idx, len(neg_idx), replace=True)
                dlatent_b = z_j[s_pos].mean() - z_j[s_neg].mean()
                boot[b] = (D_flat[s_pos].mean(axis=0) - D_flat[s_neg].mean(axis=0)) #/ dlatent_b
            lo = np.percentile(boot, (100-ci_level)/2, axis=0)
            hi = np.percentile(boot, 100-(100-ci_level)/2, axis=0)
            alpha_sig = ~((lo <= 0) & (hi >= 0))
        alpha_dict[j] = alpha_j  # store for later

        # Restore spatial shape
        alpha_maps.append(
            xr.DataArray(
                alpha_j.reshape([D[d].size for d in spatial_dims] + [1]),
                coords={**{d: D[d] for d in spatial_dims}},
                dims=spatial_dims + [latent_dim]
            )
        )

        alpha_sig_maps.append(
            xr.DataArray(
                alpha_sig.reshape([D[d].size for d in spatial_dims] + [1]),
                coords={**{d: D[d] for d in spatial_dims}},
                dims=spatial_dims + [latent_dim]
            )
        )

    alpha = xr.concat(alpha_maps, latent_dim).assign_coords({latent_dim: latent_vals}).rename({latent_dim: tendency_dim})
    alpha_sig = xr.concat(alpha_sig_maps, latent_dim).assign_coords({latent_dim: latent_vals}).rename({latent_dim: tendency_dim})

    # --- Beta ---
    beta_maps, beta_sig_maps = [], []
    for t_idx, t in enumerate(latent_vals):
        z_t = Z.sel({latent_dim: t}).values.astype(np.float32)
        t_pos_idx = np.where(z_t > upper_thresh[t_idx])[0]
        t_neg_idx = np.where(z_t < lower_thresh[t_idx])[0]
        t_neutral_idx = np.where((z_t <= upper_thresh[t_idx]) & (z_t >= lower_thresh[t_idx]))[0]
        dlatent_t = z_t[t_pos_idx].mean() - z_t[t_neg_idx].mean() if len(t_pos_idx) and len(t_neg_idx) else np.nan

        beta_pred_list, beta_sig_list = [], []
        for p_idx, p in enumerate(latent_vals):
            z_p = Z.sel({latent_dim: p}).values.astype(np.float32)
            p_pos_idx = np.where(z_p > upper_thresh[p_idx])[0]
            p_neg_idx = np.where(z_p < lower_thresh[p_idx])[0]
            dlatent_p = z_p[p_pos_idx].mean() - z_p[p_neg_idx].mean() if len(p_pos_idx) and len(p_neg_idx) else np.nan

            # Diagonal
            if t == p:
                if len(t_pos_idx)==0 or len(t_neg_idx)==0 or len(t_neutral_idx)==0:
                    beta_tp = np.full(n_features, np.nan, dtype=np.float32)
                    beta_sig_tp = np.zeros(n_features, dtype=bool)
                else:
                    beta_tp = (D_flat[t_pos_idx].mean(axis=0) +
                               D_flat[t_neg_idx].mean(axis=0) -
                               0*D_flat[t_neutral_idx].mean(axis=0)) #/ (dlatent_t)**2
                    boot = np.empty((n_bootstrap, n_features), dtype=np.float32)
                    for b in range(n_bootstrap):
                        s_pos = np.random.choice(t_pos_idx, len(t_pos_idx), replace=True)
                        s_neg = np.random.choice(t_neg_idx, len(t_neg_idx), replace=True)
                        s_neutral = np.random.choice(t_neutral_idx, len(t_neutral_idx), replace=True)
                        dlatent_b = z_t[s_pos].mean() - z_t[s_neg].mean()
                        boot[b] = (D_flat[s_pos].mean(axis=0) +
                                   D_flat[s_neg].mean(axis=0) -
                                   0*D_flat[s_neutral].mean(axis=0)) #/ (dlatent_b)**2
                    lo = np.percentile(boot, (100-ci_level)/2, axis=0)
                    hi = np.percentile(boot, 100-(100-ci_level)/2, axis=0)
                    beta_sig_tp = ~((lo <= 0) & (hi >= 0))
            else:
                # Off-diagonal
                PP_idx = np.intersect1d(t_pos_idx, p_pos_idx)
                PN_idx = np.intersect1d(t_pos_idx, p_neg_idx)
                NP_idx = np.intersect1d(t_neg_idx, p_pos_idx)
                NN_idx = np.intersect1d(t_neg_idx, p_neg_idx)
                if min(len(PP_idx), len(PN_idx), len(NP_idx), len(NN_idx))==0:
                    beta_tp = np.full(n_features, np.nan, dtype=np.float32)
                    beta_sig_tp = np.zeros(n_features, dtype=bool)
                else:
                    beta_raw = ((D_flat[PP_idx].mean(axis=0)-D_flat[PN_idx].mean(axis=0))-
                                (D_flat[NP_idx].mean(axis=0)-D_flat[NN_idx].mean(axis=0))) #/ (dlatent_t*dlatent_p)
                    dz_t_given_p = z_t[p_pos_idx].mean() - z_t[p_neg_idx].mean()
                    beta_tp = beta_raw - alpha_dict[t]*(dz_t_given_p/dlatent_p)

                    boot = np.empty((n_bootstrap, n_features), dtype=np.float32)
                    for b in range(n_bootstrap):
                        s_PP = np.random.choice(PP_idx, len(PP_idx), replace=True)
                        s_PN = np.random.choice(PN_idx, len(PN_idx), replace=True)
                        s_NP = np.random.choice(NP_idx, len(NP_idx), replace=True)
                        s_NN = np.random.choice(NN_idx, len(NN_idx), replace=True)
                        dz_t_b = (z_t[np.r_[s_PP, s_PN]].mean() - z_t[np.r_[s_NP, s_NN]].mean())
                        dz_p_b = (z_p[np.r_[s_PP, s_NP]].mean() - z_p[np.r_[s_PN, s_NN]].mean())
                        dz_t_given_p_b = (z_t[np.r_[s_PP, s_NP]].mean() - z_t[np.r_[s_PN, s_NN]].mean())
                        D_PP = D_flat[s_PP].mean(axis=0)
                        D_PN = D_flat[s_PN].mean(axis=0)
                        D_NP = D_flat[s_NP].mean(axis=0)
                        D_NN = D_flat[s_NN].mean(axis=0)
                        beta_raw_b = ((D_PP-D_PN)-(D_NP-D_NN)) / (dz_t_b*dz_p_b)
                        boot[b] = beta_raw_b  - alpha_dict[t]*(dz_t_given_p_b/dz_p_b)
                    lo = np.percentile(boot, (100-ci_level)/2, axis=0)
                    hi = np.percentile(boot, 100-(100-ci_level)/2, axis=0)
                    beta_sig_tp = ~((lo <= 0) & (hi >= 0))

            beta_pred_list.append(beta_tp)
            beta_sig_list.append(beta_sig_tp)

        # reshape to (predictor, lat, lon)
        beta_maps.append(
            xr.DataArray(np.array(beta_pred_list).reshape([len(latent_vals)]+[D[d].size for d in spatial_dims]),
                         coords={predictor_dim: latent_vals, **{d: D[d] for d in spatial_dims}},
                         dims=[predictor_dim]+spatial_dims)
        )
        beta_sig_maps.append(
            xr.DataArray(np.array(beta_sig_list).reshape([len(latent_vals)]+[D[d].size for d in spatial_dims]),
                         coords={predictor_dim: latent_vals, **{d: D[d] for d in spatial_dims}},
                         dims=[predictor_dim]+spatial_dims)
        )

    beta = xr.concat(beta_maps, dim=tendency_dim).assign_coords({tendency_dim: latent_vals})
    beta_sig = xr.concat(beta_sig_maps, dim=tendency_dim).assign_coords({tendency_dim: latent_vals})

    return alpha, beta, alpha_sig, beta_sig


def cross_composites(
        D: xr.DataArray,
        Z: xr.DataArray,
        sample_dim: str = 'time',
        latent_dim: str = 'mode',
        mode1: int = 2,
        mode2: int = 5,
        lower_q: float = 0.33,
        upper_q: float = 0.67,
        n_bootstrap: int = 1000,
        ci_level: float = 95
    ):
    """
    Conditional composite alpha and beta maps using xarray.concat.
    Includes baseline subtraction to isolate state-dependent modulation.

    D: [sample_dim, lat, lon]
    Z: [sample_dim, latent_dim]
    """

    # Quantile thresholds (simplified, using +/- std as in your code)
    lower_thresh = Z.quantile(lower_q, dim=sample_dim) #
    upper_thresh = Z.quantile(upper_q, dim=sample_dim) #

    # produce the mode1 < BN, mode2 < BN composite maps 
    mode1_ = Z.sel({latent_dim: mode1}, drop=True)
    mode2_ = Z.sel({latent_dim: mode2}, drop=True)

    pos_pos = ( mode1_ > upper_thresh.sel({latent_dim: mode1}, drop=True) ) & ( mode2_ > upper_thresh.sel({latent_dim: mode2}, drop=True) )
    neg_neg = ( mode1_ < lower_thresh.sel({latent_dim: mode1}, drop=True) ) & ( mode2_ < lower_thresh.sel({latent_dim: mode2}, drop=True) )
    pos_neg = ( mode1_ > upper_thresh.sel({latent_dim: mode1}, drop=True) ) & ( mode2_ < lower_thresh.sel({latent_dim: mode2}, drop=True) )
    neg_pos = ( mode1_ < lower_thresh.sel({latent_dim: mode1}, drop=True) ) & ( mode2_ > upper_thresh.sel({latent_dim: mode2}, drop=True) )

    composites, significances = [], []
    for condition in [pos_pos, neg_neg, pos_neg, neg_pos]:
        D_pos = D.sel({sample_dim: condition  })

        alpha_j = D_pos.mean(sample_dim) 

        # Bootstrap
        pos_vals = D_pos.values
        boot = np.empty((n_bootstrap,) + alpha_j.shape)
        for b in range(n_bootstrap):
            boot[b] = (
                pos_vals[np.random.randint(0, pos_vals.shape[0], pos_vals.shape[0])].mean(axis=0)
            )
        lo = np.percentile(boot, (100 - ci_level) / 2, axis=0)
        hi = np.percentile(boot, 100 - (100 - ci_level) / 2, axis=0)
        alpha_sig_j = ~((lo <= 0) & (hi >= 0))

        composites.append(alpha_j)
        significances.append( xr.DataArray(alpha_sig_j, coords=alpha_j.coords, dims=alpha_j.dims))

    mode1_pos = xr.concat([ composites[2], composites[0]], dim='mode2').assign_coords({'mode2': [ 'mode2_neg', 'mode2_pos']})
    mode1_neg = xr.concat([ composites[1], composites[3]], dim='mode2').assign_coords({'mode2': [ 'mode2_neg', 'mode2_pos',]})
    alpha = xr.concat([mode1_neg, mode1_pos ], dim='mode1').assign_coords({'mode1': ['mode1_neg', 'mode1_pos']})

    mode1_pos = xr.concat([significances[2], significances[0]], dim='mode2').assign_coords({'mode2': ['mode2_neg', 'mode2_pos']})
    mode1_neg = xr.concat([significances[1], significances[3]], dim='mode2') .assign_coords({'mode2': ['mode2_neg', 'mode2_pos']})
    alpha_sig = xr.concat([mode1_neg, mode1_pos], dim='mode1').assign_coords({'mode1': ['mode1_neg', 'mode1_pos']})
    return alpha, alpha_sig


def cross_composite_diffs(
    D: xr.DataArray,
    Z: xr.DataArray,
    sample_dim: str = 'time',
    latent_dim: str = 'mode',
    mode1: int = 2,
    mode2: int = 5,
    lower_q: float = 0.30,
    upper_q: float = 0.67,
    n_bootstrap: int = 1000,
    ci_level: float = 95
    ):
    """
    Return composites of differences between the four specified subsets.

    Produces a 2x2 layout of difference composites:
      - Row 0 ("mode2_across_mode1"): differences across mode1 for each mode2 state
      [ pos_pos - neg_pos,  pos_neg - neg_neg ]
      - Row 1 ("mode1_across_mode2"): differences across mode2 for each mode1 state
      [ pos_pos - pos_neg,  neg_pos - neg_neg ]

    D: [sample_dim, lat, lon]
    Z: [sample_dim, latent_dim]
    """
    lower_thresh = Z.quantile(lower_q, dim=sample_dim)
    upper_thresh = Z.quantile(upper_q, dim=sample_dim)

    m1 = Z.sel({latent_dim: mode1}, drop=True)
    m2 = Z.sel({latent_dim: mode2}, drop=True)

    pos_pos_mask = (m1 > upper_thresh.sel({latent_dim: mode1}, drop=True)) & (m2 > upper_thresh.sel({latent_dim: mode2}, drop=True))
    neg_neg_mask = (m1 < lower_thresh.sel({latent_dim: mode1}, drop=True)) & (m2 < lower_thresh.sel({latent_dim: mode2}, drop=True))
    pos_neg_mask = (m1 > upper_thresh.sel({latent_dim: mode1}, drop=True)) & (m2 < lower_thresh.sel({latent_dim: mode2}, drop=True))
    neg_pos_mask = (m1 < lower_thresh.sel({latent_dim: mode1}, drop=True)) & (m2 > upper_thresh.sel({latent_dim: mode2}, drop=True))

    # composites (means)
    C_pp = D.sel({sample_dim: pos_pos_mask}).mean(sample_dim)
    C_nn = D.sel({sample_dim: neg_neg_mask}).mean(sample_dim)
    C_pn = D.sel({sample_dim: pos_neg_mask}).mean(sample_dim)
    C_np = D.sel({sample_dim: neg_pos_mask}).mean(sample_dim)

    # values for bootstrapping
    vals_pp = D.sel({sample_dim: pos_pos_mask}).values
    vals_nn = D.sel({sample_dim: neg_neg_mask}).values
    vals_pn = D.sel({sample_dim: pos_neg_mask}).values
    vals_np = D.sel({sample_dim: neg_pos_mask}).values

    # prepare difference composites
    diff_mode2_pos = C_pp - C_np      # pos_pos - neg_pos (difference across mode1 when mode2 positive)
    diff_mode2_neg = C_pn - C_nn      # pos_neg - neg_neg (difference across mode1 when mode2 negative)
    diff_mode1_pos = C_pp - C_pn      # pos_pos - pos_neg (difference across mode2 when mode1 positive)
    diff_mode1_neg = C_np - C_nn      # neg_pos - neg_neg (difference across mode2 when mode1 negative)

    # bootstrap significance for each diff
    def boot_diff(a_vals, b_vals):
        boot = np.empty((n_bootstrap,) + a_vals.shape[1:])
        for b in range(n_bootstrap):
            a_idx = np.random.randint(0, a_vals.shape[0], a_vals.shape[0]) if a_vals.shape[0] > 0 else np.array([], dtype=int)
            b_idx = np.random.randint(0, b_vals.shape[0], b_vals.shape[0]) if b_vals.shape[0] > 0 else np.array([], dtype=int)
            a_mean = a_vals[a_idx].mean(axis=0) if a_idx.size > 0 else np.full(a_vals.shape[1:], np.nan)
            b_mean = b_vals[b_idx].mean(axis=0) if b_idx.size > 0 else np.full(b_vals.shape[1:], np.nan)
            boot[b] = a_mean - b_mean
        lo = np.nanpercentile(boot, (100 - ci_level) / 2, axis=0)
        hi = np.nanpercentile(boot, 100 - (100 - ci_level) / 2, axis=0)
        return ~((lo <= 0) & (hi >= 0))

    sig_mode2_pos = boot_diff(vals_pp, vals_np)
    sig_mode2_neg = boot_diff(vals_pn, vals_nn)
    sig_mode1_pos = boot_diff(vals_pp, vals_pn)
    sig_mode1_neg = boot_diff(vals_np, vals_nn)

    # assemble into 2x2 layout
    row0 = xr.concat([diff_mode2_pos, diff_mode2_neg], dim='mode2').assign_coords({'mode2': ['mode2_pos', 'mode2_neg']})
    row1 = xr.concat([diff_mode1_pos, diff_mode1_neg], dim='mode2').assign_coords({'mode2': ['mode2_pos', 'mode2_neg']})
    alpha = xr.concat([row0, row1], dim='mode1').assign_coords({'mode1': ['mode2_across_mode1', 'mode1_across_mode2']})

    s0 = xr.concat([xr.DataArray(sig_mode2_pos, coords=diff_mode2_pos.coords, dims=diff_mode2_pos.dims),
            xr.DataArray(sig_mode2_neg, coords=diff_mode2_neg.coords, dims=diff_mode2_neg.dims)], dim='mode2')\
         .assign_coords({'mode2': ['mode2_pos', 'mode2_neg']})
    s1 = xr.concat([xr.DataArray(sig_mode1_pos, coords=diff_mode1_pos.coords, dims=diff_mode1_pos.dims),
            xr.DataArray(sig_mode1_neg, coords=diff_mode1_neg.coords, dims=diff_mode1_neg.dims)], dim='mode2')\
         .assign_coords({'mode2': ['mode2_pos', 'mode2_neg']})
    alpha_sig = xr.concat([s0, s1], dim='mode1').assign_coords({'mode1': ['mode2_across_mode1', 'mode1_across_mode2']})

    return alpha, alpha_sig