#!/usr/bin/env python3
"""
OBS-composite Jacobian validation vs regression-predicted Jacobian.

- X = observed SST anomaly (SSTA)
- z = model-encoded latents from X
- Define 4 regimes in (z_i,z_j): ++, +-, -+, --
- Estimate Jacobian-like maps from OBS composites:
    J_i_obs ≈ 0.5 * [ (X++ - X-+) / (zi++ - zi-+)  +  (X+- - X--) / (zi+- - zi--) ]
  (i.e., finite-difference derivative wrt z_i, averaged over z_j sign)

- Regression: fit J_i_model(t) (encoder-Jacobian / tendencies target) on TRAIN:
    J_i_model(t,x) = alpha_i(x) + sum_p beta_i,p(x) * phi_p(t)
  where phi_p are quadratic predictors of z (default: zi^2, zj^2, zi*zj).
  Then predict regime-mean Jacobian maps on TEST and form the same stencil:
    J_i_reg(stencil) built from predicted J_i_reg(++) etc, normalized by (zi++ - zi-+) etc.

Outputs:
- prints counts + denominators (mean zi separations) per regime
- prints pattern corr / RMSE between J_obs and J_reg for i and j
- plots maps: [J_obs, J_reg, diff] for i and j
- also plots interaction curvature B_ij (optional) using 4-corner stencil

This validates composite observations against regression estimates.
"""

import numpy as np
import xarray as xr
import torch
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import kgae


# -----------------------------
# SETTINGS
# -----------------------------
ERA5_SST_PATH = "~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc"
MODEL_PATH    = "./large-ensemble-3-1940-2014/seed0/split0/model.pkl"

PERIOD = ("1940-01-01", "2014-12-31")
TRAIN  = ("1940-01-01", "1990-12-31")
TEST   = ("1991-01-01", "2014-12-31")

# latent modes (1-based)
MODE_I = 4
MODE_J = 5
THRESH = 0.5

# if True, also require |other modes|<sigma to reduce contamination
GATE_OTHER = False
GATE_SIGMA = 1.5

# regression predictors for J_i(z)
# Default predictors: zi^2, zj^2, zi*zj
PREDICTOR_SET = "quadratic_ij"   # or "all_zizj"

central_longitude = 180
proj = ccrs.PlateCarree(central_longitude=central_longitude)
pc   = ccrs.PlateCarree(central_longitude=180)
ADD_STATES = False
CMAP = "RdBu_r"

# plot levels (None => auto symmetric)
LEVELS_J = None
LEVELS_B = None


# -----------------------------
# Helpers
# -----------------------------
def _ensure_latlon_names(da: xr.DataArray) -> xr.DataArray:
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims:
        da = da.rename({"longitude": "lon"})
    return da

def coast(ax):
    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.6)
    if ADD_STATES:
        ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)

def area_weights(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    return w / w.mean()

def pattern_corr(a: xr.DataArray, b: xr.DataArray) -> float:
    a, b = xr.align(a, b, join="inner")
    w = area_weights(a["lat"]).broadcast_like(a)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) == 0:
        return np.nan
    a0 = a.where(m); b0 = b.where(m); w0 = w.where(m)
    ma = (a0*w0).sum(("lat","lon")) / w0.sum(("lat","lon"))
    mb = (b0*w0).sum(("lat","lon")) / w0.sum(("lat","lon"))
    da = a0 - ma; db = b0 - mb
    num = (da*db*w0).sum(("lat","lon"))
    den = np.sqrt((da**2*w0).sum(("lat","lon")) * (db**2*w0).sum(("lat","lon")))
    return float((num/den).item())

def rmse_weighted(a: xr.DataArray, b: xr.DataArray) -> float:
    a, b = xr.align(a, b, join="inner")
    w = area_weights(a["lat"]).broadcast_like(a)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) == 0:
        return np.nan
    e2 = ((a-b)**2).where(m)
    w0 = w.where(m)
    return float(np.sqrt((e2*w0).sum(("lat","lon")) / w0.sum(("lat","lon"))).item())

def auto_levels(*das, q=0.99, n=21):
    vals = []
    for da in das:
        v = np.asarray(da.values).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(np.quantile(np.abs(v), q))
    m = float(np.max(vals)) if vals else 1.0
    if m == 0:
        m = 1.0
    return np.linspace(-m, m, n)

def gate_other_modes(latents: xr.DataArray, keep_modes: list[int]) -> xr.DataArray:
    if not GATE_OTHER:
        return xr.ones_like(latents.isel(mode=0), dtype=bool)
    gate = xr.ones_like(latents.isel(mode=0), dtype=bool)
    for m in latents["mode"].values:
        mi = int(m)
        if mi in keep_modes:
            continue
        gate = gate & (np.abs(latents.sel(mode=mi)) < GATE_SIGMA)
    return gate

def make_regimes(zi: xr.DataArray, zj: xr.DataArray, th: float, base_gate=None):
    if base_gate is None:
        base_gate = xr.ones_like(zi, dtype=bool)
    return {
        "++": base_gate & (zi >  th) & (zj >  th),
        "+-": base_gate & (zi >  th) & (zj < -th),
        "-+": base_gate & (zi < -th) & (zj >  th),
        "--": base_gate & (zi < -th) & (zj < -th),
    }

def composite_mean(X: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    return X.where(mask).mean("time", skipna=True)

def mean_of(da: xr.DataArray, mask: xr.DataArray) -> float:
    return float(da.where(mask).mean("time", skipna=True).item())

def stencil_J_from_quadrants(Xpp, Xmp, Xpm, Xmm, dpp, dmp, dpm, dmm):
    """
    Return J_i estimate:
      0.5 * [(Xpp-Xmp)/(dpp-dmp) + (Xpm-Xmm)/(dpm-dmm)]
    Where d are mean zi in those regimes.
    """
    term1 = (Xpp - Xmp) / (dpp - dmp)
    term2 = (Xpm - Xmm) / (dpm - dmm)
    return 0.5 * (term1 + term2)

def stencil_Bij_from_quadrants(Xpp, Xmp, Xpm, Xmm, dpp, dmp, dpm, dmm):
    """
    Interaction curvature-like stencil (four-corner):
      (Xpp - Xmp) - (Xpm - Xmm)
    scaled by mean zj separation between (+) and (-) sides, *optionally*.
    Here we implement a derivative wrt zj of the derivative wrt zi:
      Bij ≈ [ (Xpp-Xmp)/(zi++-zi-+) - (Xpm-Xmm)/(zi+- - zi--) ] / (zj+ - zj-)
    We'll compute zj+ and zj- as means across the two relevant regimes.
    """
    Ji_pos = (Xpp - Xmp) / (dpp - dmp)   # z_j > 0 side
    Ji_neg = (Xpm - Xmm) / (dpm - dmm)   # z_j < 0 side
    return Ji_pos - Ji_neg


# -----------------------------
# Load data + model latents + model Jacobian targets (tendencies)
# -----------------------------
def load_all():
    # OBS SSTA
    ds = xr.open_dataset(ERA5_SST_PATH)
    sst = _ensure_latlon_names(ds.sst).sel(time=slice(PERIOD[0], PERIOD[1]))

    ssta, *_ = kgae.global_detrend(sst, deg=2)
    X, _ = kgae.remove_climo(ssta)
    X = X.sortby("time")

    # stack features for encoder / jacobian
    Xstk = (
        X.stack(feature=("lat","lon"))
         .dropna("feature", how="any")
         .transpose("time","feature")
    )
    template = Xstk.isel(time=0)

    # model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    model.eval()
    try:
        model.to("cpu")
    except Exception:
        pass

    # encode latents
    with torch.no_grad():
        z = model.encoder(torch.as_tensor(Xstk.values, dtype=torch.float32))
        if getattr(model, "is_variational", False):
            z = z[:, :model.latent_dim]
    z = z.detach().cpu().numpy()

    latents = xr.DataArray(
        z,
        coords={"time": Xstk["time"].values, "mode": np.arange(1, model.latent_dim+1)},
        dims=("time","mode"),
    ).sortby("time")

    # canonical orientation for analysis
    latents = latents * model.flip_signs

    # model Jacobian targets: tendencies = dD/dz_i evaluated along observed trajectory
    # kgae.jacobian gives the Jacobian with respect to encoded SST input.
    tend = kgae.jacobian(model, torch.as_tensor(Xstk.values, dtype=torch.float32))
    tend = tend.detach().cpu().numpy()

    tendencies = xr.DataArray(
        tend,
        coords={"time": Xstk["time"].values,
                "mode": np.arange(1, model.latent_dim+1),
                "feature": template["feature"]},
        dims=("time","mode","feature"),
    ).sortby("time").unstack("feature").sortby(["time","lat","lon"])

    # Align all
    X, latents, tendencies = xr.align(X, latents, tendencies, join="inner")
    return X, latents, tendencies


# -----------------------------
# Fit regression for J_i(z) on TRAIN
# -----------------------------
def fit_regression_J(lat_tr: xr.DataArray, tend_tr: xr.DataArray, mi: int, mj: int):
    zi = lat_tr.sel(mode=mi, drop=True)
    zj = lat_tr.sel(mode=mj, drop=True)

    if PREDICTOR_SET == "quadratic_ij":
        predictors = xr.concat([np.abs(zi), np.abs(zj), zi*zj], dim="mode").assign_coords(
            mode=["zi2","zj2","zizj"]
        ).transpose("time","mode")
    else:
        raise ValueError(f"Unknown PREDICTOR_SET={PREDICTOR_SET}")

    # regress_tendencies_r2 expects:
    #   z : (time, mode)  predictors
    #   T : (time, mode, lat, lon)  with "mode" = tendency modes being fit
    T = tend_tr.sel(mode=[mi, mj]).transpose("time","mode","lat","lon")

    alpha, beta, r2, *_ = kgae.regress_tendencies_r2(predictors, T)
    # alpha: (tendency_mode, lat, lon)
    # beta : (tendency_mode, predictor_mode, lat, lon)
    return alpha, beta, r2


def predict_J_map(alpha, beta, reg_means: dict, tm: int) -> xr.DataArray:
    """
    alpha/beta predict J_tm at a regime by plugging in regime mean predictors.
    reg_means must have keys matching beta.predictor_mode coords (zi2,zj2,zizj)
    """
    out = alpha.sel(tendency_mode=tm)
    for pname, val in reg_means.items():
        out = out + beta.sel(tendency_mode=tm, predictor_mode=pname) * float(val)
    return out


# -----------------------------
# Main validation
# -----------------------------
def main():
    X, latents, tendencies = load_all()

    # split
    X_tr  = X.sel(time=slice(TRAIN[0], TRAIN[1]))
    Z_tr  = latents.sel(time=slice(TRAIN[0], TRAIN[1]))
    T_tr  = tendencies.sel(time=slice(TRAIN[0], TRAIN[1]))

    X_te  = X.sel(time=slice(TEST[0], TEST[1]))
    Z_te  = latents.sel(time=slice(TEST[0], TEST[1]))

    # fit regression for model Jacobians on TRAIN
    alpha, beta, r2 = fit_regression_J(Z_tr, T_tr, MODE_I, MODE_J)

    print("\nRegression diagnostics (TRAIN) for Jacobian targets:")
    print("Area-mean R2:",
          r2.mean(("lat","lon"), skipna=True),
          "  median:",
          r2.median(("lat","lon"), skipna=True))

    # regimes on TEST
    zi = Z_te.sel(mode=MODE_I)
    zj = Z_te.sel(mode=MODE_J)
    base_gate = gate_other_modes(Z_te, keep_modes=[MODE_I, MODE_J])
    reg = make_regimes(zi, zj, THRESH, base_gate)

    # quadrant composites of OBS field
    Xpp = composite_mean(X_te, reg["++"])
    Xmp = composite_mean(X_te, reg["-+"])
    Xpm = composite_mean(X_te, reg["+-"])
    Xmm = composite_mean(X_te, reg["--"])

    # mean zi per regime (for derivative scaling)
    zi_pp = mean_of(zi, reg["++"]); zi_mp = mean_of(zi, reg["-+"])
    zi_pm = mean_of(zi, reg["+-"]); zi_mm = mean_of(zi, reg["--"])

    zj_pp = mean_of(zj, reg["++"]); zj_mp = mean_of(zj, reg["-+"])
    zj_pm = mean_of(zj, reg["+-"]); zj_mm = mean_of(zj, reg["--"])

    # sample counts
    n_pp = int(reg["++"].sum().item())
    n_pm = int(reg["+-"].sum().item())
    n_mp = int(reg["-+"].sum().item())
    n_mm = int(reg["--"].sum().item())

    print("\nTEST regime counts and mean latent values:")
    print(f" ++ n={n_pp:3d}  zi={zi_pp:+.3f} zj={zj_pp:+.3f}")
    print(f" +- n={n_pm:3d}  zi={zi_pm:+.3f} zj={zj_pm:+.3f}")
    print(f" -+ n={n_mp:3d}  zi={zi_mp:+.3f} zj={zj_mp:+.3f}")
    print(f" -- n={n_mm:3d}  zi={zi_mm:+.3f} zj={zj_mm:+.3f}")

    print("\nDenominators (mean separations):")
    print(" zi sep at zj>0:", (zi_pp - zi_mp))
    print(" zi sep at zj<0:", (zi_pm - zi_mm))
    print(" zj sep at zi>0:", (zj_pp - zj_pm))
    print(" zj sep at zi<0:", (zj_mp - zj_mm))

    # -----------------------------
    # 1) OBS-composite Jacobian estimates (from X)
    # -----------------------------
    Ji_obs = stencil_J_from_quadrants(Xpp, Xmp, Xpm, Xmm, zi_pp, zi_mp, zi_pm, zi_mm)
    Jj_obs = stencil_J_from_quadrants(Xpp, Xpm, Xmp, Xmm, zj_pp, zj_pm, zj_mp, zj_mm)  # swap roles carefully

    # For Jj_obs, we want derivative wrt zj holding zi sign fixed:
    #   0.5 * [ (X++-X+-)/(zj++-zj+-) + (X-+ - X--)/(zj-+ - zj--) ]
    # The call above uses Xpp-Xpm and Xmp-Xmm as intended.

    # interaction curvature-like from OBS:
    Bij_obs = stencil_Bij_from_quadrants(Xpp, Xmp, Xpm, Xmm, zi_pp, zi_mp, zi_pm, zi_mm)
    Bji_obs = stencil_Bij_from_quadrants(Xpp, Xpm, Xmp, Xmm, zj_pp, zj_pm, zj_mp, zj_mm)

    # -----------------------------
    # 2) Regression-predicted Jacobian estimates
    # -----------------------------
    # regime means of predictors for regression
    # predictors: zi2, zj2, zizj
    def reg_means(mask):
        return dict(
            zi2=float((zi.where(mask)**2).mean("time", skipna=True).item()),
            zj2=float((zj.where(mask)**2).mean("time", skipna=True).item()),
            zizj=float((zi.where(mask)*zj.where(mask)).mean("time", skipna=True).item()),
        )

    mu_pp = reg_means(reg["++"])
    mu_pm = reg_means(reg["+-"])
    mu_mp = reg_means(reg["-+"])
    mu_mm = reg_means(reg["--"])

    # predicted J maps in each quadrant for tm=i and tm=j
    Ji_pp = predict_J_map(alpha, beta, mu_pp, tm=MODE_I)
    Ji_pm = predict_J_map(alpha, beta, mu_pm, tm=MODE_I)
    Ji_mp = predict_J_map(alpha, beta, mu_mp, tm=MODE_I)
    Ji_mm = predict_J_map(alpha, beta, mu_mm, tm=MODE_I)

    Jj_pp = predict_J_map(alpha, beta, mu_pp, tm=MODE_J)
    Jj_pm = predict_J_map(alpha, beta, mu_pm, tm=MODE_J)
    Jj_mp = predict_J_map(alpha, beta, mu_mp, tm=MODE_J)
    Jj_mm = predict_J_map(alpha, beta, mu_mm, tm=MODE_J)

    # Now build the same stencil combinations but applied to predicted J-quadrant maps.
    # For Ji(z): average over zj sign:
    Ji_reg = 0.5 * (Ji_pp + Ji_pm)  # Ji_pp is "at ++", Ji_pm is at "+-"; these are values of Ji.
    Ji_reg_alt = stencil_J_from_quadrants(
        # Alternative: use predicted *D* fields directly,
        # but here we are validating Jacobian directly: Ji_reg is the regime-average of Ji(z).
        # We'll keep Ji_reg as a regime-average Jacobian, because OBS Ji_obs came from a derivative of X.
        # To mirror the OBS stencil, use Ji_reg_pos = Ji_pp (zj>0, zi>0) vs Ji_mp (zj>0, zi<0), etc.
        # but Ji_reg is already a Jacobian, not a field to be differenced in zi.
        # The correct comparable object is:
        #   Ji_reg_pos ≈ E[Ji | zj>0, zi>th] and Ji_reg_neg ≈ E[Ji | zj>0, zi<-th]
        # However our regression does not condition on zi sign separately unless predictors include sign.
        # So the clean comparison is:
        #   Ji_obs (stencil derivative of X wrt zi)  vs  E[Ji_reg(z) | regimes used in stencil]
        # We'll do that explicitly below.
        Xpp, Xmp, Xpm, Xmm, zi_pp, zi_mp, zi_pm, zi_mm
    )

    # Better: compare "side-specific" predicted Jacobian means:
    #   Ji_reg_pos = 0.5*(Ji_pp + Ji_pm)  (zi>0, both zj signs)
    #   Ji_reg_neg = 0.5*(Ji_mp + Ji_mm)  (zi<0, both zj signs)
    # then predicted derivative-like contrast:
    Ji_reg_derivlike = ( (0.5*(Ji_pp + Ji_pm)) - (0.5*(Ji_mp + Ji_mm)) ) / (
        (0.5*(zi_pp + zi_pm)) - (0.5*(zi_mp + zi_mm))
    )

    # For Jj:
    Jj_reg_derivlike = ( (0.5*(Jj_pp + Jj_mp)) - (0.5*(Jj_pm + Jj_mm)) ) / (
        (0.5*(zj_pp + zj_mp)) - (0.5*(zj_pm + zj_mm))
    )

    # Interaction curvature from regression (mirrors OBS four-corner stencil on Ji wrt zj):
    Bij_reg = ( (Ji_pp - Ji_mp)/(zi_pp - zi_mp) ) - ( (Ji_pm - Ji_mm)/(zi_pm - zi_mm) )
    Bji_reg = ( (Jj_pp - Jj_pm)/(zj_pp - zj_pm) ) - ( (Jj_mp - Jj_mm)/(zj_mp - zj_mm) )

    # -----------------------------
    # Report metrics
    # -----------------------------
    print("\nVALIDATION: OBS-composite Jacobian vs Regression-predicted Jacobian")
    c_i = pattern_corr(Ji_reg_derivlike, Ji_obs)
    r_i = rmse_weighted(Ji_reg_derivlike, Ji_obs)
    c_j = pattern_corr(Jj_reg_derivlike, Jj_obs)
    r_j = rmse_weighted(Jj_reg_derivlike, Jj_obs)
    print(f"J_i (mode {MODE_I})  corr={c_i:.3f}  rmse={r_i:.4g}")
    print(f"J_j (mode {MODE_J})  corr={c_j:.3f}  rmse={r_j:.4g}")

    c_bij = pattern_corr(Bij_reg, Bij_obs)
    r_bij = rmse_weighted(Bij_reg, Bij_obs)
    print(f"B_ij (dJ_i/dz_j)  corr={c_bij:.3f}  rmse={r_bij:.4g}")

    c_bji = pattern_corr(Bji_reg, Bji_obs)
    r_bji = rmse_weighted(Bji_reg, Bji_obs)
    print(f"B_ji (dJ_j/dz_i)  corr={c_bji:.3f}  rmse={r_bji:.4g}")

    # -----------------------------
    # Plots
    # -----------------------------
    def plot_triptych(A, B, title, out, levels=None):
        B = B/B.std(['lat', 'lon']) * A.std(['lat', 'lon'])
        if levels is None:
            levels = auto_levels(A, B, B-A, q=0.99, n=21)
        fig = plt.figure(figsize=(12.6, 3.7))
        axs = [fig.add_subplot(1,3,k+1, projection=proj) for k in range(3)]
        for ax in axs: coast(ax)

        m = axs[0].contourf(A["lon"], A["lat"], A, transform=pc, cmap=CMAP, levels=levels, extend="both")
        axs[0].set_title("OBS composite-estimate", fontsize=10)
        axs[1].contourf(B["lon"], B["lat"], B, transform=pc, cmap=CMAP, levels=levels, extend="both")
        axs[1].set_title("Regression-predicted", fontsize=10)
        axs[2].contourf((B-A)["lon"], (B-A)["lat"], (B-A), transform=pc, cmap=CMAP, levels=levels, extend="both")
        axs[2].set_title("Pred - OBS", fontsize=10)

        fig.suptitle(title, y=1.02)
        cb = fig.colorbar(m, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
        cb.set_label("K per latent-unit (approx)")
        fig.savefig(out, dpi=250, bbox_inches="tight")
        print("Saved:", out)
        plt.show()

    plot_triptych(
        Ji_obs, Ji_reg_derivlike,
        title=f"Jacobian wrt mode {MODE_I}: OBS stencil vs regression",
        out=f"J_validation_mode{MODE_I}.png",
        levels=LEVELS_J
    )

    plot_triptych(
        Jj_obs, Jj_reg_derivlike,
        title=f"Jacobian wrt mode {MODE_J}: OBS stencil vs regression",
        out=f"J_validation_mode{MODE_J}.png",
        levels=LEVELS_J
    )

    plot_triptych(
        Bij_obs, Bij_reg,
        title=f"Interaction curvature B_ij: dJ_i/dz_j  (i={MODE_I}, j={MODE_J})",
        out=f"Bij_validation_i{MODE_I}_j{MODE_J}.png",
        levels=LEVELS_B
    )
    plot_triptych(
        Bji_obs, Bji_reg,
        title=f"Interaction curvature B_ji: dJ_j/dz_i  (j={MODE_I}, i={MODE_J})",
        out=f"Bji_validation_i{MODE_I}_j{MODE_J}.png",
        levels=LEVELS_B
    )

if __name__ == "__main__":
    main()
