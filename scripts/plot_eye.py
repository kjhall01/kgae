import pickle
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import kgae

# ---- edit these ----
experiment = "/Users/kylehall/Desktop/kgae/scripts/kgvae_cv1940-2014"  # root: seed*/split*/model.pkl
n_ensemble = 30
splits = [0, 1, 2, 3, 4]         # folds to average over

sst_path   = os.path.expanduser("~/Desktop/Data/era5/era5.sst.pacific.1x1.1940-2023.nc")
decoder_col_ind = None           # e.g. [2,0,1,3,4] (same for all models), or None
device = "cpu"                   # "cpu" | "cuda" | "mps"
central_lon = 180
cmap = "RdBu_r"
# --------------------

# grid + mask from ssta (only used for coords/unstack)
sst = xr.open_dataset(sst_path).sst
sst = sst.rename({k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in sst.dims})
ssta, *_ = kgae.global_detrend(sst, deg=2)
ssta, _  = kgae.remove_climo(ssta)
tmpl2d = ssta.isel(time=0)

full_feat = tmpl2d.stack(feature=("lat", "lon"))
keep = ~np.isnan(full_feat)
feat = full_feat.where(keep, drop=True)

# -----------------------------
# Ensemble + fold average of D(+I) and D(-I)
# -----------------------------
Yp_sum = None  # decoder(+I)
Yn_sum = None  # decoder(-I)
K = None
n_models = 0
scale = 1
shift= 0.05

for seed in range(n_ensemble):
    for split in splits:
        model_path = os.path.join(experiment, f"seed{seed}", f"split{split}", "model.pkl")
        if not os.path.exists(model_path):
            print(f"missing: {model_path}  (skipping)")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model.eval()
        if hasattr(model, "to"):
            model = model.to(device)

        W0 = model.decoder.network[0].weight.data
        if K is None:
            K = W0.shape[1]
            Zp = torch.eye(K, device=device)*scale + shift    # +I
            Zn = -torch.eye(K, device=device)*scale + shift   # -I

        if decoder_col_ind is not None:
            model.decoder.network[0].weight.data = W0[:, decoder_col_ind]

        with torch.no_grad():
            Yp = model.decoder(Zp)
            Yn = model.decoder(Zn)
        if isinstance(Yp, (tuple, list)): Yp = Yp[0]
        if isinstance(Yn, (tuple, list)): Yn = Yn[0]

        Yp = Yp.detach().cpu().numpy()
        Yn = Yn.detach().cpu().numpy()

        if Yp.ndim == 3: Yp = Yp.reshape(K, -1)
        if Yn.ndim == 3: Yn = Yn.reshape(K, -1)

        if Yp.shape[1] != feat.sizes["feature"]:
            raise ValueError(f"[seed{seed} split{split}] Yp has {Yp.shape[1]} features but grid has {feat.sizes['feature']}.")
        if Yn.shape[1] != feat.sizes["feature"]:
            raise ValueError(f"[seed{seed} split{split}] Yn has {Yn.shape[1]} features but grid has {feat.sizes['feature']}.")

        flips = np.asarray(getattr(model, "flip_signs", np.ones(K)))
        Yp = Yp * flips[:, None]
        Yn = Yn * flips[:, None]

        if Yp_sum is None:
            Yp_sum = Yp.copy()
            Yn_sum = Yn.copy()
        else:
            Yp_sum += Yp
            Yn_sum += Yn

        n_models += 1

Yp_mean = Yp_sum / n_models   # (K, n_feature)
Yn_mean = Yn_sum / n_models
Ydiff = Yp_mean - Yn_mean
Ysum  = Yp_mean + Yn_mean

print(f"Averaged over {n_models} models = {n_ensemble} seeds × {len(splits)} splits (minus any missing).")

# -----------------------------
# Helper to unstack back to (lat,lon)
# -----------------------------
proj = ccrs.PlateCarree(central_longitude=central_lon)
data_crs = ccrs.PlateCarree(central_longitude=central_lon)

def vec_to_field(vec1d):
    vec_full = xr.full_like(full_feat, np.nan).astype(np.float32)
    vec_full.loc[dict(feature=feat.feature)] = vec1d
    return vec_full.unstack("feature")

rows = [Yp_mean, Yn_mean, Ydiff, Ysum]
row_titles = ["D(+I)", "D(−I)", "D(+I) − D(−I)", "D(+I) + D(−I)"]

# -----------------------------
# Plot: 4 rows × K columns with pcolormesh
# -----------------------------
fig, axes = plt.subplots(
    4, K,
    figsize=(3.2*K, 3.0*4),
    subplot_kw={"projection": proj},
    constrained_layout=True
)

for r in range(4):
    for i in range(K):
        ax = axes[r, i] if K > 1 else axes[r]

        field = vec_to_field(rows[r][i])

        # --- key change: pcolormesh instead of contourf ---
        im = ax.pcolormesh(
            field.lon, field.lat, field,
            cmap=cmap,
            transform=data_crs,
            shading="auto"
        )

        ax.coastlines(linewidth=0.6)

        if r == 0:
            ax.set_title(f"mode {i+1}")

        if i == 0:
            ax.text(
                -0.10, 0.5, row_titles[r],
                transform=ax.transAxes,
                rotation=90, va="center", ha="center", fontsize=11
            )

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

plt.show()
