import numpy as np
import matplotlib.pyplot as plt

def plot_geometry_cv(train, val, sig=None,
                     geometry='alpha',
                     tendency_modes=None,
                     predictor_modes=None,
                     mode_labels=None,
                     colors=dict(unmasked="tab:blue", masked="tab:orange"),
                     figsize=(15, 4)):
    """
    Plot CV reproducibility of geometry (alpha or beta), showing
    regression slope/intercept + R² with 90% CI per seed, using mean across seeds for scatter.

    Parameters
    ----------
    train, val : xarray.DataArray
        Train and validation geometry. For beta, shape (point, seed, tendency_mode, predictor_mode)
        For alpha, shape (point, seed, tendency_mode)
    sig : xarray.DataArray or None
        Same shape as train, boolean mask where geometry is significant
    geometry : str
        'alpha' or 'beta'
    tendency_modes : list
        indices of tendency modes to plot
    predictor_modes : list or None
        indices of predictor modes (for beta)
    mode_labels : list
        Labels for tendency modes
    colors : dict
        colors for unmasked/masked
    figsize : tuple
        figure size
    """

    # ---- determine plotting layout ----
    n_tm = len(tendency_modes)
    if geometry == 'alpha':
        fig, axes = plt.subplots(
            1, n_tm,
            figsize=figsize,
            sharex=True, sharey=True
        )
        if n_tm == 1:
            axes = [axes]  # make iterable
    else:  # beta
        n_pm = len(predictor_modes)
        fig, axes = plt.subplots(
            n_tm, n_pm,
            figsize=(3.5*n_pm, 3.0*n_tm),
            sharex=True, sharey=True
        )

    # ---- helper functions ----
    def linregress_with_r2(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 5:
            return np.nan, np.nan, np.nan
        m, b = np.polyfit(x, y, 1)
        yhat = m*x + b
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return m, b, r2

    def regress_across_seeds(train_arr, val_arr, sig_arr=None):
        slopes, intercepts, r2s = [], [], []
        for s in range(train_arr.shape[1]):
            x = train_arr[:, s]
            y = val_arr[:, s]
            if sig_arr is not None:
                mask = sig_arr[:, s]
                x = x[mask]
                y = y[mask]
            m, b, r2 = linregress_with_r2(x, y)
            slopes.append(m)
            intercepts.append(b)
            r2s.append(r2)
        return np.array(slopes), np.array(intercepts), np.array(r2s)

    # ---- plotting ----
    if geometry == 'alpha':
        for i_tm, tm in enumerate(tendency_modes):
            ax = axes[i_tm]

            x = train.sel(tendency_mode=tm).values
            y = val.sel(tendency_mode=tm).values
            s = sig.sel(tendency_mode=tm).values if sig is not None else None

            # compute regressions per seed
            m_u, b_u, r2_u = regress_across_seeds(x, y)
            m_m, b_m, r2_m = regress_across_seeds(x, y, sig_arr=s)

            # scatter: mean across seeds
            x_mean = np.nanmean(x, axis=1)
            y_mean = np.nanmean(y, axis=1)
            ax.scatter(x_mean, y_mean, s=5, alpha=0.4, color=colors['unmasked'])
            if s is not None:
                mask_flat = np.nanmean(s, axis=1) > 0.5  # majority of seeds
                ax.scatter(x_mean[mask_flat], y_mean[mask_flat], s=5, alpha=0.6, color=colors['masked'])

            # 1–1 line
            lim = np.nanpercentile(np.r_[x_mean, y_mean], [1, 99])
            ax.plot(lim, lim, 'k--', lw=0.8)
            xx = np.linspace(*lim, 100)
            ax.plot(xx, np.nanmean(m_u)*xx + np.nanmean(b_u), color=colors['unmasked'], lw=1.8)
            if s is not None:
                ax.plot(xx, np.nanmean(m_m)*xx + np.nanmean(b_m), color=colors['masked'], lw=1.8)

            # --- compute 90% CI ---
            def ci(arr): return np.nanpercentile(arr, [5, 50, 95])
            m_u_ci = ci(m_u)
            b_u_ci = ci(b_u)
            r2_u_ci = ci(r2_u)
            if s is not None:
                m_m_ci = ci(m_m)
                b_m_ci = ci(b_m)
                r2_m_ci = ci(r2_m)

            # annotation: top-left
            txt = f"Unmasked: y={m_u_ci[1]:.2f}x+{b_u_ci[1]:.2f}\n"
            txt += f"R²={r2_u_ci[1]:.2f} [{r2_u_ci[0]:.2f},{r2_u_ci[2]:.2f}]\n"
            if s is not None:
                txt += f"Masked: y={m_m_ci[1]:.2f}x+{b_m_ci[1]:.2f}\n"
                txt += f"R²={r2_m_ci[1]:.2f} [{r2_m_ci[0]:.2f},{r2_m_ci[2]:.2f}]"

            ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                    ha='left', va='top', fontsize=8)

            ax.set_title(mode_labels[i_tm] if mode_labels is not None else f"TM {tm}")
            ax.set_xlabel("Train α")
            if i_tm == 0:
                ax.set_ylabel("Validation α")

    else:  # beta
        for i_tm, tm in enumerate(tendency_modes):
            for i_pm, pm in enumerate(predictor_modes):
                ax = axes[i_tm, i_pm]

                x = train.sel(tendency_mode=tm, predictor_mode=pm).values
                y = val.sel(tendency_mode=tm, predictor_mode=pm).values
                s = sig.sel(tendency_mode=tm, predictor_mode=pm).values if sig is not None else None

                # regressions per seed
                m_u, b_u, r2_u = regress_across_seeds(x, y)
                m_m, b_m, r2_m = regress_across_seeds(x, y, sig_arr=s)

                # scatter: mean across seeds
                x_mean = np.nanmean(x, axis=1)
                y_mean = np.nanmean(y, axis=1)
                ax.scatter(x_mean, y_mean, s=5, alpha=0.4, color=colors['unmasked'])
                if s is not None:
                    mask_flat = np.nanmean(s, axis=1) > 0.5
                    ax.scatter(x_mean[mask_flat], y_mean[mask_flat], s=5, alpha=0.6, color=colors['masked'])

                # 1–1 line
                lim = np.nanpercentile(np.r_[x_mean, y_mean], [1, 99])
                ax.plot(lim, lim, 'k--', lw=0.8)
                xx = np.linspace(*lim, 100)
                ax.plot(xx, np.nanmean(m_u)*xx + np.nanmean(b_u), color=colors['unmasked'], lw=1.8)
                if s is not None:
                    ax.plot(xx, np.nanmean(m_m)*xx + np.nanmean(b_m), color=colors['masked'], lw=1.8)

                # --- 90% CI ---
                def ci(arr): return np.nanpercentile(arr, [5, 50, 95])
                m_u_ci, b_u_ci, r2_u_ci = ci(m_u), ci(b_u), ci(r2_u)
                if s is not None:
                    m_m_ci, b_m_ci, r2_m_ci = ci(m_m), ci(b_m), ci(r2_m)

                # annotation: top-left
                txt = f"Unmasked: y={m_u_ci[1]:.2f}x+{b_u_ci[1]:.2f}\n"
                txt += f"R²={r2_u_ci[1]:.2f} [{r2_u_ci[0]:.2f},{r2_u_ci[2]:.2f}]\n"
                if s is not None:
                    txt += f"Masked: y={m_m_ci[1]:.2f}x+{b_m_ci[1]:.2f}\n"
                    txt += f"R²={r2_m_ci[1]:.2f} [{r2_m_ci[0]:.2f},{r2_m_ci[2]:.2f}]"

                ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                        ha='left', va='top', fontsize=7)

                if i_tm == n_tm-1:
                    ax.set_xlabel(f"Train β (PM={pm})")
                if i_pm == 0:
                    ax.set_ylabel(f"Val β\n{mode_labels[i_tm] if mode_labels is not None else tm}")

    # --- final touches ---
    handles = [
        plt.Line2D([], [], color=colors['unmasked'], label="Unmasked"),
        plt.Line2D([], [], color=colors['masked'], label="Masked"),
        plt.Line2D([], [], color='k', linestyle='--', label="1–1")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
