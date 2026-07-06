import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------------
# Example PSD (asymmetric logistic with more prominent secondary peak)
# ---------------------------
freqs = torch.linspace(0, 10, 500)
# Primary asymmetric peak
psd_primary = torch.exp(-(freqs-5)**2 / 2) * (1 + 0.5 * torch.sigmoid(freqs-5))
# Slightly more prominent secondary bump
psd_bump = 0.25 * torch.exp(-(freqs-7.5)**2 / 0.2)
psd = psd_primary + psd_bump
psd = psd / psd.sum()          # normalize

# Compute cumulative sum (B)
B = torch.cumsum(psd, dim=0) * (1 - 1e-5)

# Logit transform
logitB = torch.log(B / (1 - B + 1e-8))

# Polynomial fit in logit space
fit_power = 4
freqs2 = torch.linspace(-5, 5, 500)
X = torch.stack([freqs2**i for i in range(1, fit_power+1)], dim=1)
beta = torch.linalg.lstsq(X, logitB).solution
fit_logit = X @ beta

# Map fit back to PSD-space via sigmoid derivative
fit_psd = F.sigmoid(fit_logit) * (1 - F.sigmoid(fit_logit))
fit_psd = fit_psd / fit_psd.sum()

# Residuals in PSD space
residuals = psd - fit_psd

# ---------------------------
# Main PSD plot with residual hatching
# ---------------------------
fig, ax = plt.subplots(figsize=(7,4))

ax.plot(freqs, psd, 'k', lw=2, label='PSD (bimodal)')
ax.plot(freqs, fit_psd, 'k--', lw=2, label='Poly fit')

# Residuals as vertical hatching
for i in range(len(freqs)-1):
    x0, x1 = freqs[i], freqs[i+1]
    y0, y1 = psd[i], psd[i+1]
    f0, f1 = fit_psd[i], fit_psd[i+1]

    if residuals[i] >= 0 and residuals[i+1] >= 0:
        ax.fill_between([x0, x1], [f0, f1], [y0, y1],
                        facecolor='white', edgecolor='k', hatch='//', linewidth=0)
    elif residuals[i] < 0 and residuals[i+1] < 0:
        ax.fill_between([x0, x1], [y0, y1], [f0, f1],
                        facecolor='white', edgecolor='k', hatch='\\\\', linewidth=0)

ax.set_xlabel("Frequency")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
ax.grid(False)
ax.legend(frameon=False, loc='upper left')

# ---------------------------
# Inset logit-space plot
# ---------------------------
axins = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=1)
axins.plot(freqs, logitB, 'k', lw=1.5, label='Logit(PSD cumsum)')
axins.plot(freqs, fit_logit, 'k--', lw=1.5, label='Poly fit')
axins.set_xticks([])
axins.set_yticks([])
axins.set_title("logit-space", fontsize=8)
axins.spines['top'].set_visible(True)
axins.spines['right'].set_visible(True)
axins.spines['left'].set_visible(True)
axins.spines['bottom'].set_visible(True)

plt.tight_layout()
plt.show()
