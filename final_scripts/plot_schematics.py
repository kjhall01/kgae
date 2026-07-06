import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch
import torch.nn.functional as F

# ======================================================
# Figure setup (wider, stacked panels with spacer)
# ======================================================
fig = plt.figure(figsize=(12.5, 11))  # <<< CHANGED (wider)

gs = fig.add_gridspec(
    5, 1,
    height_ratios=[1.1, 0.4, 0.8, 0.4, 1.1],  # <<< CHANGED (spacer row)
    hspace=0.0
)

axes = [
    fig.add_subplot(gs[0]),
    fig.add_subplot(gs[2]),
    fig.add_subplot(gs[4])
]

# ======================================================
# Helper: panel labels
# ======================================================
def panel_label(ax, label):
    ax.text(
        -0.06, 1.04, label,
        transform=ax.transAxes,
        fontsize=14, fontweight='bold',
        ha='left', va='top',
        clip_on=False
    )

# ======================================================
# PANEL 1 — VAE DIAGRAM
# ======================================================
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Encoder trapezoid
encoder_coords = [(0.5,0.5),(3,1.2),(3,4.8),(0.5,5.5)]
ax.add_patch(Polygon(
    encoder_coords, closed=True,
    edgecolor='black', facecolor='white', lw=1.5
))
ax.text(1.7,3, r"Encoder $q_\phi(z|x)$",
        ha='center', va='center', fontsize=12)

# Decoder trapezoid
decoder_coords = [(7,1.2),(9.5,0.5),(9.5,5.5),(7,4.8)]
ax.add_patch(Polygon(
    decoder_coords, closed=True,
    edgecolor='black', facecolor='white', lw=1.5
))
ax.text(8.3,3, r"Decoder $p_\theta(x|z)$",
        ha='center', va='center', fontsize=12)

# Latent circle
latent_center = (4.5,3)
latent_radius = 0.6
ax.add_patch(Ellipse(
    latent_center, 1.2, 1.2,
    edgecolor='black', facecolor='white', lw=1.5
))

theta = np.deg2rad([40, 220])
ax.add_line(Line2D(
    latent_center[0] + latent_radius*np.cos(theta),
    latent_center[1] + latent_radius*np.sin(theta),
    color='black', lw=1.2
))

def rotate_point(x, y, cx, cy, angle_deg):
    ang = np.deg2rad(angle_deg)
    xs, ys = x-cx, y-cy
    return (xs*np.cos(ang)-ys*np.sin(ang)+cx,
            xs*np.sin(ang)+ys*np.cos(ang)+cy)

mu_x, mu_y = rotate_point(latent_center[0]+0.24, latent_center[1]+0.24,
                          *latent_center, 90)
sig_x, sig_y = rotate_point(latent_center[0]-0.24, latent_center[1]-0.24,
                           *latent_center, 90)

ax.text(mu_x, mu_y, r"$\mu$", ha='center', va='center', fontsize=12)
ax.text(sig_x, sig_y, r"$\sigma$", ha='center', va='center', fontsize=12)

arrowprops = dict(arrowstyle="->", lw=2)
ax.annotate("", xy=(0.5,3), xytext=(0,3), arrowprops=arrowprops)
ax.text(-0.2,3, r"$x$", ha='center', va='center', fontsize=12)

ax.annotate("", xy=(4.0,2.5), xytext=(3,1.8), arrowprops=arrowprops)
ax.annotate("", xy=(4.0,3.5), xytext=(3,4.2), arrowprops=arrowprops)

ax.annotate("", xy=(7,3), xytext=(5.15,3), arrowprops=arrowprops)
ax.text(6.1, 3.3, r"$z = \mu + \sigma \odot \epsilon$",
        ha='center', va='bottom', fontsize=12)

ax.annotate("", xy=(10,3), xytext=(9.5,3), arrowprops=arrowprops)
ax.text(10.2,3, r"$\hat{x}$", ha='center', va='center', fontsize=12)

panel_label(ax, "a)")

# ======================================================
# PANEL 2 — IoU POWER SPECTRA
# ======================================================
ax = axes[1]

x = np.linspace(0, 10, 500)
spec_i = np.exp(-(x-4)**2 / 2)
spec_j = 0.8*np.exp(-(x-5)**2 / 1.5)

intersection = np.minimum(spec_i, spec_j)
union = np.maximum(spec_i, spec_j)

ax.fill_between(x, 0, union, facecolor='white',
                edgecolor='black', hatch='/', linewidth=0)
ax.fill_between(x, 0, intersection, facecolor='white',
                edgecolor='black', hatch='xx', linewidth=0)

ax.plot(x, spec_i, 'k', lw=2, label=r'$\hat{P}(\mu_i)$')
ax.plot(x, spec_j, 'k--', lw=2, label=r'$\hat{P}(\mu_j)$')

ax.spines[['top','right','left']].set_visible(False)
ax.yaxis.set_visible(False)
ax.set_xticks([])
ax.set_xlabel("Frequency")
ax.legend(frameon=False, loc='upper right')

# --- IoU hatch legend (axes-relative inset; confined here) ---
legax = inset_axes(ax, width="22%", height="38%", loc="upper left")
legax.axis('off')

legax.add_patch(Rectangle((0.0,0.6),0.35,0.25,
                          facecolor='white', edgecolor='black', hatch='///'))
legax.text(0.45,0.72,"union",va='center')

legax.add_patch(Rectangle((0.0,0.15),0.15,0.25,
                          facecolor='white', edgecolor='black', hatch='///'))
legax.add_patch(Rectangle((0.2,0.15),0.15,0.25,
                          facecolor='white', edgecolor='black', hatch='\\\\'))
legax.text(0.175,0.28,"+",ha='center',va='center')
legax.text(0.45,0.28,"intersection",va='center')
ax.set_ylim(bottom=0)
ax.spines['bottom'].set_position(('data', 0))

panel_label(ax, "b)")

# ======================================================
# PANEL 3 — CURVE-FIT LOSS
# ======================================================
ax = axes[2]

freqs = torch.linspace(0, 10, 500)
psd_primary = torch.exp(-(freqs-5)**2/2)*(1+0.5*torch.sigmoid(freqs-5))
psd_bump = 0.25*torch.exp(-(freqs-7.5)**2/0.2)
psd = (psd_primary + psd_bump)
psd /= psd.sum()

B = torch.cumsum(psd, dim=0)*(1-1e-5)
logitB = torch.log(B/(1-B+1e-8))

freqs2 = torch.linspace(-5,5,500)
X = torch.stack([freqs2**i for i in range(1,5)], dim=1)
beta = torch.linalg.lstsq(X, logitB).solution
fit_logit = X @ beta

fit_psd = torch.sigmoid(fit_logit)*(1-torch.sigmoid(fit_logit))
fit_psd /= fit_psd.sum()
residuals = psd - fit_psd

ax.plot(freqs, psd, 'k', lw=2, label='Latent PSD')
ax.plot(freqs, fit_psd, 'k--', lw=2, label='Transformed Poly-Fit')

for i in range(len(freqs)-1):
    if residuals[i]>=0 and residuals[i+1]>=0:
        ax.fill_between(freqs[i:i+2], fit_psd[i:i+2], psd[i:i+2],
                        facecolor='white', edgecolor='k', hatch='//', linewidth=0)
    elif residuals[i]<0 and residuals[i+1]<0:
        ax.fill_between(freqs[i:i+2], psd[i:i+2], fit_psd[i:i+2],
                        facecolor='white', edgecolor='k', hatch='\\\\', linewidth=0)

ax.spines[['top','right','left']].set_visible(False)
ax.set_yticks([])
ax.set_xlabel("Frequency")
ax.set_xticks([])  # <<< CHANGED (keep label, remove ticks)
ax.legend(frameon=False, loc='upper left')

axins = inset_axes(ax, width="33%", height="33%", loc='upper right')
axins.plot(freqs, logitB, 'k', lw=1.2)
axins.plot(freqs, fit_logit, 'k--', lw=1.2)
axins.set_xticks([])
axins.set_yticks([])
axins.set_title("logit-space poly-fit", fontsize=8)
ax.set_ylim(bottom=0)
ax.spines['bottom'].set_position(('data', 0))

panel_label(ax, "c)")
plt.savefig('schematics.png', dpi=300)
plt.show()
