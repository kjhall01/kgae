import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Example "power spectral densities"
x = np.linspace(0, 10, 500)
spec_i = np.exp(-(x-4)**2 / 2)          # Gaussian for μ_i
spec_j = 0.8 * np.exp(-(x-5)**2 / 1.5)  # Gaussian for μ_j

# Intersection and union
intersection = np.minimum(spec_i, spec_j)
union = np.maximum(spec_i, spec_j)

fig, ax = plt.subplots(figsize=(8,2.5))

# Union area (single-hatch)
ax.fill_between(x, 0, union, facecolor='white', edgecolor='black', hatch='/', linewidth=0)

# Intersection area (cross-hatch)
ax.fill_between(x, 0, intersection, facecolor='white', edgecolor='black', hatch='xx', linewidth=0)

# Spectra outlines
ax.plot(x, spec_i, 'k', lw=2, label=r'$\hat{P}(\mu_i)$')
ax.plot(x, spec_j, 'k--', lw=2, label=r'$\hat{P}(\mu_j)$')

# Remove bounding box and y-axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_visible(False)

# Keep x-axis but remove tick labels
ax.spines['bottom'].set_position('zero')
ax.set_xticks([])
ax.set_xlabel("Frequency", labelpad=5)

# Legend without box for spectra
ax.legend(frameon=False, loc='upper right')

# Custom inset for union and intersection patterns (top-left)
inset_x = 0.04
inset_y = 0.79
box_w = 0.02   # bigger boxes
box_h = 0.04
spacing = 0.03  # horizontal spacing
vspacing = 0.04
# Union box with single hatch
union_box = Rectangle((inset_x + box_w + spacing, inset_y), box_w, box_h, transform=fig.transFigure,
                      facecolor='white', edgecolor='black', hatch='////', linewidth=0)
fig.patches.append(union_box)
ax.text(inset_x + 2*box_w + spacing + 0.01, inset_y + box_h/2, "union", transform=fig.transFigure, va='center')

# Intersection boxes with two overlapping hatches
inter_box1 = Rectangle((inset_x, inset_y - box_h - vspacing), box_w, box_h, transform=fig.transFigure,
                       facecolor='white', edgecolor='black', hatch='////', linewidth=0)
inter_box2 = Rectangle((inset_x + box_w + spacing, inset_y - box_h - vspacing), box_w, box_h, transform=fig.transFigure,
                       facecolor='white', edgecolor='black', hatch='\\\\\\\\', linewidth=0)
fig.patches.extend([inter_box1, inter_box2])

# Plus sign between the two intersection patches
ax.text(inset_x + box_w + spacing/2, inset_y - vspacing - box_h/2, "+", transform=fig.transFigure,
        va='center', ha='center', fontsize=12)

ax.text(inset_x + 2*box_w + spacing + 0.01, inset_y - vspacing - box_h/2, "intersection",
        transform=fig.transFigure, va='center')

plt.tight_layout()
plt.show()
