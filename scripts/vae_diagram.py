import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.lines import Line2D
import numpy as np

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(8,4))  # slightly less wide
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# --- Encoder trapezoid (more taper) ---
encoder_coords = [
    (0.5,0.5),   # bottom-left (outer)
    (3,1.2),     # bottom-right (inner)
    (3,4.8),     # top-right (inner)
    (0.5,5.5)    # top-left (outer)
]
encoder_trap = Polygon(encoder_coords, closed=True, edgecolor='black', facecolor='white', lw=1.5)
ax.add_patch(encoder_trap)
ax.text(1.7,3, r"Encoder $q_\phi(z|x)$", ha='center', va='center', fontsize=12)

# --- Decoder trapezoid (more taper) ---
decoder_coords = [
    (7,1.2),    # bottom-left (inner)
    (9.5,0.5),  # bottom-right (outer)
    (9.5,5.5),  # top-right (outer)
    (7,4.8)     # top-left (inner)
]
decoder_trap = Polygon(decoder_coords, closed=True, edgecolor='black', facecolor='white', lw=1.5)
ax.add_patch(decoder_trap)
ax.text(8.3,3, r"Decoder $p_\theta(x|z)$", ha='center', va='center', fontsize=12)

# --- Latent circle (shifted left) ---
latent_center = (4.5,3)
latent_radius = 0.6
latent_circle = Ellipse(latent_center, width=1.2, height=1.2, edgecolor='black', facecolor='white', lw=1.5)
ax.add_patch(latent_circle)

# --- Diagonal bisecting line ---
theta = np.deg2rad([40, 220])  # diagonal tilt
r = latent_radius
x0 = latent_center[0] + r * np.cos(theta)
y0 = latent_center[1] + r * np.sin(theta)
ax.add_line(Line2D(x0, y0, color='black', lw=1.2))

# --- μ and σ positions revolved 90° around circle center ---
def rotate_point(x, y, cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    x_shifted, y_shifted = x-cx, y-cy
    xr = x_shifted*np.cos(angle_rad) - y_shifted*np.sin(angle_rad)
    yr = x_shifted*np.sin(angle_rad) + y_shifted*np.cos(angle_rad)
    return xr+cx, yr+cy

# Original positions along diagonal
mu_x, mu_y = latent_center[0] + 0.4*latent_radius, latent_center[1] + 0.4*latent_radius
sigma_x, sigma_y = latent_center[0] - 0.4*latent_radius, latent_center[1] - 0.4*latent_radius

# Revolve 90° around circle center
mu_x, mu_y = rotate_point(mu_x, mu_y, latent_center[0], latent_center[1], 90)
sigma_x, sigma_y = rotate_point(sigma_x, sigma_y, latent_center[0], latent_center[1], 90)

ax.text(mu_x, mu_y, r"$\mu$", ha='center', va='center', fontsize=12)
ax.text(sigma_x, sigma_y, r"$\sigma$", ha='center', va='center', fontsize=12)

# --- Arrows ---
arrowprops = dict(arrowstyle="->", color='black', lw=2)

# x -> Encoder
ax.annotate("", xy=(0.5,3), xytext=(0,3), arrowprops=arrowprops)
ax.text(-0.2,3, r"$x$", ha='center', va='center', fontsize=12)

# Encoder -> Latent (tapered)
ax.annotate("", xy=(4.0,2.5), xytext=(3,1.8), arrowprops=arrowprops)
ax.annotate("", xy=(4.0,3.5), xytext=(3,4.2), arrowprops=arrowprops)

# Latent -> Decoder (starts just outside circle)
start_x = latent_center[0] + latent_radius + 0.05
ax.annotate("", xy=(7,3), xytext=(start_x,3), arrowprops=arrowprops)
ax.text((start_x+7)/2,3.3, r"$z = \mu + \sigma \odot \epsilon$", ha='center', va='bottom', fontsize=12)

# Decoder -> x_hat
ax.annotate("", xy=(10,3), xytext=(9.5,3), arrowprops=arrowprops)
ax.text(10.2,3, r"$\hat{x}$", ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()
