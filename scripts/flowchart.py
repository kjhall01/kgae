import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def add_box(ax, xy, w, h, text, fontsize=12, fc="white", ec="black", lw=1.8, rounding=0.12):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return box

def add_arrow(ax, p0, p1, text=None, text_offset=(0, 0), fontsize=11, color="black",
              connectionstyle="arc3,rad=0.0", arrowstyle="-|>", lw=1.8, mutation_scale=18):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle=arrowstyle,
        linewidth=lw, color=color,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle
    )
    ax.add_patch(arr)
    if text:
        mx = 0.5*(p0[0] + p1[0]) + text_offset[0]
        my = 0.5*(p0[1] + p1[1]) + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize)
    return arr

# -----------------------------
# Build flowchart
# -----------------------------
fig, ax = plt.subplots(figsize=(10.5, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# Boxes
dec = add_box(
    ax, (1.0, 6.6), 8.0, 2.2,
    "DECADAL MODE (slow background state)\n"
    "• sets mean state / envelope\n"
    "• modulates ENSO response",
    fontsize=12
)

enso = add_box(
    ax, (1.0, 1.4), 8.0, 2.2,
    "INTERANNUAL MODE (ENSO-like)\n"
    "• dominant spatial pattern\n"
    "• strong intrinsic nonlinearity",
    fontsize=12
)

# Helper: box edge anchor points
def top_center(box):   return (box.get_x() + box.get_width()/2, box.get_y() + box.get_height())
def bot_center(box):   return (box.get_x() + box.get_width()/2, box.get_y())
def left_mid(box):     return (box.get_x(), box.get_y() + box.get_height()/2)
def right_mid(box):    return (box.get_x() + box.get_width(), box.get_y() + box.get_height()/2)

# Main coupling arrows
add_arrow(
    ax, bot_center(dec), top_center(enso),
    text="Decadal → ENSO modulation\n$\\beta_{\\mathrm{int,dec}}$\n(amplitude + nonlinear character)",
    text_offset=(2.35, 0.0),
    connectionstyle="arc3,rad=-0.10"
)
add_arrow(
    ax, top_center(enso), bot_center(dec),
    text="ENSO → decadal perturbation\n$\\beta_{\\mathrm{dec,int}}$\n(distinct interaction pattern)",
    text_offset=(-2.35, 0.0),
    connectionstyle="arc3,rad=-0.10"
)

# Self terms (side loops)
# Decadal self-state dependence
add_arrow(
    ax, right_mid(dec), (9.6, 7.7),
    text=None, connectionstyle="arc3,rad=0.35"
)
add_arrow(
    ax, (9.6, 7.7), right_mid(dec),
    text="$\\beta_{\\mathrm{dec,dec}}$\n(projects onto ENSO subspace)",
    text_offset=(0.0, 0.55),
    connectionstyle="arc3,rad=0.35",
    fontsize=10
)

# ENSO self-nonlinearity
add_arrow(
    ax, right_mid(enso), (9.6, 2.5),
    text=None, connectionstyle="arc3,rad=0.35"
)
add_arrow(
    ax, (9.6, 2.5), right_mid(enso),
    text="$\\beta_{\\mathrm{int,int}}$\n(ENSO self-nonlinearity)",
    text_offset=(0.0, 0.55),
    connectionstyle="arc3,rad=0.35",
    fontsize=10
)

# Optional: a small legend / note box summarizing your key empirical finding
note = add_box(
    ax, (1.0, 9.1), 8.0, 0.7,
    "Empirical summary:  $\\beta_{\\mathrm{dec,dec}}\\;\\approx\\; a\\,\\alpha_{\\mathrm{int}}\\; +\\; c\\,\\beta_{\\mathrm{int,int}}$   (dominant ENSO subspace)",
    fontsize=10, lw=1.2
)

plt.tight_layout()
plt.show()
