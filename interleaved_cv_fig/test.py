#!/usr/bin/env python3
"""
Generate a color-coded schematic of interleaved non-overlapping 5-fold
time-aware cross-validation for monthly climate samples.

Outputs:
  interleaved_cv_schematic.svg
  interleaved_cv_schematic.pdf
  interleaved_cv_schematic.png
"""

import colorsys
import os
from pathlib import Path
import tempfile

cache_root = Path(tempfile.gettempdir()) / "interleaved_cv_matplotlib_cache"
mpl_cache = cache_root / "matplotlib"
xdg_cache = cache_root / "xdg"
for cache_dir in (mpl_cache, xdg_cache):
    cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


FOLD_COLORS = {
    1: "#0072B2",  # blue
    2: "#E69F00",  # orange
    3: "#009E73",  # green
    4: "#CC79A7",  # rose
    5: "#D55E00",  # vermillion
}
TEXT_COLOR = "#1F252D"
MUTED_TEXT = "#5B6572"


def sample_label(index, predicted=False):
    if predicted:
        return fr"$\hat{{t}}_{{{index}}}$"
    return fr"$t_{{{index}}}$"


def fold_for_sample(index, n_folds):
    return ((index - 1) % n_folds) + 1


def muted_fold_color(fold_id):
    hex_color = FOLD_COLORS[fold_id].lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(
        hue,
        min(0.82, lightness + 0.20),
        saturation * 0.34,
    )
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def fold_sample_map(n_months, n_folds):
    return {
        fold_id: [
            index
            for index in range(1, n_months + 1)
            if fold_for_sample(index, n_folds) == fold_id
        ]
        for fold_id in range(1, n_folds + 1)
    }


def x_positions(n_months, n_folds, x0, cell_w, cycle_gap):
    return [
        x0 + i * cell_w + (i // n_folds) * cycle_gap
        for i in range(n_months)
    ]


def draw_box(
    ax,
    x,
    y,
    width,
    height,
    facecolor,
    edgecolor="white",
    linewidth=0.7,
):
    rect = Rectangle(
        (x, y),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(rect)
    return rect


def draw_sample_cell(
    ax,
    x,
    y,
    width,
    height,
    index,
    fold_id,
    fontsize=7.2,
    predicted=False,
    facecolor=None,
    text_color="white",
):
    draw_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor or FOLD_COLORS[fold_id],
        linewidth=0.6,
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        sample_label(index, predicted=predicted),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
    )


def draw_panel_label(ax, label, x, y):
    ax.text(
        x,
        y,
        label,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=TEXT_COLOR,
    )


def draw_time_arrow(ax, x0, x1, y, label, fontsize=5.8):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops={"arrowstyle": "->", "lw": 0.75, "color": MUTED_TEXT},
    )
    ax.text(
        (x0 + x1) / 2,
        y - 0.10,
        label,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=MUTED_TEXT,
    )


def make_interleaved_cv_schematic(
    output_stem="interleaved_cv_schematic",
    n_months=15,
    n_folds=5,
):
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_months < n_folds:
        raise ValueError("n_months must be at least n_folds")
    if n_folds > len(FOLD_COLORS):
        raise ValueError(f"Define colors before plotting more than {len(FOLD_COLORS)} folds")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "mathtext.default": "it",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fold_samples = fold_sample_map(n_months, n_folds)

    fig_w = 6.8
    fig_h = 6.1
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.0, 1.10, 1.0, 1.55], "hspace": 0.18},
    )
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.axis("off")

    # a) Interleaved fold assignment in the original time order.
    ax = axes[0]
    timeline_x0 = 0.0
    timeline_y = 0.34
    timeline_cell_w = 0.42
    timeline_h = 0.34
    cycle_gap = 0.14
    xs = x_positions(n_months, n_folds, timeline_x0, timeline_cell_w, cycle_gap)
    matrix_right = xs[-1] + timeline_cell_w
    panel_xmax = matrix_right + 0.42
    ax.set_xlim(-0.08, panel_xmax)
    ax.set_ylim(0.02, 0.82)
    draw_panel_label(ax, "a)", -0.08, timeline_y + timeline_h + 0.025)

    for month_index, x in enumerate(xs, start=1):
        fold_id = fold_for_sample(month_index, n_folds)
        draw_sample_cell(
            ax,
            x,
            timeline_y,
            timeline_cell_w,
            timeline_h,
            month_index,
            fold_id,
            fontsize=6.8,
        )

    draw_time_arrow(
        ax,
        timeline_x0,
        matrix_right,
        timeline_y - 0.12,
        "time (interval = 1 month)",
        fontsize=6.3,
    )

    # b) Fold pools after interleaving.
    ax = axes[1]
    pool_y = 0.52
    pool_cell_w = timeline_cell_w
    pool_cell_h = timeline_h
    pool_col_gap = 0.0
    samples_per_fold = max(len(samples) for samples in fold_samples.values())
    pool_group_w = samples_per_fold * pool_cell_w + (samples_per_fold - 1) * pool_col_gap
    pool_group_gap = (matrix_right - n_folds * pool_group_w) / (n_folds - 1)
    ax.set_xlim(-0.08, panel_xmax)
    ax.set_ylim(0.14, 1.05)
    draw_panel_label(ax, "b)", -0.08, pool_y + pool_cell_h + 0.07)

    for fold_id in range(1, n_folds + 1):
        group_x = timeline_x0 + (fold_id - 1) * (pool_group_w + pool_group_gap)
        ax.text(
            group_x + pool_group_w / 2,
            pool_y + pool_cell_h + 0.13,
            f"Fold {fold_id} ({'validation' if fold_id == 1 else 'minibatch'})",
            ha="center",
            va="center",
            fontsize=5.8,
            fontweight="bold",
            color=FOLD_COLORS[fold_id],
        )

        for sample_number, month_index in enumerate(fold_samples[fold_id]):
            x = group_x + sample_number * (pool_cell_w + pool_col_gap)
            draw_sample_cell(
                ax,
                x,
                pool_y,
                pool_cell_w,
                pool_cell_h,
                month_index,
                fold_id,
                fontsize=6.9,
            )
        draw_time_arrow(
            ax,
            group_x,
            group_x + pool_group_w,
            pool_y - 0.12,
            "time (interval = 5 months)",
            fontsize=4.5,
        )

    # c) Out-of-sample predictions from the first held-out fold.
    ax = axes[2]
    ax.set_xlim(-0.08, panel_xmax)
    ax.set_ylim(0.02, 0.82)
    draw_panel_label(ax, "c)", -0.08, timeline_y + timeline_h + 0.025)
    muted_prediction_color = "#D9DEE3"
    muted_prediction_text = "#74808C"

    for month_index, x in enumerate(xs, start=1):
        fold_id = fold_for_sample(month_index, n_folds)
        is_validation = fold_id == 1
        draw_sample_cell(
            ax,
            x,
            timeline_y,
            timeline_cell_w,
            timeline_h,
            month_index,
            fold_id,
            fontsize=6.4,
            predicted=True,
            facecolor=FOLD_COLORS[fold_id] if is_validation else muted_prediction_color,
            text_color="white" if is_validation else muted_prediction_text,
        )

    draw_time_arrow(
        ax,
        timeline_x0,
        matrix_right,
        timeline_y - 0.12,
        "time (interval = 1 month)",
        fontsize=6.3,
    )

    # d) Fold roles across the five cross-validation splits.
    ax = axes[3]
    role_x0 = 0.42
    role_y0 = 0.12
    role_cell_h = 0.28
    role_gap = 0.06
    role_label_x = 0.0
    role_cell_w = (matrix_right - role_x0 - (n_folds - 1) * role_gap) / n_folds
    grid_h = n_folds * role_cell_h + (n_folds - 1) * role_gap
    ax.set_xlim(-0.08, panel_xmax)
    ax.set_ylim(0.04, role_y0 + grid_h + 0.26)
    draw_panel_label(ax, "d)", -0.08, role_y0 + grid_h + 0.07)

    for fold_id in range(1, n_folds + 1):
        x = role_x0 + (fold_id - 1) * (role_cell_w + role_gap)
        ax.text(
            x + role_cell_w / 2,
            role_y0 + grid_h + 0.13,
            f"Fold {fold_id}",
            ha="center",
            va="center",
            fontsize=6.4,
            color=MUTED_TEXT,
        )

    for split_id in range(1, n_folds + 1):
        y = role_y0 + (n_folds - split_id) * (role_cell_h + role_gap)
        ax.text(
            role_label_x,
            y + role_cell_h / 2,
            f"Split {split_id}",
            ha="left",
            va="center",
            fontsize=6.4,
            color=MUTED_TEXT,
        )

        for fold_id in range(1, n_folds + 1):
            x = role_x0 + (fold_id - 1) * (role_cell_w + role_gap)
            is_validation = fold_id == split_id
            draw_box(
                ax,
                x,
                y,
                role_cell_w,
                role_cell_h,
                FOLD_COLORS[fold_id] if is_validation else muted_fold_color(fold_id),
                edgecolor=TEXT_COLOR if is_validation else "white",
                linewidth=1.2 if is_validation else 0.6,
            )

    output_stem = Path(output_stem)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(output_stem.with_suffix(".svg"), **save_kwargs)
    fig.savefig(output_stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, **save_kwargs)

    plt.close(fig)


if __name__ == "__main__":
    make_interleaved_cv_schematic()
