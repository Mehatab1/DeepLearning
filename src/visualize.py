from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no screen needed, we only save images

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

from .classes import CLASS_COLORS, CLASS_IDS, CLASS_NAMES

# colormap where class 1 -> first colour, class 2 -> second colour, ...
CMAP = ListedColormap([CLASS_COLORS[c] for c in CLASS_IDS])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], CMAP.N)

LEGEND = []
for c in CLASS_IDS:
    LEGEND.append(Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c]))


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)]


def colorize(class_map):
    # turns a map of class ids into an RGB image
    rgb = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    for c in CLASS_IDS:
        rgb[class_map == c] = hex_to_rgb(CLASS_COLORS[c])
    return rgb


def monthly_grid(stack, months, title=None):
    # 12 maps in a 3 x 4 grid
    cols = 4
    rows = int(np.ceil(len(months) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.7))
    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < len(months):
            axes[i].imshow(stack[i], cmap=CMAP, norm=NORM, interpolation="nearest")
            axes[i].set_title(months[i], fontsize=11)

    fig.legend(handles=LEGEND, loc="lower center", ncol=6, frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def composition_chart(percentages):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for c in CLASS_IDS:
        name = CLASS_NAMES[c]
        ax.plot(percentages.index, percentages[name], marker="o", linewidth=2,
                color=CLASS_COLORS[c], label=name)

    ax.set_xlabel("Month")
    ax.set_ylabel("Share of area (%)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=9, frameon=False)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def month_to_month_heatmap(m2m):
    data = m2m[[CLASS_NAMES[c] for c in CLASS_IDS]]
    values = data.values

    # same colour scale on both sides of zero
    limit = np.abs(values).max()
    if limit < 0.1:
        limit = 0.1

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(data) + 1.5))
    im = ax.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    # write the number in each cell
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            ax.text(c, r, f"{values[r, c]:+.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Change (percentage points)")
    ax.set_title("Month-to-month change in class share (pp)")
    fig.tight_layout()
    return fig


def total_change_bars(total):
    values = total["Change (pp)"]
    colors = [CLASS_COLORS[c] for c in CLASS_IDS]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(values.index, values.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)

    for i in range(len(values)):
        v = values.values[i]
        if v >= 0:
            ax.text(v, i, f" {v:+.2f} pp ", va="center", ha="left", fontsize=9)
        else:
            ax.text(v, i, f" {v:+.2f} pp ", va="center", ha="right", fontsize=9)

    ax.set_xlabel("Change from first to last month (percentage points)")
    ax.invert_yaxis()

    # some space for the labels
    pad = max(abs(values).max() * 0.35, 0.5)
    ax.set_xlim(min(values.min(), 0) - pad, max(values.max(), 0) + pad)
    fig.tight_layout()
    return fig


def change_maps(stack, months, switches):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    axes[0].imshow(stack[0], cmap=CMAP, norm=NORM, interpolation="nearest")
    axes[0].set_title("First month (" + months[0] + ")")

    axes[1].imshow(stack[-1], cmap=CMAP, norm=NORM, interpolation="nearest")
    axes[1].set_title("Last month (" + months[-1] + ")")

    im = axes[2].imshow(switches, cmap="magma", interpolation="nearest")
    axes[2].set_title("Class switches over 12 months")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    fig.legend(handles=LEGEND, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def save_gif(stack, months, path, scale=None, duration_ms=800):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = stack.shape[1], stack.shape[2]
    if scale is None:
        # make small maps bigger so the gif is readable
        scale = max(1, 480 // max(h, w))

    frames = []
    for i in range(len(months)):
        img = Image.fromarray(colorize(stack[i]))
        img = img.resize((w * scale, h * scale), Image.NEAREST)

        # white strip on top with the month
        frame = Image.new("RGB", (img.width, img.height + 28), "white")
        frame.paste(img, (0, 28))
        draw = ImageDraw.Draw(frame)
        draw.text((8, 7), "LULC  " + months[i], fill="black")
        frames.append(frame)

    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return path


def save_all(results, output_dir, title=None):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    stack = results["stack"]
    months = results["summary"]["months"]

    figures = {}
    figures["lulc_12_months.png"] = monthly_grid(stack, months, title)
    figures["class_composition.png"] = composition_chart(results["percentages"])
    figures["month_to_month_change_pp.png"] = month_to_month_heatmap(results["month_to_month_pp"])
    figures["total_change_pp.png"] = total_change_bars(results["total_change_pp"])
    figures["change_maps.png"] = change_maps(stack, months, results["switches"])

    paths = {}
    for name in figures:
        figures[name].savefig(folder / name, dpi=130, bbox_inches="tight")
        plt.close(figures[name])
        paths[name] = folder / name

    paths["lulc_12_months.gif"] = save_gif(stack, months, folder / "lulc_12_months.gif")
    return paths
