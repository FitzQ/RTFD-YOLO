"""Plot latency–model size–MAP trade-offs for fall-detection models.

Replace the example values in the DATA DEFINITION section with measured data,
then run:

    python plot_3d_fall_models.py

The figure is saved beside this script as fall_models_3d.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# DATA DEFINITION — replace these example values with your real measurements.
# Units used below: latency = ms/image, size = million parameters (M), MAP = %.
# =============================================================================
pose_n_latency, pose_n_size, pose_n_map = 18.0, 3.5, 86.0
pose_s_latency, pose_s_size, pose_s_map = 31.0, 11.2, 91.5

seg_n_latency, seg_n_size, seg_n_map = 22.0, 4.1, 84.5
seg_s_latency, seg_s_size, seg_s_map = 38.0, 12.8, 90.0

poseg_n_latency, poseg_n_size, poseg_n_map = 26.0, 5.2, 89.5
poseg_s_latency, poseg_s_size, poseg_s_map = 45.0, 15.6, 94.0


# View and output settings.
ELEVATION = 24       # Vertical viewing angle in degrees.
AZIMUTH = -55        # Horizontal viewing angle in degrees.
OUTPUT_DPI = 240
SHOW_FIGURE = True


def plot_model_pair(ax, points, labels, color, line_width=1.8, zorder=3):
    """Draw one n→s model pair as a solid 3D line with small circular points."""
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    ax.plot(
        x,
        y,
        z,
        color=color,
        linestyle="-",
        linewidth=line_width,
        marker="o",
        markersize=5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=zorder,
    )

    # Slight offsets keep labels from covering the circular markers.
    for index, (xi, yi, zi, label) in enumerate(zip(x, y, z, labels)):
        # The n label extends to the right; the s label extends to the left.
        # This keeps the largest point away from the outer z-axis tick labels.
        is_last = index == len(labels) - 1
        display_label = f"{label}  " if is_last else f"  {label}"
        ax.text(
            xi,
            yi,
            zi,
            display_label,
            color=color,
            fontsize=9,
            weight="semibold",
            horizontalalignment="right" if is_last else "left",
        )


def main():
    # PoseG is deliberately dark; Pose and Seg use lighter colors.
    colors = {
        "Pose": "#9CC9E2",
        "Seg": "#A9D9C0",
        "PoseG": "#123B73",
    }

    series = {
        "Pose": {
            "points": [
                (pose_n_latency, pose_n_size, pose_n_map),
                (pose_s_latency, pose_s_size, pose_s_map),
            ],
            "labels": ["Pose-n", "Pose-s"],
        },
        "Seg": {
            "points": [
                (seg_n_latency, seg_n_size, seg_n_map),
                (seg_s_latency, seg_s_size, seg_s_map),
            ],
            "labels": ["Seg-n", "Seg-s"],
        },
        "PoseG": {
            "points": [
                (poseg_n_latency, poseg_n_size, poseg_n_map),
                (poseg_s_latency, poseg_s_size, poseg_s_map),
            ],
            "labels": ["PoseG-n", "PoseG-s"],
        },
    }

    fig = plt.figure(figsize=(10.5, 7.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Draw light series first, then the dark PoseG series on top.
    for name in ("Pose", "Seg", "PoseG"):
        plot_model_pair(
            ax,
            series[name]["points"],
            series[name]["labels"],
            colors[name],
            line_width=2.4 if name == "PoseG" else 1.7,
            zorder=5 if name == "PoseG" else 3,
        )

    ax.set_xlabel("Inference latency (ms/image)", labelpad=12)
    ax.set_ylabel("Model size (M parameters)", labelpad=12)
    ax.set_zlabel("Competition MAP (%)", labelpad=10)
    ax.set_title("Fall-detection Model Trade-off", fontsize=15, weight="bold", pad=18)

    # Add data-dependent padding so labels at the minimum/maximum values are
    # not clipped against the 3D axes after real measurements are inserted.
    all_points = [point for item in series.values() for point in item["points"]]
    for setter, values in (
        (ax.set_xlim, [point[0] for point in all_points]),
        (ax.set_ylim, [point[1] for point in all_points]),
        (ax.set_zlim, [point[2] for point in all_points]),
    ):
        low, high = min(values), max(values)
        span = high - low
        padding = span * 0.09 if span else max(abs(low) * 0.09, 1.0)
        setter(low - padding, high + padding)

    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9, pad=2)

    # Light panes and subtle grid lines keep the 3D frame readable.
    pane_color = (0.97, 0.98, 0.99, 1.0)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.75, 0.78, 0.82, 0.38)
        axis._axinfo["grid"]["linewidth"] = 0.7

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[name],
            marker="o",
            markersize=5,
            linewidth=2.4 if name == "PoseG" else 1.7,
            label=name,
        )
        for name in ("Pose", "Seg", "PoseG")
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.96), frameon=True)

    output_path = Path(__file__).with_name("fall_models_3d.png")
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to: {output_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
