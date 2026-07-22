"""Plot the audited latency–model size–MAP trade-off for fall detection.

The six values below correspond to the saved best checkpoints from the common
fall-video-dataset architecture-selection experiment.  Latency is a raw
vision-front-end CUDA kernel measurement on the development TITAN RTX; it is
neither application end-to-end latency nor Hi3516CV610 latency.  Run:

    python plot_3d_fall_models.py

The figure is saved beside this script as fall_models_3d.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# DATA DEFINITION — audited on 2026-07-22; see
# runs/model_family_comparison/model_family_audit.json.
# Units: raw front-end latency = mean ms/image (batch 1, FP16, 640×640,
# 50 warm-ups + 300 timed iterations with torch.cuda.Event); size = total
# checkpoint model parameters in millions; MAP = saved-best internal-val score.
# =============================================================================
pose_n_latency, pose_n_size, pose_n_map = 14.819, 4.947546, 97.58788
pose_s_latency, pose_s_size, pose_s_map = 14.813, 13.077434, 96.28549

seg_n_latency, seg_n_size, seg_n_map = 15.198, 4.374539, 88.68564
seg_s_latency, seg_s_size, seg_s_map = 15.588, 12.755947, 87.48694

poseg_n_latency, poseg_n_size, poseg_n_map = 17.022, 5.982699, 92.61648
poseg_s_latency, poseg_s_size, poseg_s_map = 17.315, 15.049547, 94.19420


# View and output settings.
ELEVATION = 24       # Vertical viewing angle in degrees.
AZIMUTH = -55        # Horizontal viewing angle in degrees.
OUTPUT_DPI = 240
SHOW_FIGURE = False


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
    # PoseFall is deliberately dark because it is the selected route.
    colors = {
        "PoseFall": "#123B73",
        "SegmentFall": "#A9D9C0",
        "PoseGFall": "#9CC9E2",
    }

    series = {
        "PoseFall": {
            "points": [
                (pose_n_latency, pose_n_size, pose_n_map),
                (pose_s_latency, pose_s_size, pose_s_map),
            ],
            "labels": ["PoseFall-n (selected)", "PoseFall-s"],
        },
        "SegmentFall": {
            "points": [
                (seg_n_latency, seg_n_size, seg_n_map),
                (seg_s_latency, seg_s_size, seg_s_map),
            ],
            "labels": ["SegmentFall-n", "SegmentFall-s"],
        },
        "PoseGFall": {
            "points": [
                (poseg_n_latency, poseg_n_size, poseg_n_map),
                (poseg_s_latency, poseg_s_size, poseg_s_map),
            ],
            "labels": ["PoseGFall-n", "PoseGFall-s"],
        },
    }

    fig = plt.figure(figsize=(10.5, 7.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Draw light series first, then the selected dark PoseFall series on top.
    for name in ("SegmentFall", "PoseGFall", "PoseFall"):
        plot_model_pair(
            ax,
            series[name]["points"],
            series[name]["labels"],
            colors[name],
            line_width=2.4 if name == "PoseFall" else 1.7,
            zorder=5 if name == "PoseFall" else 3,
        )

    ax.set_xlabel("Front-end kernel latency (ms/image)", labelpad=12)
    ax.set_ylabel("Total parameters (M)", labelpad=12)
    ax.set_zlabel("Internal-val competition MAP (%)", labelpad=10)
    ax.set_title("Fall-detection Representation Route Trade-off", fontsize=15, weight="bold", pad=18)

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
            linewidth=2.4 if name == "PoseFall" else 1.7,
            label=name,
        )
        for name in ("PoseFall", "SegmentFall", "PoseGFall")
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
