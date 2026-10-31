"""Create high-resolution SciVis report figures and a seven-sample GIF."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from scivis_utils import (
    DATA_PATH,
    GIF_SAMPLE_DAYS,
    display_field,
    frame_range,
    make_quiver,
    marching_squares_plot,
    map_axes,
    norm_for,
    open_dataset,
    scalar_mesh,
    temporal_range,
    units_for,
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent / "images/scivis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def color_experiments(ds):
    index = 44  # 2022-02-14: near the center of the selected quarter.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    # Continuous sequential mapping of maximum temperature.
    field = display_field(ds, "tmmx", index)
    limits = temporal_range(ds, "tmmx")
    im = scalar_mesh(axes[0, 0], ds, field, "viridis", norm_for(field, "Continuous", limits))
    map_axes(axes[0, 0], ds, "Continuous · sequential · Tmax (°C)")
    fig.colorbar(im, ax=axes[0, 0], orientation="horizontal", fraction=0.05, pad=0.08, label="°C · quarter-global limits")

    # Discrete palette with the same physical limits.
    im = scalar_mesh(
        axes[0, 1],
        ds,
        field,
        "YlOrRd",
        norm_for(field, "Discrete (8 intervals)", limits),
    )
    map_axes(axes[0, 1], ds, "Discrete · 8 intervals · Tmax (°C)")
    discrete_bar = fig.colorbar(
        im,
        ax=axes[0, 1],
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        label="°C · quarter-global limits",
    )
    discrete_bar.set_ticks(np.linspace(limits[0], limits[1], 5))

    # Logarithmic precipitation mapping; zero precipitation is masked.
    rain = display_field(ds, "pr", index)
    rain_limits = temporal_range(ds, "pr")
    positive_min = float(ds.pr.where(ds.pr > 0).min(skipna=True).values)
    log_limits = (positive_min, rain_limits[1])
    im = scalar_mesh(axes[1, 0], ds, rain, "magma", norm_for(rain, "Logarithmic", log_limits))
    map_axes(axes[1, 0], ds, "Logarithmic · precipitation (mm/day)")
    fig.colorbar(im, ax=axes[1, 0], orientation="horizontal", fraction=0.05, pad=0.08, label="mm/day · log scale")
    axes[1, 0].text(-124, 25.7, "Gray = 0 mm/day · white = outside grid", fontsize=9, color="#263746")

    # Diverging mapping of anomaly from quarter mean, centered at zero.
    anomaly = display_field(ds, "tmmx", index, anomaly=True)
    amax = max(abs(temporal_range(ds, "tmmx", anomaly=True)[0]), abs(temporal_range(ds, "tmmx", anomaly=True)[1]))
    anomaly_norm = mcolors.TwoSlopeNorm(vmin=-amax, vcenter=0, vmax=amax)
    im = scalar_mesh(axes[1, 1], ds, anomaly, "RdBu_r", anomaly_norm)
    map_axes(axes[1, 1], ds, "Diverging · Tmax anomaly from quarter mean")
    fig.colorbar(im, ax=axes[1, 1], orientation="horizontal", fraction=0.05, pad=0.08, label="°C anomaly · symmetric limits")

    fig.suptitle("Color mapping experiments · 2022-02-14", fontsize=18, weight="bold")
    fig.savefig(OUT_DIR / "fig1_color_mapping_experiments.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def global_local_comparison(ds):
    index = 44
    field = display_field(ds, "tmmx", index)
    limits = temporal_range(ds, "tmmx")
    local = frame_range(field)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    for ax, scale, title in (
        (axes[0], limits, "Quarter-global min/max"),
        (axes[1], local, "Per-day local min/max"),
    ):
        im = scalar_mesh(ax, ds, field, "YlOrRd", norm_for(field, "Continuous", scale))
        map_axes(ax, ds, f"{title} · same Tmax field")
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.1, label="°C")
    fig.suptitle("Temporal normalization changes what colors mean · 2022-02-14", fontsize=17, weight="bold")
    fig.savefig(OUT_DIR / "fig2_global_vs_local_scaling.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def contour_figure(ds):
    index = 44
    variable = "tmmx"
    field = display_field(ds, variable, index)
    low, high = frame_range(field)
    levels = np.linspace(low, high, 9)
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
    fig, ax = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    filled = ax.contourf(lon, lat, field, levels=levels, cmap="YlOrRd", extend="both", corner_mask=False)
    lines = ax.contour(lon, lat, field, levels=levels, colors="#263746", linewidths=0.6, alpha=0.86)
    ax.clabel(lines, levels=lines.levels[::2], inline=True, fontsize=8, fmt="%.1f")
    map_axes(ax, ds, "Filled contours and labeled isolines · Tmax · 2022-02-14")
    fig.colorbar(filled, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, label="Maximum temperature (°C)")
    fig.savefig(OUT_DIR / "fig3_contour_mapping.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def contour_algorithm_comparison(ds):
    index = 44
    variable = "tmmx"
    field = display_field(ds, variable, index)
    low, high = frame_range(field)
    levels = np.linspace(low, high, 11)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True)

    lines, stats = marching_squares_plot(axes[0], field, ds, levels, "YlOrRd")
    map_axes(axes[0], ds, f"Marching Squares · {stats['levels']} isoline thresholds")
    fig.colorbar(
        lines,
        ax=axes[0],
        orientation="horizontal",
        fraction=0.05,
        pad=0.09,
        label="Isoline value (°C)",
    )

    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
    filled = axes[1].contourf(lon, lat, field, levels=levels, cmap="YlOrRd", extend="both", corner_mask=False)
    map_axes(axes[1], ds, f"Contour fill · {len(levels) - 1} value bands")
    fig.colorbar(
        filled,
        ax=axes[1],
        orientation="horizontal",
        fraction=0.05,
        pad=0.09,
        label="Maximum temperature (°C)",
    )
    fig.suptitle("Two contour methods · same Tmax field · 2022-02-14", fontsize=17, weight="bold")
    fig.text(
        0.5,
        -0.02,
        f"Marching Squares: {stats['segments']:,} segments, {stats['vertices']:,} vertices. "
        "Isolines emphasize boundary shape; filled bands emphasize regional value zones.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(OUT_DIR / "fig6_contour_algorithm_comparison.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quiver_experiments(ds):
    index = 44
    speed_max = float(ds.vs.max(skipna=True).values)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    q_for_bar = None
    settings = [
        (axes[0, 0], 16, True, "Magnitude-proportional · stride 16"),
        (axes[0, 1], 32, True, "Magnitude-proportional · stride 32"),
        (axes[1, 0], 16, False, "Same-sized · stride 16"),
        (axes[1, 1], 32, False, "Same-sized · stride 32"),
    ]
    for ax, stride, magnitude_scaled, title in settings:
        q = make_quiver(ax, ds, index, stride, magnitude_scaled, speed_vmax=speed_max)
        q_for_bar = q
        map_axes(ax, ds, title)
    fig.colorbar(q_for_bar, ax=axes.ravel().tolist(), shrink=0.72, label=f"Wind speed (m/s), fixed 0–{speed_max:.1f} scale")
    fig.suptitle(
        "Wind quiver experiments · 2022-02-14\n"
        "`th` is a wind-from bearing; arrows point toward the direction of travel",
        fontsize=17,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "fig4_quiver_sampling_and_scaling.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def precipitation_gif(ds):
    frames = []
    positive_min = float(ds.pr.where(ds.pr > 0).min(skipna=True).values)
    quarter_max = float(ds.pr.max(skipna=True).values)
    log_norm = mcolors.LogNorm(vmin=positive_min, vmax=quarter_max, clip=True)
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
    for index in GIF_SAMPLE_DAYS:
        rain = display_field(ds, "pr", int(index))
        fig, ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
        image = ax.pcolormesh(
            lon,
            lat,
            np.ma.masked_less_equal(np.ma.masked_invalid(rain), 0),
            cmap="YlGnBu",
            norm=log_norm,
            shading="auto",
            rasterized=True,
        )
        zeros = np.ma.masked_where(~(np.isfinite(rain) & (rain == 0)), np.ones_like(rain))
        ax.pcolormesh(
            lon,
            lat,
            zeros,
            cmap=mcolors.ListedColormap(["#d7dce5"]),
            vmin=0,
            vmax=1,
            shading="auto",
            rasterized=True,
        )
        date_label = np.datetime_as_string(ds.time.values[int(index)], unit="D")
        map_axes(ax, ds, f"Daily precipitation · {date_label}")
        fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.05, pad=0.09, label="mm/day · log scale")
        ax.text(-124, 25.7, "Gray = 0 mm/day · white = outside grid", fontsize=8, color="#263746")
        fig.text(0.5, -0.035, "Seven approximately fortnightly samples; quarter-global limits", ha="center", fontsize=9)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(
        OUT_DIR / "fig5_precipitation_7sample_animation.gif",
        frames,
        duration=0.85,
        loop=0,
    )


def main():
    print("Opening prepared dataset…", flush=True)
    with open_dataset(DATA_PATH) as ds:
        print("Writing color mapping figure…", flush=True)
        color_experiments(ds)
        print("Writing global/local scale comparison…", flush=True)
        global_local_comparison(ds)
        print("Writing contour figure…", flush=True)
        contour_figure(ds)
        print("Writing contour algorithm comparison…", flush=True)
        contour_algorithm_comparison(ds)
        print("Writing quiver experiments…", flush=True)
        quiver_experiments(ds)
        print("Rendering precipitation animation…", flush=True)
        precipitation_gif(ds)
    print(f"Created report figures and GIF in {OUT_DIR}")
    print("Uniform day indexes:", ", ".join(str(int(value)) for value in GIF_SAMPLE_DAYS))


if __name__ == "__main__":
    main()
