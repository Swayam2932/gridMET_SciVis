"""Interactive SciVis dashboard for the gridMET Jan-Mar 2022 slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from scivis_utils import (
    DATA_PATH,
    SCALAR_VARIABLES,
    VARIABLE_LABELS,
    dates_as_strings,
    display_field,
    frame_range,
    make_quiver,
    marching_squares_plot,
    map_axes,
    norm_for,
    open_dataset,
    scalar_mesh,
    units_for,
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from skimage.measure import find_contours


st.set_page_config(page_title="gridMET SciVis", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: none; width: 100%; padding: 1.1rem clamp(1.4rem, 3vw, 3.5rem) 2rem; }
      [data-testid="stMarkdownContainer"] p,
      [data-testid="stCaptionContainer"] p,
      [data-testid="stWidgetLabel"] p,
      [data-testid="stTab"] button {
        font-size: 1.14rem !important;
        line-height: 1.45 !important;
      }
      [data-testid="stWidgetLabel"] p { font-weight: 500 !important; }
      h1 { font-size: 2.9rem !important; }
      h2 { font-size: 1.85rem !important; }
      h3 { font-size: 1.45rem !important; }
      [data-testid="stTabs"] [role="tablist"] { width:100%; justify-content:space-evenly; gap:0; }
      [data-testid="stTabs"] button[role="tab"] { flex:1 1 0; justify-content:center; padding:1rem 1.15rem; font-size:1.3rem !important; }
      .scivis-hero { box-sizing:border-box; width:100%; text-align:center; margin:0 auto .65rem; padding:1.35rem 1rem 1.5rem;
        border-bottom:1px solid #d9d9d9; }
      .scivis-hero h1 { display:block; margin:0; font-size:3.55rem !important; }
      [data-testid="stToolbar"] { visibility: hidden; height: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="scivis-hero"><h1>gridMET SciVis</h1></div>',
    unsafe_allow_html=True,
)


if not DATA_PATH.exists():
    st.error(f"Prepared dataset not found: {DATA_PATH}")
    st.stop()


@st.cache_resource(show_spinner="Opening prepared gridMET dataset…")
def get_dataset(path: str):
    return open_dataset(path)


@st.cache_data(show_spinner="Calculating quarter-wide color limits…")
def cached_temporal_range(path: str, variable: str, anomaly: bool, positive_only: bool = False):
    with open_dataset(path) as ds:
        field = ds[variable]
        if anomaly:
            field = field - field.mean("time", skipna=True)
        if variable in ("tmmx", "tmmn") and not anomaly:
            field = field - 273.15
        if positive_only:
            field = field.where(field > 0)
        return float(field.min(skipna=True).values), float(field.max(skipna=True).values)


@st.cache_data(show_spinner="Summarizing daily extrema across the quarter…")
def cached_daily_extrema(path: str, variable: str, anomaly: bool):
    with open_dataset(path) as ds:
        field = ds[variable]
        if variable in ("tmmx", "tmmn"):
            field = field - 273.15
        if anomaly:
            field = field - field.mean("time", skipna=True)
        spatial_dims = [dim for dim in field.dims if dim != "time"]
        daily_min = field.min(dim=spatial_dims, skipna=True).values.astype(float)
        daily_max = field.max(dim=spatial_dims, skipna=True).values.astype(float)
        return daily_min, daily_max


@st.cache_data(show_spinner="Loading selected spatial field…")
def cached_display_field(path: str, variable: str, index: int, anomaly: bool):
    with open_dataset(path) as ds:
        return display_field(ds, variable, index, anomaly=anomaly)


@st.cache_data(show_spinner="Calculating quarter-wide wind speed limit…")
def cached_wind_max(path: str):
    with open_dataset(path) as ds:
        return float(ds.vs.max(skipna=True).values)


ds = get_dataset(str(DATA_PATH))
date_labels = dates_as_strings(ds)


def date_control(key: str, default_index: int = 44) -> int:
    return st.select_slider(
        "Daily sample",
        options=list(range(len(date_labels))),
        value=default_index,
        format_func=lambda i: date_labels[i],
        key=key,
    )


def variable_control(key: str, default: str = "tmmx") -> str:
    return st.selectbox(
        "Scalar field",
        SCALAR_VARIABLES,
        index=SCALAR_VARIABLES.index(default),
        format_func=lambda value: f"{VARIABLE_LABELS.get(value, value)} ({value})",
        key=key,
    )


def _choose_palette(key: str, palette: str) -> None:
    st.session_state[key] = palette


def palette_selector(label: str, palettes: list[str], key: str) -> str:
    selected = st.session_state.get(key, palettes[0])
    if selected not in palettes:
        selected = palettes[0]
    with st.popover(f"{label}: {selected}", use_container_width=True):
        st.caption("Select a palette")
        for palette in palettes:
            color_list = [mcolors.to_hex(plt.get_cmap(palette)(x)) for x in np.linspace(0, 1, 16)]
            gradient = ", ".join(color_list)
            left, right = st.columns([1.05, 1.6], vertical_alignment="center")
            with left:
                st.button(
                    ("✓  " if palette == selected else "") + palette,
                    key=f"{key}_option_{palette}",
                    on_click=_choose_palette,
                    args=(key, palette),
                    use_container_width=True,
                )
            with right:
                st.markdown(
                    f'<div style="height:1.15rem;border:1px solid #bbb;border-radius:4px;'
                    f'background:linear-gradient(90deg,{gradient});"></div>',
                    unsafe_allow_html=True,
                )
    return selected


def contour_fill_plot(ax, field, ds, levels, cmap):
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
    filled = ax.contourf(
        lon,
        lat,
        field,
        levels=levels,
        cmap=cmap,
        extend="both",
        corner_mask=False,
    )
    return filled


def title_for(variable: str, index: int, extra: str = "") -> str:
    label = VARIABLE_LABELS.get(variable, variable)
    suffix = f" · {extra}" if extra else ""
    return f"{label} ({variable}) · {date_labels[index]}{suffix}"


def color_norm(field, norm_name: str, limits: tuple[float, float], anomaly: bool):
    if anomaly:
        bound = max(abs(limits[0]), abs(limits[1]))
        symmetric_limits = (-bound, bound)
        if norm_name == "Discrete (8 intervals)":
            return norm_for(field, norm_name, symmetric_limits)
        return mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    return norm_for(field, norm_name, limits)


def plotly_colorscale(cmap: str, discrete: bool = False, bins: int = 8):
    color_map = plt.get_cmap(cmap)
    if not discrete:
        return [[i / 31, mcolors.to_hex(color_map(i / 31))] for i in range(32)]
    stops = []
    for i in range(bins):
        left, right = i / bins, (i + 1) / bins
        color = mcolors.to_hex(color_map((i + 0.5) / bins))
        stops.extend([[left, color], [right, color]])
    return stops


def interactive_heatmap(field, ds, cmap: str, limits: tuple[float, float], norm_name: str, title: str, anomaly: bool, height: int = 900):
    stride = 2
    sampled = np.asarray(field)[::stride, ::stride]
    lon = ds.lon.values[::stride]
    lat = ds.lat.values[::stride]
    z = sampled.copy()
    zmin, zmax = limits
    colorbar = {"title": units_for(ds, variable, anomaly), "thickness": 18, "len": 0.82}
    if norm_name == "Logarithmic":
        positive = sampled[np.isfinite(sampled) & (sampled > 0)]
        if positive.size:
            z = np.where(sampled > 0, np.log10(sampled), np.nan)
            zmin = float(np.log10(max(zmin, float(positive.min()))))
            zmax = float(np.log10(max(zmax, zmin + 1e-6)))
            tickvals = np.linspace(zmin, zmax, 5)
            colorbar.update(tickvals=tickvals, ticktext=[f"{10**x:.3g}" for x in tickvals])
    colorscale = plotly_colorscale(cmap, norm_name == "Discrete (8 intervals)")
    fig = go.Figure(go.Heatmap(
        x=lon, y=lat, z=z, customdata=sampled, colorscale=colorscale,
        zmin=zmin, zmax=zmax, zsmooth=False, connectgaps=False,
        colorbar=colorbar,
        hovertemplate=("Longitude: %{x:.3f}°<br>Latitude: %{y:.3f}°N<br>"
                       f"{VARIABLE_LABELS.get(variable, variable)}: %{{customdata:.3g}} {units_for(ds, variable, anomaly)}<extra></extra>"),
    ))
    return style_geo_figure(fig, title, ds, height=height)


def style_geo_figure(fig, title: str, ds, height: int = 900):
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        height=height,
        margin={"l": 55, "r": 35, "t": 65, "b": 55},
        paper_bgcolor="white", plot_bgcolor="white",
        font={"color": "#111", "size": 13},
    )
    lat_mid = float((np.nanmin(ds.lat.values) + np.nanmax(ds.lat.values)) / 2)
    fig.update_xaxes(title="Longitude (°)", range=[float(ds.lon.min()), float(ds.lon.max())], showgrid=True, gridcolor="#e6e6e6", zeroline=False)
    fig.update_yaxes(title="Latitude (°N)", range=[float(ds.lat.min()), float(ds.lat.max())], showgrid=True, gridcolor="#e6e6e6", zeroline=False, scaleanchor="x", scaleratio=float(1 / np.cos(np.deg2rad(lat_mid))), constrain="domain")
    return fig


def interactive_contour(field, ds, levels, cmap: str, title: str, mode: str, height: int = 900):
    stride = 2
    sampled = np.asarray(field)[::stride, ::stride]
    low, high = float(levels[0]), float(levels[-1])
    if mode == "Marching Squares":
        fig = go.Figure()
        color_map = plt.get_cmap(cmap)
        lon_index = np.arange(ds.sizes["lon"], dtype=float)
        lat_index = np.arange(ds.sizes["lat"], dtype=float)
        fig.add_trace(go.Heatmap(
            x=ds.lon.values[::stride], y=ds.lat.values[::stride], z=sampled,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)" ]],
            zmin=low, zmax=high, showscale=False, hoverongaps=False,
            hovertemplate=("Longitude: %{x:.3f}°<br>Latitude: %{y:.3f}°N<br>"
                           f"{VARIABLE_LABELS.get(variable, variable)}: %{{z:.3g}} {units_for(ds, variable)}<extra></extra>"),
        ))
        for level in levels[1:-1]:
            x_values, y_values, level_values = [], [], []
            for contour in find_contours(field, level=float(level), fully_connected="low", positive_orientation="low", mask=np.isfinite(field)):
                if len(contour) < 2:
                    continue
                xs = np.interp(contour[:, 1], lon_index, ds.lon.values)
                ys = np.interp(contour[:, 0], lat_index, ds.lat.values)
                x_values.extend(xs.tolist() + [None])
                y_values.extend(ys.tolist() + [None])
                level_values.extend([float(level)] * len(xs) + [None])
            if x_values:
                fig.add_trace(go.Scattergl(
                    x=x_values, y=y_values, mode="lines",
                    line={"color": mcolors.to_hex(color_map((float(level) - low) / (high - low or 1))), "width": 1},
                    customdata=level_values,
                    hovertemplate=("Longitude: %{x:.3f}°<br>Latitude: %{y:.3f}°N<br>"
                                   f"Isoline value: %{{customdata:.3g}} {units_for(ds, variable)}<extra></extra>"),
                    showlegend=False,
                ))
        fig.add_trace(go.Scatter(
            x=[None] * len(levels[1:-1]), y=[None] * len(levels[1:-1]), mode="markers",
            marker={"size": 0, "color": levels[1:-1], "colorscale": plotly_colorscale(cmap),
                    "cmin": low, "cmax": high, "showscale": True,
                    "colorbar": {"title": f"Isoline ({units_for(ds, variable)})", "thickness": 18, "len": 0.82}},
            hoverinfo="skip", showlegend=False,
        ))
        return style_geo_figure(fig, title, ds, height=height)
    color_mode = "lines" if mode == "Marching Squares" else "fill"
    fig = go.Figure(go.Contour(
        x=ds.lon.values[::stride], y=ds.lat.values[::stride], z=sampled,
        colorscale=plotly_colorscale(cmap), zmin=low, zmax=high,
        contours={"start": low, "end": high, "size": (high - low) / max(len(levels) - 1, 1), "coloring": color_mode, "showlines": True},
        line={"width": 1.1}, connectgaps=False,
        colorbar={"title": units_for(ds, variable), "thickness": 18, "len": 0.82},
        hovertemplate=("Longitude: %{x:.3f}°<br>Latitude: %{y:.3f}°N<br>"
                       f"{VARIABLE_LABELS.get(variable, variable)}: %{{z:.3g}} {units_for(ds, variable)}<extra></extra>"),
    ))
    return style_geo_figure(fig, title, ds, height=height)


def interactive_quiver(ds, index: int, stride: int, magnitude_scaled: bool, speed_max: float):
    speed = ds.vs.isel(time=index).values.astype(np.float32, copy=False)
    direction = ds.th.isel(time=index).values.astype(np.float32, copy=False)
    radians = np.deg2rad(direction)
    u, v = -speed * np.sin(radians), -speed * np.cos(radians)
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
    sel = np.s_[::stride, ::stride]
    sx, sy, su, sv, ss, sd = (arr[sel] for arr in (lon, lat, u, v, speed, direction))
    valid = np.isfinite(ss) & np.isfinite(sd)
    sx, sy, su, sv, ss, sd = (arr[valid] for arr in (sx, sy, su, sv, ss, sd))
    angle_lat_scale = np.cos(np.deg2rad(sy))
    if magnitude_scaled:
        length = np.clip(ss / max(speed_max, 1e-6), 0.15, 1.0) * 1.15
    else:
        length = np.full_like(ss, 0.75, dtype=float)
    norm = np.hypot(su, sv)
    safe = np.where(norm > 0, norm, 1)
    dx = (su / safe) * length / angle_lat_scale
    dy = (sv / safe) * length
    ex, ey = sx + dx, sy + dy
    xs, ys = [], []
    for x0, y0, x1, y1 in zip(sx, sy, ex, ey):
        xs.extend([x0, x1, None]); ys.extend([y0, y1, None])
    colorscale = plotly_colorscale("viridis")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line={"color": "#333", "width": 1.1}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattergl(
        x=sx, y=sy, mode="markers",
        marker={"size": 9, "color": ss, "colorscale": colorscale, "cmin": 0, "cmax": speed_max,
                "colorbar": {"title": "Wind speed (m/s)", "thickness": 18, "len": 0.82}},
        customdata=np.column_stack([ss, sd, su, sv]),
        hovertemplate=("Longitude: %{x:.3f}°<br>Latitude: %{y:.3f}°N<br>"
                       "Wind speed: %{customdata[0]:.2f} m/s<br>Wind-from direction: %{customdata[1]:.1f}°"
                       "<br>Eastward component: %{customdata[2]:.2f} m/s<br>Northward component: %{customdata[3]:.2f} m/s<extra></extra>"),
        showlegend=False,
    ))
    annotations = []
    for x0, y0, x1, y1 in zip(sx, sy, ex, ey):
        annotations.append({"x": float(x1), "y": float(y1), "ax": float(x0), "ay": float(y0),
                            "xref": "x", "yref": "y", "axref": "x", "ayref": "y", "showarrow": True,
                            "arrowhead": 2, "arrowsize": 0.8, "arrowwidth": 1, "arrowcolor": "#333"})
    fig.update_layout(annotations=annotations)
    return style_geo_figure(fig, f"Wind vectors · {date_labels[index]} · stride {stride}", ds, height=900)


color_tab, contour_tab, quiver_tab = st.tabs(
    ["1 · Color mapping", "2 · Contour mapping", "3 · Quiver plot"]
)


with color_tab:
    st.subheader("Color mapping experiments")
    st.markdown(
        "Explore palette, normalization, and temporal scaling choices. Quarter-global limits "
        "make values comparable across dates; local limits reveal each day’s spatial contrast."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        variable = variable_control("color_variable", "tmmx")
    with c2:
        index = date_control("color_date")
    with c3:
        value_mode = st.radio(
            "Mapped value",
            ["Field value", "Anomaly from quarter mean"],
            horizontal=False,
            key="color_mode",
        )
    with c4:
        anomaly = value_mode.startswith("Anomaly")
        scale_options = ["Continuous", "Discrete (8 intervals)"]
        if not anomaly:
            scale_options.append("Logarithmic")
        scaling = st.selectbox(
            "Normalization",
            scale_options,
            key="color_norm",
        )

    anomaly = value_mode.startswith("Anomaly")

    p1, p2, p3 = st.columns(3)
    with p1:
        scale_mode = st.selectbox(
            "Temporal limits",
            ["Compare quarter-global and per-day", "Quarter-global", "Per-day local"],
            key="color_scale_mode",
        )
    with p2:
        palette_options = (
            ["RdBu_r", "PuOr", "BrBG", "coolwarm", "viridis"]
            if anomaly
            else ["viridis", "cividis", "plasma", "YlOrRd", "Blues", "RdBu_r"]
        )
        cmap = palette_selector("Palette", palette_options, "color_palette")
    with p3:
        st.caption("Temperature fields are shown in °C. Precipitation is in mm/day.")

    field = cached_display_field(str(DATA_PATH), variable, index, anomaly)
    global_limits = cached_temporal_range(str(DATA_PATH), variable, anomaly, scaling == "Logarithmic")
    local_limits = frame_range(field)
    norm_name = scaling if scaling != "Continuous" else "Continuous"

    daily_minima, daily_maxima = cached_daily_extrema(str(DATA_PATH), variable, anomaly)
    with st.expander("How parameter-to-color mapping changes across time", expanded=True):
        st.markdown(
            f"**Quarter-global domain:** {global_limits[0]:.3g} to {global_limits[1]:.3g} "
            f"{units_for(ds, variable, anomaly)}. Every daily map uses this same domain, so the "
            "same color represents the same value on every date. **Selected-day domain "
            f"({date_labels[index]}):** {local_limits[0]:.3g} to {local_limits[1]:.3g}. "
            "Per-day scaling remaps that day’s minimum and maximum to the palette endpoints, "
            "which improves local contrast but can make two days with different absolute values "
            "look similar."
        )
        trend_fig, trend_ax = plt.subplots(figsize=(11, 2.8), constrained_layout=True)
        day_numbers = np.arange(len(date_labels))
        trend_ax.fill_between(day_numbers, daily_minima, daily_maxima, color="#d8d8d8", alpha=0.7, label="Daily min–max")
        trend_ax.plot(day_numbers, daily_minima, color="#222", linewidth=0.8, label="Daily minimum")
        trend_ax.plot(day_numbers, daily_maxima, color="#777", linewidth=0.8, label="Daily maximum")
        trend_ax.axhline(global_limits[0], color="#111", linestyle=":", linewidth=1, label="Quarter-global min/max")
        trend_ax.axhline(global_limits[1], color="#111", linestyle=":", linewidth=1)
        trend_ax.scatter([index, index], [daily_minima[index], daily_maxima[index]], color="#d62728", zorder=4)
        trend_ax.set_xticks([0, 15, 30, 45, 60, 75, 89], [date_labels[i][5:] for i in [0, 15, 30, 45, 60, 75, 89]])
        trend_ax.set_xlabel("Date in 2022")
        trend_ax.set_ylabel(units_for(ds, variable, anomaly))
        trend_ax.set_title("Daily spatial minima and maxima across all 90 timesteps")
        trend_ax.tick_params(axis="both")
        trend_ax.grid(axis="y", color="#ddd", linewidth=0.6)
        trend_ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
        st.pyplot(trend_fig, width="stretch")
        plt.close(trend_fig)

    if scale_mode == "Compare quarter-global and per-day":
        col_a, col_b = st.columns(2)
        for col, limits, heading in (
            (col_a, global_limits, "Quarter-global limits"),
            (col_b, local_limits, "Per-day limits"),
        ):
            fig = interactive_heatmap(
                field, ds, cmap, limits, norm_name,
                title_for(variable, index, f"{heading} · {value_mode}"), anomaly, height=780,
            )
            col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        limits = global_limits if scale_mode == "Quarter-global" else local_limits
        fig = interactive_heatmap(field, ds, cmap, limits, norm_name, title_for(variable, index, f"{scaling} · {scale_mode}"), anomaly)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    animation_path = Path(__file__).resolve().parent / "images/scivis/fig5_precipitation_7sample_animation.gif"
    if animation_path.exists():
        st.image(str(animation_path), caption="Seven evenly spaced precipitation samples across the quarter.", width="stretch")


with contour_tab:
    st.subheader("Contour mapping")
    st.markdown(
        "Compare two ways to represent the same scalar field: Marching Squares extracts threshold "
        "isolines, while contour fill colors the intervals between thresholds."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        variable = variable_control("contour_variable", "tmmx")
    with c2:
        index = date_control("contour_date")
    with c3:
        n_levels = st.slider("Fill intervals / threshold count", min_value=5, max_value=18, value=10, key="contour_levels")
    with c4:
        cmap = palette_selector("Palette", ["YlOrRd", "viridis", "cividis", "coolwarm", "Blues"], "contour_palette")
    algorithm = st.radio(
        "Contour algorithm",
        ["Marching Squares", "Contour fill", "Compare both"],
        horizontal=True,
        key="contour_algorithm",
    )
    field = cached_display_field(str(DATA_PATH), variable, index, False)
    low, high = frame_range(field)
    levels = np.linspace(low, high, n_levels + 1)
    if algorithm == "Compare both":
        stats_fig, stats_ax = plt.subplots()
        _, marching_stats = marching_squares_plot(stats_ax, field, ds, levels, cmap)
        plt.close(stats_fig)
        col_a, col_b = st.columns(2)
        col_a.plotly_chart(interactive_contour(field, ds, levels, cmap, title_for(variable, index, f"Marching Squares · {len(levels) - 2} thresholds"), "Marching Squares", height=780), width="stretch", config={"displayModeBar": False})
        col_b.plotly_chart(interactive_contour(field, ds, levels, cmap, title_for(variable, index, f"Contour fill · {n_levels} bands"), "Contour fill", height=780), width="stretch", config={"displayModeBar": False})
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Marching Squares line segments", f"{marching_stats['segments']:,}")
            st.metric("Vertices across isolines", f"{marching_stats['vertices']:,}")
        with col_b:
            st.metric("Approximate total isoline length", f"{marching_stats['length_km']:,.0f} km")
            st.metric("Filled value bands", n_levels)
        st.info(
            f"Both panels use the same {n_levels} intervals on {date_labels[index]}. Marching Squares "
            f"returns {marching_stats['segments']:,} line segments at {marching_stats['levels']} "
            "internal thresholds; it exposes boundary shape but leaves the regions unfilled. "
            "Contour fill shows the spatial extent of each value band at a glance, but values "
            "within a band share a color. For this temperature map, use filled contours for the "
            "regional overview and Marching Squares when the boundary locations are the focus."
        )
    elif algorithm == "Marching Squares":
        stats_fig, stats_ax = plt.subplots()
        _, marching_stats = marching_squares_plot(stats_ax, field, ds, levels, cmap)
        plt.close(stats_fig)
        fig = interactive_contour(field, ds, levels, cmap, title_for(variable, index, f"Marching Squares · {marching_stats['levels']} thresholds"), "Marching Squares")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        m1, m2, m3 = st.columns(3)
        m1.metric("Line segments", f"{marching_stats['segments']:,}")
        m2.metric("Vertices", f"{marching_stats['vertices']:,}")
        m3.metric("Approximate total length", f"{marching_stats['length_km']:,.0f} km")
        st.info(
            "Marching Squares linearly interpolates where each chosen threshold crosses a grid "
            "cell edge, then connects those crossings into isolines. The line map preserves "
            "boundary detail without assigning filled colors to the regions between lines."
        )
    else:
        fig = interactive_contour(field, ds, levels, cmap, title_for(variable, index, f"Contour fill · {n_levels} value bands"), "Contour fill")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.info(
            "Contour fill divides the value range into bands and colors the area in each band. "
            "It is usually easier to read for a regional overview; more intervals preserve finer "
            "variation but create more visual boundaries."
        )


with quiver_tab:
    st.subheader("Wind vector field")
    st.markdown(
        "Wind speed (`vs`) and wind-from direction (`th`) are converted to eastward and northward "
        "components. The direction conversion reverses the meteorological bearing so arrows point "
        "where the wind travels."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        index = date_control("quiver_date")
    with c2:
        stride = st.slider("Grid sampling stride", min_value=8, max_value=48, value=32, step=4, key="quiver_stride")
    with c3:
        arrow_mode = st.radio(
            "Arrow length",
            ["Magnitude-proportional", "Same-sized (direction only)"],
            key="quiver_mode",
        )
    magnitude_scaled = arrow_mode == "Magnitude-proportional"
    speed_max = cached_wind_max(str(DATA_PATH))
    fig = interactive_quiver(ds, index, stride, magnitude_scaled, speed_max)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        f"Quarter maximum wind speed: {speed_max:.1f} m/s. Compare sparse and dense sampling, "
        "then switch arrow length to separate direction from magnitude."
    )


