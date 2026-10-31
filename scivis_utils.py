"""Shared data and plotting helpers for the gridMET SciVis dashboard."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache/matplotlib"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import xarray as xr
from skimage.measure import find_contours


DATA_PATH = PROJECT_DIR / "data/processed/gridmet_2022-01_to_2022-03.nc"
GIF_SAMPLE_DAYS = np.array([0, 15, 30, 45, 60, 75, 89], dtype=int)
SCALAR_VARIABLES = [
    "tmmx", "tmmn", "pr", "srad", "vpd", "sph", "etr", "pet",
    "bi", "erc", "fm1", "fm10", "fm100", "fm1000", "rmin", "rmax",
]
VARIABLE_LABELS = {
    "tmmx": "Maximum temperature",
    "tmmn": "Minimum temperature",
    "pr": "Daily precipitation",
    "srad": "Downward shortwave radiation",
    "vpd": "Vapor pressure deficit",
    "sph": "Specific humidity",
    "etr": "Reference evapotranspiration (alfalfa)",
    "pet": "Reference evapotranspiration (short grass)",
    "bi": "Burning index",
    "erc": "Energy release component",
    "fm1": "1-hour dead fuel moisture",
    "fm10": "10-hour dead fuel moisture",
    "fm100": "100-hour dead fuel moisture",
    "fm1000": "1000-hour dead fuel moisture",
    "rmin": "Minimum relative humidity",
    "rmax": "Maximum relative humidity",
}


def open_dataset(path: Path | str = DATA_PATH) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4", decode_times=True, cache=False)


def units_for(ds: xr.Dataset, variable: str, anomaly: bool = False) -> str:
    if anomaly:
        return f"{units_for(ds, variable)} anomaly"
    if variable in ("tmmx", "tmmn"):
        return "°C"
    return str(ds[variable].attrs.get("units", "value"))


def display_field(ds: xr.Dataset, variable: str, index: int, anomaly: bool = False) -> np.ndarray:
    field = ds[variable].isel(time=index).values.astype(np.float32, copy=False)
    if variable in ("tmmx", "tmmn"):
        field = field - np.float32(273.15)
    if anomaly:
        mean = ds[variable].mean("time", skipna=True).values.astype(np.float32, copy=False)
        if variable in ("tmmx", "tmmn"):
            mean = mean - np.float32(273.15)
        field = field - mean
    return field


def temporal_range(ds: xr.Dataset, variable: str, anomaly: bool = False) -> tuple[float, float]:
    source = ds[variable]
    if anomaly:
        source = source - source.mean("time", skipna=True)
    if variable in ("tmmx", "tmmn") and not anomaly:
        source = source - 273.15
    low = float(source.min(skipna=True).values)
    high = float(source.max(skipna=True).values)
    return low, high


def frame_range(field: np.ndarray) -> tuple[float, float]:
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def dates_as_strings(ds: xr.Dataset) -> list[str]:
    return [np.datetime_as_string(value, unit="D") for value in ds.time.values]


def map_axes(ax: plt.Axes, ds: xr.Dataset, title: str | None = None) -> None:
    lon = ds.lon.values
    lat = ds.lat.values
    lat_mid = float((np.nanmin(lat) + np.nanmax(lat)) / 2)
    ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.set_aspect(1 / np.cos(np.deg2rad(lat_mid)), adjustable="box")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_xticks(np.arange(-125, -65, 10))
    ax.set_yticks(np.arange(25, 55, 5))
    ax.grid(color="white", linewidth=0.45, alpha=0.55)
    if title:
        ax.set_title(title)


def norm_for(
    field: np.ndarray,
    method: str,
    scale: tuple[float, float] | None = None,
    bins: int = 8,
) -> mcolors.Normalize:
    low, high = scale or frame_range(field)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        high = low + 1.0
    if method == "Logarithmic":
        positive = field[np.isfinite(field) & (field > 0)]
        if positive.size == 0:
            raise ValueError("Logarithmic scaling needs positive data values.")
        if scale is None or low <= 0:
            low = float(positive.min())
        if high <= low:
            high = float(positive.max())
        return mcolors.LogNorm(vmin=low, vmax=high, clip=True)
    if method == "Discrete (8 intervals)":
        levels = np.linspace(low, high, bins + 1)
        return mcolors.BoundaryNorm(levels, ncolors=256, clip=True)
    return mcolors.Normalize(vmin=low, vmax=high, clip=True)


def scalar_mesh(
    ax: plt.Axes,
    ds: xr.Dataset,
    field: np.ndarray,
    cmap: str,
    norm: mcolors.Normalize,
):
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
    masked = np.ma.masked_invalid(field)
    if isinstance(norm, mcolors.LogNorm):
        masked = np.ma.masked_less_equal(masked, 0)
    image = ax.pcolormesh(
        lon,
        lat,
        masked,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    if isinstance(norm, mcolors.LogNorm):
        zeros = np.ma.masked_where(~(np.isfinite(field) & (field == 0)), np.ones_like(field))
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
    return image


def make_quiver(
    ax: plt.Axes,
    ds: xr.Dataset,
    index: int,
    stride: int,
    magnitude_scaled: bool,
    cmap: str = "viridis",
    speed_vmax: float = 15.0,
):
    speed = ds.vs.isel(time=index).values.astype(np.float32, copy=False)
    direction = ds.th.isel(time=index).values.astype(np.float32, copy=False)
    radians = np.deg2rad(direction)
    # gridMET th is a meteorological "from" bearing (clockwise from north).
    u_east = -speed * np.sin(radians)
    v_north = -speed * np.cos(radians)
    magnitude = np.hypot(u_east, v_north)
    valid = np.isfinite(u_east) & np.isfinite(v_north) & np.isfinite(speed)
    if magnitude_scaled:
        u_draw, v_draw = u_east, v_north
        scale = 60
    else:
        safe = np.where(magnitude > 0, magnitude, 1)
        u_draw, v_draw = u_east / safe, v_north / safe
        scale = 4
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)
    q = ax.quiver(
        lon[::stride, ::stride][valid[::stride, ::stride]],
        lat[::stride, ::stride][valid[::stride, ::stride]],
        u_draw[::stride, ::stride][valid[::stride, ::stride]],
        v_draw[::stride, ::stride][valid[::stride, ::stride]],
        speed[::stride, ::stride][valid[::stride, ::stride]],
        cmap=cmap,
        norm=mcolors.Normalize(vmin=0, vmax=max(speed_vmax, 1.0)),
        angles="uv",
        scale_units="inches",
        scale=scale,
        units="inches",
        width=0.006,
        headwidth=3.4,
        headlength=4.2,
        headaxislength=3.8,
        pivot="middle",
    )
    return q


def marching_squares_plot(ax, field, ds, levels, cmap):
    """Extract colored Marching Squares isolines and return their summary metrics."""
    valid = np.isfinite(field)
    segment_points = []
    segment_values = []
    total_length_km = 0.0
    lon_index = np.arange(ds.sizes["lon"], dtype=float)
    lat_index = np.arange(ds.sizes["lat"], dtype=float)
    lon_values = ds.lon.values
    lat_values = ds.lat.values

    for level in levels[1:-1]:
        contours = find_contours(
            field,
            level=float(level),
            fully_connected="low",
            positive_orientation="low",
            mask=valid,
        )
        for contour in contours:
            if len(contour) < 2:
                continue
            contour_lon = np.interp(contour[:, 1], lon_index, lon_values)
            contour_lat = np.interp(contour[:, 0], lat_index, lat_values)
            points = np.column_stack((contour_lon, contour_lat))
            segment_points.append(points)
            segment_values.append(float(level))
            mean_lat = np.deg2rad(np.nanmean(contour_lat))
            dx = np.diff(contour_lon) * 111.32 * np.cos(mean_lat)
            dy = np.diff(contour_lat) * 111.32
            total_length_km += float(np.nansum(np.hypot(dx, dy)))

    line_norm = mcolors.Normalize(vmin=float(levels[1]), vmax=float(levels[-2]))
    lines = LineCollection(segment_points, cmap=cmap, norm=line_norm, linewidths=0.65)
    lines.set_array(np.asarray(segment_values, dtype=np.float32))
    ax.add_collection(lines)
    return lines, {
        "segments": len(segment_points),
        "vertices": sum(len(points) for points in segment_points),
        "length_km": total_length_km,
        "levels": len(levels) - 2,
    }
