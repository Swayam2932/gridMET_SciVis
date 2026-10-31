#!/usr/bin/env python3
"""Download all annual 2022 gridMET NetCDFs and combine Jan-Mar slices."""

from __future__ import annotations

import concurrent.futures
import re
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import xarray as xr


BASE_URL = "https://www.northwestknowledge.net/metdata/data/"
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "gridmet_2022"
OUTPUT = ROOT / "data" / "processed" / "gridmet_2022-01_to_2022-03.nc"


def get_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "gridMET-assignment-data-prep/1.0"})
    with urlopen(req, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def annual_files() -> list[str]:
    listing = get_text(BASE_URL)
    names = sorted(set(re.findall(r"\b([a-z0-9]+_2022\.nc)\b", listing)))
    if not names:
        raise RuntimeError("Could not find 2022 variable files in the official archive listing")
    return names


def download_one(name: str) -> tuple[str, int]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / name
    if target.exists() and target.stat().st_size > 0:
        return name, target.stat().st_size
    temp = target.with_suffix(target.suffix + ".part")
    url = BASE_URL + name
    for attempt in range(4):
        try:
            req = Request(url, headers={"User-Agent": "gridMET-assignment-data-prep/1.0"})
            with urlopen(req, timeout=180) as response, temp.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if temp.stat().st_size == 0:
                raise RuntimeError(f"Downloaded an empty file: {name}")
            temp.replace(target)
            return name, target.stat().st_size
        except (URLError, TimeoutError, OSError) as exc:
            temp.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Failed to download {name}: {exc}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def build_dataset(names: list[str]) -> None:
    datasets = []
    variable_encodings = {}
    try:
        for name in names:
            path = RAW_DIR / name
            ds = xr.open_dataset(path, engine="netcdf4")
            abbreviation = name.removesuffix("_2022.nc")
            if "day" in ds.dims and "time" not in ds.dims:
                ds = ds.rename({"day": "time"})
            if "time" not in ds.dims:
                ds.close()
                raise ValueError(f"No daily time dimension in {name}")
            ds = ds.sel(time=slice("2022-01-01", "2022-03-31"))
            if ds.sizes.get("time") != 90:
                ds.close()
                raise ValueError(f"Expected 90 Jan-Mar 2022 days in {name}, got {ds.sizes.get('time')}")
            primary = [
                var for var in ds.data_vars
                if ds[var].attrs.get("standard_name") == abbreviation
                or ds[var].attrs.get("long_name") == abbreviation
            ]
            if len(primary) != 1:
                ds.close()
                raise ValueError(f"Could not identify the {abbreviation} measurement in {name}")
            field = ds[[primary[0]]].rename({primary[0]: abbreviation})
            source_encoding = field[abbreviation].encoding
            # Keep the source storage type and packing so the merged file retains
            # gridMET's native precision without inflating packed fields to float64.
            variable_encodings[abbreviation] = {
                key: source_encoding[key]
                for key in ("dtype", "_FillValue", "scale_factor", "add_offset", "chunksizes")
                if key in source_encoding
            }
            variable_encodings[abbreviation].update(
                {"zlib": True, "complevel": 4, "shuffle": True}
            )
            # Retain the CRS metadata variable used by the source fields' grid_mapping.
            if "crs" in ds.variables:
                field = field.assign_coords(crs=ds["crs"])
            datasets.append(field)
            ds.close()

        combined = xr.merge(datasets, compat="no_conflicts", join="exact")
        for var in combined.data_vars:
            if "dimensions" in combined[var].attrs:
                combined[var].attrs["dimensions"] = "time lat lon"
        combined.attrs["source"] = "gridMET official direct NetCDF archive"
        combined.attrs["source_url"] = BASE_URL
        combined.attrs["processing"] = (
            "All annual gridMET variable files available in the archive for 2022 were sliced "
            "to daily dates 2022-01-01 through 2022-03-31 inclusive and merged by coordinates. "
            "Original variable values and scale/offset decoding were preserved by xarray."
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_netcdf(
            OUTPUT, engine="netcdf4", format="NETCDF4", encoding=variable_encodings
        )
        combined.close()
    finally:
        for ds in datasets:
            ds.close()

    with xr.open_dataset(OUTPUT, engine="netcdf4") as check:
        if check.sizes.get("time") != 90:
            raise RuntimeError("Combined output validation failed: wrong time dimension")
        if len(check.data_vars) != len(names):
            raise RuntimeError(
                f"Combined output validation failed: expected {len(names)} variables, "
                f"found {len(check.data_vars)}"
            )
        print(f"Output: {OUTPUT}")
        print(f"Dates: {str(check.time.values[0])[:10]} through {str(check.time.values[-1])[:10]} (90 days)")
        print(f"Dimensions: {dict(check.sizes)}")
        print(f"Variables ({len(check.data_vars)}): {', '.join(check.data_vars)}")
        print(f"Output size: {OUTPUT.stat().st_size:,} bytes")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = annual_files()
    print(f"Found {len(names)} official 2022 variable files: {', '.join(names)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download_one, name) for name in names]
        for future in concurrent.futures.as_completed(futures):
            name, size = future.result()
            print(f"Ready: {name} ({size:,} bytes)", flush=True)
    build_dataset(names)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
