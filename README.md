# gridMET SciVis

An interactive Streamlit dashboard for three SciVis tasks using the gridMET CONUS daily weather dataset: color mapping, contour mapping, and wind vector (quiver) visualization.

## Requirements

- Python 3.10 or newer
- The prepared NetCDF file `gridmet_2022-01_to_2022-03.nc` (90 daily records from 2022-01-01 through 2022-03-31)

The NetCDF file is intentionally not stored in this Git repository because it is large (about 438 MB). Obtain the prepared file from your project files and place it at:

```text
data/processed/gridmet_2022-01_to_2022-03.nc
```

## Setup and run

Clone the repository and enter its folder:

```bash
git clone https://github.com/Swayam2932/gridMET_SciVis.git
cd gridMET_SciVis
```

Create the data folder and copy the NetCDF file into it. On macOS or Linux, if the file is in your Downloads folder, run:

```bash
mkdir -p data/processed
cp ~/Downloads/gridmet_2022-01_to_2022-03.nc data/processed/
```

On Windows PowerShell, create the folder and copy the file from its current location, for example:

```powershell
New-Item -ItemType Directory -Force data/processed
Copy-Item "$HOME\Downloads\gridmet_2022-01_to_2022-03.nc" "data/processed/"
```

Create and activate a virtual environment, then install all Python requirements:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually [http://localhost:8501](http://localhost:8501). Keep the terminal running while you use the dashboard; press `Ctrl+C` there to stop it.

## Dashboard tabs

1. **Color mapping** — inspect daily scalar fields, palettes, normalization, quarter-global versus daily limits, and anomalies. Hover the map to see coordinates and the field value.
2. **Contour mapping** — view Marching Squares isolines, filled contours, or a comparison of both. Hover to inspect locations and values.
3. **Quiver plot** — explore wind speed and direction with adjustable arrow sampling and scaling. Hover a vector for wind speed, wind-from direction, and eastward/northward components.

If the prepared NetCDF file is missing, the dashboard displays the expected path. To recreate the quarter dataset instead, install the requirements and run `python process_gridmet.py`; that optional script downloads the 2022 source variables from the official gridMET archive and writes the prepared file to `data/processed/`.

## Project files

- `app.py` — Streamlit dashboard.
- `scivis_utils.py` — data loading, field conversion, plotting, and vector helpers.
- `requirements.txt` — Python dependencies for running the dashboard.
- `process_gridmet.py` — optional source-data download and preprocessing script.
- `generate_scivis_artifacts.py` — optional script for generating static assignment figures and the GIF; it requires the prepared NetCDF file.
- `images/scivis/fig5_precipitation_7sample_animation.gif` — animation shown in the dashboard.
