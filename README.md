# Sentinel-1 LULC

Monthly land use / land cover (LULC) maps from Sentinel-1 radar only.

You give a location (latitude, longitude) and a start month. The prototype downloads
12 monthly Sentinel-1 VV/VH composites from Google Earth Engine, runs a U-Net trained
on ESA WorldCover labels, and returns:

- 12 monthly LULC maps (GeoTIFF, figure and GIF)
- the share of the area covered by each class for every month (%)
- the month-to-month change of each class in percentage points (pp)
- the total change from the first to the last month in percentage points

Classes: tree cover, grassland, cropland, built-up, bare/sparse vegetation, permanent water.

## Deliverables

| Item | Location |
|---|---|
| Dataset | **[download link](ADD_DATASET_LINK)** (see [Dataset](#dataset)) |
| Data collection notebook | `notebooks/01_data_collection.ipynb` |
| Training notebook (results and training curves) | `notebooks/02_train_evaluate_unet.ipynb` |
| Prototype notebook (full run, two cities) | `notebooks/03_prototype_run.ipynb` |
| Prototype source code | `app.py`, `run_lulc.py`, `src/` |
| Trained model weights | `weights/best_unet_v5.pt` |
| Screen recording (Stuttgart, full run) | **[video link](ADD_VIDEO_LINK)** |

---

## Setup

Python 3.10 to 3.12.

```bash
git clone https://github.com/Mehatab1/sentinel1-lulc.git
cd sentinel1-lulc
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is not needed for the prototype. Predicting a 2 km × 2 km area for 12 months
takes a few seconds on CPU; most of the run time is the Earth Engine download.

### Required accounts and keys

Only one account is needed: **Google Earth Engine** with a Google Cloud project.
No other API keys are required. Sentinel-1 and WorldCover are both read through Earth Engine.

1. Sign up at <https://earthengine.google.com> and register a Cloud project for
   Earth Engine (free for non-commercial use). Note the project ID.
2. Log in once on the machine that runs the code:
   ```bash
   earthengine authenticate
   ```
   On a server or JupyterHub without a browser use
   `python -c "import ee; ee.Authenticate(auth_mode='notebook')"`,
   open the printed link, and paste the code back.
3. Set the project ID:
   ```bash
   export EE_PROJECT=your-gcp-project-id
   ```
   (or copy `.env.example` to `.env`; the app also has a field for it)

For headless use you can use a service account instead: give it the
*Earth Engine Resource Viewer* role, download a JSON key, and set `EE_SERVICE_ACCOUNT`
and `GOOGLE_APPLICATION_CREDENTIALS` (see `.env.example`).

### Model weights

`weights/best_unet_v5.pt` (about 8 MB) is included in the repository. It contains the
model weights, the architecture settings and the VV/VH normalisation values, and is
loaded by `src/model.py`. Notebook 02 writes a new copy after retraining.

---

## Running the prototype

### Web app

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

- Choose the location by **clicking on the map**, picking a preset city, or typing coordinates.
- Choose the start month (2017-01 or later) and the area size (1–5 km).
- Press **Run analysis**.

The results are shown in four tabs: monthly maps, class share per month, change (pp),
and a download of all outputs as a zip.

When running on a remote server (e.g. JupyterHub), the port has to be forwarded.
One option is a Cloudflare quick tunnel:

```bash
nohup streamlit run app.py --server.port 8501 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false > streamlit.log 2>&1 &
cloudflared tunnel --url http://localhost:8501 --protocol http2
```

and open the printed `https://....trycloudflare.com` link.

### Command line

```bash
python run_lulc.py --lat 48.7758 --lon 9.1829 --start 2025-10 --area-km 2
```

### Notebook

`notebooks/03_prototype_run.ipynb` runs the same pipeline for Stuttgart and Frankfurt
and shows all tables and figures.

### Output files

Each run is saved in `runs/<lat>_<lon>_<start>_<size>km/`:

```
inputs/        s1_YYYY-MM.tif      monthly Sentinel-1 VV/VH median composites (dB)
predictions/   lulc_YYYY-MM.tif    predicted class ids 1-6, same grid
tables/        class_percentages_by_month.csv
               month_to_month_change_pp.csv
               total_change_first_to_last_pp.csv
               transition_matrix_first_to_last_pixels.csv
figures/       lulc_12_months.png / .gif, class_composition.png,
               month_to_month_change_pp.png, total_change_pp.png, change_maps.png
run_metadata.json
```

Downloaded inputs are reused if the same location and period is run again. Every input
image (new or reused) is checked before use: it must open as a 2-band GeoTIFF with the
expected CRS, 10 m pixels and the same top-left corner as the analysis grid. A broken
cached file is downloaded again; a bad download stops the run with a clear message.

### Processing details

- **Orbit choice.** All 12 months should come from the same viewing geometry, otherwise
  part of the month-to-month change would come from the radar angle instead of the land.
  The order is: one descending relative orbit present in every month → one ascending
  relative orbit → mixed descending orbits → mixed ascending orbits. The choice is
  shown in the app and saved in `run_metadata.json`.
- **Missing pixels.** NaN and ±infinity are treated as missing and filled with the band
  median. If more than half of the image is missing, that month is not predicted.
- **Tiling.** The model works on 128 × 128 tiles. Tiles overlap by 64 pixels and the
  class probabilities are averaged, which avoids visible seams every 1.28 km.
  (`TILE_STRIDE` in `src/inference.py`; 128 gives non-overlapping tiles.)

### How the numbers are calculated

For month *m* and class *c*:

- share(*m*, *c*) = pixels of class *c* / all pixels × 100
- month-to-month change = share(*m*, *c*) − share(*m*−1, *c*), in percentage points
- total change = share(last month, *c*) − share(first month, *c*), in percentage points

The changes of all classes add up to zero for each period. The column
`Pixels changed (%)` is the share of pixels whose class differs from the previous
month (gross change). It can be large even when the net pp changes are small.

---

## Dataset

**Download:** [ADD_DATASET_LINK](ADD_DATASET_LINK) — size and sha256 are printed by
`python scripts/package_dataset.py`.

| File | Contents |
|---|---|
| `patches_v5/train_v5.npz` | 8,280 samples (690 locations × 12 months) |
| `patches_v4/val_v4.npz` | spatially held-out validation locations × 12 months |
| `patches_v4/test_v4.npz` | 324 samples (27 locations × 12 months) |
| `monthly_inputs/` | 12 Sentinel-1 composites of the 20 km Stuttgart area (2001 × 2001 px) |

Each `.npz` file contains `X` (VV and VH, normalised, shape N × 2 × 128 × 128),
`y` (class ids 1–6, 255 = ignore), `class_medians`, `class_iqr`, and `metadata`
with the month of each sample.

- **Input:** `COPERNICUS/S1_GRD`, IW mode, VV + VH in dB, monthly median of one
  descending relative orbit, 10 m, UTM zone 32N, October 2025 – September 2026.
- **Labels:** `ESA/WorldCover/v200` (2021): 10 → tree cover, 30 → grassland,
  40 → cropland, 50 → built-up, 60 → bare/sparse, 80 → water. Shrubland, snow,
  wetland, mangroves and moss are set to 255 and ignored.
- **Split:** by location, so the same area never appears in two splits. Validation and
  test come from the Stuttgart core; 600 extra training locations come from a ring
  around it (V5).

Rebuild the dataset from scratch with `notebooks/01_data_collection.ipynb`.

---

## Model and training

- U-Net, 4 down/up levels, 16 base channels, batch norm, 1.94 M parameters
- Input: 2 channels (VV, VH), 128 × 128 tiles; output: 6 classes
- Normalisation per band: (dB − median) / IQR, values stored in the checkpoint
- Loss: weighted cross-entropy (weights 0.48 for the four common classes, 2.40 bare, 1.68 water)
- Patches with ≥ 100 bare or water pixels are sampled twice as often
- AdamW (lr 1e-3, weight decay 1e-4), batch size 4, 30 epochs
- Best checkpoint chosen by validation mIoU (epoch 6, validation mIoU 0.545)

## Results

All scores compare the prediction with ESA WorldCover 2021. Pixels labelled 255 are
not counted.

- **Pixel accuracy** = correctly classified pixels / all labelled pixels
- **IoU** per class = TP / (TP + FP + FN)
- **mIoU** = mean of the six class IoUs

### Held-out test set (27 locations × 12 months)

Pixel accuracy **0.719**, mIoU **0.473**

| Class | IoU | Precision | Recall |
|---|---:|---:|---:|
| Tree cover | 0.539 | 0.754 | 0.653 |
| Grassland | 0.329 | 0.573 | 0.436 |
| Cropland | 0.656 | 0.752 | 0.838 |
| Built-up | 0.632 | 0.711 | 0.850 |
| Bare/sparse vegetation | 0.000 | 0.000 | 0.000 |
| Permanent water | 0.684 | 0.817 | 0.808 |

By month, mIoU ranges from 0.455 (June) to 0.484 (December). The summer months are
the hardest because growing vegetation changes the radar signal.

### Prototype runs (12 months, October 2025 – September 2026)

| Area | Main classes in WorldCover | Accuracy | mIoU | Tree IoU | Built-up IoU |
|---|---|---:|---:|---:|---:|
| Stuttgart centre, 2 km × 2 km | 73 % built-up, 26 % trees | 0.791 | 0.202 | 0.29 | 0.78 |
| Stuttgart centre, 5 km × 5 km | 52 % trees, 45 % built-up | 0.788 | 0.340 | 0.64 | 0.68 |
| South-west of Stuttgart (48.69, 9.06), 5 km × 5 km | 84 % trees, 11 % built-up | 0.911 | 0.411 | 0.92 | 0.56 |

Reproduce with:

```bash
python scripts/check_accuracy.py --project your-gcp-project-id
python scripts/check_accuracy.py --project your-gcp-project-id --skip-test-set --lat 48.69 --lon 9.06 --area-km 5
```

### Observations

- Tree cover and built-up are mapped reliably when they cover large areas. Built-up
  recall is 0.71–0.97 at all sites.
- In the dense city centre, street trees and small parks are often predicted as
  built-up (tree recall 0.31), because the radar return from nearby buildings dominates
  at 10 m.
- Grassland is weak everywhere (IoU about 0.3); it is confused with trees and cropland.
- Water is sometimes predicted on smooth artificial surfaces such as roads and car
  parks (precision 0.25 at the south-west site).
- Bare/sparse vegetation is almost absent around Stuttgart and is not learned.
- mIoU depends strongly on classes that cover only a few pixels in a given area, so
  for single locations pixel accuracy and the IoU of the main classes are more
  informative.

## Limitations

- WorldCover 2021 is a single annual map used as the label for every month of
  2025–26. Monthly scores measure agreement with this static map, not monthly ground truth.
- Month-to-month changes mix real land cover change with seasonal radar effects
  (soil moisture, crop growth, frozen ground, snow). Changes of a few pp in stable
  areas are mostly noise.
- The model was trained around Stuttgart on descending passes and has only been
  checked against WorldCover around Stuttgart. The prototype accepts any coordinates
  with Sentinel-1 IW VV+VH data, but it is **not validated** for other regions;
  results elsewhere, and on ascending passes, should be treated as indicative only.
- Before 2017 Sentinel-1 coverage has gaps in some months, so the app offers start
  months from 2017 onward.

## Tests

```bash
pytest -q
```

The tests use fake radar images and an untrained model, so they run without Earth
Engine or the real weights. `scripts/check_real_data.py` and `scripts/check_accuracy.py`
run checks on real data.

## Repository structure

```
app.py                       Streamlit app
run_lulc.py                  command line version
src/
  gee.py                     Earth Engine: months, orbit choice, composites, download
  location.py                lat/lon -> UTM grid
  model.py                   U-Net and checkpoint loading
  inference.py               tiling, normalisation, filling gaps, prediction
  analysis.py                class shares, pp changes, transitions
  visualize.py               maps, charts, GIF
  pipeline.py                full run, used by the app, CLI and notebook
  classes.py                 class names and colours
notebooks/
  01_data_collection.ipynb
  02_train_evaluate_unet.ipynb
  03_prototype_run.ipynb
  archive/                   earlier exploration notebooks
scripts/
  package_dataset.py         zip the dataset for upload
  check_real_data.py         compare results on real data
  check_accuracy.py          accuracy against WorldCover
tests/test_pipeline.py
weights/best_unet_v5.pt
docs/RECORDING.md            plan for the screen recording
```
