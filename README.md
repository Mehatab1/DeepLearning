# Sentinel-1 LULC

Monthly land use / land cover maps from Sentinel-1 radar alone.

Give it a latitude, longitude and start month. It pulls 12 monthly Sentinel-1 VV/VH
composites from Google Earth Engine, runs a U-Net trained on ESA WorldCover labels,
and returns:

- 12 monthly LULC maps (GeoTIFF + figures + GIF)
- the share of the area covered by each class, per month (%)
- the month-to-month change per class, in percentage points
- the total change from the first to the last month, in percentage points

Classes: tree cover, grassland, cropland, built-up, bare/sparse vegetation, permanent water.

| Deliverable | Where |
|---|---|
| Dataset | **[download link — add after upload](#dataset)** · regenerate with `notebooks/01_data_collection.ipynb` |
| Training notebook (results + training curves) | `notebooks/02_train_evaluate_unet.ipynb` |
| Prototype notebook (full run, two cities) | `notebooks/03_prototype_run.ipynb` |
| Prototype source | `app.py`, `run_lulc.py`, `src/` |
| Trained weights | `weights/best_unet_v5.pt` |

---

## Setup

Python 3.10–3.12.

```bash
git clone https://github.com/Mehatab1/sentinel1-lulc.git
cd sentinel1-lulc
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is optional. A 2 km × 2 km area (12 × 200 × 200 px) predicts in a few seconds on CPU;
most of the run time is the Earth Engine download.

### Accounts and keys

Only one account is needed: **Google Earth Engine**, attached to a Google Cloud project.

1. Sign up at <https://earthengine.google.com> and register a Cloud project for Earth Engine
   (non-commercial use is free). Note the project ID.
2. Authenticate once on the machine that runs the code:
   ```bash
   earthengine authenticate
   ```
3. Tell the code which project to use:
   ```bash
   cp .env.example .env               # edit EE_PROJECT
   export EE_PROJECT=your-gcp-project-id
   ```
   The app also has a field for the project ID.

For servers without a browser, use a service account instead: grant it the
*Earth Engine Resource Viewer* role, download a JSON key, and set
`EE_SERVICE_ACCOUNT` and `GOOGLE_APPLICATION_CREDENTIALS` (see `.env.example`).

No other API keys are required. Sentinel-1 and WorldCover are read through Earth Engine.

### Model weights

`weights/best_unet_v5.pt` (≈ 8 MB) is in the repository. It is a dict with the
`model_state_dict`, the architecture settings and the VV/VH normalisation statistics,
loaded by `src/model.py::load_v5_model`. Notebook 02 rewrites it after retraining.

---

## Running the prototype

**Web app**

```bash
streamlit run app.py
```

Pick a preset city or type coordinates, choose the start month and area size, press
*Run analysis*. Tabs show the 12 maps, the class shares, the change tables and a zip
download of everything.

**Command line**

```bash
python run_lulc.py --lat 48.7758 --lon 9.1829 --start 2025-10 --area-km 2
```

**Notebook**: `notebooks/03_prototype_run.ipynb`.

Each run writes to `runs/<lat>_<lon>_<start>_<size>km/`:

```
inputs/        s1_YYYY-MM.tif       monthly VV/VH median composites (dB)
predictions/   lulc_YYYY-MM.tif     class IDs 1–6, same grid
tables/        class_percentages_by_month.csv
               month_to_month_change_pp.csv
               total_change_first_to_last_pp.csv
               transition_matrix_first_to_last_pixels.csv
figures/       lulc_12_months.png / .gif, class_composition.png,
               month_to_month_change_pp.png, total_change_pp.png, change_maps.png
run_metadata.json
```

Inputs are cached, so re-running the same location and period skips the download.

### How the numbers are defined

For month *m* and class *c*, share(*m*, *c*) = pixels of class *c* / all pixels × 100.

- Month-to-month change = share(*m*, *c*) − share(*m*−1, *c*), in percentage points (pp).
  The rows sum to zero across classes.
- Total change = share(last, *c*) − share(first, *c*), in pp.
- `Pixels changed (%)` in the month-to-month table is the gross share of pixels whose class
  differs from the previous month; it can be large while net pp changes are small.

---

## Dataset

**Download:** _add the Zenodo / Google Drive link here_ (≈ size, sha256 printed by
`scripts/package_dataset.py`).

| File | Shape | Content |
|---|---|---|
| `patches_v5/train_v5.npz` | X (8280, 2, 128, 128) | 690 locations × 12 months |
| `patches_v4/val_v4.npz` | — | spatially held-out validation locations |
| `patches_v4/test_v4.npz` | X (324, 2, 128, 128) | 27 locations × 12 months |
| `monthly_inputs/` | 12 × 2001 × 2001 × 2 | Stuttgart 20 km composites used for full-area maps |

Each `.npz` holds `X` (VV/VH, already normalised), `y` (class IDs 1–6, 255 = ignore),
`class_medians`, `class_iqr` and a `metadata` record array with the month of each sample.

- **Input:** `COPERNICUS/S1_GRD`, IW mode, descending pass, VV + VH in dB, monthly
  median of one relative orbit, 10 m, UTM 32N. Oct 2025 – Sep 2026.
- **Labels:** `ESA/WorldCover/v200` (2021). 10→tree, 30→grassland, 40→cropland,
  50→built-up, 60→bare/sparse, 80→water. Shrubland, snow, wetland, mangrove and moss
  are set to 255 and ignored in the loss.
- **Split:** by location, so no patch area appears in two splits. Validation and test
  come from the Stuttgart core; V5 added 600 training locations from a ring around it.

To build the archive: `python scripts/package_dataset.py`.
To rebuild the dataset from scratch: `notebooks/01_data_collection.ipynb`.

---

## Model

U-Net, 4 down/up levels, 16 base channels, batch norm, 1.94 M parameters.
2 input channels (VV, VH), 6 output classes, 128 × 128 tiles.

Training: weighted cross-entropy (weights 0.48 ×4, 2.40 bare, 1.68 water), patches
with ≥ 100 bare or water pixels oversampled 2×, AdamW (lr 1e-3, wd 1e-4), batch 4,
30 epochs, best checkpoint by validation mIoU (epoch 6).

Inputs are normalised per band: (dB − median) / IQR, with the statistics stored in the checkpoint.

### Results

| Split | Pixel accuracy | mIoU |
|---|---:|---:|
| Validation (best epoch) | — | 0.545 |
| Test (held-out locations) | 0.719 | 0.473 |

Validation IoU per class: tree 0.83 · grassland 0.39 · cropland 0.74 · built-up 0.61 ·
bare/sparse 0.00 · water 0.70. Test mIoU ranges 0.455–0.484 across months, lowest in
early summer.

Training curves, confusion matrices and per-month scores are in notebook 02.

---

## Repository layout

```
app.py                     Streamlit app
run_lulc.py                CLI
src/
  gee.py                   Earth Engine: month windows, orbit selection, composites, download
  location.py              lat/lon → pixel-aligned UTM grid
  model.py                 U-Net + checkpoint loader
  inference.py             tiling, normalisation, NoData filling, prediction
  analysis.py              class shares, pp changes, transitions, stability
  visualize.py             maps, charts, GIF
  pipeline.py              end-to-end run used by app, CLI and notebook
notebooks/
  01_data_collection.ipynb
  02_train_evaluate_unet.ipynb
  03_prototype_run.ipynb
  archive/                 earlier exploratory notebooks (V5 prototype, arbitrary-location tests)
scripts/package_dataset.py
tests/test_pipeline.py     offline tests (synthetic rasters, random weights): pytest -q
weights/best_unet_v5.pt
```
