# full run for one location:
# lat/lon + start month -> 12 Sentinel-1 composites -> U-Net -> statistics -> figures

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import analysis, gee, visualize
from .inference import predict_array, read_radar, write_prediction
from .location import create_utm_grid
from .model import load_v5_model

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "weights" / "best_unet_v5.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "runs"

# keep the model in memory so the app doesn't load it again for every run
loaded_models = {}


def get_model(checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    key = str(checkpoint_path) + "|" + str(device)

    if key not in loaded_models:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                "Model weights not found at " + str(checkpoint_path) + ". See README -> 'Model weights'."
            )
        loaded_models[key] = load_v5_model(checkpoint_path, device=device)

    return loaded_models[key]


def run_name(lat, lon, start_month, area_km):
    return f"{lat:.4f}_{lon:.4f}_{start_month}_{area_km:g}km"


def fit_to_grid(radar, grid):
    # Earth Engine sometimes gives 1 pixel more or less, so cut or pad to the grid size
    h = grid["height"]
    w = grid["width"]
    radar = radar[:, :h, :w]

    missing_h = h - radar.shape[1]
    missing_w = w - radar.shape[2]
    if missing_h > 0 or missing_w > 0:
        radar = np.pad(radar, ((0, 0), (0, missing_h), (0, missing_w)), mode="edge")

    return radar


def show_progress(progress, message, fraction):
    # progress is a function from the app or the cli (can be None)
    if progress is not None:
        progress(message, fraction)


def run(latitude, longitude, start_month, area_km=2.0,
        checkpoint_path=DEFAULT_CHECKPOINT, ee_project=None,
        output_root=DEFAULT_OUTPUT_ROOT, input_dir=None,
        device=None, progress=None, make_figures=True):
    # input_dir: folder with already downloaded s1_YYYY-MM.tif files,
    # then Earth Engine is not used (for testing without internet)

    area_m = int(round(area_km * 1000))
    grid = create_utm_grid(latitude, longitude, area_size_m=area_m)
    schedule = gee.month_schedule(start_month)
    months = [m["label"] for m in schedule]

    run_dir = Path(output_root) / run_name(latitude, longitude, start_month, area_km)
    if input_dir is None:
        inputs_dir = run_dir / "inputs"
    else:
        inputs_dir = Path(input_dir)

    acquisition = {"note": "pre-downloaded inputs", "scene_counts": {}}

    # step 1: download the Sentinel-1 images
    if input_dir is None:
        show_progress(progress, "Connecting to Earth Engine", 0.02)
        gee.initialize(ee_project)
        region = gee.ee_region(grid)

        show_progress(progress, "Checking Sentinel-1 coverage for all 12 months", 0.05)
        acquisition = gee.select_acquisition(region, schedule)

        for i in range(len(schedule)):
            month = schedule[i]
            path = inputs_dir / ("s1_" + month["label"] + ".tif")
            if path.exists():
                continue   # already downloaded in an earlier run
            show_progress(progress, "Downloading Sentinel-1 composite " + month["label"],
                          0.08 + 0.5 * i / len(schedule))
            image = gee.monthly_composite(region, month, acquisition)
            gee.download_image(image, grid, path)

    # step 2: predict every month
    model, checkpoint = get_model(checkpoint_path, device)
    maps = {}
    invalid = {}
    for i in range(len(months)):
        label = months[i]
        show_progress(progress, "Predicting LULC for " + label, 0.6 + 0.25 * i / len(months))

        radar, profile = read_radar(inputs_dir / ("s1_" + label + ".tif"))
        radar = fit_to_grid(radar, grid)
        profile.update(height=grid["height"], width=grid["width"])

        prediction, invalid_fraction = predict_array(radar, model, checkpoint)
        write_prediction(prediction, profile, run_dir / "predictions" / ("lulc_" + label + ".tif"))

        maps[label] = prediction
        invalid[label] = invalid_fraction

    # step 3: percentages and changes
    show_progress(progress, "Calculating class percentages and changes", 0.88)
    results = analysis.analyze(maps, months, grid["pixel_size_m"], run_dir / "tables")

    # step 4: figures
    figures = {}
    if make_figures:
        show_progress(progress, "Rendering maps and charts", 0.94)
        title = (f"LULC {months[0]} – {months[-1]}  |  {latitude:.4f}, {longitude:.4f}  |  "
                 f"{area_km:g} km × {area_km:g} km")
        figures = visualize.save_all(results, run_dir / "figures", title)

    # save some info about the run
    summary = dict(results["summary"])
    del summary["months"]

    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "latitude": latitude,
        "longitude": longitude,
        "start_month": start_month,
        "months": months,
        "area_km": area_km,
        "crs": grid["crs"],
        "pixels": [grid["height"], grid["width"]],
        "acquisition": acquisition,
        "invalid_pixel_fraction": invalid,
        "checkpoint": str(checkpoint_path),
        "summary": summary,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    show_progress(progress, "Done", 1.0)

    return {
        "run_dir": run_dir,
        "grid": grid,
        "months": months,
        "maps": maps,
        "results": results,
        "figures": figures,
        "metadata": metadata,
    }
