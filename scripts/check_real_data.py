# checks the new code on real data
# 1) re-runs Stuttgart with the Sentinel-1 images that the old code downloaded
#    and compares maps + tables with the old results (they should be identical)
# 2) does a fresh Earth Engine run for Frankfurt to test the download part
#
# usage: python scripts/check_real_data.py --project ee-sentinel1-lulc

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True)
args = parser.parse_args()

old_run = PROJECT_DIR / "runs" / "48.7758_9.1829_2025-10_2km"
check_root = PROJECT_DIR / "runs_check"
all_ok = True

# ---------- part 1: same inputs, old vs new results ----------
print("=== Part 1: Stuttgart, comparing with the old run ===")

if not (old_run / "inputs").is_dir():
    print("Old run not found at", old_run, "- skipping part 1")
else:
    new = pipeline.run(48.7758, 9.1829, "2025-10", area_km=2.0,
                       output_root=check_root, input_dir=old_run / "inputs")

    for month in new["months"]:
        with rasterio.open(old_run / "predictions" / ("lulc_" + month + ".tif")) as src:
            old_map = src.read(1)
        same = np.array_equal(old_map, new["maps"][month])
        print(month, "map identical:", same)
        if not same:
            all_ok = False

    for name in ["class_percentages_by_month.csv", "month_to_month_change_pp.csv",
                 "total_change_first_to_last_pp.csv", "transition_matrix_first_to_last_pixels.csv"]:
        old_table = pd.read_csv(old_run / "tables" / name, index_col=0)
        new_table = pd.read_csv(new["run_dir"] / "tables" / name, index_col=0)
        same = old_table.shape == new_table.shape and np.allclose(old_table.values, new_table.values)
        print(name, "identical:", same)
        if not same:
            all_ok = False

# ---------- part 2: fresh download from Earth Engine ----------
print("\n=== Part 2: Frankfurt, fresh Earth Engine run ===")


def show(message, fraction):
    print(f"[{fraction * 100:5.1f}%] {message}")


fresh = pipeline.run(50.1120, 8.7050, "2025-10", area_km=2.0, ee_project=args.project,
                     output_root=check_root, progress=show)

meta = fresh["metadata"]
print("Acquisition:", meta["acquisition"]["note"])
print("Scenes per month:", meta["acquisition"]["scene_counts"])
print("Invalid pixel fraction:", meta["invalid_pixel_fraction"])
print(fresh["results"]["percentages"].round(2).to_string())
print(fresh["results"]["total_change_pp"].round(2).to_string())

stack = fresh["results"]["stack"]
if stack.shape != (12, 200, 200):
    print("Unexpected map size:", stack.shape)
    all_ok = False

print("\nFigures:", fresh["run_dir"] / "figures")
if all_ok:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED - send me the output above")
