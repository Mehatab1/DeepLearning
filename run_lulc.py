# run the whole analysis from the terminal, example:
# python run_lulc.py --lat 48.7758 --lon 9.1829 --start 2025-10

import argparse

import pandas as pd

from src import pipeline


def print_progress(message, fraction):
    print(f"[{fraction * 100:5.1f}%] {message}")


parser = argparse.ArgumentParser(description="12-month Sentinel-1 LULC analysis")
parser.add_argument("--lat", type=float, required=True)
parser.add_argument("--lon", type=float, required=True)
parser.add_argument("--start", required=True, help="first month, YYYY-MM")
parser.add_argument("--area-km", type=float, default=2.0, help="side length of the square (1-10 km)")
parser.add_argument("--project", default=None, help="Earth Engine project (or set EE_PROJECT)")
parser.add_argument("--weights", default=str(pipeline.DEFAULT_CHECKPOINT))
parser.add_argument("--out", default=str(pipeline.DEFAULT_OUTPUT_ROOT))
parser.add_argument("--input-dir", default=None,
                    help="use already downloaded s1_YYYY-MM.tif files instead of Earth Engine")
args = parser.parse_args()

if args.area_km < 1 or args.area_km > 10:
    parser.error("--area-km must be between 1 and 10")

result = pipeline.run(
    args.lat, args.lon, args.start,
    area_km=args.area_km,
    checkpoint_path=args.weights,
    ee_project=args.project,
    output_root=args.out,
    input_dir=args.input_dir,
    progress=print_progress,
)

tables = result["results"]
pd.set_option("display.width", 160)

print("\nClass share per month (%)")
print(tables["percentages"].round(2).to_string())

print("\nMonth-to-month change (percentage points)")
print(tables["month_to_month_pp"].round(2).to_string())

print("\nTotal change, first to last month (percentage points)")
print(tables["total_change_pp"].round(2).to_string())

print("\nOutputs:", result["run_dir"])
