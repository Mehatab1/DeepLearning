# accuracy check on real data
# part 1: held-out test patches (patches_v4/test_v4.npz) vs WorldCover labels
# part 2: the 12 monthly maps of a prototype run vs WorldCover for the same area
#
# usage:
#   python scripts/check_accuracy.py --project ee-sentinel1-lulc
#   python scripts/check_accuracy.py --project ee-sentinel1-lulc --lat 50.1120 --lon 8.7050 --start 2025-10

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src import gee, pipeline
from src.classes import CLASS_IDS, CLASS_NAMES
from src.location import create_utm_grid

NAMES = [CLASS_NAMES[c] for c in CLASS_IDS]


def confusion_matrix(reference, prediction):
    # rows = WorldCover class, columns = predicted class
    # pixels whose reference is not 1-6 (255 = ignored class) are skipped
    cm = np.zeros((6, 6), dtype=np.int64)
    valid = (reference >= 1) & (reference <= 6)
    ref = reference[valid].astype(np.int64) - 1
    pred = prediction[valid].astype(np.int64) - 1
    np.add.at(cm, (ref, pred), 1)
    return cm


def metrics(cm):
    tp = np.diag(cm).astype(float)
    actual = cm.sum(axis=1).astype(float)      # row sums
    predicted = cm.sum(axis=0).astype(float)   # column sums

    iou = np.zeros(6)
    precision = np.zeros(6)
    recall = np.zeros(6)
    for i in range(6):
        union = actual[i] + predicted[i] - tp[i]
        if union > 0:
            iou[i] = tp[i] / union
        if predicted[i] > 0:
            precision[i] = tp[i] / predicted[i]
        if actual[i] > 0:
            recall[i] = tp[i] / actual[i]

    # mIoU only over classes that exist in the reference or the prediction
    present = (actual + predicted) > 0
    accuracy = tp.sum() / max(actual.sum(), 1)
    miou = iou[present].mean() if present.any() else 0.0

    table = pd.DataFrame({"IoU": iou, "Precision": precision, "Recall": recall,
                          "Reference pixels": actual.astype(int)}, index=NAMES)
    return accuracy, miou, table


def plot_confusion(cm, title, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(NAMES, rotation=35, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("WorldCover reference")
    for r in range(6):
        for c in range(6):
            color = "white" if norm[r, c] > 0.5 else "black"
            ax.text(c, r, f"{norm[r, c]:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised (recall)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def test_set_accuracy(test_path, out_dir):
    out_dir = out_dir / "test_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Part 1: held-out test set ===")
    print("File:", test_path)

    model, checkpoint = pipeline.get_model()
    device = next(model.parameters()).device

    data = np.load(test_path)
    X = data["X"].astype(np.float32)     # already normalised
    y = data["y"]                        # 1-6, 255 = ignore
    months = data["metadata"]["month"] if "metadata" in data.files else None
    print("Patches:", X.shape[0])

    # predict in small batches
    preds = np.zeros(y.shape, dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 16):
            batch = torch.from_numpy(X[i:i + 16]).to(device)
            preds[i:i + 16] = model(batch).argmax(dim=1).cpu().numpy() + 1

    cm = confusion_matrix(y, preds)
    accuracy, miou, table = metrics(cm)
    print(f"\nPixel accuracy: {accuracy:.4f}")
    print(f"mIoU:           {miou:.4f}  (classes present in reference or prediction)")
    print(f"mIoU, all 6:    {table['IoU'].mean():.4f}  (same as notebook 02)")
    print(table.round(4).to_string())

    table.to_csv(out_dir / "per_class.csv")
    plot_confusion(cm, "Test set confusion matrix", out_dir / "confusion.png")

    if months is not None:
        rows = []
        for month in sorted(set(months.tolist())):
            idx = months == month
            acc_m, miou_m, _ = metrics(confusion_matrix(y[idx], preds[idx]))
            rows.append({"month": month, "accuracy": acc_m, "mIoU": miou_m})
        by_month = pd.DataFrame(rows).set_index("month")
        print("\nBy month:")
        print(by_month.round(4).to_string())
        by_month.to_csv(out_dir / "by_month.csv")


def location_accuracy(lat, lon, start, area_km, project, out_dir):
    # one folder per location, so runs for different places do not overwrite each other
    out_dir = out_dir / pipeline.run_name(lat, lon, start, area_km)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Part 2: prototype run at {lat}, {lon}, start {start} ===")

    run = pipeline.run(lat, lon, start, area_km=area_km, ee_project=project)

    # WorldCover for exactly the same grid
    grid = create_utm_grid(lat, lon, int(round(area_km * 1000)))
    wc_path = run["run_dir"] / "worldcover_reference.tif"
    if not wc_path.exists():
        print("Downloading WorldCover reference ...")
        gee.download_image(gee.worldcover_labels(gee.ee_region(grid)), grid, wc_path)

    with rasterio.open(wc_path) as src:
        reference = src.read(1)[:grid["height"], :grid["width"]]

    labelled = np.mean((reference >= 1) & (reference <= 6)) * 100
    print(f"Pixels with a usable WorldCover label: {labelled:.1f} %")

    rows = []
    total_cm = np.zeros((6, 6), dtype=np.int64)
    for month in run["months"]:
        cm = confusion_matrix(reference, run["maps"][month])
        total_cm += cm
        acc, miou, _ = metrics(cm)
        rows.append({"month": month, "accuracy": acc, "mIoU": miou})

    by_month = pd.DataFrame(rows).set_index("month")
    print(by_month.round(4).to_string())

    accuracy, miou, table = metrics(total_cm)
    print(f"\nAll 12 months together: accuracy {accuracy:.4f}, mIoU {miou:.4f}")
    print(table.round(4).to_string())

    by_month.to_csv(out_dir / "by_month.csv")
    table.to_csv(out_dir / "per_class.csv")
    plot_confusion(total_cm, "Prototype run vs WorldCover (12 months)", out_dir / "confusion.png")


parser = argparse.ArgumentParser()
parser.add_argument("--project", default=None, help="Earth Engine project (or set EE_PROJECT)")
parser.add_argument("--test-file", default=str(PROJECT_DIR / "patches_v4" / "test_v4.npz"))
parser.add_argument("--lat", type=float, default=48.7758)
parser.add_argument("--lon", type=float, default=9.1829)
parser.add_argument("--start", default="2025-10")
parser.add_argument("--area-km", type=float, default=2.0)
parser.add_argument("--skip-test-set", action="store_true")
parser.add_argument("--skip-location", action="store_true")
args = parser.parse_args()

out_dir = PROJECT_DIR / "results" / "accuracy"
out_dir.mkdir(parents=True, exist_ok=True)

if not args.skip_test_set:
    if Path(args.test_file).exists():
        test_set_accuracy(args.test_file, out_dir)
    else:
        print("Test file not found:", args.test_file)

if not args.skip_location:
    location_accuracy(args.lat, args.lon, args.start, args.area_km, args.project, out_dir)

print("\nResults saved in", out_dir)