# tests that run without Earth Engine and without the real weights
# (fake radar images + an untrained model)
# run with: pytest -q

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analysis, gee, pipeline
from src.location import create_utm_grid
from src.model import UNet

TODAY = date(2026, 11, 6)


def make_checkpoint(folder):
    torch.manual_seed(0)
    path = folder / "random.pt"
    torch.save({
        "epoch": 0,
        "model_state_dict": UNet().state_dict(),
        "best_val_miou": 0.0,
        "architecture": "UNet",
        "in_channels": 2,
        "num_classes": 6,
        "base_channels": 16,
        "class_medians": [-9.042964, -15.390633],
        "class_iqr": [3.967108, 4.4696884],
    }, path)
    return path


def make_fake_inputs(folder, grid, months, add_nodata=False):
    rng = np.random.default_rng(1)
    folder.mkdir(parents=True, exist_ok=True)

    for i in range(len(months)):
        vv = rng.normal(-9 + i * 0.1, 4, (grid["height"], grid["width"]))
        vh = rng.normal(-15, 4, (grid["height"], grid["width"]))
        data = np.stack([vv, vh]).astype(np.float32)
        if add_nodata:
            data[:, :5, :5] = gee.NODATA   # small gap in the corner

        with rasterio.open(folder / ("s1_" + months[i] + ".tif"), "w", driver="GTiff",
                           count=2, dtype="float32", width=grid["width"], height=grid["height"],
                           crs=grid["crs"], transform=grid["affine"], nodata=gee.NODATA) as dst:
            dst.write(data)


def test_month_schedule():
    schedule = gee.month_schedule("2025-10", today=TODAY)
    labels = [m["label"] for m in schedule]

    assert len(schedule) == 12
    assert labels[:4] == ["2025-10", "2025-11", "2025-12", "2026-01"]
    assert schedule[-1]["label"] == "2026-09"
    assert schedule[-1]["end"] == "2026-10-01"


def test_month_schedule_not_finished_yet():
    with pytest.raises(ValueError):
        gee.month_schedule("2026-01", today=TODAY)
    assert gee.latest_start_month(today=TODAY) == "2025-11"
    # the latest start month itself must be accepted
    assert len(gee.month_schedule("2025-11", today=TODAY)) == 12


def test_grid():
    grid = create_utm_grid(48.7758, 9.1829, 2000)
    assert grid["epsg"] == 32632
    assert grid["width"] == 200 and grid["height"] == 200
    assert grid["transform"][2] % 10 == 0
    assert grid["transform"][5] % 10 == 0


def test_percentage_points():
    a = np.full((10, 10), 4, np.uint8)   # 100 % built-up
    b = a.copy()
    b[:2] = 1                             # 80 % built-up, 20 % tree
    c = b.copy()
    c[2:3] = 3                            # 70 % built-up, 20 % tree, 10 % crop

    r = analysis.analyze({"m1": a, "m2": b, "m3": c}, ["m1", "m2", "m3"])

    pct = r["percentages"]
    assert pct.loc["m1", "Built-up"] == 100
    assert np.allclose(pct.sum(axis=1), 100)

    m2m = r["month_to_month_pp"]
    assert m2m.loc["m1 → m2", "Built-up"] == pytest.approx(-20)
    assert m2m.loc["m1 → m2", "Tree cover"] == pytest.approx(20)
    assert m2m.loc["m2 → m3", "Cropland"] == pytest.approx(10)
    assert m2m.loc["m2 → m3", "Pixels changed (%)"] == pytest.approx(10)

    total = r["total_change_pp"]["Change (pp)"]
    assert total["Built-up"] == pytest.approx(-30)
    assert total.sum() == pytest.approx(0)

    assert r["transition_first_last"].loc["Built-up", "Tree cover"] == 20
    assert r["summary"]["largest_net_loss"][0] == "Built-up"


def test_full_run_offline(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    lat, lon, start = 48.7758, 9.1829, "2025-10"

    grid = create_utm_grid(lat, lon, 1500)
    months = [m["label"] for m in gee.month_schedule(start)]
    make_fake_inputs(tmp_path / "inputs", grid, months, add_nodata=True)

    result = pipeline.run(lat, lon, start, area_km=1.5, checkpoint_path=checkpoint,
                          output_root=tmp_path / "runs", input_dir=tmp_path / "inputs")

    assert len(result["maps"]) == 12
    for month in result["maps"]:
        assert result["maps"][month].shape == (150, 150)
        assert result["maps"][month].min() >= 1 and result["maps"][month].max() <= 6

    assert result["metadata"]["invalid_pixel_fraction"]["2025-10"] > 0

    for name in ["lulc_12_months.png", "lulc_12_months.gif", "month_to_month_change_pp.png",
                 "total_change_pp.png", "class_composition.png", "change_maps.png"]:
        assert result["figures"][name].exists()

    tables = result["run_dir"] / "tables"
    assert (tables / "month_to_month_change_pp.csv").exists()
    assert (tables / "total_change_first_to_last_pp.csv").exists()

    with rasterio.open(result["run_dir"] / "predictions" / "lulc_2026-09.tif") as src:
        assert src.crs.to_epsg() == 32632
        assert src.count == 1
