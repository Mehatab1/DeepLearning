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


# ---------- tests for the review findings ----------

from src import inference

MONTHS_3 = ["m1", "m2", "m3"]


def test_orbit_prefers_single_ascending_over_mixed_descending():
    # descending has scenes every month but from different orbits, ascending has orbit 117 every month
    desc = {"m1": {15: 2}, "m2": {66: 2}, "m3": {15: 1, 66: 1}}
    asc = {"m1": {117: 3, 44: 1}, "m2": {117: 2}, "m3": {117: 2}}
    acq = gee.choose_acquisition(desc, asc, MONTHS_3)
    assert acq["orbit_pass"] == "ASCENDING"
    assert acq["relative_orbit"] == 117
    assert acq["scene_counts"] == {"m1": 3, "m2": 2, "m3": 2}


def test_orbit_order_of_fallbacks():
    single_desc = {"m1": {15: 2, 66: 1}, "m2": {15: 2}, "m3": {15: 1}}
    mixed_desc = {"m1": {15: 2}, "m2": {66: 2}, "m3": {15: 1}}
    mixed_asc = {"m1": {117: 1}, "m2": {44: 1}, "m3": {117: 1}}
    empty_month = {"m1": {117: 1}, "m2": {}, "m3": {117: 1}}

    # one descending orbit wins even if ascending also has one
    assert gee.choose_acquisition(single_desc, {"m1": {1: 1}, "m2": {1: 1}, "m3": {1: 1}},
                                  MONTHS_3)["orbit_pass"] == "DESCENDING"
    # no single orbit anywhere -> mixed descending
    acq = gee.choose_acquisition(mixed_desc, mixed_asc, MONTHS_3)
    assert acq["orbit_pass"] == "DESCENDING" and acq["relative_orbit"] is None
    # descending has a gap -> mixed ascending
    acq = gee.choose_acquisition(empty_month, mixed_asc, MONTHS_3)
    assert acq["orbit_pass"] == "ASCENDING" and acq["relative_orbit"] is None
    # both have a gap -> error that names the month
    with pytest.raises(RuntimeError, match="m2"):
        gee.choose_acquisition(empty_month, empty_month, MONTHS_3)


def write_tif(path, grid, crs=None, shift_m=0, bands=2):
    data = np.full((bands, grid["height"], grid["width"]), -10, dtype=np.float32)
    transform = rasterio.Affine(10, 0, grid["transform"][2] + shift_m, 0, -10, grid["transform"][5])
    with rasterio.open(path, "w", driver="GTiff", count=bands, dtype="float32",
                       width=grid["width"], height=grid["height"],
                       crs=crs or grid["crs"], transform=transform) as dst:
        dst.write(data)


def test_check_input_rejects_wrong_georeferencing(tmp_path):
    grid = create_utm_grid(48.7758, 9.1829, 1000)

    write_tif(tmp_path / "ok.tif", grid)
    assert pipeline.check_input(tmp_path / "ok.tif", grid) is None

    write_tif(tmp_path / "crs.tif", grid, crs="EPSG:32633")
    assert "CRS" in pipeline.check_input(tmp_path / "crs.tif", grid)

    write_tif(tmp_path / "shift.tif", grid, shift_m=10)
    assert "aligned" in pipeline.check_input(tmp_path / "shift.tif", grid)

    write_tif(tmp_path / "bands.tif", grid, bands=1)
    assert "bands" in pipeline.check_input(tmp_path / "bands.tif", grid)

    (tmp_path / "broken.tif").write_bytes(b"II*\x00 not really a tiff")
    assert "cannot be opened" in pipeline.check_input(tmp_path / "broken.tif", grid)


def test_run_rejects_misaligned_input(tmp_path):
    checkpoint = make_checkpoint(tmp_path)
    grid = create_utm_grid(48.7758, 9.1829, 1000)
    months = [m["label"] for m in gee.month_schedule("2025-10")]
    make_fake_inputs(tmp_path / "inputs", grid, months)
    write_tif(tmp_path / "inputs" / "s1_2026-03.tif", grid, crs="EPSG:32633")

    with pytest.raises(ValueError, match="2026-03"):
        pipeline.run(48.7758, 9.1829, "2025-10", area_km=1, checkpoint_path=checkpoint,
                     output_root=tmp_path / "runs", input_dir=tmp_path / "inputs")


def fake_earth_engine(monkeypatch, good_download=True):
    downloads = []
    monkeypatch.setattr(gee, "initialize", lambda project=None: project)
    monkeypatch.setattr(gee, "ee_region", lambda grid: None)
    monkeypatch.setattr(gee, "select_acquisition",
                        lambda region, schedule: {"note": "fake", "scene_counts": {}})
    monkeypatch.setattr(gee, "monthly_composite", lambda region, month, acq: month["label"])

    def download(image, grid, path):
        downloads.append(image)
        if good_download:
            write_tif(path, grid)
        else:
            Path(path).write_text("<html>error</html>")
        return path

    monkeypatch.setattr(gee, "download_image", download)
    return downloads


def test_broken_cached_file_is_downloaded_again(tmp_path, monkeypatch):
    checkpoint = make_checkpoint(tmp_path)
    downloads = fake_earth_engine(monkeypatch)
    grid = create_utm_grid(48.7758, 9.1829, 1000)

    # first run fills the cache
    pipeline.run(48.7758, 9.1829, "2025-10", area_km=1, checkpoint_path=checkpoint,
                 output_root=tmp_path / "runs", make_figures=False)
    assert len(downloads) == 12

    # truncate one cached file -> only that month is downloaded again
    run_dir = tmp_path / "runs" / pipeline.run_name(48.7758, 9.1829, "2025-10", 1)
    (run_dir / "inputs" / "s1_2026-01.tif").write_bytes(b"truncated")
    pipeline.run(48.7758, 9.1829, "2025-10", area_km=1, checkpoint_path=checkpoint,
                 output_root=tmp_path / "runs", make_figures=False)
    assert downloads[12:] == ["2026-01"]
    assert pipeline.check_input(run_dir / "inputs" / "s1_2026-01.tif", grid) is None


def test_bad_download_stops_with_clear_error(tmp_path, monkeypatch):
    checkpoint = make_checkpoint(tmp_path)
    fake_earth_engine(monkeypatch, good_download=False)
    with pytest.raises(RuntimeError, match="cannot be opened"):
        pipeline.run(48.7758, 9.1829, "2025-10", area_km=1, checkpoint_path=checkpoint,
                     output_root=tmp_path / "runs", make_figures=False)
    run_dir = tmp_path / "runs" / pipeline.run_name(48.7758, 9.1829, "2025-10", 1)
    assert not (run_dir / "inputs" / "s1_2025-10.tif").exists()   # bad file not cached


def test_infinite_and_missing_values():
    radar = np.full((2, 10, 10), -10.0, dtype=np.float32)
    radar[0, 0, :3] = np.inf
    radar[1, 1, :2] = -np.inf
    radar[0, 2, 0] = np.nan
    filled, fraction = inference.fill_invalid(radar)
    assert np.all(np.isfinite(filled))
    assert fraction == pytest.approx(6 / 100)

    empty_band = np.full((2, 10, 10), -10.0, dtype=np.float32)
    empty_band[1] = np.nan
    with pytest.raises(ValueError):
        inference.fill_invalid(empty_band)

    with pytest.raises(ValueError):
        inference.fill_invalid(np.full((2, 10, 10), np.inf, dtype=np.float32))


def test_bad_normalisation_values():
    radar = np.zeros((2, 4, 4), dtype=np.float32)
    for medians, iqr in [([0, 0], [0, 1]), ([0, 0], [-1, 1]), ([np.nan, 0], [1, 1]), ([0, 0], [1, np.inf])]:
        with pytest.raises(ValueError):
            inference.normalize(radar, {"class_medians": medians, "class_iqr": iqr})


def test_overlapping_tiles(tmp_path):
    torch.manual_seed(3)
    net = UNet()
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    net.eval()
    ckpt = {"class_medians": [-9.0, -15.0], "class_iqr": [4.0, 4.5]}
    rng = np.random.default_rng(0)
    radar = rng.normal(-10, 4, (2, 200, 150)).astype(np.float32)

    # stride 128 = old behaviour: each tile predicted on its own
    old, _ = inference.predict_array(radar, net, ckpt, stride=128)
    x = inference.normalize(radar, ckpt)
    x = np.pad(x, ((0, 0), (0, 56), (0, 106)), mode="reflect")
    expected = np.zeros((256, 256), dtype=np.uint8)
    with torch.no_grad():
        for r in [0, 128]:
            for c in [0, 128]:
                tile = torch.from_numpy(np.ascontiguousarray(x[None, :, r:r + 128, c:c + 128]))
                expected[r:r + 128, c:c + 128] = net(tile).argmax(1)[0].numpy() + 1
    assert np.array_equal(old, expected[:200, :150])

    new, _ = inference.predict_array(radar, net, ckpt)   # default stride 64
    assert new.shape == (200, 150)
    assert new.min() >= 1 and new.max() <= 6

    with pytest.raises(ValueError):
        inference.predict_array(radar, net, ckpt, stride=0)


def test_model_reloaded_when_weights_file_changes(tmp_path):
    path = make_checkpoint(tmp_path)
    first = pipeline.get_model(path)
    assert pipeline.get_model(path) is first          # cached
    import os
    os.utime(path, (1, 1))                             # pretend the file was replaced
    assert pipeline.get_model(path) is not first


def test_analysis_needs_two_months():
    with pytest.raises(ValueError):
        analysis.analyze({"m1": np.ones((5, 5), np.uint8)}, ["m1"])
