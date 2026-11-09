from pathlib import Path

import numpy as np
import rasterio
import torch

from .model import load_v5_model

NODATA = -9999.0

# if more than half of the image has no data we don't predict
MAX_INVALID_FRACTION = 0.5


def read_radar(path):
    # reads the 2 band VV/VH GeoTIFF, missing values become NaN
    with rasterio.open(path) as src:
        if src.count != 2:
            raise ValueError(str(path) + ": expected 2 bands (VV, VH), found " + str(src.count))
        radar = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        radar[radar == nodata] = np.nan
    radar[radar == NODATA] = np.nan

    return radar, profile


def fill_invalid(radar):
    # small gaps (e.g. at the swath edge) are filled with the median of the band
    # NaN and +/- infinity both count as missing
    bad_pixels = ~np.isfinite(radar[0]) | ~np.isfinite(radar[1])
    fraction = float(bad_pixels.mean())

    # this also stops the case of a completely empty band (fraction would be 100 %)
    if fraction > MAX_INVALID_FRACTION:
        raise ValueError(
            str(round(fraction * 100)) + "% of pixels have no radar data; refusing to predict."
        )

    if fraction > 0:
        radar = radar.copy()
        for band in range(2):
            good = np.isfinite(radar[band])
            median_value = np.median(radar[band][good])
            radar[band][~good] = median_value

    return radar, fraction


def normalize(radar, checkpoint):
    # same normalisation as in training: (value - median) / IQR
    medians = np.array(checkpoint["class_medians"], dtype=np.float32)
    iqr = np.array(checkpoint["class_iqr"], dtype=np.float32)

    if medians.shape != (2,) or iqr.shape != (2,):
        raise ValueError("Checkpoint must have 2 median and 2 IQR values (VV, VH).")
    if not np.all(np.isfinite(medians)) or not np.all(np.isfinite(iqr)):
        raise ValueError("Checkpoint normalisation values are not finite.")
    if np.any(iqr <= 0):
        raise ValueError("Checkpoint IQR normalisation values must be positive.")

    normalized = np.zeros_like(radar)
    for band in range(2):
        normalized[band] = (radar[band] - medians[band]) / iqr[band]
    return normalized


# tiles overlap by half a tile and the class probabilities are averaged,
# so there are no hard edges every 128 pixels. Set to 128 for the old non-overlapping tiles.
TILE_STRIDE = 64


def tile_positions(size, patch_size, stride):
    # start positions so that tiles of patch_size cover 0..size
    padded = patch_size
    while padded < size:
        padded += stride
    return list(range(0, padded - patch_size + 1, stride)), padded


def predict_array(radar, model, checkpoint, patch_size=128, batch_size=16, stride=None):
    # radar has shape (2, H, W) in dB, returns class ids 1-6 with shape (H, W)
    if stride is None:
        stride = TILE_STRIDE
    if patch_size <= 0 or batch_size <= 0 or stride <= 0 or stride > patch_size:
        raise ValueError("Need patch_size > 0, batch_size > 0 and 0 < stride <= patch_size.")
    if radar.ndim != 3 or radar.shape[0] != 2 or radar.shape[1] == 0 or radar.shape[2] == 0:
        raise ValueError("Radar array must have shape (2, H, W).")

    radar, invalid_fraction = fill_invalid(radar)
    x = normalize(radar, checkpoint)

    height = x.shape[1]
    width = x.shape[2]

    rows, padded_h = tile_positions(height, patch_size, stride)
    cols, padded_w = tile_positions(width, patch_size, stride)
    pad_h = padded_h - height
    pad_w = padded_w - width

    # reflect padding only works if the padding is smaller than the image
    if pad_h < height and pad_w < width:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    else:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

    # list of top-left corners of all tiles
    tiles = []
    for row in rows:
        for col in cols:
            tiles.append((row, col))

    device = next(model.parameters()).device
    prob_sum = np.zeros((6, padded_h, padded_w), dtype=np.float32)
    counts = np.zeros((padded_h, padded_w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]

            batch = []
            for row, col in batch_tiles:
                batch.append(x[:, row:row + patch_size, col:col + patch_size])
            batch = torch.from_numpy(np.array(batch, dtype=np.float32)).to(device)

            probs = torch.softmax(model(batch), dim=1).cpu().numpy()

            for j in range(len(batch_tiles)):
                row, col = batch_tiles[j]
                prob_sum[:, row:row + patch_size, col:col + patch_size] += probs[j]
                counts[row:row + patch_size, col:col + patch_size] += 1

    mean_probs = prob_sum / counts
    output = mean_probs.argmax(axis=0).astype(np.uint8) + 1   # 0-5 -> 1-6

    prediction = output[:height, :width]
    return prediction, invalid_fraction


def write_prediction(prediction, profile, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="LZW")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction, 1)

    return output_path


def predict_raster(input_path, output_path, checkpoint_path, device=None):
    # file in, file out (handy for testing single months)
    model, checkpoint = load_v5_model(checkpoint_path, device=device)
    radar, profile = read_radar(input_path)
    prediction, invalid_fraction = predict_array(radar, model, checkpoint)
    write_prediction(prediction, profile, output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "shape": prediction.shape,
        "invalid_fraction": invalid_fraction,
    }
