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
    bad_pixels = np.isnan(radar[0]) | np.isnan(radar[1])
    fraction = float(bad_pixels.mean())

    if fraction > MAX_INVALID_FRACTION:
        raise ValueError(
            str(round(fraction * 100)) + "% of pixels have no radar data; refusing to predict."
        )

    if fraction > 0:
        radar = radar.copy()
        for band in range(2):
            median_value = np.nanmedian(radar[band])
            radar[band][np.isnan(radar[band])] = median_value

    return radar, fraction


def normalize(radar, checkpoint):
    # same normalisation as in training: (value - median) / IQR
    medians = np.array(checkpoint["class_medians"], dtype=np.float32)
    iqr = np.array(checkpoint["class_iqr"], dtype=np.float32)

    if iqr[0] == 0 or iqr[1] == 0:
        raise ValueError("Checkpoint has a zero IQR normalisation value.")

    normalized = np.zeros_like(radar)
    for band in range(2):
        normalized[band] = (radar[band] - medians[band]) / iqr[band]
    return normalized


def predict_array(radar, model, checkpoint, patch_size=128, batch_size=16):
    # radar has shape (2, H, W) in dB, returns class ids 1-6 with shape (H, W)
    radar, invalid_fraction = fill_invalid(radar)
    x = normalize(radar, checkpoint)

    height = x.shape[1]
    width = x.shape[2]

    # pad so height and width are multiples of 128
    padded_h = int(np.ceil(height / patch_size)) * patch_size
    padded_w = int(np.ceil(width / patch_size)) * patch_size
    pad_h = padded_h - height
    pad_w = padded_w - width

    # reflect padding only works if the padding is smaller than the image
    if pad_h < height and pad_w < width:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    else:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

    # list of top-left corners of all tiles
    tiles = []
    for row in range(0, padded_h, patch_size):
        for col in range(0, padded_w, patch_size):
            tiles.append((row, col))

    device = next(model.parameters()).device
    output = np.zeros((padded_h, padded_w), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]

            batch = []
            for row, col in batch_tiles:
                batch.append(x[:, row:row + patch_size, col:col + patch_size])
            batch = torch.from_numpy(np.array(batch)).to(device)

            logits = model(batch)
            pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8) + 1   # 0-5 -> 1-6

            for j in range(len(batch_tiles)):
                row, col = batch_tiles[j]
                output[row:row + patch_size, col:col + patch_size] = pred[j]

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
