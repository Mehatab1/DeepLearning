from affine import Affine
from pyproj import Transformer


def utm_epsg_for(latitude, longitude):
    # UTM zone number from the longitude
    zone = int((longitude + 180) // 6) + 1
    if zone > 60:
        zone = 60

    if latitude >= 0:
        return 32600 + zone   # northern hemisphere
    else:
        return 32700 + zone   # southern hemisphere


def create_utm_grid(latitude, longitude, area_size_m=2000, pixel_size_m=10):
    # makes a square grid in UTM around the point
    if latitude < -80 or latitude > 84:
        raise ValueError("Latitude must be between -80 and 84 (UTM coverage).")
    if longitude < -180 or longitude > 180:
        raise ValueError("Longitude must be between -180 and 180.")
    if area_size_m <= 0 or pixel_size_m <= 0:
        raise ValueError("Area and pixel sizes must be positive.")
    if area_size_m % pixel_size_m != 0:
        raise ValueError("Area size must be divisible by pixel size.")

    epsg = utm_epsg_for(latitude, longitude)
    crs = "EPSG:" + str(epsg)

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(longitude, latitude)

    # snap the corner to full pixels so every month uses the same grid
    half = area_size_m / 2
    left = round((x - half) / pixel_size_m) * pixel_size_m
    top = round((y + half) / pixel_size_m) * pixel_size_m
    right = left + area_size_m
    bottom = top - area_size_m

    size = int(area_size_m / pixel_size_m)

    grid = {
        "crs": crs,
        "epsg": epsg,
        "transform": [pixel_size_m, 0, left, 0, -pixel_size_m, top],
        "affine": Affine(pixel_size_m, 0, left, 0, -pixel_size_m, top),
        "bounds": (left, bottom, right, top),
        "width": size,
        "height": size,
        "center_x": x,
        "center_y": y,
        "latitude": latitude,
        "longitude": longitude,
        "area_size_m": area_size_m,
        "pixel_size_m": pixel_size_m,
    }
    return grid


def grid_bounds_latlon(grid):
    # returns (south, west, north, east) in lat/lon, used for the map in the app
    transformer = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    left, bottom, right, top = grid["bounds"]

    lons = []
    lats = []
    for px, py in [(left, bottom), (left, top), (right, bottom), (right, top)]:
        lon, lat = transformer.transform(px, py)
        lons.append(lon)
        lats.append(lat)

    return min(lats), min(lons), max(lats), max(lons)
