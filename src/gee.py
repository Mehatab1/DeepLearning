import io
import os
import time
import zipfile
from datetime import date
from pathlib import Path

import ee
import requests

# value used for pixels without radar data
NODATA = -9999.0

# ESA WorldCover code -> our class id (everything else becomes 255)
WORLDCOVER_TO_MODEL = {10: 1, 30: 2, 40: 3, 50: 4, 60: 5, 80: 6}


def initialize(project=None):
    # connect to Earth Engine, project can also come from the EE_PROJECT variable
    if project is None:
        project = os.environ.get("EE_PROJECT")
    if not project:
        raise RuntimeError("No Earth Engine project set. Pass project=... or set EE_PROJECT.")

    service_account = os.environ.get("EE_SERVICE_ACCOUNT")
    key_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if service_account and key_file:
        # service account login (for servers without a browser)
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials, project=project)
    else:
        # normal login saved by "earthengine authenticate"
        ee.Initialize(project=project)

    return project


def next_month(year, month):
    if month == 12:
        return year + 1, 1
    return year, month + 1


def month_schedule(start_month, n_months=12, today=None):
    # start_month looks like "2025-10", returns a list of 12 monthly date ranges
    try:
        parts = start_month.split("-")
        year = int(parts[0])
        month = int(parts[1])
    except (ValueError, IndexError, AttributeError):
        raise ValueError("start_month must look like 'YYYY-MM'.")

    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12.")
    if year < 2014 or (year == 2014 and month < 10):
        raise ValueError("Sentinel-1 data starts in October 2014.")

    if today is None:
        today = date.today()

    # the last month of the window must be finished already
    end_year, end_month = year, month
    for i in range(n_months - 1):
        end_year, end_month = next_month(end_year, end_month)

    if (end_year, end_month) >= (today.year, today.month):
        raise ValueError(
            "A " + str(n_months) + "-month window from " + start_month + " is not complete yet. "
            "Latest valid start month is " + latest_start_month(n_months, today) + "."
        )

    schedule = []
    y, m = year, month
    for i in range(n_months):
        ny, nm = next_month(y, m)
        schedule.append({
            "label": f"{y}-{m:02d}",
            "start": f"{y}-{m:02d}-01",
            "end": f"{ny}-{nm:02d}-01",
        })
        y, m = ny, nm

    return schedule


def latest_start_month(n_months=12, today=None):
    if today is None:
        today = date.today()

    # last finished month is the month before today
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12

    # go back n_months - 1 months from there
    for i in range(n_months - 1):
        m = m - 1
        if m == 0:
            y, m = y - 1, 12

    return f"{y}-{m:02d}"


def ee_region(grid):
    left, bottom, right, top = grid["bounds"]
    return ee.Geometry.Rectangle([left, bottom, right, top], proj=grid["crs"], geodesic=False)


def s1_collection(region, start, end, orbit_pass="DESCENDING"):
    collection = ee.ImageCollection("COPERNICUS/S1_GRD")
    collection = collection.filterBounds(region)
    collection = collection.filterDate(start, end)
    collection = collection.filter(ee.Filter.eq("instrumentMode", "IW"))
    collection = collection.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    collection = collection.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    collection = collection.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    return collection.select(["VV", "VH"])


def orbit_histograms(region, schedule, orbit_pass):
    # number of scenes per relative orbit for each month
    # (all months are asked in one getInfo call because it is much faster)
    requests_per_month = {}
    for m in schedule:
        collection = s1_collection(region, m["start"], m["end"], orbit_pass)
        requests_per_month[m["label"]] = collection.aggregate_histogram("relativeOrbitNumber_start")

    result = ee.Dictionary(requests_per_month).getInfo()

    hists = {}
    for label in result:
        hists[label] = {}
        for orbit, count in result[label].items():
            hists[label][int(float(orbit))] = int(count)
    return hists


def best_common_orbit(hists, months):
    # relative orbit that has scenes in every month (the one with most scenes), or None
    common = set(hists[months[0]].keys())
    for label in months:
        common = common & set(hists[label].keys())

    best_orbit = None
    best_total = -1
    for orbit in sorted(common):
        total = 0
        for label in months:
            total += hists[label][orbit]
        if total > best_total:
            best_total = total
            best_orbit = orbit
    return best_orbit


def months_without_scenes(hists, months):
    missing = []
    for label in months:
        if len(hists[label]) == 0:
            missing.append(label)
    return missing


def choose_acquisition(hists_desc, hists_asc, months):
    # one fixed orbit keeps the viewing geometry the same in every month,
    # so it is preferred over mixing orbits, even if it means using ascending passes
    # order: descending one orbit, ascending one orbit, descending mixed, ascending mixed
    for orbit_pass, hists in [("DESCENDING", hists_desc), ("ASCENDING", hists_asc)]:
        orbit = best_common_orbit(hists, months)
        if orbit is not None:
            counts = {}
            for label in months:
                counts[label] = hists[label][orbit]
            return {
                "orbit_pass": orbit_pass,
                "relative_orbit": orbit,
                "scene_counts": counts,
                "note": orbit_pass.lower() + " pass, relative orbit " + str(orbit),
            }

    for orbit_pass, hists in [("DESCENDING", hists_desc), ("ASCENDING", hists_asc)]:
        if len(months_without_scenes(hists, months)) == 0:
            counts = {}
            for label in months:
                counts[label] = sum(hists[label].values())
            return {
                "orbit_pass": orbit_pass,
                "relative_orbit": None,
                "scene_counts": counts,
                "note": orbit_pass.lower() + " pass, mixed relative orbits "
                        "(viewing geometry may differ between months)",
            }

    desc = ", ".join(months_without_scenes(hists_desc, months)) or "none"
    asc = ", ".join(months_without_scenes(hists_asc, months)) or "none"
    raise RuntimeError(
        "No Sentinel-1 IW VV+VH scenes for some months at this location "
        "(descending missing: " + desc + "; ascending missing: " + asc + "). "
        "Pick a start month after the gap, ideally 2017 or later."
    )


def select_acquisition(region, schedule):
    months = [m["label"] for m in schedule]

    hists_desc = orbit_histograms(region, schedule, "DESCENDING")
    # most places have one descending orbit in every month, then one request is enough
    if best_common_orbit(hists_desc, months) is not None:
        return choose_acquisition(hists_desc, {}, months)

    hists_asc = orbit_histograms(region, schedule, "ASCENDING")
    return choose_acquisition(hists_desc, hists_asc, months)


def monthly_composite(region, month, acquisition):
    collection = s1_collection(region, month["start"], month["end"], acquisition["orbit_pass"])

    if acquisition["relative_orbit"] is not None:
        collection = collection.filter(
            ee.Filter.eq("relativeOrbitNumber_start", acquisition["relative_orbit"])
        )

    image = collection.median().select(["VV", "VH"]).toFloat()
    image = image.unmask(NODATA)   # empty pixels get -9999
    return image.clip(region)


def worldcover_labels(region):
    codes = list(WORLDCOVER_TO_MODEL.keys())
    ids = list(WORLDCOVER_TO_MODEL.values())

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    labels = worldcover.remap(codes, ids, 255).toUint8()
    return labels.clip(region)


def download_image(image, grid, output_path, retries=3):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = image.getDownloadURL({
        "name": output_path.stem,
        "crs": grid["crs"],
        "crs_transform": grid["transform"],
        "region": ee_region(grid),
        "format": "GEO_TIFF",
    })

    # sometimes the download fails, so try a few times
    content = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            content = response.content
            break
        except requests.RequestException as e:
            print("Download attempt", attempt + 1, "failed:", e)
            time.sleep(2 * (attempt + 1))

    if content is None:
        raise RuntimeError("Download failed for " + output_path.name)

    # Earth Engine sometimes sends a zip instead of a tif
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            tif_name = None
            for name in archive.namelist():
                if name.lower().endswith(".tif") or name.lower().endswith(".tiff"):
                    tif_name = name
                    break
            if tif_name is None:
                raise ValueError("No GeoTIFF found in the download archive.")
            output_path.write_bytes(archive.read(tif_name))
    else:
        output_path.write_bytes(content)

    return output_path
