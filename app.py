# Sentinel-1 LULC Explorer
# start with:  streamlit run app.py

import io
import os
import zipfile
from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

from src import gee, pipeline
from src.classes import CLASS_COLORS, CLASS_IDS, CLASS_NAMES
from src.location import create_utm_grid, grid_bounds_latlon
from src.visualize import colorize

PRESETS = {
    "Custom": None,
    "Stuttgart – city centre": (48.7758, 9.1829),
    "Frankfurt am Main – Ostend": (50.1120, 8.7050),
    "Munich – Riem (new district)": (48.1340, 11.6930),
    "Berlin – BER airport area": (52.3667, 13.5033),
    "Heidelberg – Bahnstadt": (49.4040, 8.6720),
}

st.set_page_config(page_title="Sentinel-1 LULC Explorer", layout="wide")

# default location is Stuttgart
if "lat" not in st.session_state:
    st.session_state["lat"] = 48.7758
if "lon" not in st.session_state:
    st.session_state["lon"] = 9.1829
if "preset" not in st.session_state:
    st.session_state["preset"] = "Stuttgart – city centre"

# if the user clicked on the map in the last run, use that point now
# (this has to happen before the number inputs are created)
if "picked" in st.session_state:
    st.session_state["lat"] = st.session_state["picked"][0]
    st.session_state["lon"] = st.session_state["picked"][1]
    st.session_state["preset"] = "Custom"
    del st.session_state["picked"]


def preset_changed():
    coords = PRESETS[st.session_state["preset"]]
    if coords is not None:
        st.session_state["lat"] = coords[0]
        st.session_state["lon"] = coords[1]


# ----- sidebar -----

st.sidebar.header("Location and period")

# no index here: the value comes from session state, set above or by a map click
st.sidebar.selectbox("Preset", list(PRESETS.keys()), key="preset", on_change=preset_changed)
lat = st.sidebar.number_input("Latitude", -80.0, 84.0, key="lat", format="%.5f")
lon = st.sidebar.number_input("Longitude", -180.0, 180.0, key="lon", format="%.5f")
st.sidebar.caption("Or click anywhere on the map.")

# start months from the newest possible one back to 2017-01
# (before 2017 Sentinel-1 coverage has gaps in many places)
start_options = []
year, month = [int(x) for x in gee.latest_start_month().split("-")]
while year >= 2017:
    start_options.append(f"{year}-{month:02d}")
    month = month - 1
    if month == 0:
        month = 12
        year = year - 1

start_month = st.sidebar.selectbox("Start month", start_options, index=0,
                                   help="The analysis covers this month and the following 11.")

area_km = st.sidebar.slider("Area (km × km)", 1.0, 5.0, 2.0, 0.5)

with st.sidebar.expander("Earth Engine"):
    ee_project = st.text_input("Cloud project ID", os.environ.get("EE_PROJECT", ""))
    st.caption("Needs `earthengine authenticate` once, or a service account (see README).")

run_clicked = st.sidebar.button("Run analysis", type="primary", width="stretch")


# ----- header and map -----

st.title("Sentinel-1 LULC Explorer")
st.caption("Monthly land use / land cover from Sentinel-1 VV/VH radar only, "
           "predicted by a U-Net trained on ESA WorldCover labels around Stuttgart. "
           "Results outside south-west Germany have not been validated.")

try:
    grid = create_utm_grid(lat, lon, int(area_km * 1000))
except ValueError as e:
    st.error(str(e))
    st.stop()

south, west, north, east = grid_bounds_latlon(grid)

m = folium.Map(location=[lat, lon], zoom_start=13, control_scale=True, tiles=None)
folium.TileLayer("OpenStreetMap", name="Map").add_to(m)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite",
).add_to(m)
folium.Rectangle([[south, west], [north, east]], color="#D95F59", weight=2,
                 fill=True, fill_opacity=0.08,
                 tooltip=f"{area_km:g} km × {area_km:g} km analysis area").add_to(m)
folium.Marker([lat, lon], tooltip=f"{lat:.5f}, {lon:.5f}").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st.markdown("**Click on the map to choose the location**")
map_data = st_folium(m, height=420, use_container_width=True,
                     returned_objects=["last_clicked"], key="picker")
st.caption(f"Centre {lat:.5f}, {lon:.5f} · {grid['width']} × {grid['height']} pixels at 10 m · {grid['crs']}")

# st_folium keeps returning the last click, so only react to a new one
click = None
if map_data is not None:
    click = map_data.get("last_clicked")

if click is not None and (click["lat"], click["lng"]) != st.session_state.get("last_click"):
    st.session_state["last_click"] = (click["lat"], click["lng"])

    new_lat = round(click["lat"], 5)
    new_lon = round(click["lng"], 5)
    # if the map was dragged around the world the longitude can be > 180
    while new_lon > 180:
        new_lon -= 360
    while new_lon < -180:
        new_lon += 360

    if -80 <= new_lat <= 84:
        st.session_state["picked"] = (new_lat, new_lon)
        st.rerun()
    else:
        st.warning("Please pick a latitude between -80° and 84°.")


# ----- run the analysis -----

if run_clicked:
    bar = st.progress(0.0, text="Starting")

    def update_bar(message, fraction):
        bar.progress(min(max(fraction, 0.0), 1.0), text=message)

    project = ee_project if ee_project else None
    try:
        st.session_state["run"] = pipeline.run(lat, lon, start_month, area_km=area_km,
                                               ee_project=project, progress=update_bar)
    except Exception as e:
        bar.empty()
        if "run" in st.session_state:
            del st.session_state["run"]
        st.error("Run failed: " + str(e))
        st.stop()
    bar.empty()

if "run" not in st.session_state:
    st.info("Choose a location and start month in the sidebar, then press **Run analysis**.")
    st.stop()


# ----- results -----

run = st.session_state["run"]
res = run["results"]
summary = res["summary"]
months = run["months"]
meta = run["metadata"]
figs = run["figures"]

st.subheader(f"{meta['latitude']:.4f}, {meta['longitude']:.4f} · {months[0]} to {months[-1]} · "
             f"{meta['area_km']:g} km × {meta['area_km']:g} km")
st.caption("Sentinel-1 acquisition: " + meta["acquisition"].get("note", ""))

gain_class, gain = summary["largest_net_gain"]
loss_class, loss = summary["largest_net_loss"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Area", f"{summary['area_km2']:.2f} km²")
col2.metric("Pixels changed (first → last)", f"{summary['changed_pct_first_last']:.2f} %")
col3.metric("Largest net gain", gain_class, f"{gain:+.2f} pp")
col4.metric("Largest net loss", loss_class, f"{loss:+.2f} pp")

tab1, tab2, tab3, tab4 = st.tabs(["Monthly maps", "Class share per month", "Change (pp)", "Downloads"])

# small coloured squares as a legend
legend = ""
for c in CLASS_IDS:
    legend += (f"<span style='display:inline-block;width:12px;height:12px;"
               f"background:{CLASS_COLORS[c]};margin:0 4px 0 12px'></span>{CLASS_NAMES[c]} ")

with tab1:
    st.image(str(figs["lulc_12_months.png"]), width="stretch")

    left, right = st.columns([3, 2])
    with left:
        selected_month = st.select_slider("Inspect a month", options=months, value=months[0])
        st.image(colorize(run["maps"][selected_month]), width="stretch", caption="LULC " + selected_month)
        st.markdown(legend, unsafe_allow_html=True)
    with right:
        st.image(str(figs["lulc_12_months.gif"]), caption="12-month animation", width="stretch")

with tab2:
    st.markdown("**Share of the area covered by each class (%)**")
    st.dataframe(res["percentages"].round(2), width="stretch")
    st.image(str(figs["class_composition.png"]), width="stretch")

with tab3:
    st.markdown("**Month-to-month change in class share (percentage points)**")
    st.dataframe(res["month_to_month_pp"].round(2), width="stretch")
    st.caption("“Pixels changed (%)” = share of pixels whose predicted class differs "
               "from the previous month (gross change, not net).")
    st.image(str(figs["month_to_month_change_pp.png"]), width="stretch")

    st.markdown(f"**Total change, {months[0]} → {months[-1]} (percentage points)**")
    st.dataframe(res["total_change_pp"].round(2), width="stretch")
    st.image(str(figs["total_change_pp.png"]), width="stretch")

    st.markdown("**First vs last month and per-pixel instability**")
    st.image(str(figs["change_maps.png"]), width="stretch")

    st.markdown("**Transition matrix, first → last month (pixels)**")
    st.dataframe(res["transition_first_last"], width="stretch")

    st.warning("These are differences between model predictions. Part of the month-to-month "
               "variation is seasonal radar response (soil moisture, crop growth, snow), not "
               "real land cover change.")

with tab4:
    run_dir = Path(run["run_dir"])

    # zip everything from the run folder in memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in run_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(run_dir.parent))

    st.download_button("Download all results (GeoTIFFs, CSVs, figures)", buffer.getvalue(),
                       file_name=run_dir.name + ".zip", mime="application/zip")
    st.caption("Saved on the server in `" + str(run_dir) + "`")
