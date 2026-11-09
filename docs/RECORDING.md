# Screen recording

Target: 2–4 minutes, one uninterrupted run for an urban area.

**Recording link:** _add here_

## Before recording

```bash
source .venv/bin/activate
export EE_PROJECT=ee-sentinel1-lulc
python run_lulc.py --lat 48.7758 --lon 9.1829 --start 2025-10 --area-km 2   # warm-up, checks auth
rm -rf runs/                                                               # so the recording shows a fresh download
streamlit run app.py
```

Use a 1920×1080 window, browser zoom 90 %. OBS Studio, the macOS screenshot tool
(Cmd+Shift+5) or the Windows Snipping Tool (Win+Shift+R) all work.

## Script

1. **Terminal (10 s)** – show `streamlit run app.py` starting.
2. **Sidebar (20 s)** – choose *Stuttgart – city centre* (or type 48.7758, 9.1829),
   start month *2025-10*, area *2 km*. Open *Selected area* to show the box on the map.
3. **Run (60–90 s)** – press *Run analysis*. Let the progress bar run through the
   orbit check, the 12 downloads and the predictions. Cut nothing here; this is the
   evidence of a live run.
4. **Monthly maps (30 s)** – the 12-panel figure, drag the month slider, the GIF.
5. **Class share per month (15 s)** – table and line chart.
6. **Change (pp) (30 s)** – month-to-month table, then the first-to-last table and bar
   chart. Point out the largest gain/loss metrics at the top.
7. **Downloads (10 s)** – download the zip, show the GeoTIFFs and CSVs inside.

Optional second city if time allows: *Frankfurt am Main – Ostend*.

Upload as an unlisted YouTube video or to Google Drive with link sharing on, and put
the link above and in the main README.
