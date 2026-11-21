#!/usr/bin/env python
"""Cross-match GUARDIAN WSE files with INDOFLOODS CWC warning/danger levels and
quantify flood (danger-level) crossings per station. Read-only analysis."""
import csv, glob, os, re, json
import openpyxl
from datetime import datetime, timedelta

BASE = os.path.expanduser("~/workspace/edge-apps-prep/semantic_encoder/india/raw")
WSE_DIR = os.path.join(BASE, "guardian/data/wse")

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

indo = {}
with open(os.path.join(BASE, "metadata_indofloods.csv")) as fh:
    for r in csv.DictReader(fh):
        indo[norm(r["Station"])] = r

# RC.xlsx coordinates for cross-check
rc = {}
wb = openpyxl.load_workbook(os.path.join(BASE, "RC.xlsx"), read_only=True)
for row in list(wb.active.iter_rows(values_only=True))[1:]:
    if row[19]:
        rc[norm(row[19])] = (row[20], row[21])

results = []
files = sorted(glob.glob(os.path.join(WSE_DIR, "*.xlsx")))
for f in files:
    stem = os.path.basename(f).replace("2020_2024_July.xlsx", "")
    key = norm(stem)
    meta = indo.get(key)
    if not meta:
        continue
    try:
        dl = float(meta["Danger Level"]); wl = float(meta["Warning Level"])
    except (TypeError, ValueError):
        continue
    try:
        wbk = openpyxl.load_workbook(f, read_only=True)
        rows = list(wbk.active.iter_rows(values_only=True))
    except Exception as e:
        print("LOADFAIL", stem, e); continue
    series = [(r[2], r[4]) for r in rows[1:] if isinstance(r[2], datetime) and isinstance(r[4], (int, float))]
    if len(series) < 500:
        continue
    series.sort()
    times = [t for t, _ in series]; vals = [v for _, v in series]
    # distinct danger-crossing events: excursions above DL, new event if gap below DL > 72h
    events, seasons = 0, set()
    above_start = None; last_below = None
    prev_above = False
    for t, v in series:
        if v >= dl:
            if not prev_above:
                if above_start is None or (t - last_below) > timedelta(hours=72) if last_below else True:
                    events += 1
                above_start = t
            prev_above = True
            if t.month in (6, 7, 8, 9):
                seasons.add(t.year)
        else:
            if prev_above:
                last_below = t
            prev_above = False
    n_above_dl = sum(1 for v in vals if v >= dl)
    n_above_wl = sum(1 for v in vals if v >= wl)
    coord_note = ""
    if key in rc:
        dlat = abs(rc[key][0] - float(meta["Latitude"])); dlon = abs(rc[key][1] - float(meta["Longitude"]))
        coord_note = f"RC-vs-INDOFLOODS dLat={dlat:.3f} dLon={dlon:.3f}"
    results.append({
        "station": meta["Station"], "file": os.path.basename(f),
        "river": meta["River Name/ Tributory/ SubTributory"], "basin": meta["Basin"],
        "state": meta["State"], "lat": float(meta["Latitude"]), "lon": float(meta["Longitude"]),
        "warning_level_m": wl, "danger_level_m": dl,
        "n_records": len(series), "start": str(times[0]), "end": str(times[-1]),
        "max_wse_m": max(vals), "n_above_danger": n_above_dl, "n_above_warning": n_above_wl,
        "danger_events": events, "monsoon_seasons_with_danger": sorted(seasons),
        "coord_check": coord_note,
    })

results.sort(key=lambda r: (-r["danger_events"], -r["n_above_danger"]))
out = os.path.join(BASE, "..", "station_analysis.json")
with open(out, "w") as fh:
    json.dump(results, fh, indent=1)
print(f"matched stations analysed: {len(results)}")
print(f"{'station':28s} {'state':16s} {'DL':>7s} {'max':>7s} {'events':>6s} {'seasons':>18s} {'records':>8s}")
for r in results[:30]:
    print(f"{r['station'][:28]:28s} {r['state'][:16]:16s} {r['danger_level_m']:7.2f} {r['max_wse_m']:7.2f} "
          f"{r['danger_events']:6d} {str(r['monsoon_seasons_with_danger']):>18s} {r['n_records']:8d}")
