"""India v2, tasks 1-2: INDOFLOODS catchment attributes + event cross-check.

Task 1: from raw/metadata_indofloods.csv, extract the catchment attributes
for the six v1 stations and build the static per-station attribute vector
used by the v2 decoder. Written to catchment_attributes_v2.json.

Task 2: cross-check v1's counted danger events (counted on the GUARDIAN
series, 2020-01-01 to 2024-07-31, >72 h-below separation rule) against
raw/floodevents_indofloods.csv. Written to event_crosscheck_v2.json.

Local analysis only; reads the already-downloaded raw files.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# v1 station order == station indices in dataset_india.npz
STATIONS = ["Jainagar", "Jhanjharpur", "Benibad",
            "Kallooppara", "Akhuapada", "Kunavaram"]
V1_EVENTS = {"Jainagar": 37, "Jhanjharpur": 27, "Benibad": 24,
             "Kallooppara": 15, "Akhuapada": 11, "Kunavaram": 8}
PERIOD = ("2020-01-01", "2024-07-31")


def main():
    meta = pd.read_csv(os.path.join(HERE, "raw", "metadata_indofloods.csv"))
    meta["Station_clean"] = meta["Station"].str.strip().str.lower()

    catchment_cols = ["Source Catchment Area", "Catchment Area",
                      "Area variation (%)", "Reliability"]
    print("catchment-attribute columns in metadata_indofloods.csv:",
          catchment_cols)
    print("file-wide missing: Catchment Area NaN =",
          int(meta["Catchment Area"].isna().sum()),
          "| Source Catchment Area NaN =",
          int(meta["Source Catchment Area"].isna().sum()),
          "| Area variation NaN =",
          int(meta["Area variation (%)"].isna().sum()),
          "| Reliability values =",
          meta["Reliability"].value_counts().to_dict())

    attrs, raw_rows = {}, {}
    for name in STATIONS:
        row = meta[meta["Station_clean"] == name.lower()]
        assert len(row) == 1, (name, len(row))
        r = row.iloc[0]
        ca = float(r["Catchment Area"])
        vec = [
            float(np.log10(ca) / 5.5),          # catchment size (log scale)
            float(r["Area variation (%)"] / 100.0),  # source-vs-DEM area disagreement
            float(r["Danger Level"] / 100.0),   # absolute danger level, m
            float(r["Latitude"] / 40.0),
            float(r["Longitude"] / 100.0),
            1.0 if str(r["Reliability"]).strip() == "Safe" else 0.0,
        ]
        attrs[name] = vec
        raw_rows[name] = {
            "gauge_id": r["GaugeID"],
            "source_catchment_area_km2": float(r["Source Catchment Area"]),
            "catchment_area_km2": ca,
            "area_variation_pct": float(r["Area variation (%)"]),
            "reliability": str(r["Reliability"]),
            "privacy": str(r["Privacy"]),
            "warning_level_m": float(r["Warning Level"]),
            "danger_level_m": float(r["Danger Level"]),
            "latitude": float(r["Latitude"]), "longitude": float(r["Longitude"]),
            "basin": str(r["Basin"]), "state": str(r["State"]),
            "metadata_record_span": [str(r["Start_date"]), str(r["End_date"])],
            "missing_catchment_values": [
                c for c in catchment_cols if pd.isna(r[c])],
            "attribute_vector_v2": vec,
        }
        print(f"{name}: CA={ca:.1f} km2 (source {float(r['Source Catchment Area']):.0f}), "
              f"variation {float(r['Area variation (%)']):.2f}%, "
              f"reliability {r['Reliability']}, privacy {r['Privacy']}, "
              f"missing={raw_rows[name]['missing_catchment_values']}")

    with open(os.path.join(HERE, "catchment_attributes_v2.json"), "w") as f:
        json.dump({
            "source": "raw/metadata_indofloods.csv (INDOFLOODS, Zenodo DOI "
                      "10.5281/zenodo.14584654, CC BY 4.0)",
            "catchment_columns_identified": catchment_cols,
            "note": ("The metadata file's catchment content is limited to "
                     "catchment area (two estimates), their % variation and "
                     "a reliability flag; it carries no land-use, soil or "
                     "elevation attributes. The Zenodo record also lists a "
                     "separate catchment_characteristics_indofloods.csv, "
                     "which was not part of the 18 Nov 2026 local download "
                     "and was not used here."),
            "attribute_vector_definition": [
                "log10(Catchment Area km2) / 5.5",
                "Area variation (%) / 100",
                "Danger Level (m) / 100",
                "Latitude / 40",
                "Longitude / 100",
                "1.0 if Reliability == 'Safe' else 0.0",
            ],
            "station_order_matches_dataset_india_npz": STATIONS,
            "raw": raw_rows,
            "vectors": attrs,
        }, f, indent=2)

    # ---- task 2: event cross-check ----
    ev = pd.read_csv(os.path.join(HERE, "raw", "floodevents_indofloods.csv"))
    ev["Start Date"] = pd.to_datetime(ev["Start Date"])
    ev["gauge"] = ev["EventID"].str.rsplit("-", n=1).str[0]
    gid2priv = dict(zip(meta["GaugeID"], meta["Privacy"]))
    priv_counts = {}
    for g in ev["gauge"].unique():
        priv_counts[gid2priv.get(g, "?")] = priv_counts.get(gid2priv.get(g, "?"), 0) + 1

    check = {"v1_counting_period": list(PERIOD),
             "v1_rule": "excursion above danger level on the GUARDIAN hourly "
                        "series, separated by >72 h below danger",
             "catalogue_gauges_by_privacy": priv_counts,
             "catalogue_event_years": [int(ev["Start Date"].dt.year.min()),
                                       int(ev["Start Date"].dt.year.max())],
             "per_station": {}}
    for name in STATIONS:
        gid = raw_rows[name]["gauge_id"]
        e = ev[ev["gauge"] == gid]
        in_period = e[(e["Start Date"] >= PERIOD[0]) &
                      (e["Start Date"] <= PERIOD[1])]
        check["per_station"][name] = {
            "gauge_id": gid, "privacy": raw_rows[name]["privacy"],
            "v1_counted_events_2020_2024": V1_EVENTS[name],
            "indofloods_events_total": int(len(e)),
            "indofloods_events_in_v1_period": int(len(in_period)),
            "indofloods_last_event_start":
                (str(e["Start Date"].max().date()) if len(e) else None),
        }
        print(f"{name} ({gid}, {raw_rows[name]['privacy']}): v1 counted "
              f"{V1_EVENTS[name]}; INDOFLOODS total {len(e)}, in-period "
              f"{len(in_period)}, last {check['per_station'][name]['indofloods_last_event_start']}")
    with open(os.path.join(HERE, "event_crosscheck_v2.json"), "w") as f:
        json.dump(check, f, indent=2)
    print("wrote catchment_attributes_v2.json and event_crosscheck_v2.json")


if __name__ == "__main__":
    main()
