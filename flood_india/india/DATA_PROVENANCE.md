# Data Provenance — Indian River Gauge Series for Semantic-Encoder Training

Retrieved: **18 November 2026**. All files under
`~/workspace/edge-apps-prep/semantic_encoder/india/`.

## Sources

### 1. Water-level series — GUARDIAN (CWC telemetry, via IIT Bombay)

- Patidar, G., Indu, J. & Karmakar, S. (2024). *ExtendinG SUb-DAily River
  Discharge data over INdia (GUARDIAN).* Scientific Data 11:1155.
  DOI: 10.1038/s41597-024-03923-8.
- Dataset: figshare, DOI **10.6084/m9.figshare.27004282** (v1, 2024-10-19).
  - `data.rar` — 577,023,190 bytes, MD5 `6c33c700e75f9d1b16d37989a4ba37e8`
    (verified against figshare's published checksum after download).
    Download URL used: https://ndownloader.figshare.com/files/49150774
  - `RC.xlsx` — 53,652 bytes (rating curves + station coordinates).
    Download URL used: https://ndownloader.figshare.com/files/49150798
- What the data is: water surface elevation (WSE, metres) recorded by Central
  Water Commission (CWC) telemetry gauges and published on the CWC flood
  portal (ffs.india-water.gov.in), extracted by the GUARDIAN authors'
  web-scraping framework (code: github.com/girishpatidar/discharge_india).
  Per-station Excel files with columns `(code, time, name, WSE)`.
- Coverage as downloaded: **1,403 station WSE files** + 203 station discharge
  files (rating-curve-derived), each spanning **2020-01-01 → 2024-07-31**.
  Full extraction preserved in `raw/guardian/`; the original archive is
  `raw/data.rar`, unmodified.
- Cadence: sub-daily, predominantly hourly (modal interval 1 h at every
  station below; secondary intervals 5 h / 14 h; occasional gaps — see notes).

### 2. Flood thresholds — INDOFLOODS (official CWC levels)

- Kuntla, S. K. & Saharia, M. (2025). *INDOFLOODS: A Comprehensive Database
  for Flood Events in India Enhanced with Catchment Attributes.* Bulletin of
  the American Meteorological Society 106(2): E333–E343.
  DOI: 10.1175/BAMS-D-24-0008.1.
- Dataset: Zenodo, DOI **10.5281/zenodo.14584654** (record 14584655,
  v1.0, 2025-01-24), licence **CC BY 4.0**.
  - `metadata_indofloods.csv` (40,058 bytes) — 214 stations with official
    CWC **Warning Level** and **Danger Level**, extracted by the authors from
    the CWC flood forecasting portal; thresholds are set by CWC in
    consultation with state governments.
  - `floodevents_indofloods.csv` (469,728 bytes) — their flood-event catalogue
    (downloaded for reference; not used in the counts below).
  - Both preserved in `raw/`.

### 3. Cross-match and verification (done here, 18 Nov 2026)

- 168 of INDOFLOODS' 214 stations match a GUARDIAN WSE file by name.
- Coordinates cross-checked between INDOFLOODS metadata and GUARDIAN's
  RC.xlsx where both exist: agreement exact to ≤0.003° for checked stations.
- Danger-level excursions counted directly on the downloaded series by
  `analyse_stations.py`; full results in `station_analysis.json`
  (168 stations). An "event" = excursion above danger level separated from
  the previous excursion by >72 h below it.

## Chosen stations (copied to `stations/`)

All series: 2020-01-01 → 2024-07-31, WSE in metres, timestamps IST as served
by the CWC portal.

| Station (file) | River | Basin | State | Lat | Lon | Warning | Danger | Records | Max WSE | Readings ≥ danger | Danger events (monsoons) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jainagar (`Jainagar2020_2024_July.xlsx`) | Kamla-Balan | Kosi / Ganga | Bihar | 26.583 | 86.148 | 66.75 m | **67.75 m** | 19,578 | 69.58 m | 3,105 | 37 (2020, 2021, 2022, 2023, 2024) |
| Jhanjharpur (`Jhanjharpur2020_2024_July.xlsx`) | Kamla-Balan | Kosi / Ganga | Bihar | 26.233 | 86.259 | 49.00 m | **50.00 m** | 19,113 | 52.81 m | 3,640 | 27 (2020–2024) |
| Benibad (`Benibad2020_2024_July.xlsx`) | Bagmati | Ganga | Bihar | 26.145 | 85.692 | 47.68 m | **48.68 m** | 19,131 | 49.99 m | 6,266 | 24 (2020–2024) |
| Kallooppara (`KALLOOPPARA2020_2024_July.xlsx`) | Manimala | Pampa / West-flowing | Kerala | 9.404 | 76.650 | 5.00 m | **6.00 m** | 24,128 | 9.12 m | 597 | 15 (2020–2024) |
| Akhuapada (`Akhuapada2020_2024_July.xlsx`) | Baitarani | Brahmani–Baitarani | Odisha | 20.916 | 86.275 | 17.83 m* | **17.83 m** | 18,088 | 20.06 m | 521 | 11 (2020–2023) |
| Kunavaram (`Kunavaram2020_2024_July.xlsx`) | Godavari | Godavari | Andhra Pradesh | 17.573 | 81.250 | 37.74 m | **39.24 m** | 18,127 | 48.87 m | 1,129 | 8 (2020–2024) |

\* Akhuapada's INDOFLOODS row repeats the danger level in the warning-level
field; only the danger level should be used for that station.

Kallooppara also has a rating-curve discharge file
(`stations/discharge_KALLOOPPARA2020_2024_July.xlsx`); the other five chosen
stations are WSE-only in GUARDIAN (no validated rating curve published).

## Known data-quality notes

- Cadence is irregular sub-daily: predominantly 1 h steps, some 5 h / 14 h
  steps; largest gaps — Kunavaram ≈ 75 days, Jainagar ≈ 36 days,
  Akhuapada ≈ 32 days, Kallooppara ≈ 27 days, Jhanjharpur ≈ 3 days.
  Window construction for training must be gap-aware.
- The wider GUARDIAN archive contains occasional gross outliers at some
  stations (e.g. Hathnur, Burhanpur — maxima ~10× their danger levels).
  None of the six chosen stations shows this pathology (maxima exceed danger
  level by 0.7–9.6 m, consistent with real floods), but spike cleaning is
  still advised before training.
- Thresholds are the CWC values as published by INDOFLOODS (2025); CWC
  revises warning/danger levels from time to time, so cite them as
  "CWC levels via INDOFLOODS (2025)".

## Files in this folder

- `INDIA_DATA_FINDINGS.md` — route-by-route investigation and verdicts.
- `DATA_PROVENANCE.md` — this file.
- `analyse_stations.py`, `station_analysis.json` — the cross-match and
  flood-crossing analysis (168 stations).
- `stations/` — WSE series for the six chosen stations (+ Kallooppara
  discharge), copied byte-for-byte from the GUARDIAN extraction.
- `raw/` — untouched downloads: `data.rar`, `guardian/` (full extraction:
  1,403 WSE + 203 discharge station files), `RC.xlsx`,
  `metadata_indofloods.csv`, `floodevents_indofloods.csv`, `zenodo_record.json`,
  `rar_listing.txt`.

## Citation requirements

- GUARDIAN: Patidar et al. (2024), Sci Data 11:1155,
  DOI 10.1038/s41597-024-03923-8; dataset DOI 10.6084/m9.figshare.27004282.
  Underlying measurements: Central Water Commission, Ministry of Jal Shakti,
  Government of India.
- INDOFLOODS: Kuntla & Saharia (2025), BAMS 106(2):E333–E343,
  DOI 10.1175/BAMS-D-24-0008.1; dataset DOI 10.5281/zenodo.14584654,
  CC BY 4.0.
