# Indian River Gauge Data for Encoder Training — Findings

Date of investigation: 18 November 2026. All requests below were made from this
machine (US-hosted VM) on that date. Companion file: `DATA_PROVENANCE.md`.
Analysis output: `station_analysis.json` (produced by `analyse_stations.py`).

**Bottom line:** Real Indian gauge data for training is obtainable today — not
from the live CWC API (unreachable from this network), but from the published
GUARDIAN archive (IIT Bombay's extract of that same CWC portal, on figshare),
with official CWC danger levels joined from the INDOFLOODS database (Zenodo,
CC BY 4.0). The result: hourly water-level series for 2020 – Jul 2024 with
dozens of verified danger-level crossings at flood-prone stations in Bihar,
Kerala, Odisha and the Godavari basin. **Route 2 verdict: VIABLE.**

---

## Route 1 — CWC Flood Forecasting System API (ffs.india-water.gov.in)

**Verdict: API is real, unauthenticated and live — but NOT REACHABLE from this
network. Not usable from here; superseded by Route 2.**

What was tried (all on 18 Nov 2026, via curl and Python, with browser
User-Agent headers):

| Request | Result |
|---|---|
| `POST /iam/api/layer-station/specification/count`, header `class-name: LayerStationDto`, JSON specification body | Connection dropped after TLS handshake ("Empty reply from server", HTTP 000) |
| `GET /iam/api/new-entry-data/specification/count`, header `class-name: NewEntryDataDto` | Same — connection dropped |
| `POST /web-api/getHGStationDataForFFS/` with `{stationCode, startDate, endDate}` | Same — connection dropped |
| `GET https://ffs.india-water.gov.in/` and `https://india-water.gov.in/` | Timeout / empty reply |
| `GET https://cwc.gov.in/` | HTTP 401 (host reachable, fronted by an auth/WAF layer) |

This is a network-egress block, not an authentication wall — no login exists
for this API. An independent live check from another network (Sep 2026,
published research notes at github.com/hack-the-future-26) confirms the API
answers unauthenticated requests, returns live readings, and that
`layer-station/specification/count` responds — consistent with these Indian
government portals restricting access by geography. No account was created and
no login attempted.

History depth (the decisive question) — established from the GUARDIAN team's
published extraction code (github.com/girishpatidar/discharge_india):

- Their v1 code queries `POST /web-api/getHGStationDataForFFS/` per station
  with explicit `startDate`/`endDate` and was run for **2020-01-01 →
  2024-07-31** — i.e. the backend retains and serves multi-year history even
  though the web UI only displays 7-day windows.
- Their v2 code queries `GET /iam/api/new-entry-data/specification/sorted`
  with a specification filter: `id.stationCode eq <code>`,
  `id.datatypeCode eq "HHS"`, `id.dataTime btn <start>,<end>` — confirming
  arbitrary date-range history queries on the IAM API as well.

Station metadata: the `layer-station` resource is the portal's station layer
and is the source from which INDOFLOODS extracted the official warning/danger
levels (see Route 2), so the fields exist; they could not be read directly
from this network.

Reading codes: `HHS` is confirmed as the sub-daily water-level (WSE) datatype
used by GUARDIAN's v2 extractor. The other codes seen on the portal (`GBD`,
`GBT`, `GEP`, …) are **not publicly documented** — the Sep 2026 research note
explicitly lists mapping them as outstanding work, and no authoritative
mapping was found. Their meanings remain unverified.

Field caveats (from the same independent check, and confirmed by our own
analysis discipline): the portal contains dead stations (e.g. "KOLHAPUR",
last reading 2017) and stations with wrong coordinates (that one plots in
Telangana). Any station chosen from this portal must be validated for recent
data and sane coordinates — which is exactly what Route 2's archive let us do.

## Route 2 — GUARDIAN dataset (Scientific Data 2024) + INDOFLOODS thresholds

**Verdict: VIABLE — this is the training source. Downloaded and verified.**

### The dataset

- Patidar, Indu & Karmakar (IIT Bombay), *Scientific Data* 11:1155 (2024),
  DOI 10.1038/s41597-024-03923-8. Data on figshare,
  DOI **10.6084/m9.figshare.27004282**:
  - `data.rar` — 577,023,190 bytes; MD5 verified against figshare's published
    checksum (`6c33c700e75f9d1b16d37989a4ba37e8`). Downloaded to `raw/data.rar`,
    extracted to `raw/guardian/`.
  - `RC.xlsx` — rating-curve table for the 210 discharge stations (station
    name, lat, lon, RC coefficients, NSE/RMSE). Downloaded to `raw/RC.xlsx`.
- Contents: per-station Excel files, columns `(code, time, name, WSE)` —
  **water-level (WSE) series for 1,403 stations** and rating-curve-derived
  discharge for 203 stations, spanning **2020-01-01 → 2024-07-31** (4½ years,
  five monsoons), scraped from the CWC flood portal (Route 1's source).
- Cadence: sub-daily; the modal interval at every chosen station is **1 hour**
  (15,000–22,000 hourly steps per station), with some 5 h / 14 h steps and
  occasional gaps (worst chosen-station gap: Kunavaram, ~75 days). Workable
  for 16-step windows with gap-aware sampling.
- The dataset contains **no warning/danger levels** — RC.xlsx carries
  coordinates and rating curves only. Thresholds come from INDOFLOODS (below).
- Also mirrored via the authors' portal indiariverflow.com (reachable, HTTP
  200; a Plotly Dash app — good for human browsing, not needed since the
  figshare bulk download works).

### The thresholds — INDOFLOODS

- Kuntla & Saharia, *Bull. Amer. Meteorol. Soc.* 106(2), 2025. Data on
  Zenodo, DOI **10.5281/zenodo.14584654**, licence **CC BY 4.0**.
  `metadata_indofloods.csv` (downloaded to `raw/`) lists 214 stations with
  **official CWC Warning Level and Danger Level** (extracted by the authors
  from the CWC flood forecasting portal), river, basin, state, coordinates.
- **168 of the 214 match a GUARDIAN WSE file by station name.** Coordinate
  cross-checks between INDOFLOODS and RC.xlsx agree (Kallooppara exact;
  Bhadrachalam Δlat 0.001°, Δlon 0.003°).

### Verified flood richness (our analysis of the actual downloaded series)

Danger-level excursions counted directly on the series; an "event" is an
excursion above the danger level separated from the previous one by >72 h
below it. Full table: `station_analysis.json`.

| Station | River / Basin | State | Danger level | Max level | Danger events | Monsoons with crossings | Records |
|---|---|---|---|---|---|---|---|
| Jainagar | Kamla-Balan (Kosi) | Bihar | 67.75 m | 69.58 m | **37** | 2020–2024 (all 5) | 19,578 |
| Jhanjharpur | Kamla-Balan (Kosi) | Bihar | 50.00 m | 52.81 m | **27** | 2020–2024 (all 5) | 19,113 |
| Benibad | Bagmati | Bihar | 48.68 m | 49.99 m | **24** | 2020–2024 (all 5) | 19,131 |
| Kallooppara | Manimala (Pampa) | Kerala | 6.00 m | 9.12 m | **15** | 2020–2024 (all 5) | 24,128 |
| Akhuapada | Baitarani | Odisha | 17.83 m | 20.06 m | **11** | 2020–2023 | 18,088 |
| Kunavaram | Godavari | Andhra Pradesh | 39.24 m | 48.87 m | **8** | 2020–2024 (all 5) | 18,127 |
| Bhadrachalam | Godavari | Telangana | 48.77 m | — | 5 | — | 18,487 |

(168 matched stations were analysed; the table shows the strongest. The
chosen six are copied into `stations/`.)

### Caveats that must travel with the data

- **Outliers exist in the raw scraped series** (e.g. Hathnur max 2143 m
  against a 214 m danger level; Burhanpur 2169 m against 221 m) — spike/datum
  cleaning is required before training. The six chosen stations show no such
  pathology (maxima are 0.7–9.6 m above danger level, i.e. plausible floods).
- Akhuapada's INDOFLOODS row repeats its danger level in the warning-level
  field (17.83 m for both) — use the danger level only for that station.
- Some INDOFLOODS Ganga-basin metadata rows are flagged "Restricted" in their
  own file (their *discharge* records are request-only). The thresholds
  themselves are published in the open CC-BY metadata, and every level series
  used here comes from GUARDIAN, not from INDOFLOODS — but the Bihar stations'
  basin (Ganga) is a *classified* basin under CWC's dissemination policy,
  which is worth knowing if this work is ever published with the data
  redistributed.
- Licence position: the figshare record sets no explicit licence; the paper
  is CC BY-NC-ND 4.0 and the underlying measurements are CWC (Government of
  India) telemetry republished by IIT Bombay for research. INDOFLOODS is
  CC BY 4.0. Training/research use with citation is the intended use;
  commercial redistribution would need a fresh look at CWC's Hydrometeorological
  Data Dissemination Policy.
- **Channel note for the encoder contract:** GUARDIAN supplies the *level*
  channel and INDOFLOODS the *threshold* (danger level) channel. The
  *rainfall* and *temperature* channels of the existing 16×4 contract are not
  in either dataset; for Indian stations they can be paired from a weather
  archive at the station coordinates (e.g. the free Open-Meteo archive API,
  verified reachable below) — a decision for the training step, not a blocker
  for the flood series itself.

## Route 3 — India-WRIS (indiawris.gov.in)

**Verdict: NOT VIABLE as the primary programmatic source (not pursued beyond
verification, per plan — Routes 1–2 were decisive).**

- Documented scope (GUARDIAN paper): daily historical WSE and discharge for
  **unclassified rivers only, up to 2020**, with significant gaps.
  Transboundary basins (Ganga, Brahmaputra, Indus) are request-only
  (INDOFLOODS). Bulk downloads are restricted and the data "requires
  extensive curation" (INDOFLOODS).
- The host itself is unreachable from this network (timeouts identical to
  Route 1). No login was attempted and no account created.
- Its historical daily data is in any case inside GUARDIAN's lineage (the
  rating curves were built from it); for training, Route 2's sub-daily
  2020–2024 series supersede it.

## Route 4 — Open-Meteo Flood API (GloFAS) — fallback, not needed

**Verdict: works, but MODELLED data — fallback only. Not used.**

- Verified live: `GET https://flood-api.open-meteo.com/v1/flood?latitude=26.58
  &longitude=86.15&daily=river_discharge&start_date=2024-07-01&end_date=
  2024-07-10` returned daily discharge for the grid point nearest Jainagar.
- It is GloFAS *modelled* discharge: daily, no gauge water level, no official
  CWC thresholds. It cannot substitute for gauge data in a claim of "trained
  on real Indian river data". Recorded here only so the fallback is known-good
  if the gauge route ever breaks.

---

## Recommendation

Train the flood case on **GUARDIAN WSE + INDOFLOODS danger levels**, starting
with Jainagar, Jhanjharpur, Benibad (Bihar), Kallooppara (Kerala), Akhuapada
(Odisha) and Kunavaram (Godavari): real CWC gauge measurements, official
documented thresholds, five monsoons, and 8–37 distinct danger-level events
per station — a stronger and more honest basis than the USGS series for a
paper set in India, and it keeps the agriculture/flood setting of the paper
intact. The live FFS API remains the route for *future* data refreshes, but
only from a network that can reach india-water.gov.in.
