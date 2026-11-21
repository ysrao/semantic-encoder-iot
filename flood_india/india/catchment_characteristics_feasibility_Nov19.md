# INDOFLOODS catchment_characteristics — feasibility check for a v3 run

**Checked: 19 Nov 2026 (PT), proactively, after the India v2 report named this
file as "the plausible next lever". Local only — nothing uploaded or sent.**

## What was fetched

From the same Zenodo record Rao was looking at tonight
(record 14584655, DOI 10.5281/zenodo.14584654), the two files that were NOT
part of the 18 Nov 2026 download:

- `raw/catchment_characteristics_indofloods.csv` — 146,653 bytes, exactly the
  size listed in the record's own file list (`raw/zenodo_record.json`).
- `raw/precipitation_variables_indofloods.csv` — 677,548 bytes, likewise exact.

## What the characteristics file actually is

- 155 gauge rows × 108 columns — it does **not** cover all 214 metadata
  stations.
- Content is morphometry and context, not richer versions of what v2 used:
  stream-order counts/lengths, relief, drainage density, ruggedness,
  bifurcation ratios; WorldClim-style climate (annual precipitation,
  temperature seasonality, Köppen-Geiger type); GDP/HDI/night-light/population
  series; and categorical Land cover, Soil type and lithology type.
  There is no elevation time series and no per-station land-use fractions.

## The decisive problem: our Bihar stations are absent

| Station | Gauge ID | In characteristics file? |
|---|---|---|
| Jainagar | INDOFLOODS-gauge-118 | **No** |
| Jhanjharpur | INDOFLOODS-gauge-119 | **No** |
| Benibad | INDOFLOODS-gauge-115 | **No** |
| Kallooppara | INDOFLOODS-gauge-427 | Yes (higher-order stream fields blank — stream order 3) |
| Akhuapada | INDOFLOODS-gauge-694 | Yes |
| Kunavaram | INDOFLOODS-gauge-31 | Yes |

The three Bihar stations — the ones where v2 moved (Jainagar improved,
Jhanjharpur worsened) and the showcase animation station — are exactly the
"Restricted" stations, and this file follows the same restriction pattern as
the event catalogue. A v3 conditioned on these attributes could cover only
3 of the 6 pooled stations; the other half would need imputed or
missing-flag inputs, which with six stations is not a clean comparison
against v1/v2 or the rule (F1 0.911).

## Precipitation variables file

Per-catalogue-event rows (4,548 events; T1d–T10d antecedent precipitation).
Event counts for our gauges: Jainagar 0, Jhanjharpur 0, Benibad 0,
Kallooppara 135, Akhuapada 1, Kunavaram 2. Same disjointness the v2 report
found in the event catalogue — it cannot label or enrich the 2020–24
GUARDIAN windows for the Bihar stations.

## Bottom line

The "richer catchment file" lever is largely **not there for the stations
that matter**. Options if Rao wants to chase granularity further:

1. **Stop at v2** — the evidence (v1 = v2 F1 0.872, rule 0.911) stands, and
   the paper draft already reports it honestly.
2. **3-station v3** — condition only Kallooppara / Akhuapada / Kunavaram on
   the morphometric + climate attributes; a narrower, weaker claim.
3. **Different granularity source** — e.g. derive catchment attributes for
   the Bihar basins from open DEM / land-cover data ourselves (more work,
   but covers all six stations uniformly).

Recommendation: option 1 unless Rao wants the DEM route (3) for its own
sake; option 2's partial coverage would muddy the paper's honest headline.
