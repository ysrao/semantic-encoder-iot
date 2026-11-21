# Semantic encoder — INDIA v2 report (catchment attributes added)

**Status: TRAINED locally, 19 Nov 2026. Nothing uploaded or sent anywhere.**
v2 changes exactly one thing vs v1: the gateway decoder is conditioned on
six static per-station catchment attributes from INDOFLOODS. Encoder,
windows, splits, seed and training protocol are v1's, unchanged — the
windows were loaded from v1's own `dataset_india.npz`, so the splits are
identical by construction (train 8,735 / val 2,927 / test 918 windows,
194 test positives).

## Headline numbers — identical 2024 test split (918 windows)

| System | Accuracy | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|---|
| **v2 NN pair (attributes at decoder)** | 0.9477 | **0.9056** | 0.8402 | **0.8717** | 0.0235 |
| v1 NN pair | 0.9455 | 0.8673 | **0.8763** | **0.8718** | 0.0359 |
| Hand-crafted rule | **0.9641** | **0.9548** | 0.8711 | **0.9111** | **0.0110** |

Confusion counts: v2 — TP 163, FP 17, FN 31. v1 — TP 170, FP 26, FN 24.
Rule — TP 169, FP 8, FN 25.

**The plain reading:** the attributes did what was asked of them in one
direction only. False alarms fell by a third (26 → 17; FAR 3.6% → 2.4%,
precision 0.867 → 0.906) — but misses rose by almost exactly the offsetting
amount (24 → 31; recall 0.876 → 0.840). **F1 is unchanged: 0.8717 vs
0.8718.** The rule still beats both NN versions overall. On a single test
monsoon of 918 windows, differences of this size are also within sampling
noise; the honest claim is a precision-for-recall trade, not an
improvement. Validation agreed it was a wash: best val F1 0.8947 at
threshold 0.45 (v1: 0.8920 at 0.35).

One secondary effect in v2's favour: it degrades less under the σ=0.03
channel noise (test F1 0.8654, −0.006 vs clean; v1 fell to 0.8329, −0.039).
One against: reconstruction MSE rose (0.0062 vs v1's 0.0040).

Per station (NN F1, precision / recall):

| Station | v2 F1 (P / R) | v1 F1 (P / R) | Verdict |
|---|---|---|---|
| Jainagar | 0.8315 (0.822 / 0.841) | 0.8081 (0.727 / 0.909) | better — precision up sharply |
| Jhanjharpur | 0.8039 (0.891 / 0.732) | 0.8440 (0.868 / 0.821) | **worse — recall fell** |
| Benibad | 0.9259 (0.943 / 0.909) | 0.9159 (0.942 / 0.891) | slightly better |
| Kallooppara | 0.8182 (0.900 / 0.750) | 0.8182 (0.900 / 0.750) | identical |
| Akhuapada | — (13 windows, 0 positives) | — | uninformative, as in v1 |
| Kunavaram | 0.9811 (1.000 / 0.963) | 0.9811 (1.000 / 0.963) | identical |

So v2 helped at two stations, hurt at one, and changed nothing at two.
There is no per-station win to claim.

## What the attributes were (Task 1)

`raw/metadata_indofloods.csv` carries only four catchment-attribute
columns: **Source Catchment Area, Catchment Area, Area variation (%)** and
**Reliability** (Safe/Caution). There are **no land-use, soil or elevation
attributes in this file** — the richer per-catchment descriptors live in a
separate `catchment_characteristics_indofloods.csv` in the same Zenodo
record, which was not part of the 18 Nov 2026 local download and was not
used. What v2 actually received per station was a 6-number vector:
log₁₀(Catchment Area)/5.5, Area variation/100, Danger Level/100,
Latitude/40, Longitude/100, and a Safe-reliability flag — appended to the
8-dim latent at the decoder's first layer ((8+6) → 24 → 48). The encoder
and the 8-byte over-the-air payload are untouched; the attributes are
station metadata applied at the gateway.

Extracted values (full record: `catchment_attributes_v2.json`):

| Station | Catchment Area (km²) | Source estimate (km²) | Variation | Reliability |
|---|---|---|---|---|
| Jainagar | 2,118.6 | 2,131 | 0.58% | Safe |
| Jhanjharpur | 3,504.4 | 2,945 | 19.00% | Safe |
| Benibad | 1,793.5 | 6,160 | **70.88%** | **Caution** |
| Kallooppara | 704.2 | 731 | 3.66% | Safe |
| Akhuapada | 10,024.2 | 10,385 | 3.47% | Safe |
| Kunavaram | 279,468.0 | 284,880 | 1.90% | Safe |

Missing values: **none for the six stations** in any catchment column.
(File-wide, Source Catchment Area and Area variation are missing for 16 of
214 stations — none of them ours.) Two honesty flags: Benibad's two area
estimates disagree by 71% and INDOFLOODS itself flags that station
"Caution" — its area input is unreliable, and v2 had no way to know which
estimate to trust beyond the variation and reliability numbers it was
given. And with only six stations, six attribute vectors is a very small
amount of conditioning signal — each station is effectively its own
category, so "attributes" here are close to a learned per-station bias.

## Event cross-check (Task 2) — could not be performed

v1's counted danger events (Jainagar 37, Jhanjharpur 27, Benibad 24,
Kallooppara 15, Akhuapada 11, Kunavaram 8; GUARDIAN series, Jan 2020 –
Jul 2024, >72 h separation) **cannot be checked against the INDOFLOODS
event catalogue, because the two are disjoint in both coverage and
period** (full record: `event_crosscheck_v2.json`):

- The catalogue contains events for **Open stations only** — all 155
  gauges in `floodevents_indofloods.csv` are "Open". The three Bihar
  stations are marked "Restricted" in INDOFLOODS' own metadata and have
  **zero catalogue events at any date**.
- For the three Open stations, the catalogue still ends before the
  GUARDIAN series begins: Kallooppara's last event starts 2019-08-08
  (135 events, 1985–2019), Kunavaram's 2019-08-07 (2 events), Akhuapada's
  2018-07-23 (1 event). The catalogue as a whole ends in 2020.
- **Events in the v1 counting period (2020-01-01 → 2024-07-31): zero, for
  all six stations.**

So the INDOFLOODS catalogue neither confirms nor refutes v1's counts; it
simply does not cover the same stations in the same years. v1's counts
stand on the GUARDIAN series alone, as before. Any future cross-check
would need an event source covering 2020–2024 for the restricted Ganga-
basin gauges, which this dataset is not.

## Where v2 did NOT help — summary

1. **Overall F1 unchanged** (0.8717 vs 0.8718); the rule (0.9111) still
   leads both NN versions, still with the lowest false-alarm rate (1.1%).
2. **Recall fell** (0.876 → 0.840) — v2 misses 7 more flood windows than
   v1. For a flood-warning task that trade is not obviously the right one;
   it is a threshold-region shift, not a strict gain.
3. **Jhanjharpur got worse** (F1 0.844 → 0.804); Kallooppara and Kunavaram
   were bit-for-bit equivalent in outcome.
4. The attribute source available locally is **thin** (area + reliability
   only) and partly unreliable (Benibad). The plausible next lever — the
   full catchment-characteristics file (land use, soil, elevation) — was
   out of local scope for this run and remains untested.
5. The event-catalogue cross-check Rao asked about **does not exist as a
   possibility with this dataset** (Task 2 above) — that avenue is closed,
   not merely unhelpful.

## Files (all new, v1 untouched)

- `india_v2_analysis.py` — Tasks 1–2 extraction and cross-check.
- `catchment_attributes_v2.json`, `event_crosscheck_v2.json` — records.
- `train_semantic_india_v2.py`, `train_log_india_v2.txt`,
  `training_report_india_v2.json` — training code, log, full metrics.
- `semantic_encoder_india_v2_trained.pt` / `.onnx` — traced encoder
  (contract unchanged: (1, 64) → (1, 8); reload-verified).
- `gateway_decoder_india_v2_trained.pt` — traced v2 decoder, task head,
  input (1, 14) = [latent, attributes] (reload-verified).

No AI Hub upload was made, and the paper draft was not touched.
