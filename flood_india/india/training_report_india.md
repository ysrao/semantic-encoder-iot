# Semantic encoder — INDIA training report (real CWC river-gauge data)

**Status: TRAINED on real data, 18 Nov 2026.** Same encoder + gateway
decoder and the identical training recipe as the simulation run
(`../training_report.md`); only the data changed — from the study's
synthetic traces to **real Central Water Commission (CWC) river-gauge
measurements in India**. This model is the companion artefact to the
paper "Taskoriented Semantic Communications" (Zenodo DOI
**10.5281/zenodo.23220770**). The flood case only is real-data here; the
agriculture case remains simulation-based (no equivalent open Indian
soil-moisture gauge archive was used).

## Data provenance

- **Water level:** GUARDIAN archive — CWC telemetry published on the CWC
  flood portal and extracted by Patidar, Indu & Karmakar (IIT Bombay),
  *Scientific Data* 11:1155 (2024). Dataset: figshare DOI
  **10.6084/m9.figshare.27004282** (MD5-verified download; see
  `DATA_PROVENANCE.md`).
- **Thresholds:** official CWC **danger levels** from INDOFLOODS
  (Kuntla & Saharia, *BAMS* 106(2), 2025), Zenodo DOI
  **10.5281/zenodo.14584654**, CC BY 4.0. Akhuapada's INDOFLOODS warning
  level duplicates its danger level (17.83 m) — danger level only is
  used, for every station.
- **Rainfall / temperature channels:** Open-Meteo archive API (ERA5),
  hourly at each station's coordinates, timezone Asia/Kolkata,
  2020-01-01 → 2024-07-31, fetched 2026-11-18 by
  `fetch_weather_india.py`, cached verbatim as `weather_<station>.json`
  (40,176 hours per station, no missing values). **No station needed the
  fallback substitution** (level rate-of-change as channel 3) — the
  weather fetch succeeded for all six stations.
- Series span: **2020-01-01 → 2024-07-31**, timestamps IST.

| Station | River | State | Danger | Records | Spikes removed | Valid hours / 40,176 | Latched episodes |
|---|---|---|---|---|---|---|---|
| Jainagar | Kamla-Balan | Bihar | 67.75 m | 19,578 | 2 | 30,754 | 41 |
| Jhanjharpur | Kamla-Balan | Bihar | 50.00 m | 19,113 | 5 | 30,962 | 34 |
| Benibad | Bagmati | Bihar | 48.68 m | 19,131 | 2 | 31,053 | 28 |
| Kallooppara | Manimala | Kerala | 6.00 m | 24,128 | 2 | 32,376 | 18 |
| Akhuapada | Baitarani | Odisha | 17.83 m | 18,088 | 1 | 28,862 | 12 |
| Kunavaram | Godavari | Andhra Pradesh | 39.24 m | 18,127 | 4 | 27,990 | 7 |

## Cleaning and windowing (documented rules)

- **Spike rule** (applied to the recorded series in time order): a
  maximal run of ≤2 consecutive readings is a telemetry/datum glitch if
  the readings immediately before and after the run agree within 0.5 m
  and every reading in the run deviates from the flank median by
  > 2.0 m; flagged readings are set to missing. This is exactly the
  pathology in the raw archive — isolated dips/spikes of 2.25–20 m
  against locally stable levels (e.g. Jainagar 46.40 m amid 66.40 m;
  Akhuapada 1.22 m amid 16.22 m). A genuine flood rise cannot return to
  the flank level within 2 readings, so real events are untouched.
  **16 readings removed across the six stations** (table above).
- **Grid:** hourly (40,176 steps). Missing runs of ≤4 h linearly
  interpolated; longer gaps stay missing. A window is used only if all
  20 timesteps it touches (16 window + 4 label horizon) are present —
  this is why only ~28–32k of 40k hours yield windows (dry-season
  reporting is 3×/day at these gauges, and there are long outages,
  e.g. Kunavaram ≈75 days).
- **Windows:** causal, length 16, stride 8; channels per timestep
  `[level / danger, 1.0 (threshold), clip(rain_mm_h/20, 0, 3),
  temp_C/50]` — the encoder's existing (1, 64) contract and the
  simulation run's normalisation.
- **Label:** the study's ground-truth latch (`gen_env_trace` semantics)
  applied to the cleaned observed level: an episode starts after
  **persist = 2** consecutive hourly readings above the danger level
  (onset backfilled to the first above reading) and ends when the level
  recedes below **danger − 0.3 m**; latch state resets across data
  gaps. A window ending at *i* is positive if the latched flood state
  is active at any of *i+1 … i+4* — a flood event within the next
  4 timesteps.
- **Rule baseline** on identical windows: the study's own
  `stage3.rule_above` (persist 2, hysteresis 0.3 m, causal smoothing 6
  — the env task parameters), threshold = station danger level,
  computed per contiguous observed segment, read at the window's final
  timestep. Causal — same information as the NN.
- **Splits by window-end time, stations pooled:** train = ends in
  2020–2022, validation = ends in 2023, test = ends in 2024 (Jan–Jul).

| Split | Windows | Positives |
|---|---|---|
| Train (2020–2022) | 8,735 | 2,068 |
| Validation (2023) | 2,927 | 318 |
| Test (2024 Jan–Jul) | 918 | 194 |

Per-station test counts: Jainagar 180 (44 pos), Jhanjharpur 180 (56),
Benibad 180 (55), Kallooppara 184 (12), Akhuapada **13 (0)**,
Kunavaram 181 (27). Akhuapada's 2024 coverage is very sparse (only 13
valid windows, no flood episode in its observed 2024 record) — its
per-station test row is uninformative and is reported for completeness.

## Model and training

Identical to the simulation run: `SemanticEncoder` MLP 64 → 48 → 24 →
8-dim tanh latent (4,496 params, architecture unchanged) + gateway
decoder 8 → 24 → 48 with task head and auxiliary reconstruction head,
trained jointly. Int8 quantisation **in the loop** via straight-through
estimator; Gaussian channel noise σ = 0.03 on the quantised latent
during training; loss = pos-weighted BCE + 0.05 × MSE(recon); Adam
lr 1e-3, batch 256, early stop on validation F1 (best val F1 **0.8920**
at epoch 16; decision threshold **0.35** selected on validation).
Training script: `train_semantic_india.py`; log: `train_log_india.txt`.

## Test results — NN pair vs the study's rule, identical windows

| System | Accuracy | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|---|
| **NN pair** | 0.9455 | 0.8673 | 0.8763 | **0.8718** | 0.0359 |
| NN pair, noisy latent (σ=0.03) | 0.9270 | 0.8068 | 0.8608 | 0.8329 | 0.0552 |
| Rule-based | 0.9641 | 0.9548 | 0.8711 | **0.9111** | 0.0110 |

Per station (NN pair vs rule):

| Station | NN F1 (P / R) | Rule F1 (P / R) |
|---|---|---|
| Jainagar | 0.8081 (0.727 / 0.909) | 0.8966 (0.907 / 0.886) |
| Jhanjharpur | 0.8440 (0.868 / 0.821) | 0.8627 (0.957 / 0.786) |
| Benibad | 0.9159 (0.942 / 0.891) | 0.9533 (0.981 / 0.927) |
| Kallooppara | 0.8182 (0.900 / 0.750) | 0.8182 (0.900 / 0.750) |
| Akhuapada | — (13 windows, 0 pos) | — |
| Kunavaram | 0.9811 (1.000 / 0.963) | 0.9811 (1.000 / 0.963) |

**Honest reading:** on real gauge data the study's rule is the stronger
of the two overall (F1 0.911 vs 0.872) — driven by precision: the rule
false-alarms on 1.1% of negative windows vs the NN's 3.6%, at
essentially equal recall (0.871 vs 0.876). The NN matches the rule
exactly at Kunavaram and Kallooppara and has the better recall at
Jainagar (0.909 vs 0.886), but it does **not** beat the rule on this
data, and the noisy-latent degradation is larger than in simulation
(F1 −0.039 vs ≤0.001). The test split is a single monsoon season
(918 windows), so these figures carry real sampling uncertainty. The
result mirrors the simulation finding (parity, rule slightly ahead on
the flood task) with a wider gap — the honest claim is that the
8-byte-latent NN pair reproduces rule-class flood-warning performance
on real CWC data at 32× smaller payload, not that it surpasses the
rule. Reconstruction (auxiliary head): MSE 0.0040 on normalised test
inputs.

## Payload

Raw window 64 float32 = **256 bytes**; NN latent 8 int8 = **8 bytes**
(32× smaller), inside the study's 20-byte semantic-report budget.

## AI Hub (Phase 3)

The trained India encoder was submitted to Qualcomm AI Hub as
**semantic_encoder_india_cwc** — described as the companion artefact to
Zenodo DOI 10.5281/zenodo.23220770 — compiled to TFLite and profiled on
a Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon v75 NPU). Both jobs
**SUCCESS**:

- Compile job `j5ml7l67g` → TFLite model `mqv4z002x`
  (https://workbench.aihub.qualcomm.com/jobs/j5ml7l67g/)
- Profile job `jpyylyj0p`
  (https://workbench.aihub.qualcomm.com/jobs/jpyylyj0p/):
  **estimated inference 42 µs**, median measured **56 µs** (100 runs),
  first load 348 ms, warm load 163 ms, estimated inference peak memory
  119,320,576 bytes (~114 MiB — the TFLite/NPU runtime footprint, as
  with the other builds).

Full record: `~/workspace/edge-apps-prep/aihub_results/trained_semantic_india.json`;
runner: `aihub_results/run_trained_semantic_india.py`.

## Animation window (Phase 4)

`animation_window_india.json`: **Jainagar (Kamla-Balan, Bihar), test
split** — the first-choice station and split. The 60-point segment
(18 Jun 19:00 → 21 Jun 06:00 IST, fully observed) shows the level
holding at 67.45 m, then a steep sustained climb — 67.65 → 67.90 →
68.25 → 68.50 → 68.60 m — crossing the 67.75 m danger level at
**2024-06-20 12:00 IST**, peaking at 68.60 m and receding slowly. The
16-step window ends 2024-06-20 15:00 IST, three steps after the
crossing (rise across the window +1.15 m); the encoder's actual int8
latent is [−72, 13, 10, 7, 25, −43, 16, −96], the decoder probability
**0.6614** vs the 0.35 threshold → prediction 1, matching the ground
truth (flood episode active in the following timesteps).

## Artifacts (this folder)

- `semantic_encoder_india_trained.pt` — traced encoder only,
  (1, 64) → (1, 8); reload-verified in a fresh process.
- `semantic_encoder_india_trained.onnx` (+ `.onnx.data` sidecar) —
  ONNX export, checker passes (opset 14).
- `gateway_decoder_india_trained.pt` — traced gateway decoder
  (task head), (1, 8) → (1, 1) logit.
- `train_semantic_india.py`, `train_log_india.txt`,
  `fetch_weather_india.py`, `weather_<station>.json` (cached weather),
  `dataset_india.npz` (all windows/labels/rule predictions),
  `training_report_india.json`, `animation_window_india.json`.

## Caveats that travel with this model

- Thresholds are the CWC values as published by INDOFLOODS (2025);
  CWC revises warning/danger levels from time to time.
- GUARDIAN's figshare record sets no explicit licence; the underlying
  measurements are CWC (Government of India) telemetry republished by
  IIT Bombay for research, and INDOFLOODS is CC BY 4.0. Training /
  research use with citation is the intended use.
- The Bihar stations sit in the Ganga basin, a *classified* basin
  under CWC's dissemination policy — worth a fresh look before any
  redistribution of the data itself (the model weights and metrics
  here are derived products).
- Weather channels are ERA5 reanalysis at the station coordinates,
  not rain-gauge observations.
