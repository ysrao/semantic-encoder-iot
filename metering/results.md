# Water metering / leak detection — THIRD simulated application of the task-oriented semantic encoder/decoder pair

**Status: TRAINED on synthetic data, 21 Nov 2026.** Same encoder +
gateway decoder and the identical training recipe as the flood-warning
build (`../semantic_encoder/train_semantic.py`) and the outage build
(`../outage_sim/train_outage.py`); only the application and data
changed. The flood case has a real-data (India) run; **this
application is simulation-only — no field metering data was used.**
Training script: `train_metering.py` (this folder); full
machine-readable record: `training_report.json`; log: `train_log.txt`.

## Task

A household water meter watches the supply flow. The gateway must
know whether a **leak** is active — a sustained low flow that never
fully stops — distinguishing true leaks from **long legitimate draws**
(garden irrigation, tank filling), cyclic appliances (laundry,
dishwasher), and ordinary draws (showers, baths, taps, toilet
flushes), none of which are leaks. The device either sends the raw
sensing window, a hand-crafted rule report, or the learned 8-byte
latent; the gateway decides from what it receives.

## Dataset design (synthetic generator, seeded)

- **Traces:** 1 sample/min, 2,880 samples (48 h) per trace.
  Background: a daily usage pattern — morning peak 05:30–09:30 and
  evening peak 17:30–23:00, sporadic daytime draws, quiet nights;
  flow measurement noise σ 0.02 L/min; supply pressure nominal 1.0
  (3 bar) with slow wander and a demand dip, noise σ 0.004.
- **Draws** (per day): showers 7–12 L/min, 5–15 min; baths 10–18
  L/min, 25–60 min (30% of days); taps 2–6 L/min, 0.5–4 min;
  toilet flushes 5–7 L/min, 1–2 min; laundry (50% of days) and
  dishwasher (40%) as cyclic bursts 3–6 L/min separated by zero-flow
  gaps over 45–90 min.
- **Confusers** (per day, NOT leaks): irrigation — 8–16 L/min
  continuous, 60–240 min, 35% of days, early morning or evening;
  tank filling — 6–14 L/min continuous, 45–150 min, 30% of days.
- **Leaks** (ground truth): 0.4–2.5 L/min sustained, **180–900 min
  (3–15 h)**, 0/1/2 per trace (p = 0.40/0.40/0.20); 30% grow
  linearly to 1.5× their initial rate.
- **Ground truth:** leak state = 1 exactly on the injected leak
  intervals (generative truth, not a threshold on the observed
  series).
- **Windows:** causal, length 16 timesteps × 4 channels
  `[flow_L/min/20, cumvol_since_midnight_L/500, pressure_norm,
  continuous_flow_min/240]` = 64 floats, stride 8; label =
  ground-truth state at the window's final timestep (the family
  convention). The fourth channel is the meter's own
  continuous-flow register — minutes since the flow last fell to
  near zero — a standard smart-meter quantity; see the design note
  under Caveats for why it is there.
- **Splits by trace seed** (never random interleaving): train 30
  traces (seeds 101–105, 501–525), validation 8 (601–608), test 20
  (201–205, 701–715).

| Split | Windows | Positives | Test-split event inventory |
|---|---|---|---|
| Train | 10,770 | 1,146 (10.6%) | 17 leaks, 22 irrigation, 22 tank fills (train totals) |
| Validation | 2,872 | 625 (21.8%) | 9 leaks |
| Test | 7,180 | 1,459 (20.3%) | **21 leaks**, 18 irrigation, 13 tank fills, 158 laundry, 106 dishwasher cycles |

## Architectures (counted exactly)

| Component | Shape | Parameters | Size |
|---|---|---|---|
| Device encoder (same architecture as flood/outage builds) | MLP 64 → 48 → 24 → 8, tanh latent | **4,496** | fp32 weights 17,984 B (**17.6 KB**); int8 weights 4,496 B (**4.4 KB**); traced TorchScript file 30,495 B; ONNX export 19,394 B (1,730 + 17,664 sidecar) |
| Gateway decoder | 8 → 24 → 48, task head 48 → 1 (+ recon head) | 4,601 | gateway-side, not device-constrained |

Training (family recipe verbatim, one principled run, nothing tuned
against the rule or the test set): encoder + decoder jointly; int8
quantisation **in the loop** (straight-through estimator); Gaussian
noise σ = 0.03 on the quantised latent during training; loss =
pos-weighted BCE + 0.05 × MSE(recon); Adam lr 1e-3, batch 256; the
run completed all 40 epochs (the patience-8 early stop never fired);
best validation F1 **0.9670** at epoch ~mid-run, decision threshold
**0.70** selected on validation. Reconstruction (auxiliary head):
MSE 0.71430 on test inputs — much higher than the outage build's
0.008, because the cumulative-volume channel spans 0–~2.4 and
dominates the MSE; the head is auxiliary and is discarded at
deployment, as in the other builds.

Encoder inference cost on this VM (CPU, torch 2.14): **0.032 ms**
per single window; 1.43 µs/window batched. On-device (AI Hub)
figures are in the section below.

## Rule baseline

The study's stage3 rule family, "above" variant (threshold +
persistence + hysteresis + causal smoothing), applied to the flow
channel only — the same information discipline as the earlier
builds: causal smoothing 4, threshold **0.3 L/min** (0.015
normalised), persistence **90 samples (90 min)**, hysteresis
0.005 normalised, read at the window's final timestep. This is the
classic minimum-flow leak rule as an engineer would set it: flow
that stays above a small threshold for an hour and a half is
reported. The rule device transmits a **20-byte task report** at
each episode start (event type, onset timestamp, mean flow,
duration, flags — the study's 20 B semantic-report budget).

## Test results — NN pair vs rule, identical windows

Window level (7,180 test windows, 1,459 positive):

| System | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| **NN pair (8 B latent)** | 0.9471 | **0.9507** | 0.7800 | 0.8569 | 59 | 321 |
| NN pair, noisy latent (σ = 0.03) | 0.9469 | 0.9499 | 0.7800 | 0.8566 | 60 | 321 |
| Rule-based (20 B report) | 0.9475 | 0.8892 | **0.8472** | **0.8677** | 154 | 223 |

Episode level (21 ground-truth leak episodes in the test traces):

| System | Detected | Missed | False-alarm episodes | Median detection delay |
|---|---|---|---|---|
| NN pair | 21 / 21 | 0 | **13** | **17 min** |
| Rule-based | 21 / 21 | 0 | 27 | 93 min |

## Payload and bytes/day (SIMULATED)

| What the device sends | Bytes |
|---|---|
| Raw window (64 float32) | **256 B** |
| Rule task report (per episode start) | **20 B** |
| Learned latent (8 int8) | **8 B** — 32× smaller than raw, 2.5× smaller than the rule report |

Bytes/day under the simulated reporting cadence — **stated
assumptions:** the device evaluates one causal window every 8 min
(the window stride at 1 sample/min) = **180 windows/day**; the raw
and learned modes stream per window; the rule mode is
event-triggered and transmits its 20 B report once per latch
episode, with episodes counted on the held-out test traces
(0.925 episodes/day, mean per trace-day, detections and false
alarms together):

| Mode | Bytes/day (simulated) |
|---|---|
| Raw window stream | 46,080 |
| Learned latent stream | **1,440** (32× below raw) |
| Rule reports (event-triggered) | 18.5 |

This **replaces the paper's metering estimate** (~400 B/day →
~10 B/day, ~40×). The absolute figures differ from the estimate
because the simulated cadence — a streamed window every 8 minutes —
is far more frequent than the cadence the estimate implicitly
assumed; the comparable quantity is the ratio, 32× for the learned
stream against raw, in line with the flood and outage builds. Note
also, plainly: the rule's event-triggered reporting sends fewer
bytes per day than either stream. That is a real property of
event-triggered reporting, not an error in the table — the learned
pair pays a streaming cost (1.4 KB/day) for gateway-side decisions
on every window.

## Honest reading

On this synthetic task the two systems finish close, with
complementary strengths, and **at window level the rule edges the
learned pair** (F1 0.8677 vs 0.8569): the rule recalls more leak
windows (0.847 vs 0.780 — its latch, once fired, holds through the
whole episode, while the NN misses some interior windows where a
leak is masked by concurrent draws), and the NN is the more precise
(0.951 vs 0.889 — it uses flow rate, not just duration, to reject
long legitimate draws). **At episode level — the level a utility
acts on — the learned pair is ahead on every operational measure:**
both systems detect all 21 leaks, but the NN raises half the rule's
false-alarm episodes (13 vs 27) and detects a median **17 min**
after onset against the rule's **93 min**, which is essentially the
rule's own 90-minute persistence floor. The σ = 0.03 latent-noise
robustness seen in the earlier builds holds here (no metric change
of consequence).

**Caveats that travel with these numbers:**

- **Design note — one input change, made for information parity,
  and reported here rather than buried.** The first configuration
  of this build used time-of-day as the fourth channel. In that
  run the NN finished well behind the rule (F1 0.6645, precision
  0.4997, 333 false-alarm episodes; the rule was unchanged at
  0.8677). The cause was structural, not a matter of tuning: the
  rule's 90-minute persistence integrates an hour and a half of
  history, while the encoder sees only a 16-minute window, so the
  rule was deciding from temporal information the NN could not
  access — breaking the comparison discipline of the earlier
  builds, where the rule's persistence (3 samples, outage) sat
  inside the window. The fourth channel was therefore changed to
  the meter's continuous-flow register (minutes since flow last
  near zero), a quantity real smart meters maintain for exactly
  this alarm; the rule's persistence counter and the NN's fourth
  channel now encode the same underlying quantity, and the NN
  additionally sees rate, volume and pressure. Run 1 is archived
  in full under `run1_tod_context/`. No hyperparameter was tuned
  against the rule in either run; the recipe is the family recipe
  verbatim in both.
- **Discrimination rests on (rate, duration).** With the
  continuous-flow register visible, leaks and confusers are largely
  separated by sustained duration at low rate versus shorter or
  higher-rate draws — but the overlap is real (tank fills run to
  150 min; the slowest leaks start at 0.4 L/min against a 0.3 L/min
  rule threshold), which is why neither system is near ceiling
  here, unlike the outage build. This is the least separable of
  the three simulated tasks, and the scores show it.
- **Simulation only.** One generator, one distribution, rectangular
  draw pulses, Gaussian noise, no meter dropouts, no seasonal or
  household-to-household variation, no field validation. The flood
  application's real-data run (where the rule finished ahead,
  F1 0.911 vs 0.872) is the standing warning against extrapolating
  simulation rankings to the field.
- Episode-level claims rest on **21 test episodes** — detection
  counts are exact but coarse.
- Split positive rates differ (train 10.6% vs validation 21.8% vs
  test 20.3%) because per-trace leak counts and durations are
  stochastic; the train seeds happened to draw fewer and shorter
  leaks. The decision threshold was selected on validation only —
  and validation (8 traces, 9 leak episodes) flattered the NN
  (val F1 0.967 vs test 0.857), a small-validation-split effect
  worth remembering when reading the threshold.
- One training run with the family recipe's hyperparameters
  verbatim; no seed sweep, no tuning against the rule.

## Qualcomm AI Hub

The trained metering encoder was submitted to Qualcomm AI Hub as
**semantic_encoder_metering_sim** — the third application's
companion model, in the same project and the same way as the flood
encoder (`semantic_encoder_india_cwc`) and the outage encoder
(`semantic_encoder_outage_sim`) — compiled to TFLite and profiled
on a Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon NPU). Both
jobs **SUCCESS**:

- Compile job `j569zko65` → TFLite model `mn7l4w1l4`
  (https://workbench.aihub.qualcomm.com/jobs/j569zko65/)
- Profile job `jpvyrjmkp`
  (https://workbench.aihub.qualcomm.com/jobs/jpvyrjmkp/):
  **estimated inference 55 µs**, median measured **58 µs**,
  first load 310 ms, estimated inference peak memory 119,689,216
  bytes (~114 MiB — the TFLite/NPU runtime footprint, as with the
  other builds, not the 4.4 KB of model weights).

Same order as the flood and outage encoders on the same device
(42 µs est / 56 µs median for the India flood build; 52/54 µs for
the outage build) — expected, since the architecture is identical.
Full record: `../aihub_results/trained_semantic_metering.json`;
runner: `../aihub_results/run_trained_semantic_metering.py`.

## Artifacts (this folder)

- `semantic_encoder_metering_trained.pt` — traced encoder, (1, 64) → (1, 8); reload-verified.
- `semantic_encoder_metering_trained.onnx` (+ `.onnx.data` sidecar) — ONNX export, checker passes.
- `gateway_decoder_metering_trained.pt` — traced gateway decoder (task head), (1, 8) → (1, 1); reload-verified.
- `train_metering.py`, `train_log.txt`, `training_report.json`, `dataset_metering.npz`.
- `run1_tod_context/` — the archived first configuration (time-of-day fourth channel): `training_report_run1.json`, `train_log_run1.txt`.
