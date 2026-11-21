# Power outage detection — SECOND simulated application of the task-oriented semantic encoder/decoder pair

**Status: TRAINED on synthetic data, 21 Nov 2026.** Same encoder +
gateway decoder and the identical training recipe as the flood-warning
build (`../semantic_encoder/train_semantic.py`); only the application
and data changed. The flood case has a real-data (CWC) run; **this
application is simulation-only — no field voltage data was used.**
Training script: `train_outage.py` (this folder); full machine-readable
record: `training_report.json`; log: `train_log.txt`.

## Task

A voltage sensor on a distribution feeder watches RMS voltage. The
gateway must know whether an **outage** (interruption) is active —
distinguishing true outages from voltage **sags**, **deep sags**,
**momentary dips** and **restoration inrush sags**, none of which are
outages. The device either sends the raw sensing window, a hand-crafted
rule report, or the learned 8-byte latent; the gateway decides from
what it receives.

## Dataset design (synthetic generator, seeded)

- **Traces:** 1 Hz RMS sampling, 3,600 samples (1 h) per trace.
  Background: slow voltage drift (±0.035 pu, sinusoids + clipped random
  walk), load-current wander 0.3–0.95 pu, frequency wander 49.7–50.3 Hz,
  measurement noise (σ V 0.006, I 0.01, f 0.008).
- **Injected events** (per trace): sags — residual 0.45–0.88 pu,
  0.3–25 s, Poisson(2.5); deep sags — residual 0.25–0.42 pu, 1–30 s,
  in 35% of traces; momentary dips — residual 0.0–0.15 pu, **1–2 s**,
  Poisson(0.8) (sub-threshold transients that are *not* outages — the
  duration requirement is what rejects them); outages — residual
  0.0–0.08 pu, **30–600 s**, 0/1/2 per trace (p = 0.25/0.45/0.30),
  half followed by a 1–3 s inrush sag to 0.65–0.8 pu at restoration
  (also not an outage).
- **Ground truth:** outage state = 1 exactly on the injected outage
  intervals (generative truth, not a threshold on the observed series).
- **Windows:** causal, length 16 timesteps × 4 channels
  `[voltage_pu, 0.20 (threshold), current_pu, (f − 49.5)]` = 64 floats,
  stride 8; label = ground-truth state at the window's final timestep
  (the flood simulation run's convention).
- **Splits by trace seed** (never random interleaving): train 30
  traces (seeds 101–105, 501–525), validation 8 (601–608), test 20
  (201–205, 701–715).

| Split | Windows | Positives | Test-split event inventory |
|---|---|---|---|
| Train | 13,470 | 1,349 (10.0%) | 35 outages, 86 sags, 9 deep sags, 19 momentary dips |
| Validation | 3,592 | 518 (14.4%) | — |
| Test | 8,980 | 672 (7.5%) | **16 outages**, 50 sags, 10 deep sags, 19 momentary dips, 11 inrush sags |

## Architectures (counted exactly)

| Component | Shape | Parameters | Size |
|---|---|---|---|
| Device encoder (SemanticEncoder, unchanged from flood build) | MLP 64 → 48 → 24 → 8, tanh latent | **4,496** | fp32 weights 17,984 B (**17.6 KB**); int8 weights 4,496 B (**4.4 KB**); traced TorchScript file 30,539 B; ONNX export 19,388 B (1,724 + 17,664 sidecar) |
| Gateway decoder | 8 → 24 → 48, task head 48 → 1 (+ recon head) | 4,601 | gateway-side, not device-constrained |

Training (flood recipe verbatim, one principled run, nothing tuned
against the rule or the test set): encoder + decoder jointly; int8
quantisation **in the loop** (straight-through estimator); Gaussian
noise σ = 0.03 on the quantised latent during training; loss =
pos-weighted BCE + 0.05 × MSE(recon); Adam lr 1e-3, batch 256; early
stop on validation F1 (best val F1 **0.9981** at epoch 5, stopped at
epoch 13; decision threshold **0.95** selected on validation).
Reconstruction (auxiliary head): MSE 0.00801 on test inputs.

Encoder inference cost on this VM (CPU, torch 2.14): **0.032 ms** per
single window; 0.95 µs/window batched. On-device (AI Hub) figures are
in the section below.

## Rule baseline

The study's own `stage3.rule_below`, imported as-is: causal smoothing
4, threshold **0.20 pu**, persistence **3 samples**, hysteresis 0.05 pu,
read at the window's final timestep — same information as the NN. The
rule device transmits a **20-byte task report** at each episode start
(event type, onset timestamp, minimum RMS, duration, flags — the
study's 20 B semantic-report budget).

## Test results — NN pair vs rule, identical windows

Window level (8,980 test windows, 672 positive):

| System | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| **NN pair (8 B latent)** | 0.9992 | 1.0000 | 0.9896 | **0.9948** | 0 | 7 |
| NN pair, noisy latent (σ = 0.03) | 0.9992 | 1.0000 | 0.9896 | **0.9948** | 0 | 7 |
| Rule-based (20 B report) | 0.9981 | 1.0000 | 0.9747 | **0.9872** | 0 | 17 |

Episode level (16 ground-truth outage episodes in the test traces):

| System | Detected | Missed | False-alarm episodes | Median detection delay |
|---|---|---|---|---|
| NN pair | 16 / 16 | 0 | 0 | **4 s** |
| Rule-based | 16 / 16 | 0 | 0 | 10 s |

## Payload

| What the device sends | Bytes |
|---|---|
| Raw window (64 float32) | **256 B** |
| Rule task report (per episode start) | **20 B** |
| Learned latent (8 int8) | **8 B** — 32× smaller than raw, 2.5× smaller than the rule report |

## Honest reading

On this synthetic task the learned pair **matches the rule's precision
(both zero false positives — every sag, deep sag, momentary dip and
inrush sag in the test set was correctly rejected by both systems) and
edges it on recall** (F1 0.9948 vs 0.9872), because the rule's causal
smoothing + persistence delays its latch at outage onsets and its
hysteresis holds it at offsets (FN 17 vs 7), which also shows up as a
slower median detection delay (10 s vs 4 s). The σ = 0.03 latent-noise
robustness seen in the flood build holds here (no metric change).

**Caveats that travel with these numbers:**

- **The task is near-separable by construction.** Outages are defined
  as residual ≤ 0.08 pu for ≥ 30 s; the only sub-threshold confusers
  are 1–2 s dips, and the rule's 0.20 pu threshold sits on the
  generative boundary. Near-ceiling scores for *both* systems are
  therefore expected; the comparison is meaningful only in the
  residuals (edge recall, detection delay), not as evidence of a hard
  discrimination problem solved.
- **Simulation only.** One generator, one distribution, Gaussian
  noise, no harmonics, no sensor dropouts or missing data, no field
  validation. The flood application's real-data run (where the rule
  finished ahead, F1 0.911 vs 0.872) is the standing warning against
  extrapolating simulation rankings to the field.
- Episode-level claims rest on **16 test episodes** — detection counts
  are exact but coarse.
- Validation and test positive rates differ (14.4% vs 7.5%) because
  per-trace outage counts are stochastic; the decision threshold was
  selected on validation only.
- One training run with the flood recipe's hyperparameters verbatim;
  no seed sweep, no tuning against the rule.

## Qualcomm AI Hub

The trained outage encoder was submitted to Qualcomm AI Hub as
**semantic_encoder_outage_sim** — the second application's companion
model, in the same project and the same way as the flood encoder
(`semantic_encoder_india_cwc`) — compiled to TFLite and profiled on a
Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon NPU). Both jobs
**SUCCESS**:

- Compile job `jpykjye0g` → TFLite model `mn0x6z8pw`
  (https://workbench.aihub.qualcomm.com/jobs/jpykjye0g/)
- Profile job `jpx86z2j5`
  (https://workbench.aihub.qualcomm.com/jobs/jpx86z2j5/):
  **estimated inference 52 µs**, median measured **54 µs**,
  first load 468 ms, estimated inference peak memory 124,682,240
  bytes (~119 MiB — the TFLite/NPU runtime footprint, as with the
  other builds, not the 4.4 KB of model weights).

Same order as the flood encoder on the same device (42 µs est /
56 µs median for the India build) — expected, since the architecture
is identical. Full record:
`../aihub_results/trained_semantic_outage.json`; runner:
`../aihub_results/run_trained_semantic_outage.py`.

## Artifacts (this folder)

- `semantic_encoder_outage_trained.pt` — traced encoder, (1, 64) → (1, 8); reload-verified.
- `semantic_encoder_outage_trained.onnx` (+ `.onnx.data` sidecar) — ONNX export, checker passes.
- `gateway_decoder_outage_trained.pt` — traced gateway decoder (task head), (1, 8) → (1, 1) logit.
- `train_outage.py`, `train_log.txt`, `training_report.json`, `dataset_outage.npz`.
