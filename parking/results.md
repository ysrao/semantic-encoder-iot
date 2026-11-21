# Parking occupancy detection — FOURTH simulated application of the task-oriented semantic encoder/decoder pair

**Status: TRAINED on synthetic data, 21 Nov 2026.** Same encoder +
gateway decoder and the identical training recipe as the flood-warning
build (`../semantic_encoder/train_semantic.py`) and the outage build
(`../outage_sim/`); only the application and data changed. The flood
case has a real-data (CWC) run; **this application is simulation-only —
no field magnetometer data was used.** Training script:
`train_parking.py` (this folder); full machine-readable record:
`training_report.json`; log: `train_log.txt`.

## Task

A magnetometer buried in a single parking bay watches the local
magnetic field. A parked vehicle distorts the field vector; the
gateway must know whether **this bay is occupied** — distinguishing
true occupancy from a **vehicle passing in the adjacent lane**, a
**vehicle parked in the neighbouring bay**, and **transient magnetic
noise**, none of which are occupancy of this bay. The device either
sends the raw sensing window, a hand-crafted rule report, or the
learned 8-byte latent; the gateway decides from what it receives.

## Dataset design (synthetic generator, seeded)

- **Traces:** 1 Hz sampling, 21,600 samples (6 h) per trace, one bay.
  Background field 42–58 µT (per-trace direction) with slow drift
  (±0.4 µT sinusoids + clipped walk per axis) and per-axis measurement
  noise σ 0.15 µT.
- **Channels:** the field **deviation vector** (ΔBx, ΔBy, ΔBz) and its
  **magnitude** |ΔB|, relative to the device's own empty-bay baseline —
  a causal gated EMA (τ = 1800 s) that adapts only while the running
  deviation is < 1.0 µT, so a parked vehicle does not drag the
  baseline. This is how magnetometer parking sensors actually work
  (install-time calibration + slow gated adaptation), and both the NN
  and the rule consume the identical deviation series. Features are
  scaled ×0.1 (10 µT = 1.0) for the encoder; the rule reads the
  magnitude unscaled.
- **Injected events** (per trace): own-bay occupancy — step change of
  the field vector, magnitude log-uniform **3–14 µT**, direction in a
  cone around a canonical own-bay axis (defined relative to the
  background-field frame), duration log-uniform **10–240 min**, gaps
  20–90 min; neighbouring-bay parking — sustained distortion
  **1.5–5 µT** around a *different* canonical axis, 10–180 min,
  Poisson(1.2); passing vehicles — raised-cosine distortion 3–12 µT,
  **2–8 s**, own-bay-like direction, Poisson(3); transient spikes —
  single-axis, 2–6 µT, 1–3 s, Poisson(6).
- **Ground truth:** occupancy state = 1 exactly on the injected
  own-bay intervals (generative truth).
- **Windows / cadence:** causal, length 16 timesteps × 4 channels =
  64 floats; the device evaluates **one window per minute** (window
  ends every 60 s); label = ground-truth state at the window's final
  timestep (the flood/outage convention).
- **Splits by trace seed** (never random interleaving): train 30
  traces (seeds 111–115, 531–555), validation 8 (611–618), test 20
  (211–215, 721–735).

| Split | Windows | Positives | Test-split event inventory |
|---|---|---|---|
| Train | 10,800 | 6,201 (57.4%) | 82 occupancy, 40 neighbour, 93 passing, 173 transients |
| Validation | 2,880 | 1,653 (57.4%) | — |
| Test | 7,200 | 3,817 (53.0%) | **62 occupancy episodes**, 22 neighbour, 49 passing, 139 transients |

Measured turnover on the test traces: 62 episodes over 5 bay-days =
**12.4 arrivals/day** (124 arrival/departure edges/day) — a busy bay.

## Architectures (counted exactly)

| Component | Shape | Parameters | Size |
|---|---|---|---|
| Device encoder (SemanticEncoder, unchanged from flood build) | MLP 64 → 48 → 24 → 8, tanh latent | **4,496** | fp32 weights 17,984 B (**17.6 KB**); int8 weights 4,496 B (**4.4 KB**); traced TorchScript file 30,694 B; ONNX export 19,391 B (1,727 + 17,664 sidecar) |
| Gateway decoder | 8 → 24 → 48, task head 48 → 1 (+ recon head) | 4,601 | gateway-side, not device-constrained |

Training (flood recipe verbatim, one principled run, nothing tuned
against the rule or the test set): encoder + decoder jointly; int8
quantisation **in the loop** (straight-through estimator); Gaussian
noise σ = 0.03 on the quantised latent during training; loss =
pos-weighted BCE + 0.05 × MSE(recon); Adam lr 1e-3, batch 256; early
stop on validation F1 (best val F1 **0.9919** at epoch 2, stopped at
epoch 10; decision threshold **0.75** selected on validation).
Reconstruction (auxiliary head): MSE 0.09391 on test inputs.

Encoder inference cost on this VM (CPU, torch 2.14): **0.037 ms** per
single window; 0.25 µs/window batched. On-device (AI Hub) figures are
in the section below.

## Rule baseline

The study's own `stage3.rule_above`, imported as-is, on the deviation
magnitude series: causal smoothing 4, threshold **2.0 µT**,
persistence **15 samples**, hysteresis 0.5 µT, read at each evaluated
window's final timestep — same information as the NN. (Persistence 15 s
is set from the scenario's physics, a priori: passing vehicles last
2–8 s, occupancy lasts ≥ 10 min.) The rule device transmits a
**20-byte task report** at each episode edge (event type, onset
timestamp, min/max deviation, duration, flags — the study's 20 B
semantic-report budget).

## Test results — NN pair vs rule, identical windows

Window level (7,200 test windows, 3,817 positive):

| System | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|
| **NN pair (8 B latent)** | 0.9642 | 0.9599 | 0.9730 | **0.9664** | 155 | 103 |
| NN pair, noisy latent (σ = 0.03) | 0.9639 | 0.9597 | 0.9728 | 0.9662 | 156 | 104 |
| Rule-based (20 B report) | 0.9221 | 0.8739 | 0.9969 | 0.9313 | 549 | 12 |

Episode level (62 ground-truth occupancy episodes in the test traces;
124 arrival/departure edges):

| System | Detected | Missed | False-alarm episodes | Median detection delay |
|---|---|---|---|---|
| NN pair | 62 / 62 | 0 | 17 | **39.5 s** |
| Rule-based | 62 / 62 | 0 | 22 | 41.0 s |

Detection delay here is dominated by the 1-minute evaluation cadence
(mean wait ≈ 30 s to the next window end) for both systems.

## Payload and bytes/day (SIMULATED)

Per unit, as in the other builds:

| What the device sends | Bytes |
|---|---|
| Raw window (64 float32) | **256 B** |
| Rule task report (per episode edge) | **20 B** |
| Learned latent (8 int8) | **8 B** — 32× smaller than raw, 2.5× smaller than the rule report |

Per day, under the simulated reporting cadence — **one evaluated
window per minute (1,440 windows/day)**; raw and learned transmit per
evaluated window; the rule transmits one 20 B report per episode edge,
at the turnover measured on the test traces (12.4 arrivals/day,
24.8 edges/day):

| Mode | Bytes/day |
|---|---|
| Raw windows | **368,640 B/day** (360 KiB) |
| Learned latent | **11,520 B/day** (11.25 KiB) — 32× below raw |
| Rule reports | **496 B/day** |

This replaces the paper's earlier parking **estimate** (~11.5 KB/day
raw → ~120 B/day, ~100×). The simulated numbers differ because the
cadence is now explicit: the old estimate implicitly assumed a much
slower raw snapshot rate and a lower turnover, and its "~120 B/day"
semantic figure corresponds to the rule's episode-edge discipline at
~3 arrivals/day, not to the learned pair's per-window latent. The
learned pair's honest simulated saving against raw at equal cadence
is the same 32× as every other application; the rule sends fewer
bytes still, because it reports only at episode edges — the same
asymmetry already present in the outage build's payload table.

## Honest reading

On this synthetic task the two systems split the honours, and neither
is at the ceiling. The **rule has the higher recall** (0.9969 vs
0.9730 — it almost never misses an occupied window) but **much lower
precision** (0.8739 vs 0.9599): its magnitude-only threshold at 2.0 µT
sits inside the neighbouring-bay distortion range (1.5–5 µT), so
neighbouring-bay parking latches it for long stretches — 549 false
positive windows and 22 false-alarm episodes. The **learned pair
wins on precision and F1** (0.9664 vs 0.9313) because it sees the
deviation *vector*, and the own-bay and neighbouring-bay signatures
point in systematically different directions; its misses concentrate
at episode edges and on the weakest own-bay signatures. At episode
level both systems detect **all 62 arrivals** (and hence all
departures), the NN with fewer false-alarm episodes (17 vs 22) and a
marginally lower median delay (39.5 s vs 41.0 s — both
cadence-dominated). The σ = 0.03 latent-noise robustness seen in the
other builds holds here (F1 0.9662 vs 0.9664).

**Caveats that travel with these numbers:**

- **Partly separable by construction.** Own-bay signatures (3–14 µT)
  and neighbouring-bay signatures (1.5–5 µT) overlap only in the
  3–5 µT band *in magnitude*, but their direction cones are distinct
  by construction, which is exactly the signal the NN exploits and
  the magnitude-only rule cannot. Real bays will have messier
  geometry: signature direction depends on vehicle type, position and
  orientation within the bay, and the own/neighbour direction
  separation will not always be this clean.
- **Simulation only.** One generator, one distribution, Gaussian
  noise, an idealised gated-EMA baseline (real baselines also face
  temperature drift, nearby ferrous infrastructure and long
  occupancies that defeat gating), no sensor dropouts or missing
  data, no field validation. The flood application's real-data run
  (where the rule finished ahead, F1 0.911 vs 0.872) is the standing
  warning against extrapolating simulation rankings to the field.
- The window-level positive rate (~53%) is high because the
  generator's bays are busy (12.4 arrivals/day, mean occupancy ≈
  62 min); a quieter bay would shift the precision/recall balance.
- Episode-level claims rest on **62 test episodes** — detection
  counts are exact but coarse.
- One training run with the flood recipe's hyperparameters verbatim;
  no seed sweep, no tuning against the rule.

## Qualcomm AI Hub

The trained parking encoder was submitted to Qualcomm AI Hub as
**semantic_encoder_parking_sim** — the fourth application's companion
model, in the same project and the same way as the flood encoder
(`semantic_encoder_india_cwc`) and the outage encoder
(`semantic_encoder_outage_sim`) — compiled to TFLite and profiled on
a Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon NPU). Both jobs
**SUCCESS**:

- Compile job `j571lqov5` → TFLite model `mm5dj18d6`
  (https://workbench.aihub.qualcomm.com/jobs/j571lqov5/)
- Profile job `jpx86w035`
  (https://workbench.aihub.qualcomm.com/jobs/jpx86w035/):
  **estimated inference 50 µs**, median measured **63 µs**,
  first load 300 ms, estimated inference peak memory 122,466,304
  bytes (~117 MiB — the TFLite/NPU runtime footprint, as with the
  other builds, not the 4.4 KB of model weights).

Same order as the other encoders on the same device (flood India
build 42 µs est / 56 µs median; outage 52 µs est / 54 µs median) —
expected, since the architecture is identical. Full record:
`../aihub_results/trained_semantic_parking.json`; runner:
`../aihub_results/run_trained_semantic_parking.py`.

## Artifacts (this folder)

- `semantic_encoder_parking_trained.pt` — traced encoder, (1, 64) → (1, 8).
- `semantic_encoder_parking_trained.onnx` (+ `.onnx.data` sidecar) — ONNX export, checker passes.
- `gateway_decoder_parking_trained.pt` — traced gateway decoder (task head), (1, 8) → (1, 1) logit.
- `train_parking.py`, `train_log.txt`, `training_report.json`, `dataset_parking.npz`.
