# Semantic encoder — training report

**Status: TRAINED (was: untrained reference build).** Trained 18 Nov 2026 on
the semantic-mIoT-NTN study's own simulation traces, jointly with a gateway
decoder. **All data is seeded synthetic simulation data from the study — not
field data, not real sensor traffic.** Every number below is a simulation
result and must be cited as such.

## Data provenance and route

- **Route taken: the study repo's own trace generators, used as-is.**
  Cloned `https://github.com/ysrao/semantic-miot-ntn` into
  `~/workspace/edge-apps-prep/semantic_traces_src/` and called
  `ntn_eval.traces.gen_agri_trace` / `gen_env_trace` with the repo's own
  `configs/agri.yaml` / `configs/env.yaml`. No generator was reimplemented.
- Only environment change: the venv shipped a pandas build that was
  binary-incompatible with its numpy 2.5.3, so pandas 2.2.3 was installed
  into `~/v2env`. The study code itself is untouched.
- **Splits by seed** (the study's own discipline — thresholds tuned on seeds
  101–105, evaluated on held-out 201–205, never random interleaving):
  train = 101–105 + 301–308 (extra seeds through the same generator),
  validation = 401–402, test = 201–205.
- **Windows:** causal, length 16 timesteps, stride 8, input
  `[primary, threshold, rainfall, temperature]` per timestep (the encoder's
  existing (1, 64) contract). Normalisation — agri: primary = soil-moisture
  obs / field capacity 0.38, threshold = stage threshold / 0.38,
  rain = clip(rain mm/h / 20, 0, 3), temp = temp obs / 50; env: primary =
  level obs / danger level 4.5 m, threshold = 1.0, rain and temp as agri.
- **Label:** the ground-truth task event at the window's final timestep
  (`irrig_need` for agriculture, `flood` for environment) — the study scores
  all task outcomes against ground truth.
- **Dataset sizes:** train 56,134 windows (7,656 positive), validation
  8,636 (1,435 positive), test 21,590 (2,670 positive). Note the imbalance:
  agri positives are rare (78 of 5,395 test windows); env positives are
  common (2,592 of 16,195).

## Model and training

- **Encoder:** the existing `SemanticEncoder` (MLP 64 → 48 → 24 → 8-dim tanh
  latent, 4,496 params) — architecture unchanged.
- **Gateway decoder (new):** mirror MLP 8 → 24 → 48 with a task head
  (48 → 1, the event decision) and a small reconstruction head (48 → 64,
  auxiliary only).
- **Joint end-to-end training**, loss = class-weighted binary cross-entropy
  on the task head + 0.05 × MSE on the reconstruction head. Adam, lr 1e-3,
  batch 256, early stop on validation F1 (best val F1 0.9714 at epoch 21;
  decision threshold 0.70 selected on validation).
- **Quantisation/channel treatment:** the latent is **int8-quantised in the
  training loop** via a straight-through estimator (forward pass rounds to
  the 1/127 grid; backward pass is identity), and Gaussian channel noise
  σ = 0.03 is added to the quantised latent during training. Evaluation
  uses true round-quantisation; a noisy-latent evaluation is also reported.
- Training script: `train_semantic.py` (PyTorch CPU, ~73 s end to end).

## Test results — NN pair vs the study's rule, identical windows

The rule baseline is the study's own scheme (threshold + persistence +
hysteresis + causal smoothing; agri persist 2 / hyst 0.02 / smooth 8, env
persist 2 / hyst 0.3 / smooth 6), computed on the full trace and read at
the same window-final timesteps. The rule is causal, so both sides see the
same information. Metrics are **per-window** (the study's stage-3 headline
metrics are per-episode and are not directly comparable).

| Split | System | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Combined (21,590) | **NN pair** | 0.9920 | 0.9619 | 0.9742 | **0.9680** |
| Combined | NN pair, noisy latent | 0.9923 | 0.9637 | 0.9742 | 0.9689 |
| Combined | Rule-based | 0.9914 | 0.9442 | 0.9891 | 0.9662 |
| Agri (5,395; 78 pos) | **NN pair** | 0.9861 | 0.5152 | 0.6538 | **0.5763** |
| Agri | Rule-based | 0.9744 | 0.3598 | **0.9872** | 0.5274 |
| Env (16,195; 2,592 pos) | NN pair | 0.9940 | 0.9789 | 0.9838 | 0.9813 |
| Env | **Rule-based** | 0.9971 | 0.9926 | 0.9892 | **0.9909** |

Honest reading:

- **Overall the NN pair matches the rule** (combined F1 0.968 vs 0.966) —
  it does not beat it decisively, and on the env/flood task the rule is
  slightly better (F1 0.991 vs 0.981).
- On agri the NN's F1 edge comes from precision; the **rule's recall is far
  higher (0.987 vs 0.654)** — the rule misses almost no irrigation-need
  windows, at the cost of many more false alarms. With only 78 positive
  test windows, the agri figures are small-sample and should not be
  over-read. (Per-window misses also overstate episode misses: an event
  spans many windows.)
- The latent is robust to the channel treatment: adding σ = 0.03 noise to
  the quantised latent at test time changes F1 by ≤ 0.001.
- Reconstruction (auxiliary head): MSE 0.031 on normalised test inputs —
  the latent retains a usable approximation of the window, but the task
  head is the product, not the reconstruction.

## Payload

- Raw window: 64 float32 values = **256 bytes** (the study's raw reporting
  payload is 200 B/report).
- NN latent: 8 int8 values = **8 bytes** — 32× smaller than the raw window,
  and inside the study's 20-byte semantic-report budget.

## Artifacts (this folder)

- `semantic_encoder_trained.pt` — traced **encoder only**, same
  (1, 64) → (1, 8) contract as the reference build.
- `semantic_encoder_trained.onnx` (+ `.onnx.data` sidecar) — ONNX export,
  checker passes.
- `gateway_decoder_trained.pt` — traced gateway decoder (task head),
  (1, 8) → (1, 1) logit, kept for the record.
- `training_report.json` — machine-readable version of this report.
- `train_semantic.py`, `train_log.txt` — full training script and log.
- Traced encoder reload was verified in a fresh process: (1, 64) → (1, 8),
  decoder (1, 8) → (1, 1).

## AI Hub re-profile

The trained encoder was re-submitted to Qualcomm AI Hub on a Samsung
Galaxy S24 (Snapdragon 8 Gen 3, TFLite target). Both jobs **SUCCESS**:

- Compile job `jp0xwo6ep` → TFLite model `mn41l4okr`
  (https://workbench.aihub.qualcomm.com/jobs/jp0xwo6ep/)
- Profile job `jp8knj18p`
  (https://workbench.aihub.qualcomm.com/jobs/jp8knj18p/):
  **estimated inference 48 µs** (median measured 55 µs), first load
  326 ms, warm load 159 ms, estimated inference peak memory
  119,422,976 bytes (~114 MiB — as with the reference build, this is the
  TFLite/NPU runtime footprint; the model's own inference memory increase
  was 0–33 KB).

For comparison, the untrained reference encoder profiled at 47 µs
estimated / 54 µs median on the same device — training changed the
weights, not the architecture, so on-device cost is unchanged, as
expected. Full record:
`~/workspace/edge-apps-prep/aihub_results/trained_semantic.json`.
