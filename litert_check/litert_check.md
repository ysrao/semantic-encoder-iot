# LiteRT (Google) runtime compatibility check — four trained semantic encoders

**Date:** 21 Nov 2026 · **Question:** do the trained encoders run under Google's
open LiteRT/TFLite runtime, not only through Qualcomm's AI Hub toolchain?
**Answer: yes — all four, with outputs indistinguishable from the PyTorch
models at the precision the system actually uses.**

## What was checked

| Application | Encoder checkpoint (PyTorch, TorchScript) | Study decision threshold |
|---|---|---|
| Flood warning — India (real CWC data) | `semantic_encoder/india/semantic_encoder_india_trained.pt` | 0.35 |
| Power outage detection (sim) | `outage_sim/semantic_encoder_outage_trained.pt` | 0.95 |
| Water metering / leak detection (sim) | `metering_sim/semantic_encoder_metering_trained.pt` | 0.70 |
| Parking occupancy (sim) | `parking_sim/semantic_encoder_parking_trained.pt` | 0.75 |

## Where the .tflite files came from (local conversion only)

Per Rao's direction, **no Qualcomm AI Hub artifact was used** (an earlier
download of the AI Hub-compiled models was deleted unused). Each `.tflite`
was produced on this VM from our own checkpoints:

- The trained weights were loaded out of each study's TorchScript encoder
  checkpoint into the studies' own `SemanticEncoder` architecture class
  (MLP 64→48→24→8, tanh latent). The rebuilt module was verified
  **bit-identical** to the checkpoint (max output difference 0.0 on probe
  inputs) before conversion.
- Conversion: Google's **litert-torch 0.10.0** converter (the current name
  of the ai-edge-torch package), default float32 conversion, sample input
  (1, 64). Files: `litert_check/semantic_encoder_{india,outage,metering,parking}_local.tflite`
  (20,716 bytes each). Scripts: `litert_check/convert_local.py`.
- (Route note: the pure `onnx2tflite` package is not available on this
  machine's pip index, so the ONNX intermediate was not needed — the
  converter works directly from PyTorch.)

## Interpreter

**ai-edge-litert 2.3.0** (Google LiteRT, the successor of the TFLite
runtime), `ai_edge_litert.interpreter.Interpreter`, CPU, on this VM.
Script: `litert_check/run_litert_check.py`; raw numbers:
`litert_check/litert_check_results.json`.

## Method

Per application, 32 real test windows from the study's own saved test set
(the first 16 test windows plus 16 positive-labelled ones, so both classes
are exercised). Identical windows went to (a) the PyTorch checkpoint and
(b) the LiteRT interpreter. Both latents were then pushed through the
study's own evaluation path — int8 quantisation `round(z·127)/127`, the
trained PyTorch gateway decoder, sigmoid, and the study's
validation-chosen threshold above — and the event decisions compared.

## Results (32 windows per application)

| Application | Latent max abs diff | Latent mean abs diff | int8 codes exact match | Decoder prob. max abs diff | Decisions agree |
|---|---|---|---|---|---|
| Flood–India | 2.0e-07 | 5.8e-08 | 100% (0 levels off) | 0.0 | **32/32** |
| Outage | 2.4e-07 | 4.5e-08 | 100% (0 levels off) | 0.0 | **32/32** |
| Metering | 2.7e-07 | 4.8e-08 | 100% (0 levels off) | 0.0 | **32/32** |
| Parking | 2.4e-07 | 3.8e-08 | 100% (0 levels off) | 0.0 | **32/32** |

The latent differences are pure float32 round-off (~1e-07). Because the
gateway decoder consumes the **quantised** latent — and every one of the
256 int8 codes per application matched exactly — the downstream event
probabilities are bit-identical and every thresholded decision agrees.

## Honest limits

- This is a **host-side compatibility check**: the LiteRT interpreter ran
  on this VM's x86 CPU, on the float32 model. It proves the models are
  valid, portable LiteRT artifacts whose outputs match PyTorch; it is
  **not** an on-device or on-MCU measurement (no LiteRT-for-Microcontrollers
  run, no latency/energy figures). The on-device numbers for these
  encoders remain the Qualcomm AI Hub Galaxy S24 profiles.
- 32 windows per application (both classes represented), not the full
  test sets; the comparison target is the PyTorch checkpoint, which is
  also the artifact the AI Hub compile started from.
- Only the encoders were converted — the gateway decoder stays
  gateway-side (PyTorch here) by design.
