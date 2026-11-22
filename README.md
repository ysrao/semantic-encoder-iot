# Semantic and Task-Oriented Encoder/Decoder Pair for IoT Sensors — simulation code

The simulation bench and training code behind the paper *"Semantic and Task-Oriented
Encoder/Decoder Pair for IoT Sensors"* (Yenamandra, Nsofu & Ram).

One software pair — a thin learned encoder on the device (64→48→24→8 MLP,
4,496 parameters, 4.4 KB quantised, 8-byte messages from 256-byte windows) and its
gateway decoder — exercised on **one simulation bench across four tasks**. Only the
task changes: per use case, a new trace generator, channel meanings and labels;
the encoder architecture, training recipe and scoring discipline are identical.

## Layout

| Folder | Contents |
|---|---|
| `flood_synthetic/` | The pair implementation (`semantic_encoder.py`), flood training on the companion study's synthetic traces (`train_semantic.py`), trained weights, training report |
| `flood_india/india/` | The real-data flood study: six Indian river stations — water levels from the GUARDIAN dataset (figshare, DOI 10.6084/m9.figshare.27004282), station metadata and danger levels from INDOFLOODS (Zenodo, DOI 10.5281/zenodo.14584654; IIT Delhi). The GUARDIAN data itself is not mirrored here: fetch the raw download from figshare and run `train_semantic_india.py` to build the processed training set (train 2020–22 / validation 2023 / test 2024). See `DATA_PROVENANCE.md` |
| `outage/` | Outage detection on synthetic 1 Hz voltage traces (`train_outage.py`), results and training report |
| `metering/` | Leak detection on synthetic household water traces (`train_metering.py`), results and training report. `run1_tod_context/` archives the first, structurally flawed run (time-of-day channel), kept for transparency |
| `parking/` | Parking occupancy on synthetic magnetometer traces (`train_parking.py`), results and training report |
| `companion_traces/` | The companion study's trace generators and rule functions (`ntn_eval` package) — the source of the flood synthetic traces and of the reference (hand-crafted) encoder reported in the paper as reference [1] |
| `litert_check/` | The LiteRT portability check: all four task encoders converted locally and verified against the reference models (identical int8 codes; decoder decisions agree on all test windows). See `litert_check.md` |
| `storm/` | **v3, real data:** cyclone landfall within 24 h from IBTrACS North Indian best tracks (`build_storm.py`, `results.json`) |
| `soil/` | **v3, real data:** soil dry-down crossing within 48 h, ERA5-Land at Indore / Ludhiana / Nagpur via Open-Meteo (`build_soil.py`, `results.json`) |
| `monsoon/` | **v3, real data:** monsoon onset over Kerala within 5 days; labels are IMD's declared onset dates 2009–2025, predictors ERA5 via Open-Meteo (`build_monsoon.py`, `results.json`) |

## Version 3 — all-real evaluation

Version 3 of the paper removes the simulations: its four applications, in the
paper's order, are soil dry-down (`soil/`), dust storms (`dust/`), cyclone
landfall (`storm/`) and flood on real Indian river data (`flood/`, trained on
the windows built by `flood_india/`) — all real public records, each compared
against a hand-crafted encoder on identical windows, with chronological splits
throughout and one common training recipe and seed across all four. A fifth
build, monsoon onset (`monsoon/`), is the boundary experiment reported in the
paper's Discussion. Each v3 script reads its public record, builds the
windows, trains the same pair with the same recipe, and writes `results.json`.
Trained weights are produced by running the scripts (not mirrored here).

## Running a study

Each task folder is self-contained: `python train_<task>.py` regenerates the
traces from fixed seeds, trains the pair jointly (encoder + gateway decoder,
int8 quantisation in the loop), scores it on held-out traces, and writes its
results. Splits are always by trace/seed, never by shuffled windows.

Requirements: Python 3, PyTorch (CPU is sufficient), NumPy.

## Honest scope

- Flood is evaluated on **real data** (India) as well as synthetic traces.
- Outage, metering and parking are evaluated on **simulated data** from the
  seeded generators in this repo, with injected events and known ground truth.
  No open real-world dataset exists for these three tasks in a form the pair
  can consume without changing the task (checked November 2026).
- The reference encoder reported alongside the learned pair is the companion
  study's hand-crafted encoder, used unchanged; it is not a design of the paper.

## Authors

Rao S. Yenamandra (DigitalTwinSim) · Mubanga Nsofu (Vodacom) · Asokan Ram
(Wireless Research Centre, Wake Forest, NC)
