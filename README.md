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
| `extras/` | Paper v2.0 additions: `robustness.py` (five training seeds, int8-weight encoders, a hand-picked 8-byte baseline; Section IV-F, Table III), `airtime.py` (LoRaWAN time on air and energy; Section V, Table IV), `figure1.py` (Figure 1), and their saved results |
| `litert_check/` | The LiteRT portability check: all four task encoders converted locally and verified against the reference models (identical int8 codes; decoder decisions agree on all test windows). See `litert_check.md` |

## Running a study

Install the requirements once (Python 3.10 or newer; CPU is enough):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run each study from the repository root. Each one regenerates its traces
from fixed seeds, trains the pair jointly (encoder + gateway decoder, int8
quantisation in the loop), scores it against the reference rule on the same
held-out windows, and writes its results and trained weights into its folder.
Splits are always by trace/seed, never by shuffled windows. Each takes seconds.

```bash
python flood_synthetic/train_semantic.py
python outage/train_outage.py
python metering/train_metering.py
python parking/train_parking.py
```

The scripts find the shared encoder (`flood_synthetic/semantic_encoder.py`) and
the companion study's rules (`companion_traces/src/ntn_eval`) by their place in
the repository, so they run from any working directory.

**India.** The river data is not mirrored here; fetch it from figshare (see
`flood_india/india/DATA_PROVENANCE.md`) and copy the six station files the
paper uses (`Jainagar2020_2024_July.xlsx`, `Jhanjharpur2020_2024_July.xlsx`,
`Benibad2020_2024_July.xlsx`, `KALLOOPPARA2020_2024_July.xlsx`,
`Akhuapada2020_2024_July.xlsx`, `Kunavaram2020_2024_July.xlsx`) into
`flood_india/india/stations/`. Then build the windows and train v1, then v2:

```bash
python flood_india/india/train_semantic_india.py      # writes dataset_india.npz
python flood_india/india/train_semantic_india_v2.py   # catchment attributes at the decoder
```

The published v1 weights (`semantic_encoder_india_trained.pt`,
`gateway_decoder_india_trained.pt`) reproduce the paper's v1 figures exactly on
that dataset.

**Paper v2.0 checks.** After the runs above (they write the `dataset_*.npz`
files it reads; India is skipped if its dataset has not been built):

```bash
python extras/robustness.py      # Table III: 5 seeds, int8 weights, hand-picked 8-byte baseline
python extras/airtime.py         # Table IV: LoRaWAN time on air and energy
```

**Reproducibility.** Re-running reproduces every reference-rule figure exactly.
The learned pair's figures reproduce exactly for outage, parking and flood
simulation; metering and India v2 differ slightly between machines (metering F1
0.857 on the original run, 0.864 on an Apple-silicon Mac), a training
difference that changes no ranking. `extras/robustness.py` reports the spread
over five training seeds.

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
