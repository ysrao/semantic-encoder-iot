"""Train the semantic encoder + gateway decoder on REAL Indian river data.

Same model and recipe as train_semantic.py (the simulation run) — encoder
64->48->24->8 unchanged, gateway decoder 8->24->48 (task + recon heads),
int8 latent in the loop via straight-through estimator, Gaussian channel
noise sigma=0.03, loss = pos-weighted BCE + 0.05*MSE, Adam lr 1e-3,
batch 256, early stop on validation F1, threshold selected on validation.
Only the DATA changes: real CWC gauge series instead of study generators.

DATA (see INDIA_DATA_FINDINGS.md / DATA_PROVENANCE.md in this folder):
  * Water level (WSE, hourly grid): GUARDIAN archive — CWC telemetry
    extracted from the CWC flood portal by Patidar et al. (IIT Bombay),
    figshare DOI 10.6084/m9.figshare.27004282; six stations, 2020-01-01
    to 2024-07-31, timestamps IST.
  * Threshold: official CWC danger level per station from INDOFLOODS
    (Kuntla & Saharia 2025), Zenodo DOI 10.5281/zenodo.14584654, CC BY 4.0.
    (Akhuapada's INDOFLOODS warning level duplicates its danger level;
    danger only is used, for all stations.)
  * Rainfall + temperature channels: Open-Meteo archive API (ERA5),
    hourly at the station coordinates, timezone Asia/Kolkata, fetched by
    fetch_weather_india.py and cached as weather_<station>.json.
    Normalisation identical to the simulation run:
    rain = clip(rain_mm_h/20, 0, 3), temp = temp_C/50.

CLEANING (documented spike rule, applied to the recorded series in time
order before gridding): a maximal run of <=2 consecutive readings is a
telemetry/datum glitch if the readings immediately before and after the
run agree within 0.5 m and every reading in the run deviates from the
flank median by > 2.0 m. Flagged readings are set to missing. (This is
the exact pathology present in the raw archive: isolated 1-2 reading
dips/spikes of 2.25-20 m against locally stable levels. A genuine flood
rise cannot return to the flank level within 2 readings, so real events
are untouched.) Counts removed per station are reported.

GRID: hourly, 2020-01-01 00:00 to 2024-07-31 23:00 IST (40,176 steps).
Missing runs of <=4 h are linearly interpolated (bridges short telemetry
dropouts); longer gaps stay missing. A window is used only if all 20
timesteps it touches (16 window + 4 label horizon) are present after
this step.

LABEL: the study's ground-truth latch (ntn_eval.traces.gen_env_trace)
applied to the cleaned observed level: an episode starts after
persist=2 consecutive hourly readings above the danger level (onset
backfilled to the first above reading) and ends when the level recedes
below danger - 0.3 m (the study's flood_hyst_m). Latch state resets
across data gaps. Label of a window ending at i = 1 if the latched
flood state is active at any of i+1..i+4 ("a flood event occurs within
the next 4 timesteps").

RULE BASELINE on identical windows: the study's own stage3.rule_above
(persist=2, hysteresis=0.3 m, causal smoothing 6 — the env task
parameters) applied per contiguous observed segment with threshold =
danger level; prediction read at the window's final timestep. Causal —
same information as the NN.

WINDOWS: causal, length 16, stride 8, channels per timestep
[level/danger, 1.0 (threshold), rain_norm, temp_norm], row-major —
the encoder's existing (1, 64) contract.

SPLITS by window-END timestamp, all stations pooled:
  train: ends in 2020-2022; validation: ends in 2023;
  test: ends in 2024 (Jan-Jul).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)                       # semantic_encoder/
ROOT = os.path.dirname(BASE)                       # edge-apps-prep/
SRC = os.path.join(ROOT, "semantic_traces_src", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, BASE)

from ntn_eval import stage3                        # noqa: E402
from semantic_encoder import SemanticEncoder, INPUT_DIM, LATENT_DIM  # noqa: E402

WINDOW = 16
STRIDE = 8
HORIZON = 4
NOISE_SIGMA = 0.03
RECON_WEIGHT = 0.05
PERSIST = 2          # hourly samples (study rule persist_confirm; GT latch)
HYST = 0.3           # m (study flood_hyst_m / task hysteresis)
SMOOTH = 6           # study env task causal smoothing

STATIONS = [
    # name, file, river, state, lat, lon, danger_m
    ("Jainagar", "Jainagar2020_2024_July.xlsx", "Kamla-Balan", "Bihar",
     26.583, 86.148, 67.75),
    ("Jhanjharpur", "Jhanjharpur2020_2024_July.xlsx", "Kamla-Balan", "Bihar",
     26.233, 86.259, 50.00),
    ("Benibad", "Benibad2020_2024_July.xlsx", "Bagmati", "Bihar",
     26.145, 85.692, 48.68),
    ("Kallooppara", "KALLOOPPARA2020_2024_July.xlsx", "Manimala", "Kerala",
     9.404, 76.650, 6.00),
    ("Akhuapada", "Akhuapada2020_2024_July.xlsx", "Baitarani", "Odisha",
     20.916, 86.275, 17.83),
    ("Kunavaram", "Kunavaram2020_2024_July.xlsx", "Godavari",
     "Andhra Pradesh", 17.573, 81.250, 39.24),
]

GRID_START = "2020-01-01 00:00"
GRID_END = "2024-07-31 23:00"


# ---------------------------------------------------------------- cleaning

def spike_clean(w):
    """Isolated-deviation rule (see module docstring). Returns (cleaned, n_flagged)."""
    w = w.astype(float).copy()
    flagged = np.zeros(len(w), bool)
    i = 1
    while i < len(w) - 1:
        hit = False
        for L in (1, 2):
            j = i + L - 1
            if j + 1 >= len(w):
                break
            flank_lo, flank_hi = w[i - 1], w[j + 1]
            if abs(flank_lo - flank_hi) <= 0.5:
                med = 0.5 * (flank_lo + flank_hi)
                if np.all(np.abs(w[i:j + 1] - med) > 2.0):
                    flagged[i:j + 1] = True
                    i = j + 1
                    hit = True
                    break
        if not hit:
            i += 1
    w[flagged] = np.nan
    return w, int(flagged.sum())


def flood_latch(level, valid, danger, persist=PERSIST, hyst=HYST):
    """Study GT latch (traces.gen_env_trace) on the cleaned observed series."""
    n = len(level)
    F = np.zeros(n, bool)
    run, in_f = 0, False
    for i in range(n):
        if not valid[i]:
            run, in_f = 0, False
            continue
        run = run + 1 if level[i] > danger else 0
        if not in_f and run >= persist:
            in_f = True
            F[i - persist + 1:i + 1] = True
        elif in_f:
            if level[i] < danger - hyst:
                in_f = False
            else:
                F[i] = True
    return F


def rule_membership(level, valid, danger):
    """Study stage3.rule_above per contiguous valid segment -> bool array."""
    n = len(level)
    mem = np.zeros(n, bool)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        seg = level[i:j]
        if len(seg) >= 2:
            eps = stage3.rule_above(np.arange(len(seg)), seg, float(danger),
                                    PERSIST, HYST, SMOOTH)
            for s, e in eps:
                mem[i + s:i + e + 1] = True
        i = j
    return mem


# ---------------------------------------------------------------- data build

def load_station(name, fname, danger):
    path = os.path.join(HERE, "stations", fname)
    df = pd.read_excel(path, usecols=["time", "WSE"])
    t = pd.to_datetime(df["time"])
    assert (t.dt.minute == 0).all() and (t.dt.second == 0).all(), name
    w_raw = pd.to_numeric(df["WSE"], errors="coerce").values
    order = np.argsort(t.values)
    t, w_raw = t.values[order], w_raw[order]
    w_clean, n_flag = spike_clean(w_raw)

    grid = pd.date_range(GRID_START, GRID_END, freq="1h")
    s = pd.Series(w_clean, index=pd.DatetimeIndex(t)).reindex(grid)
    n_obs = int(s.notna().sum())
    s_interp = s.interpolate(method="linear", limit=4, limit_area="inside")
    level = s_interp.values.astype(float)
    valid = ~np.isnan(level)
    n_interp = int(valid.sum() - n_obs)

    with open(os.path.join(HERE, f"weather_{name}.json")) as f:
        wx = json.load(f)["hourly"]
    wx_t = pd.DatetimeIndex(pd.to_datetime(wx["time"]))
    assert len(wx_t) == len(grid) and wx_t[0] == grid[0] and wx_t[-1] == grid[-1], name
    rain = np.asarray(wx["rain"], dtype=float)          # mm/h (ERA5)
    temp = np.asarray(wx["temperature_2m"], dtype=float)  # deg C

    F = flood_latch(level, valid, danger)
    rule = rule_membership(level, valid, danger)
    info = {"station": name, "records_raw": int(len(w_raw)),
            "spikes_removed": n_flag, "grid_hours": int(len(grid)),
            "hours_observed": n_obs, "hours_interpolated_le4h": n_interp,
            "hours_valid": int(valid.sum()),
            "latched_flood_hours": int(F.sum()),
            "latched_episodes": int((np.diff(F.astype(int)) == 1).sum() + (1 if F[0] else 0))}
    return grid, level, valid, rain, temp, F, rule, info


def build_all():
    Xs, ys, rs, ss, ee = [], [], [], [], []   # features, label, rule, station idx, end idx
    per_station = {}
    station_data = {}
    for si, (name, fname, river, state, lat, lon, danger) in enumerate(STATIONS):
        grid, level, valid, rain, temp, F, rule, info = load_station(name, fname, danger)
        station_data[name] = {"grid": grid, "level": level, "valid": valid,
                              "F": F, "danger": danger, "river": river}
        rain_n = np.clip(rain / 20.0, 0, 3)
        temp_n = temp / 50.0
        feats = np.stack([level / danger, np.ones(len(grid)), rain_n, temp_n], axis=-1)
        cnt, pos = 0, 0
        for i in range(WINDOW - 1, len(grid) - HORIZON, STRIDE):
            if not valid[i - WINDOW + 1:i + HORIZON + 1].all():
                continue
            Xs.append(feats[i - WINDOW + 1:i + 1].reshape(-1))
            lab = int(F[i + 1:i + HORIZON + 1].any())
            ys.append(lab); pos += lab
            rs.append(int(rule[i]))
            ss.append(si); ee.append(i)
            cnt += 1
        info["windows"] = cnt
        info["windows_positive"] = pos
        per_station[name] = info
        print(f"{name}: raw {info['records_raw']}, spikes {info['spikes_removed']}, "
              f"valid hours {info['hours_valid']}/{info['grid_hours']}, "
              f"episodes {info['latched_episodes']}, windows {cnt} (pos {pos})",
              flush=True)
    return (np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64),
            np.asarray(rs, dtype=np.int64), np.asarray(ss, dtype=np.int64),
            np.asarray(ee, dtype=np.int64), per_station, station_data)


# ---------------------------------------------------------------- model

class GatewayDecoder(nn.Module):
    """Identical to train_semantic.py: 8 -> 24 -> 48 -> task head (+ recon)."""

    def __init__(self, latent_dim: int = LATENT_DIM, input_dim: int = INPUT_DIM):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent_dim, 24), nn.ReLU(),
            nn.Linear(24, 48), nn.ReLU(),
        )
        self.task_head = nn.Linear(48, 1)
        self.recon_head = nn.Linear(48, input_dim)

    def forward(self, z):
        h = self.body(z)
        return self.task_head(h).squeeze(-1), self.recon_head(h)


def quantise_ste(z):
    q = torch.clamp(torch.round(z * 127.0) / 127.0, -1.0, 1.0)
    return z + (q - z).detach()


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return {"accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(f1), 4),
            "false_alarm_rate": round(float(far), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true)),
            "positives": int(y_true.sum())}


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    torch.manual_seed(20261118)
    np.random.seed(20261118)

    print("building dataset from CWC/GUARDIAN series...", flush=True)
    X, y, r, sidx, eidx, per_station, station_data = build_all()
    # split by window-end timestamp
    end_year = np.array([station_data[STATIONS[si][0]]["grid"][ei].year
                         for si, ei in zip(sidx, eidx)])
    splits = {"train": end_year <= 2022, "val": end_year == 2023,
              "test": end_year >= 2024}
    counts = {}
    for k, m in splits.items():
        counts[k] = {"n": int(m.sum()), "positives": int(y[m].sum())}
        per_st = {}
        for si, st in enumerate(STATIONS):
            mm = m & (sidx == si)
            per_st[st[0]] = {"n": int(mm.sum()), "positives": int(y[mm].sum())}
        counts[k]["per_station"] = per_st
    print(json.dumps(counts, indent=2), flush=True)
    np.savez(os.path.join(HERE, "dataset_india.npz"), X=X, y=y, rule=r,
             station=sidx, end_idx=eidx, end_year=end_year)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    Xtr_t = torch.from_numpy(X[tr]); ytr_t = torch.from_numpy(y[tr]).float()
    Xva_t = torch.from_numpy(X[va]); Xte_t = torch.from_numpy(X[te])
    pos = float(y[tr].sum()); neg = float(tr.sum() - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos))

    enc = SemanticEncoder()
    dec = GatewayDecoder()
    params = list(enc.parameters()) + list(dec.parameters())
    print(f"encoder params {sum(p.numel() for p in enc.parameters())}, "
          f"decoder params {sum(p.numel() for p in dec.parameters())}", flush=True)
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()

    def predict(X_t, noisy=False):
        enc.eval(); dec.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), 4096):
                xb = X_t[i:i + 4096]
                z = enc(xb)
                zq = torch.clamp(torch.round(z * 127.0) / 127.0, -1, 1)
                if noisy:
                    zq = zq + torch.randn_like(zq) * NOISE_SIGMA
                logit, _ = dec(zq)
                probs.append(torch.sigmoid(logit).numpy())
        return np.concatenate(probs)

    best_f1, best_state, best_thr, bad = -1.0, None, 0.5, 0
    n = len(Xtr_t)
    for epoch in range(40):
        enc.train(); dec.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            z = enc(xb)
            zq = quantise_ste(z) + torch.randn_like(z) * NOISE_SIGMA
            logit, recon = dec(zq)
            loss = bce(logit, yb) + RECON_WEIGHT * mse(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        pv = predict(Xva_t)
        thrs = np.linspace(0.05, 0.95, 19)
        f1s = [metrics(y[va], (pv >= t).astype(int))["f1"] for t in thrs]
        k = int(np.argmax(f1s))
        print(f"epoch {epoch}: loss {tot/n:.4f} val_f1 {f1s[k]:.4f} @thr {thrs[k]:.2f}",
              flush=True)
        if f1s[k] > best_f1:
            best_f1, best_thr, bad = f1s[k], float(thrs[k]), 0
            best_state = ({k2: v.clone() for k2, v in enc.state_dict().items()},
                          {k2: v.clone() for k2, v in dec.state_dict().items()})
        else:
            bad += 1
            if bad >= 8:
                print("early stop", flush=True)
                break

    enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    print(f"best val F1 {best_f1:.4f} @ threshold {best_thr}", flush=True)

    # ---- test evaluation: NN pair vs rule, identical windows ----
    p_clean = predict(Xte_t, noisy=False)
    p_noisy = predict(Xte_t, noisy=True)
    yte, rte, ste = y[te], r[te], sidx[te]
    res = {"overall": {
        "nn_pair": metrics(yte, (p_clean >= best_thr).astype(int)),
        "nn_pair_noisy_latent": metrics(yte, (p_noisy >= best_thr).astype(int)),
        "rule_based": metrics(yte, rte)}}
    per_st_metrics = {}
    for si, st in enumerate(STATIONS):
        m = ste == si
        if m.sum() == 0:
            continue
        per_st_metrics[st[0]] = {
            "nn_pair": metrics(yte[m], (p_clean[m] >= best_thr).astype(int)),
            "rule_based": metrics(yte[m], rte[m])}
    res["per_station"] = per_st_metrics
    enc.eval(); dec.eval()
    with torch.no_grad():
        zq = torch.clamp(torch.round(enc(Xte_t) * 127.0) / 127.0, -1, 1)
        _, recon = dec(zq)
        recon_mse = float(mse(recon, Xte_t))
    print(json.dumps(res, indent=2), flush=True)
    print(f"recon MSE (test): {recon_mse:.5f}", flush=True)

    # ---- exports ----
    enc.eval(); dec.eval()
    example = torch.zeros(1, INPUT_DIM)
    with torch.no_grad():
        _ = enc(example)
    traced_enc = torch.jit.trace(enc, example)
    enc_path = os.path.join(HERE, "semantic_encoder_india_trained.pt")
    traced_enc.save(enc_path)

    class DecoderTaskOnly(nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
        def forward(self, z):
            logit, _ = self.d(z)
            return logit.unsqueeze(-1)

    traced_dec = torch.jit.trace(DecoderTaskOnly(dec), torch.zeros(1, LATENT_DIM))
    dec_path = os.path.join(HERE, "gateway_decoder_india_trained.pt")
    traced_dec.save(dec_path)

    onnx_path = os.path.join(HERE, "semantic_encoder_india_trained.onnx")
    torch.onnx.export(enc, example, onnx_path, opset_version=14,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    # ---- animation window (Phase 4): Bihar first, Jainagar preferred ----
    anim = pick_animation_window(X, y, sidx, eidx, end_year, station_data,
                                 enc, dec, best_thr)
    with open(os.path.join(HERE, "animation_window_india.json"), "w") as f:
        json.dump(anim, f, indent=2)
    print("animation window:", json.dumps({k: anim[k] for k in
          ("station", "split", "window_end_time_ist", "decoder_probability",
           "prediction_nn", "rise_over_window_m")}), flush=True)

    report = {
        "data_route": (
            "REAL gauge data. Water level: GUARDIAN archive (CWC telemetry "
            "from the CWC flood portal, extracted by Patidar et al., IIT "
            "Bombay), figshare DOI 10.6084/m9.figshare.27004282. Thresholds: "
            "official CWC danger levels from INDOFLOODS, Zenodo DOI "
            "10.5281/zenodo.14584654 (CC BY 4.0). Weather channels: Open-Meteo "
            "archive API (ERA5) hourly rain + temperature_2m at station "
            "coordinates, timezone Asia/Kolkata, fetched 2026-11-18 by "
            "fetch_weather_india.py (cached weather_<station>.json). No "
            "station needed the documented rate-of-change substitution — "
            "weather fetched successfully for all six stations."),
        "data_nature": "Real CWC gauge measurements 2020-01-01 to 2024-07-31 (IST).",
        "stations": [{"name": st[0], "river": st[2], "state": st[3],
                      "lat": st[4], "lon": st[5], "danger_level_m": st[6],
                      **per_station[st[0]]} for st in STATIONS],
        "cleaning": {
            "spike_rule": ("run of <=2 consecutive readings whose flanking "
                           "readings agree within 0.5 m while every run "
                           "reading deviates > 2.0 m from the flank median "
                           "-> set to missing"),
            "grid": "hourly 2020-01-01 00:00 to 2024-07-31 23:00 IST (40,176 h)",
            "gap_rule": ("missing runs <= 4 h linearly interpolated; longer "
                         "gaps stay missing; a window is used only if all 20 "
                         "timesteps (16 window + 4 horizon) are present")},
        "windows": {"length": WINDOW, "stride": STRIDE, "horizon": HORIZON,
                    "label": ("latched flood state (study GT latch: persist 2 "
                              "above danger, onset backfilled, episode ends "
                              "below danger - 0.3 m; state resets across gaps) "
                              "active at any of the next 4 timesteps"),
                    "normalisation": ("primary = level / station danger level; "
                                      "threshold channel = 1.0; rain = "
                                      "clip(rain_mm_h/20, 0, 3); temp = temp_C/50")},
        "splits_by_window_end_year": {"train": "2020-2022", "val": "2023",
                                      "test": "2024-01-01 to 2024-07-31"},
        "dataset_sizes": counts,
        "model": {"encoder": "SemanticEncoder MLP 64->48->24->8 tanh (4,496 params, architecture unchanged)",
                  "decoder": "GatewayDecoder 8->24->48, task head 48->1, recon head 48->64",
                  "training": ("identical recipe to the simulation run: joint "
                               "end-to-end; int8 quantisation IN THE LOOP via "
                               "straight-through estimator; Gaussian channel "
                               "noise sigma=0.03 on quantised latent during "
                               "training; loss = pos-weighted BCE + 0.05*MSE"
                               "(recon); Adam lr 1e-3, batch 256, early stop "
                               "on val F1 (patience 8, max 40 epochs)"),
                  "decision_threshold_from_val": best_thr,
                  "best_val_f1": round(float(best_f1), 4)},
        "test_metrics": res,
        "recon_mse_test": round(recon_mse, 5),
        "payload_comparison": {"raw_window": "64 float32 values = 256 bytes",
                                "nn_latent": "8 int8 values = 8 bytes",
                                "ratio_vs_raw_window_bytes": 32.0},
        "animation_window": {k: anim[k] for k in
                             ("station", "river", "split", "window_end_time_ist",
                              "crossing_time_ist", "decoder_probability",
                              "prediction_nn", "ground_truth_label")},
        "artifacts": {"encoder_traced": enc_path, "encoder_onnx": onnx_path,
                      "decoder_traced": dec_path,
                      "dataset": os.path.join(HERE, "dataset_india.npz"),
                      "animation": os.path.join(HERE, "animation_window_india.json")},
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "training_report_india.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


def pick_animation_window(X, y, sidx, eidx, end_year, station_data, enc, dec, thr):
    """Best visual crossing: Bihar first (Jainagar > Benibad > Jhanjharpur),
    test split preferred. Window ends 0-4 steps after a latched-episode
    onset (the danger crossing); steepest rise wins; NN-correct preferred.
    Segment: 60 hourly points ending 15 steps after the window end."""
    enc.eval(); dec.eval()
    name_of = {i: st[0] for i, st in enumerate(STATIONS)}
    pref_station = ["Jainagar", "Benibad", "Jhanjharpur",
                    "Kallooppara", "Akhuapada", "Kunavaram"]
    pref_split = ["test", "val", "train"]

    def split_of(si, ei):
        yr = station_data[name_of[si]]["grid"][ei].year
        return "test" if yr >= 2024 else ("val" if yr == 2023 else "train")

    # map (station, end_idx) -> row in X
    row_of = {(int(s), int(e)): k for k, (s, e) in enumerate(zip(sidx, eidx))}
    candidates = []
    for si, st in enumerate(STATIONS):
        name = st[0]
        sd = station_data[name]
        F, level, valid = sd["F"], sd["level"], sd["valid"]
        onsets = np.where(np.diff(F.astype(int)) == 1)[0] + 1
        if F[0]:
            onsets = np.concatenate([[0], onsets])
        for c in onsets:
            for i in range(c, c + 5):
                k = row_of.get((si, int(i)))
                if k is None:
                    continue
                seg_lo, seg_hi = i - 44, i + 15   # 60 points: [i-44, i+15]
                if seg_lo < 0 or seg_hi >= len(level):
                    continue
                if not valid[seg_lo:seg_hi + 1].all():
                    continue
                rise = float(level[i] - level[i - WINDOW + 1])
                candidates.append({"station": name, "si": si, "end_idx": int(i),
                                   "crossing_idx": int(c), "row": int(k),
                                   "split": split_of(si, int(i)), "rise": rise})
    if not candidates:
        raise RuntimeError("no animation candidates found")
    with torch.no_grad():
        for cd in candidates:
            xb = torch.from_numpy(X[cd["row"]:cd["row"] + 1])
            z = enc(xb)
            zq = torch.clamp(torch.round(z * 127.0) / 127.0, -1, 1)
            logit, _ = dec(zq)
            cd["prob"] = float(torch.sigmoid(logit))
            cd["latent_int8"] = [int(v) for v in
                                 torch.clamp(torch.round(z * 127.0), -127, 127)[0].tolist()]
            cd["pred"] = int(cd["prob"] >= thr)
    def sort_key(cd):
        return (pref_station.index(cd["station"]), pref_split.index(cd["split"]),
                0 if cd["pred"] == 1 else 1, -cd["rise"])
    candidates.sort(key=sort_key)
    best = candidates[0]
    sd = station_data[best["station"]]
    grid, level = sd["grid"], sd["level"]
    i, c = best["end_idx"], best["crossing_idx"]
    seg_lo, seg_hi = i - 44, i + 15
    fmt = "%Y-%m-%dT%H:%M:%S"
    return {
        "station": best["station"],
        "river": sd["river"],
        "state": dict((st[0], st[3]) for st in STATIONS)[best["station"]],
        "danger_level_m": sd["danger"],
        "split": best["split"],
        "window_end_time_ist": grid[i].strftime(fmt),
        "crossing_time_ist": grid[c].strftime(fmt),
        "window_levels_m": [round(float(v), 3) for v in level[i - WINDOW + 1:i + 1]],
        "segment_times_ist": [grid[j].strftime(fmt) for j in range(seg_lo, seg_hi + 1)],
        "segment_levels_m": [round(float(v), 3) for v in level[seg_lo:seg_hi + 1]],
        "segment_window_position": {"window_start_in_segment": 44 - WINDOW + 1,
                                    "window_end_in_segment": 44},
        "latent_int8": best["latent_int8"],
        "decoder_probability": round(best["prob"], 4),
        "decision_threshold": thr,
        "prediction_nn": best["pred"],
        "ground_truth_label": 1,
        "ground_truth": ("latched flood episode onset (danger-level crossing) "
                         f"at {grid[c].strftime(fmt)} IST; flood state active "
                         "within the 4 timesteps after the window end"),
        "rise_over_window_m": round(best["rise"], 3),
        "selection_note": ("chosen by preference order: Bihar stations first "
                           "(Jainagar > Benibad > Jhanjharpur), test split "
                           "before val/train, NN-correct detections first, "
                           "then steepest level rise across the window; "
                           "60-point segment fully observed (no gaps)"),
    }


if __name__ == "__main__":
    main()
