"""Third simulated application for the task-oriented semantic
encoder/decoder paper: WATER METERING / LEAK DETECTION from a
household supply meter.

Mirrors the flood-warning build (../semantic_encoder/train_semantic.py)
and the outage build (../outage_sim/train_outage.py) in architecture,
training recipe, rule baseline and scoring discipline:

  * SAME encoder architecture (MLP 64 -> 48 -> 24 -> 8 tanh latent)
    and SAME gateway decoder design, trained jointly.
  * SAME recipe: int8 quantisation in the loop (straight-through
    estimator), Gaussian latent noise sigma=0.03 in training,
    pos-weighted BCE + 0.05 * MSE(recon), Adam lr 1e-3, batch 256,
    early stop on validation F1 (patience 8, max 40 epochs), decision
    threshold swept on validation.
  * SAME rule-baseline family: threshold + persistence + hysteresis +
    causal smoothing (the study's stage3 rule family; the "above"
    variant here, since a leak is flow that stays ABOVE a minimum),
    applied to the primary channel only, read at the window's final
    timestep — same information discipline as the earlier builds.
  * Splits by trace seed, never random interleaving.

DATA: fully SYNTHETIC household water traces generated here (no field
data). 1 sample per minute, 2,880 samples (48 h) per trace. Channels
per timestep (the encoder's 16x4 = 64-float window contract):
  ch0 flow rate, normalised (L/min / 20)
  ch1 cumulative volume since midnight, normalised (L / 500)
  ch2 supply pressure, normalised (nominal 1.0 = 3 bar)
  ch3 continuous-flow duration: minutes since the flow last fell to
      (near) zero, normalised (/240, clipped at 1) — the continuous-
      flow register a commercial smart meter itself maintains, and
      the meter's time/context term.

DESIGN NOTE (information parity): the first configuration of this
build used time-of-day as ch3. It was replaced before the headline
run because it broke the family's comparison discipline: the rule's
90-minute persistence integrates an hour and a half of history,
while the encoder sees only a 16-minute window — the rule was given
temporal information the NN could not access (in the outage build
the rule's persistence of 3 samples sat inside the window: "same
information as the NN"). The continuous-flow register restores
parity — the rule's persistence counter and the NN's ch3 encode the
same underlying quantity, and the NN additionally sees rate, volume
and pressure. Run 1 (time-of-day ch3) is archived under
run1_tod_context/ with its numbers reported in results.md; the
change is to the input design, not to any hyperparameter, and no
hyperparameter was tuned against the rule in either run.
Trace construction: a daily usage pattern (morning and evening peaks,
quiet nights) built from discrete draws — showers, baths, taps,
toilet flushes, cyclic laundry/dishwasher loads — plus two long-draw
CONFUSERS that are not leaks (garden irrigation, tank filling) and
injected LEAKS: a sustained low flow (0.4-2.5 L/min) that never fully
stops, lasting 3-15 h.
GROUND TRUTH: leak state = 1 exactly on the injected leak intervals
(generative truth, not a threshold on the observed series). Window
label = ground-truth state at the window's final timestep (the family
convention).

TASK (paper framing): the device transmits either the raw window
(256 B), a hand-crafted rule report (20 B, study budget: event type,
onset, mean flow, duration, flags) at each rule episode start, or the
learned 8-byte int8 latent per window; the gateway decides "leak
active" from what it receives.

One principled run. Hyperparameters are the family recipe verbatim —
nothing is tuned against the test set or against the rule.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))

WINDOW = 16
STRIDE = 8
TRACE_LEN = 2880          # 48 h at 1 sample/min
DAY = 1440
FLOW_SCALE = 20.0         # L/min -> normalised flow
LEAK_THR = 0.015          # normalised flow = 0.3 L/min (rule threshold)
RULE_PERSIST = 90         # samples (minutes) above thr before firing
RULE_HYST = 0.005         # normalised flow
RULE_SMOOTH = 4           # causal moving-average length
NOISE_SIGMA = 0.03
RECON_WEIGHT = 0.05

TRAIN_SEEDS = [101, 102, 103, 104, 105] + list(range(501, 526))   # 30 traces
VAL_SEEDS = list(range(601, 609))                                  # 8 traces
TEST_SEEDS = [201, 202, 203, 204, 205] + list(range(701, 716))    # 20 traces


# ---------------------------------------------------------------- models
# Architecture identical to the flood/outage builds (SemanticEncoder in
# ../semantic_encoder/semantic_encoder.py): MLP 64 -> 48 -> 24 -> 8,
# tanh latent; gateway decoder 8 -> 24 -> 48 with task + recon heads.

class SemanticEncoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48), nn.ReLU(),
            nn.Linear(48, 24), nn.ReLU(),
            nn.Linear(24, latent_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class GatewayDecoder(nn.Module):
    """Gateway-side mirror decoder: 8 -> 24 -> 48 -> task head (+ recon).
    Identical to the flood/outage builds' decoder."""

    def __init__(self, latent_dim=8, input_dim=64):
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
    """int8 quantisation on the tanh range with straight-through gradient."""
    q = torch.clamp(torch.round(z * 127.0) / 127.0, -1.0, 1.0)
    return z + (q - z).detach()


# ---------------------------------------------------------------- rule
# The study's stage3 rule family (threshold + persistence + hysteresis
# + causal smoothing), "above" variant, re-implemented with the same
# semantics as stage3.rule_above: causal moving average first; fire
# after `persist` consecutive above-threshold samples; latch resets
# only when the smoothed series falls below thr - hyst.

def _smooth_causal(x, w):
    x = np.asarray(x, float)
    w = int(w)
    if w <= 1 or len(x) == 0:
        return x
    w = min(w, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:w] / np.arange(1, w)
    tail = (c[w:] - c[:-w]) / w
    return np.concatenate([head, tail])


def rule_above(obs, thr, persist, hyst=0.0, smooth=1):
    """Episodes where (smoothed) obs stays > thr for >= persist samples.
    Returns list of (start_idx, end_idx)."""
    obs = _smooth_causal(obs, smooth)
    above = obs > thr
    run = np.zeros(len(obs), dtype=int)
    if len(obs):
        run[0] = 1 if above[0] else 0
    for i in range(1, len(obs)):
        run[i] = run[i - 1] + 1 if above[i] else 0
    fire = run >= persist
    reset = obs < thr - hyst
    episodes, start, in_ev = [], None, False
    for i in range(len(obs)):
        if fire[i] and not in_ev:
            in_ev, start = True, i
        elif in_ev and reset[i]:
            episodes.append((start, i))
            in_ev = False
    if in_ev:
        episodes.append((start, len(obs) - 1))
    return episodes


def _episode_membership(n, episodes):
    m = np.zeros(n, dtype=bool)
    for s, e in episodes:
        m[s:e + 1] = True
    return m


# ---------------------------------------------------------------- generator

def gen_trace(seed):
    """One synthetic 48-hour household water trace. Returns dict of
    channels, the ground-truth leak state, and the injected event log.
    All draws are rectangular pulses (1-min cadence); flow is additive."""
    rng = np.random.default_rng(seed)
    n = TRACE_LEN
    flow = np.zeros(n)            # L/min
    gt = np.zeros(n, dtype=np.int64)
    events = []

    def add_pulse(start, dur, rate, kind):
        s = int(start); d = max(1, int(round(dur)))
        e = min(n, s + d)
        if s >= n or s < 0:
            return
        flow[s:e] += rate
        events.append({"kind": kind, "start": s, "end": e - 1,
                       "rate_lpm": round(float(rate), 2)})

    def cyclic(start, total_dur, kind):
        """Laundry/dishwasher: bursts of flow separated by zero-flow
        gaps (the gaps are what a persistence rule sees)."""
        tcur = int(start)
        end = min(n, int(start) + int(total_dur))
        while tcur < end:
            burst = rng.integers(2, 5)
            add_pulse(tcur, burst, rng.uniform(3.0, 6.0), kind)
            tcur += int(burst) + int(rng.integers(3, 9))

    for day in range(2):
        base = day * DAY
        # --- morning peak 05:30-09:30 ---
        for _ in range(rng.integers(1, 4)):                      # showers
            add_pulse(base + rng.integers(330, 570),
                      rng.uniform(5, 15), rng.uniform(7, 12), "shower")
        if rng.random() < 0.30:                                  # bath
            add_pulse(base + rng.integers(360, 600),
                      rng.uniform(25, 60), rng.uniform(10, 18), "bath")
        # --- daytime sporadic ---
        for _ in range(rng.poisson(7)):                          # taps
            add_pulse(base + rng.integers(480, 1080),
                      rng.uniform(0.5, 4), rng.uniform(2, 6), "tap")
        for _ in range(rng.poisson(4)):                          # toilet
            add_pulse(base + rng.integers(360, 1380),
                      rng.uniform(1, 2), rng.uniform(5, 7), "toilet")
        # --- evening peak 17:30-23:00 ---
        for _ in range(rng.integers(1, 3)):
            add_pulse(base + rng.integers(1050, 1380),
                      rng.uniform(5, 15), rng.uniform(7, 12), "shower")
        for _ in range(rng.poisson(5)):
            add_pulse(base + rng.integers(1020, 1410),
                      rng.uniform(0.5, 4), rng.uniform(2, 6), "tap")
        for _ in range(rng.poisson(3)):
            add_pulse(base + rng.integers(1020, 1410),
                      rng.uniform(1, 2), rng.uniform(5, 7), "toilet")
        # --- cyclic appliances ---
        if rng.random() < 0.50:
            cyclic(base + rng.integers(420, 1260),
                   rng.uniform(45, 90), "laundry")
        if rng.random() < 0.40:
            cyclic(base + rng.integers(1020, 1320),
                   rng.uniform(45, 90), "dishwasher")
        # --- long-draw confusers (NOT leaks) ---
        if rng.random() < 0.35:                                  # irrigation
            start_h = rng.choice([rng.uniform(270, 420),         # 04:30-07:00
                                  rng.uniform(1080, 1320)])      # 18:00-22:00
            add_pulse(base + int(start_h),
                      rng.uniform(60, 240), rng.uniform(8, 16), "irrigation")
        if rng.random() < 0.30:                                  # tank fill
            add_pulse(base + rng.integers(420, 1260),
                      rng.uniform(45, 150), rng.uniform(6, 14), "tank_fill")
        # --- rare night toilet flush (still not a leak) ---
        if rng.random() < 0.25:
            add_pulse(base + rng.integers(30, 330),
                      rng.uniform(1, 2), rng.uniform(5, 7), "toilet")

    # --- leaks (ground truth) ---
    n_leak = rng.choice([0, 1, 2], p=[0.40, 0.40, 0.20])
    placed = []
    for _ in range(n_leak):
        dur = int(rng.uniform(180, 900))                         # 3-15 h
        for _try in range(20):
            s = int(rng.integers(30, 2400))
            if all(s > e + 120 or s + dur < b - 120 for b, e in placed):
                break
        else:
            continue
        placed.append((s, s + dur))
        rate0 = rng.uniform(0.4, 2.5)                            # L/min
        e = min(n, s + dur)
        seg = np.full(e - s, rate0)
        if rng.random() < 0.30:                                  # growing leak
            seg = rate0 * np.linspace(1.0, 1.5, e - s)
        flow[s:e] += seg
        gt[s:e] = 1
        events.append({"kind": "leak", "start": s, "end": e - 1,
                       "rate_lpm": round(float(rate0), 2)})

    # --- observed channels ---
    flow_obs = np.clip(flow + rng.normal(0, 0.02, n), 0, None)   # L/min
    cumvol = np.zeros(n)                                         # L since midnight
    for d0 in range(0, n, DAY):
        seg = flow_obs[d0:d0 + DAY]
        cumvol[d0:d0 + DAY] = np.cumsum(seg)
    t = np.arange(n, dtype=float)
    pressure = (1.0 + 0.015 * np.sin(2 * np.pi * t / rng.uniform(1200, 2000)
                                     + rng.uniform(0, 2 * np.pi))
                - 0.12 * np.minimum(flow / 18.0, 1.0)
                + rng.normal(0, 0.004, n))
    pressure = np.clip(pressure, 0.8, 1.1)
    # continuous-flow register: minutes since flow last near zero
    # (deadband 0.05 L/min), computed causally from the observed flow,
    # as a meter's own continuous-flow alarm register would.
    dur = np.zeros(n)
    runlen = 0
    for i in range(n):
        if flow_obs[i] < 0.05:
            runlen = 0
        else:
            runlen += 1
        dur[i] = runlen
    contflow = np.minimum(dur, 240.0) / 240.0
    return {"flow": flow_obs / FLOW_SCALE, "cumvol": cumvol / 500.0,
            "pressure": pressure, "contflow": contflow,
            "gt": gt, "events": events}


# ---------------------------------------------------------------- data

def build_split(seeds):
    X, y, r, ends_all, trace_id = [], [], [], [], []
    ev_counts = {}
    rule_eps_per_trace = []
    for k, seed in enumerate(seeds):
        tr = gen_trace(seed)
        for ev in tr["events"]:
            ev_counts[ev["kind"]] = ev_counts.get(ev["kind"], 0) + 1
        eps = rule_above(tr["flow"], LEAK_THR, RULE_PERSIST,
                         RULE_HYST, RULE_SMOOTH)
        rule_eps_per_trace.append(len(eps))
        rule = _episode_membership(TRACE_LEN, eps)
        feats = np.stack([tr["flow"], tr["cumvol"], tr["pressure"],
                          tr["contflow"]], axis=-1)              # (n,4)
        for i in range(WINDOW - 1, TRACE_LEN, STRIDE):
            X.append(feats[i - WINDOW + 1:i + 1].reshape(-1))
            y.append(int(tr["gt"][i]))
            r.append(int(rule[i]))
            ends_all.append(i)
            trace_id.append(k)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(r, dtype=np.int64), np.asarray(ends_all),
            np.asarray(trace_id), ev_counts, rule_eps_per_trace)


# ---------------------------------------------------------------- scoring

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
    return {"accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(f1), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true)),
            "positives": int(y_true.sum())}


def episode_scores(seeds, nn_pred, rule_pred, ends, trace_id):
    """Episode-level scoring on the test traces: a ground-truth leak
    episode is DETECTED if the system flags >= 1 window ending inside
    it; a false-alarm episode is a contiguous run of positive window
    decisions (per trace, in window order) overlapping no GT episode.
    Also median detection delay (onset -> first flagged window end),
    in minutes (1 sample = 1 min)."""
    out = {}
    for name, pred in [("nn_pair", nn_pred), ("rule_based", rule_pred)]:
        detected, delays, fa = 0, [], 0
        n_eps = 0
        for k, seed in enumerate(seeds):
            tr = gen_trace(seed)
            eps = [(ev["start"], ev["end"]) for ev in tr["events"]
                   if ev["kind"] == "leak"]
            n_eps += len(eps)
            m = trace_id == k
            e_k, p_k = ends[m], pred[m]
            order = np.argsort(e_k)
            e_k, p_k = e_k[order], p_k[order]
            for s, e in eps:
                hit = np.where((e_k >= s) & (e_k <= e) & (p_k == 1))[0]
                if len(hit):
                    detected += 1
                    delays.append(int(e_k[hit[0]] - s))
            in_ep = np.zeros(len(e_k), dtype=bool)
            for s, e in eps:
                in_ep |= (e_k >= s) & (e_k <= e)
            run = False
            for j in range(len(e_k)):
                if p_k[j] == 1 and not in_ep[j]:
                    if not run:
                        fa += 1
                        run = True
                else:
                    run = False
        out[name] = {"gt_episodes": int(n_eps), "detected": int(detected),
                     "missed": int(n_eps - detected),
                     "false_alarm_episodes": int(fa),
                     "median_detection_delay_min": (float(np.median(delays))
                                                    if delays else None)}
    return out


# ---------------------------------------------------------------- train

def main():
    t0 = time.time()
    torch.manual_seed(20261121)
    np.random.seed(20261121)

    print("generating synthetic household water traces...", flush=True)
    Xtr, ytr, _rtr, _etr, _ttr, ctr, _ = build_split(TRAIN_SEEDS)
    Xva, yva, _rva, _eva, _tva, cva, _ = build_split(VAL_SEEDS)
    Xte, yte, rte, ete, tte, cte, rule_eps_test = build_split(TEST_SEEDS)
    print(f"train {len(ytr)} (pos {ytr.sum()}), val {len(yva)} (pos {yva.sum()}), "
          f"test {len(yte)} (pos {yte.sum()})", flush=True)
    print(f"train events {ctr}", flush=True)
    print(f"test events {cte}", flush=True)

    Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr).float()
    Xva_t = torch.from_numpy(Xva)
    Xte_t = torch.from_numpy(Xte)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos))

    enc = SemanticEncoder()
    dec = GatewayDecoder()
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in dec.parameters())
    print(f"encoder params {enc_params}, decoder params {dec_params}", flush=True)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=1e-3, weight_decay=1e-4)
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
        f1s = [metrics(yva, (pv >= t).astype(int))["f1"] for t in thrs]
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
    nn_pred = (p_clean >= best_thr).astype(int)
    res = {
        "nn_pair": metrics(yte, nn_pred),
        "nn_pair_noisy_latent": metrics(yte, (p_noisy >= best_thr).astype(int)),
        "rule_based": metrics(yte, rte),
    }
    eps_res = episode_scores(TEST_SEEDS, nn_pred, rte.astype(int), ete, tte)
    enc.eval(); dec.eval()
    with torch.no_grad():
        zq = torch.clamp(torch.round(enc(Xte_t) * 127.0) / 127.0, -1, 1)
        _, recon = dec(zq)
        recon_mse = float(mse(recon, Xte_t))
    print(json.dumps(res, indent=2), flush=True)
    print(json.dumps(eps_res, indent=2), flush=True)
    print(f"recon MSE (test): {recon_mse:.5f}", flush=True)

    # ---- bytes/day under the simulated reporting cadence ----
    # Cadence: the device evaluates one causal window every STRIDE
    # minutes (stride 8 at 1 sample/min) -> 1440/8 = 180 windows/day.
    # Raw and learned modes stream per window; the rule mode is
    # event-triggered and transmits its 20 B report once per latch
    # episode (episodes counted on the test traces, per trace-day).
    windows_per_day = DAY // STRIDE
    rule_eps_per_day = float(np.sum(rule_eps_test)) / (len(TEST_SEEDS) * 2.0)
    bytes_day = {
        "windows_per_day": windows_per_day,
        "raw_window_stream": windows_per_day * 256,
        "nn_latent_stream": windows_per_day * 8,
        "rule_report_event_triggered": round(rule_eps_per_day * 20, 1),
        "rule_episodes_per_day_test": round(rule_eps_per_day, 3),
    }
    print(f"bytes/day {bytes_day}", flush=True)

    # ---- encoder size + CPU inference cost (measured on this VM) ----
    sd = enc.state_dict()
    fp32_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    int8_bytes = sum(v.numel() for v in sd.values())
    enc.eval()
    x1 = torch.zeros(1, 64)
    with torch.no_grad():
        for _ in range(50):
            enc(x1)
        t1 = time.perf_counter()
        for _ in range(1000):
            enc(x1)
        single_ms = (time.perf_counter() - t1) / 1000 * 1000
        xb = torch.zeros(1024, 64)
        for _ in range(10):
            enc(xb)
        t1 = time.perf_counter()
        for _ in range(50):
            enc(xb)
        batch_us = (time.perf_counter() - t1) / 50 / 1024 * 1e6
    print(f"encoder fp32 weights {fp32_bytes} B, int8 weights {int8_bytes} B, "
          f"CPU single-window {single_ms:.3f} ms, batched {batch_us:.2f} us/window",
          flush=True)

    # ---- exports ----
    example = torch.zeros(1, 64)
    with torch.no_grad():
        _ = enc(example)
    traced_enc = torch.jit.trace(enc, example)
    enc_path = os.path.join(HERE, "semantic_encoder_metering_trained.pt")
    traced_enc.save(enc_path)

    class DecoderTaskOnly(nn.Module):
        def __init__(self, d):
            super().__init__(); self.d = d
        def forward(self, z):
            logit, _ = self.d(z)
            return logit.unsqueeze(-1)

    traced_dec = torch.jit.trace(DecoderTaskOnly(dec), torch.zeros(1, 8))
    dec_path = os.path.join(HERE, "gateway_decoder_metering_trained.pt")
    traced_dec.save(dec_path)

    onnx_path = os.path.join(HERE, "semantic_encoder_metering_trained.onnx")
    torch.onnx.export(enc, example, onnx_path, opset_version=14,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    np.savez_compressed(os.path.join(HERE, "dataset_metering.npz"),
                        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                        Xte=Xte, yte=yte, rte=rte)

    sizes = {p: os.path.getsize(p) for p in (enc_path, dec_path, onnx_path)}
    print(f"artifact sizes {sizes}", flush=True)

    report = {
        "application": "Water metering / leak detection from a household "
                       "supply meter (THIRD simulated application; flood "
                       "warning was the first, outage detection the second)",
        "data_nature": "Fully SYNTHETIC traces generated by gen_trace() in "
                       "this script (seeded). No field data.",
        "generator": {
            "sampling": "1 sample/min, 2,880 samples (48 h) per trace",
            "channels_per_timestep": ["flow_lpm/20", "cumvol_since_midnight_L/500",
                                      "pressure_norm",
                                      "continuous_flow_min/240_clip1"],
            "design_note": "ch3 is the meter's continuous-flow register "
                           "(minutes since flow last near zero). An earlier "
                           "configuration used time-of-day; it was replaced "
                           "before the headline run to restore information "
                           "parity with the rule's 90-min persistence (the "
                           "rule otherwise integrated history the 16-min "
                           "window cannot contain). Run 1 (time-of-day) "
                           "archived in run1_tod_context/ and reported in "
                           "results.md. No hyperparameter tuned in either run.",
            "background": "daily usage pattern: morning peak 05:30-09:30 and "
                          "evening peak 17:30-23:00 (showers 7-12 L/min 5-15 "
                          "min, baths 10-18 L/min 25-60 min), sporadic taps "
                          "2-6 L/min and toilet flushes 5-7 L/min, cyclic "
                          "laundry/dishwasher bursts 3-6 L/min with zero-flow "
                          "gaps over 45-90 min, quiet nights; flow noise sigma "
                          "0.02 L/min; pressure dips with demand, noise sigma "
                          "0.004",
            "events": {
                "irrigation": "8-16 L/min continuous, 60-240 min, 35% of days, "
                              "early morning or evening — NOT a leak (confuser)",
                "tank_fill": "6-14 L/min continuous, 45-150 min, 30% of days "
                             "— NOT a leak (confuser)",
                "leak": "0.4-2.5 L/min sustained, 180-900 min (3-15 h), 0/1/2 "
                        "per trace (p 0.40/0.40/0.20); 30% grow linearly to "
                        "1.5x rate — GROUND TRUTH"},
            "ground_truth": "leak state = 1 exactly on injected leak "
                            "intervals; window label = state at final timestep",
            "train_event_counts": ctr, "val_event_counts": cva,
            "test_event_counts": cte},
        "splits_by_seed": {"train": TRAIN_SEEDS, "val": VAL_SEEDS,
                           "test": TEST_SEEDS},
        "windows": {"length": WINDOW, "stride": STRIDE,
                    "sizes": {"train": {"n": int(len(ytr)), "positives": int(ytr.sum())},
                              "val": {"n": int(len(yva)), "positives": int(yva.sum())},
                              "test": {"n": int(len(yte)), "positives": int(yte.sum())}}},
        "model": {"encoder": "MLP 64->48->24->8 tanh (architecture unchanged "
                             "from the flood/outage builds)",
                  "encoder_params": enc_params, "decoder_params": dec_params,
                  "encoder_fp32_weight_bytes": fp32_bytes,
                  "encoder_int8_weight_bytes": int8_bytes,
                  "training": "joint end-to-end; int8 quantisation in the loop "
                              "(STE); latent noise sigma=0.03 in training; "
                              "pos-weighted BCE + 0.05*MSE(recon); Adam lr 1e-3, "
                              "batch 256, early stop on val F1 (patience 8, "
                              "max 40 epochs) — family recipe verbatim",
                  "decision_threshold_from_val": best_thr,
                  "best_val_f1": round(float(best_f1), 4)},
        "rule_baseline": {"rule": "stage3-family rule_above (threshold + "
                                  "persistence + hysteresis + causal "
                                  "smoothing) on the flow channel only",
                          "params": {"threshold_norm_flow": LEAK_THR,
                                     "threshold_lpm": LEAK_THR * FLOW_SCALE,
                                     "persist_samples_min": RULE_PERSIST,
                                     "hysteresis_norm": RULE_HYST,
                                     "causal_smooth": RULE_SMOOTH},
                          "report_bytes": 20,
                          "report_layout": "event type 1 B, onset timestamp "
                                           "4 B, mean flow 2 B, duration 2 B, "
                                           "flags/confidence + header/CRC 11 B "
                                           "(study's 20 B semantic-report budget)"},
        "test_metrics_window_level": res,
        "test_metrics_episode_level": eps_res,
        "recon_mse_test": round(recon_mse, 5),
        "payload_comparison": {
            "raw_window": "64 float32 values = 256 bytes",
            "rule_report": "20 bytes per episode start (event-triggered)",
            "nn_latent": "8 int8 values = 8 bytes per window (streamed)",
            "ratio_latent_vs_raw_window": 32.0},
        "bytes_per_day_simulated": {
            "cadence_assumption": "device evaluates one causal window every "
                                  "8 min (window stride at 1 sample/min) = "
                                  "180 windows/day; raw and learned stream "
                                  "per window; rule transmits 20 B once per "
                                  "latch episode, episodes counted on the "
                                  "held-out test traces (mean per trace-day)",
            **bytes_day},
        "encoder_cpu_inference_this_vm": {
            "single_window_ms": round(single_ms, 3),
            "batched_us_per_window": round(batch_us, 2)},
        "artifacts": {"encoder_traced": enc_path, "encoder_onnx": onnx_path,
                      "decoder_traced": dec_path, "artifact_sizes": sizes},
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
