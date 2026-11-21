"""Robustness checks for the paper, v2.0 (Section IV-F, Table III).

Run the four task scripts and flood_synthetic/train_semantic.py first: they write
the dataset_*.npz files this script reads. Then: python extras/robustness.py

Uses the published training recipe verbatim (semantic_encoder-iot train_*.py):
joint encoder+decoder, int8 STE quantisation, noise 0.03, loss = pos-weighted
BCE + 0.05 MSE, Adam lr 1e-3 wd 1e-4, batch 256, early stop on val F1
(patience 8, max 40 epochs), threshold from validation.

Per task and per training seed:
  * pair            - the learned pair, float weights (as published)
  * pair_int8w      - the same trained pair, encoder weights quantised to int8
                      (per-tensor symmetric; biases kept at higher precision)
  * crafted8        - a hand-picked 8-byte summary (last value and window mean of
                      each of the 4 channels, each scaled to int8 on the training
                      range) fed to the same gateway decoder, trained by the same
                      recipe; no learned device encoder
The reference rule is deterministic and scored once.
"""
import json, sys, os
import numpy as np, torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "flood_synthetic"))
from semantic_encoder import SemanticEncoder  # noqa

W = ROOT
NOISE, RECON = 0.03, 0.05


class Decoder(nn.Module):
    def __init__(s, din=8, dout=64):
        super().__init__()
        s.body = nn.Sequential(nn.Linear(din, 24), nn.ReLU(), nn.Linear(24, 48), nn.ReLU())
        s.task_head = nn.Linear(48, 1); s.recon_head = nn.Linear(48, dout)
    def forward(s, z):
        h = s.body(z); return s.task_head(h).squeeze(-1), s.recon_head(h)


def qste(z):
    q = torch.clamp(torch.round(z * 127.0) / 127.0, -1, 1); return z + (q - z).detach()


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(P, 4), "recall": round(R, 4),
            "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0,
            "far": round(fp / (fp + tn), 4) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn}


def load(task):
    if task == "india":
        d = np.load(f"{W}/flood_india/india/dataset_india.npz"); yr = d["end_year"]
        m = {"tr": yr <= 2022, "va": yr == 2023, "te": yr >= 2024}
        return ({k: d["X"][v].astype(np.float32) for k, v in m.items()},
                {k: d["y"][v].astype(int) for k, v in m.items()}, d["rule"][m["te"]].astype(int))
    f = {"flood_sim": "flood_synthetic/dataset_flood_syn.npz", "outage": "outage/dataset_outage.npz",
         "metering": "metering/dataset_metering.npz", "parking": "parking/dataset_parking.npz"}[task]
    d = np.load(f"{W}/{f}")
    return ({"tr": d["Xtr"].astype(np.float32), "va": d["Xva"].astype(np.float32), "te": d["Xte"].astype(np.float32)},
            {"tr": d["ytr"].astype(int), "va": d["yva"].astype(int), "te": d["yte"].astype(int)}, d["rte"].astype(int))


def crafted(X, lo=None, hi=None):
    w = X.reshape(-1, 16, 4)
    f = np.concatenate([w[:, -1, :], w.mean(1)], axis=1)          # last + mean per channel
    if lo is None:
        lo, hi = f.min(0), f.max(0)
    q = np.round((np.clip((f - lo) / np.maximum(hi - lo, 1e-9), 0, 1) * 2 - 1) * 127) / 127
    return q.astype(np.float32), lo, hi


def train(X, Y, seed, use_encoder=True):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(X["tr"]); ytr = torch.from_numpy(Y["tr"]).float()
    pw = torch.tensor((len(ytr) - float(ytr.sum())) / max(1.0, float(ytr.sum())))
    enc = SemanticEncoder() if use_encoder else None
    dec = Decoder(dout=64 if use_encoder else 8)
    params = (list(enc.parameters()) if enc else []) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw); mse = nn.MSELoss()

    def z_of(xb, train_mode):
        if enc is None:
            return xb + (torch.randn_like(xb) * NOISE if train_mode else 0)
        z = enc(xb)
        return (qste(z) + torch.randn_like(z) * NOISE) if train_mode else torch.clamp(torch.round(z * 127) / 127, -1, 1)

    def predict(Xn):
        if enc: enc.eval()
        dec.eval()
        with torch.no_grad():
            return torch.sigmoid(dec(z_of(torch.from_numpy(Xn), False))[0]).numpy()

    best, state, thr, bad, n = -1, None, 0.5, 0, len(Xtr)
    for ep in range(40):
        if enc: enc.train()
        dec.train(); perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]; xb, yb = Xtr[idx], ytr[idx]
            logit, rec = dec(z_of(xb, True))
            loss = bce(logit, yb) + RECON * mse(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv = predict(X["va"]); ts = np.linspace(0.05, 0.95, 19)
        f1s = [metrics(Y["va"], (pv >= t).astype(int))["f1"] for t in ts]; k = int(np.argmax(f1s))
        if f1s[k] > best:
            best, thr, bad = f1s[k], float(ts[k]), 0
            state = ({a: b.clone() for a, b in enc.state_dict().items()} if enc else None,
                     {a: b.clone() for a, b in dec.state_dict().items()})
        else:
            bad += 1
            if bad >= 8: break
    if enc: enc.load_state_dict(state[0])
    dec.load_state_dict(state[1])
    out = {"float": metrics(Y["te"], (predict(X["te"]) >= thr).astype(int)), "thr": thr}
    if enc:   # int8 weights, per-tensor symmetric
        with torch.no_grad():
            for mod in enc.net:
                if isinstance(mod, nn.Linear):
                    s = mod.weight.abs().max() / 127.0
                    mod.weight.copy_(torch.round(mod.weight / s) * s)
        out["int8w"] = metrics(Y["te"], (predict(X["te"]) >= thr).astype(int))
    return out


SEEDS = {"india": 20261118, "flood_sim": 20261118, "outage": 20261121, "metering": 20261121, "parking": 20261121}
if __name__ == "__main__":
    tasks = sys.argv[1:] or list(SEEDS)
    res = {}
    for t in tasks:
        try:
            X, Y, rule = load(t)
        except FileNotFoundError as e:
            print(f"{t}: skipped - {e.filename} not found (run that task's training script first)")
            continue
        Xc = {}; Xc["tr"], lo, hi = crafted(X["tr"]); Xc["va"], _, _ = crafted(X["va"], lo, hi); Xc["te"], _, _ = crafted(X["te"], lo, hi)
        seeds = [SEEDS[t] + k for k in range(5)]
        runs = [train(X, Y, s) for s in seeds]
        craft = [train(Xc, Y, s, use_encoder=False) for s in seeds]
        res[t] = {"seeds": seeds, "reference": metrics(Y["te"], rule),
                  "pair": [r["float"] for r in runs], "pair_int8w": [r["int8w"] for r in runs],
                  "crafted8": [r["float"] for r in craft]}
        f = lambda L: f"{np.mean([x['f1'] for x in L]):.4f} ± {np.std([x['f1'] for x in L], ddof=1):.4f} [{min(x['f1'] for x in L):.4f}-{max(x['f1'] for x in L):.4f}]"
        print(f"{t}: reference {res[t]['reference']['f1']}  pair {f(res[t]['pair'])}  int8w {f(res[t]['pair_int8w'])}  crafted8 {f(res[t]['crafted8'])}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness_results.json")
    if os.path.exists(out):
        res = {**json.load(open(out)), **res}
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness_results.json"), "w"), indent=1)
