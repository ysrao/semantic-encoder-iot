"""Dust case — REAL data (AERONET Kanpur, Level 1.5, Mar 1 - Jul 15 seasons).

Task: is this hour a desert-dust hour over the Indo-Gangetic plain?
Ground-truth signature (the accepted physical identification): AOD_500 >= 0.6
AND Angstrom(440-870) <= 0.6 (heavy load, coarse particles).
Window: last 16 valid hourly bins x 4 measured channels:
  [AOD500/2, AOD1020/2, Angstrom/2, precipitable water/6]
Hand-crafted baseline (the naive device rule): fire on AOD_500 alone above a
threshold tuned on TRAIN years — it cannot tell dust from fine-mode haze.
Splits by year: train 2019+2021+2022, val 2023, test 2024 (2020 excluded:
lockdown year, anomalous aerosol).
"""
import sys, json, glob
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder')
import numpy as np, pandas as pd, torch
from semantic_encoder import SemanticEncoder
from train_semantic import GatewayDecoder, quantise_ste, metrics

torch.manual_seed(7); np.random.seed(7)
BASE = '/home/hatch/workspace/edge-apps-prep/real_tasks/dust'

def load_year(y):
    raw = open(f'{BASE}/kanpur_{y}.csv', encoding='utf-8', errors='replace').read()
    i = raw.find('AERONET_Site')
    lines = [l for l in raw[i:].splitlines() if l.strip() and not l.startswith('<')]
    import io
    df = pd.read_csv(io.StringIO('\n'.join(lines)))
    cols = {c: c for c in df.columns}
    aod = next(c for c in df.columns if c.startswith('AOD_500'))
    aod2 = next(c for c in df.columns if c.startswith('AOD_1020'))
    ang = next(c for c in df.columns if 'ngstrom' in c and '440' in c)
    wv = next(c for c in df.columns if 'Precipitable_Water' in c or 'Water_Vapor' in c)
    out = pd.DataFrame({
        'ts': pd.to_datetime(df['Date(dd:mm:yyyy)'] + ' ' + df['Time(hh:mm:ss)'],
                             format='%d:%m:%Y %H:%M:%S'),
        'aod': pd.to_numeric(df[aod], errors='coerce'),
        'aod2': pd.to_numeric(df[aod2], errors='coerce'),
        'ang': pd.to_numeric(df[ang], errors='coerce'),
        'wv': pd.to_numeric(df[wv], errors='coerce'),
    })
    out = out[(out['aod'] > 0) & (out['ang'] > -1)].dropna(subset=['aod', 'ang'])
    out['wv'] = out['wv'].where(out['wv'] > 0)
    return out

def hourly(df):
    g = df.set_index('ts').resample('1h').median(numeric_only=True).dropna(subset=['aod', 'ang'])
    g['aod2'] = g['aod2'].ffill(limit=2); g['wv'] = g['wv'].ffill(limit=2)
    return g.dropna()

def windows(g, year):
    ch = np.stack([np.clip(g['aod'].values / 2, 0, 1.5), np.clip(g['aod2'].values / 2, 0, 1.5),
                   np.clip(g['ang'].values / 2, 0, 1.5), np.clip(g['wv'].values / 6, 0, 1.5)], axis=1)
    dust = ((g['aod'].values >= 0.6) & (g['ang'].values <= 0.6)).astype(int)
    t = g.index.values
    rows = []
    for end in range(15, len(g), 4):
        span = (t[end] - t[end - 15]) / np.timedelta64(1, 'h')
        if span > 40:
            continue
        rows.append(dict(year=year, X=ch[end-15:end+1].astype(np.float32),
                         y=int(dust[end]), aod_end=float(g['aod'].values[end])))
    return rows

all_rows = []
for y in (2019, 2021, 2022, 2023, 2024):
    g = hourly(load_year(y))
    r = windows(g, y)
    print(f'{y}: hourly bins {len(g)}, windows {len(r)}, dust rate {np.mean([x["y"] for x in r]):.3f}')
    all_rows += r

def split(years):
    s = [r for r in all_rows if r['year'] in years]
    return s, np.stack([r['X'] for r in s]).reshape(len(s), -1), np.array([r['y'] for r in s])

tr_r, Xtr, ytr = split((2019, 2021, 2022))
va_r, Xva, yva = split((2023,))
te_r, Xte, yte = split((2024,))
print('windows train/val/test:', len(ytr), len(yva), len(yte),
      '| positive rates: %.3f %.3f %.3f' % (ytr.mean(), yva.mean(), yte.mean()))

def rule_pred(rs, T):
    return np.array([1 if r['aod_end'] >= T else 0 for r in rs])
best = None
for T in (0.5, 0.6, 0.7, 0.8, 1.0):
    m = metrics(ytr, rule_pred(tr_r, T))
    if best is None or m['f1'] > best[1]['f1']:
        best = (T, m)
T_star = best[0]
rule_te = metrics(yte, rule_pred(te_r, T_star))
print('rule: AOD threshold', T_star, '(tuned on train) | test:', rule_te)

enc, dec = SemanticEncoder(), GatewayDecoder()
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32)
pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(1, ytr.sum()))
bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
for ep in range(40):
    perm = torch.randperm(len(ytr))
    for i in range(0, len(ytr), 256):
        idx = perm[i:i+256]
        xb, yb = Xtr_t[idx], ytr_t[idx]
        z = quantise_ste(enc(xb))
        zn = z + 0.03 * torch.randn_like(z)
        logit, recon = dec(zn)
        loss = bce(logit.squeeze(-1), yb) + 0.05 * torch.mean((recon - xb) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()

def probs(X):
    with torch.no_grad():
        return torch.sigmoid(dec(quantise_ste(enc(torch.tensor(X))))[0]).squeeze(-1).numpy()
pva, pte = probs(Xva), probs(Xte)
ths = np.linspace(0.05, 0.95, 19)
thr = max(ths, key=lambda t: metrics(yva, (pva >= t).astype(int))['f1'])
pair_te = metrics(yte, (pte >= thr).astype(int))
print('pair: threshold %.2f (on val) | test:' % thr, pair_te)

torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()}, f'{BASE}/pair_weights.pt')
np.savez(f'{BASE}/test_windows.npz', X=Xte, y=yte, p=pte, thr=thr)
json.dump(dict(rule_T=float(T_star), rule_test=rule_te, pair_thr=float(thr), pair_test=pair_te,
               counts=dict(train=int(len(ytr)), val=int(len(yva)), test=int(len(yte)))),
          open(f'{BASE}/results.json', 'w'), indent=1)
print('saved results.json')
