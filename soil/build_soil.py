"""Soil dry-down case — REAL data (ERA5-Land via Open-Meteo archive).

Task: topsoil is above its dry threshold now — will it cross below within 48h?
Sites: Indore (MP), Ludhiana (Punjab), Nagpur (Maharashtra), 2019-2024, daily.
Window: last 16 days x 4 channels [SM 0-7cm, SM 7-28cm, SM 28-100cm, rain/20].
Dry threshold theta_site = 20th percentile of daily SM(0-7cm) over TRAIN years
(2019-2022) at that site. Windows ending already below theta are excluded:
the event is the crossing.
Hand-crafted rule: fire if SM_end < theta + delta, SM fell over the last 4 days,
and the last 4 days brought < 2 mm rain; delta tuned on TRAIN only.
Splits chronological: train 2019-2022, val 2023, test 2024 (sites pooled).
"""
import sys, json, urllib.request
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder')
import numpy as np, pandas as pd, torch
from semantic_encoder import SemanticEncoder
from train_semantic import GatewayDecoder, quantise_ste, metrics

torch.manual_seed(7); np.random.seed(7)
BASE = '/home/hatch/workspace/edge-apps-prep/real_tasks/soil'
SITES = {'Indore': (22.72, 75.86), 'Ludhiana': (30.90, 75.86), 'Nagpur': (21.15, 79.09)}

def pull(name, lat, lon):
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s"
           "&start_date=2019-01-01&end_date=2024-12-31&hourly=soil_moisture_0_to_7cm,"
           "soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,precipitation"
           "&timezone=Asia%%2FKolkata" % (lat, lon))
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    h = d['hourly']
    df = pd.DataFrame({'ts': pd.to_datetime(h['time']),
                       'sm1': h['soil_moisture_0_to_7cm'], 'sm2': h['soil_moisture_7_to_28cm'],
                       'sm3': h['soil_moisture_28_to_100cm'], 'rain': h['precipitation']})
    day = df.set_index('ts').resample('1D').agg({'sm1': 'mean', 'sm2': 'mean', 'sm3': 'mean', 'rain': 'sum'})
    return day.dropna()

rows = []
for name, (lat, lon) in SITES.items():
    day = pull(name, lat, lon)
    tr_mask = day.index.year <= 2022
    theta = float(day.loc[tr_mask, 'sm1'].quantile(0.20))
    sm1 = day['sm1'].values
    ch = np.stack([sm1, day['sm2'].values, day['sm3'].values,
                   np.clip(day['rain'].values / 20.0, 0, 1.5)], axis=1)
    site_rows = 0
    for end in range(15, len(day) - 2, 2):
        if sm1[end] < theta:
            continue  # already dry: not a crossing-forecast window
        label = 1 if min(sm1[end + 1], sm1[end + 2]) < theta else 0
        rows.append(dict(site=name, year=int(day.index[end].year),
                         X=ch[end-15:end+1].astype(np.float32), y=label,
                         sm_end=float(sm1[end]), theta=theta,
                         fall=float(sm1[end-3] - sm1[end]),
                         rain4=float(day['rain'].values[end-3:end+1].sum())))
        site_rows += 1
    print(f'{name}: days {len(day)}, theta {theta:.3f}, windows {site_rows}')

def split(lo, hi):
    s = [r for r in rows if lo <= r['year'] <= hi]
    return s, np.stack([r['X'] for r in s]).reshape(len(s), -1), np.array([r['y'] for r in s])

tr_r, Xtr, ytr = split(2019, 2022)
va_r, Xva, yva = split(2023, 2023)
te_r, Xte, yte = split(2024, 2024)
print('windows train/val/test:', len(ytr), len(yva), len(yte),
      '| positive rates: %.3f %.3f %.3f' % (ytr.mean(), yva.mean(), yte.mean()))

def rule_pred(rs, delta):
    return np.array([1 if (r['sm_end'] < r['theta'] + delta and r['fall'] > 0 and r['rain4'] < 2.0)
                     else 0 for r in rs])
best = None
for delta in (0.01, 0.02, 0.03, 0.05):
    m = metrics(ytr, rule_pred(tr_r, delta))
    if best is None or m['f1'] > best[1]['f1']:
        best = (delta, m)
d_star = best[0]
rule_te = metrics(yte, rule_pred(te_r, d_star))
print('rule: delta', d_star, '(tuned on train) | test:', rule_te)

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
json.dump(dict(rule_delta=float(d_star), rule_test=rule_te, pair_thr=float(thr), pair_test=pair_te,
               counts=dict(train=int(len(ytr)), val=int(len(yva)), test=int(len(yte)))),
          open(f'{BASE}/results.json', 'w'), indent=1)
print('saved results.json')
