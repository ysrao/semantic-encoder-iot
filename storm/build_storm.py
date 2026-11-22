"""Storm landfall case — REAL data (IBTrACS North Indian basin).

Task: a storm at sea now — will it make landfall within 24 hours?
Window: last 16 best-track points (3-hourly) x 4 channels:
  [wind/150, (1010-pres)/100, dist2land/1000, translation speed/50]
Label at window end t: 1 if the storm's next DIST2LAND==0 point is <=24h ahead
(computed from the track itself; the IBTrACS LANDFALL column is km, not minutes).
Splits are chronological by storm season: train <=2012, val 2013-2018, test >=2019.
Pair: the semantic paper's exact architecture and training recipe.
Rule (hand-crafted forecaster's rule): fire if dist2land < D* and the storm is
closing (dist fell across the window) and wind >= 34 kt; D* tuned on TRAIN only.
"""
import sys, json, math
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder')
import numpy as np, pandas as pd, torch
from semantic_encoder import SemanticEncoder
from train_semantic import GatewayDecoder, quantise_ste, metrics

torch.manual_seed(7); np.random.seed(7)
SRC = '/home/hatch/workspace/edge-apps-prep/real_tasks/storm/ibtracs_NI.csv'

def hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

df = pd.read_csv(SRC, skiprows=[1], low_memory=False)
for c in ('LANDFALL', 'DIST2LAND', 'WMO_WIND', 'WMO_PRES', 'LAT', 'LON'):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
df = df[df['SEASON'] >= 1990].copy()
strong = df.groupby('SID')['WMO_WIND'].max()
keep = strong[strong >= 34].index
df = df[df['SID'].isin(keep)].sort_values(['SID', 'ISO_TIME'])

rows = []
for sid, g in df.groupby('SID'):
    g = g.reset_index(drop=True)
    if len(g) < 16:
        continue
    dt_h = g['ISO_TIME'].diff().dt.total_seconds() / 3600.0
    dist_step = [np.nan] + [hav_km(g['LAT'].iloc[i-1], g['LON'].iloc[i-1],
                                   g['LAT'].iloc[i], g['LON'].iloc[i]) for i in range(1, len(g))]
    speed = pd.Series(dist_step) / dt_h
    speed = speed.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    pres_def = ((1010 - g['WMO_PRES']) / 100.0).ffill()
    pres_def = pres_def.fillna(pres_def.median() if pres_def.notna().any() else 0.2)
    ch = np.stack([
        (g['WMO_WIND'] / 150.0).fillna(0).values,
        pres_def.clip(0, 1.2).values,
        (g['DIST2LAND'] / 1000.0).values,
        (speed / 50.0).clip(0, 1.5).values,
    ], axis=1)
    span_h = (g['ISO_TIME'].iloc[15:] .values - g['ISO_TIME'].iloc[:-15].values) / np.timedelta64(1, 'h')
    # NOTE: IBTrACS LANDFALL column is in km (along-track distance), NOT minutes.
    # Label from first principles: time from t to the storm's next DIST2LAND==0 point.
    times = g['ISO_TIME'].values
    on_land = (g['DIST2LAND'] == 0).values
    hours_to_lf = np.full(len(g), np.inf)
    for i in range(len(g) - 1, -1, -1):
        if on_land[i]:
            hours_to_lf[i] = 0.0
        elif i + 1 < len(g):
            step = (times[i + 1] - times[i]) / np.timedelta64(1, 'h')
            hours_to_lf[i] = step + hours_to_lf[i + 1]
    for end in range(15, len(g), 2):
        if end - 15 >= len(span_h) or span_h[end - 15] > 72:
            continue
        d2l = g['DIST2LAND'].iloc[end]; w = g['WMO_WIND'].iloc[end]
        if pd.isna(d2l) or d2l <= 0 or pd.isna(w) or w < 25:
            continue
        label = 1 if hours_to_lf[end] <= 24.0 else 0
        rows.append(dict(season=int(g['SEASON'].iloc[end]), sid=sid,
                         X=ch[end-15:end+1].astype(np.float32), y=label,
                         d2l=float(d2l), wind=float(w),
                         closing=float(ch[end-15, 2] - ch[end, 2])))

def split(lo, hi):
    s = [r for r in rows if lo <= r['season'] <= hi]
    X = np.stack([r['X'] for r in s]).reshape(len(s), -1)
    y = np.array([r['y'] for r in s])
    return s, X, y

tr_r, Xtr, ytr = split(1990, 2012)
va_r, Xva, yva = split(2013, 2018)
te_r, Xte, yte = split(2019, 2026)
print('windows train/val/test:', len(ytr), len(yva), len(yte),
      '| positive rates: %.3f %.3f %.3f' % (ytr.mean(), yva.mean(), yte.mean()))

# ---- hand-crafted rule, threshold distance tuned on TRAIN only ----
def rule_pred(rs, D):
    return np.array([1 if (r['d2l'] < D and r['closing'] > 0 and r['wind'] >= 34) else 0 for r in rs])
best = None
for D in (200, 300, 400, 500, 600):
    m = metrics(ytr, rule_pred(tr_r, D))
    if best is None or m['f1'] > best[1]['f1']:
        best = (D, m)
D_star = best[0]
rule_te = metrics(yte, rule_pred(te_r, D_star))
print('rule: D* =', D_star, 'km (tuned on train) | test:', rule_te)

# ---- learned pair, same recipe as the semantic paper ----
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

# ---- storm-level warning: landfalling test storms flagged >=6h ahead? ----
def storm_level(rs, pred):
    out = {}
    for r, p in zip(rs, pred):
        out.setdefault(r['sid'], []).append((r, p))
    flagged = leads = 0; total = 0
    for sid, lst in out.items():
        lf_rows = [r for r, _ in lst if r['y'] == 1]
        if not lf_rows:
            continue
        total += 1
        hits = [r for r, p in lst if p == 1 and r['y'] == 1]
        if hits:
            flagged += 1
    return flagged, total
rf, rt = storm_level(te_r, rule_pred(te_r, D_star))
pf, pt = storm_level(te_r, (pte >= thr).astype(int))
print('test storms with a landfall approach:', rt, '| rule flagged:', rf, '| pair flagged:', pf)

torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()},
           '/home/hatch/workspace/edge-apps-prep/real_tasks/storm/pair_weights.pt')
np.savez('/home/hatch/workspace/edge-apps-prep/real_tasks/storm/test_windows.npz',
         X=Xte, y=yte, p=pte, thr=thr)
json.dump(dict(rule_D=D_star, rule_test=rule_te, pair_thr=float(thr), pair_test=pair_te,
               counts=dict(train=int(len(ytr)), val=int(len(yva)), test=int(len(yte))),
               storm_level=dict(rule=[rf, rt], pair=[pf, pt])),
          open('/home/hatch/workspace/edge-apps-prep/real_tasks/storm/results.json', 'w'), indent=1)
print('saved results.json')
