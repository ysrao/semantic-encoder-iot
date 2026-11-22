"""Flood case under the COMMON template (same recipe as soil/dust/storm:
fixed 40 epochs, seed 7, threshold on validation) -- adopted Nov 22, 2026 so
all four cases are treated identically. Data: the saved India windows
(semantic_encoder/india/dataset_india.npz; GUARDIAN levels + INDOFLOODS
danger levels, 6 stations pooled, splits by window-end year). The
hand-crafted reference is the companion study's encoder, whose per-window
decisions are the npz 'rule' column. Also recomputes the Jainagar crossing
window's decoder probability for the paper's figure caption."""
import sys, json, os
sys.path.insert(0, '/home/hatch/workspace/edge-apps-prep/semantic_encoder')
import numpy as np, torch
from semantic_encoder import SemanticEncoder
from train_semantic import GatewayDecoder, quantise_ste, metrics
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
d = np.load('/home/hatch/workspace/edge-apps-prep/semantic_encoder/india/dataset_india.npz')
X, y, rule, yr, st = d['X'], d['y'], d['rule'], d['end_year'], d['station']
splits = {'train': yr <= 2022, 'val': yr == 2023, 'test': yr == 2024}
print({k: (int(m.sum()), int(y[m].sum())) for k, m in splits.items()})
rule_te = metrics(y[splits['test']], rule[splits['test']])
print('rule test:', {k: round(v, 4) for k, v in rule_te.items()})

torch.manual_seed(7); np.random.seed(7)
enc, dec = SemanticEncoder(), GatewayDecoder()
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
Xtr = torch.tensor(X[splits['train']]); ytr = torch.tensor(y[splits['train']], dtype=torch.float32)
pos = ytr.sum(); pw = ((len(ytr) - pos) / pos).clone()
n = len(Xtr)
for ep in range(40):
    perm = torch.randperm(n)
    for i in range(0, n, 256):
        idx = perm[i:i + 256]
        xb, yb = Xtr[idx], ytr[idx]
        z = quantise_ste(enc(xb))
        zn = z + 0.03 * torch.randn_like(z)
        logit, recon = dec(zn)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logit, yb, pos_weight=pw) + 0.05 * nn.functional.mse_loss(recon, xb)
        opt.zero_grad(); loss.backward(); opt.step()

def probs(mask):
    with torch.no_grad():
        return torch.sigmoid(dec(quantise_ste(enc(torch.tensor(X[mask]))))[0]).numpy()
pv, pte = probs(splits['val']), probs(splits['test'])
thr, best = 0.05, -1
for t in np.arange(0.05, 0.96, 0.05):
    f = metrics(y[splits['val']], (pv >= t).astype(int))['f1']
    if f > best: best, thr = f, t
pair_te = metrics(y[splits['test']], (pte >= thr).astype(int))
far = pair_te['fp'] / max(1, pair_te['fp'] + pair_te['tn'])
print('pair thr', thr, 'test:', {k: round(v, 4) for k, v in pair_te.items()}, 'FAR', round(float(far), 4))

torch.save({'enc': enc.state_dict(), 'dec': dec.state_dict()}, f'{BASE}/pair_weights.pt')
np.savez(f'{BASE}/test_windows.npz', X=X[splits['test']], y=y[splits['test']], p=pte, thr=thr)
json.dump(dict(seed=7, rule_test=rule_te, pair_thr=float(thr), pair_test=pair_te, pair_far=float(far),
               splits={k: [int(m.sum()), int(y[m].sum())] for k, m in splits.items()}),
          open(f'{BASE}/results.json', 'w'), indent=1)

# Jainagar crossing window: match the animation window's 16 levels by correlation
anim = json.load(open('/home/hatch/workspace/edge-apps-prep/semantic_encoder/india/animation_window_india.json'))
levels = np.array(anim['window_levels_m'])
te_idx = np.where(splits['test'])[0]
best_i, best_c = None, -2
for i in te_idx:
    lv = X[i].reshape(16, 4)[:, 0]
    c = np.corrcoef(lv, levels)[0, 1]
    if c > best_c: best_c, best_i = c, i
print('jainagar match: window', int(best_i), 'station', int(st[best_i]), 'corr', round(float(best_c), 5),
      'label', int(y[best_i]))
with torch.no_grad():
    z = quantise_ste(enc(torch.tensor(X[best_i:best_i + 1])))
    p_new = torch.sigmoid(dec(z)[0]).item()
    latent = (z.numpy() * 127).round().astype(int)
print('jainagar new decoder probability:', round(p_new, 4), 'vs thr', thr, '-> flagged:', bool(p_new >= thr))
print('jainagar latent int8:', latent.tolist())
print('saved results.json')
