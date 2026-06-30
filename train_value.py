"""Train a learned value function P(win | end-of-my-turn state) from self-play data.

Logistic regression (standardized) for: fast, embeddable (coeffs in main.py),
no model file, low per-eval latency (called many times inside the rollout).
Exports JSON with feature order, means, stds, coefs, intercept.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_parquet('/kaggle/working/value_train_data.parquet')
print('rows:', len(df), 'won_rate:', round(df['won'].mean(), 3))

FEATS = ['prize_diff', 'my_prize', 'op_prize', 'my_active_hp', 'op_active_hp',
         'my_total_hp', 'op_total_hp', 'my_hand', 'op_hand', 'my_deck', 'op_deck',
         'my_bench', 'op_bench', 'my_active_energy', 'my_alakazam', 'my_kadabra',
         'my_abra', 'turn']
FEATS = [f for f in FEATS if f in df.columns]

X = df[FEATS].astype(float).values
y = df['won'].astype(int).values

# train/val split (by row; games are shuffled enough)
n = len(X); idx = np.arange(n)
rng = np.random.RandomState(0); rng.shuffle(idx)
cut = int(n * 0.8)
tr, va = idx[:cut], idx[cut:]

mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1.0
Xtr = (X[tr] - mu) / sd
Xva = (X[va] - mu) / sd

clf = LogisticRegression(max_iter=2000, C=1.0)
clf.fit(Xtr, y[tr])
auc_tr = roc_auc_score(y[tr], clf.predict_proba(Xtr)[:, 1])
auc_va = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
print(f'logistic AUC train={auc_tr:.4f} valid={auc_va:.4f}')

# feature importance (|standardized coef|)
imp = sorted(zip(FEATS, clf.coef_[0]), key=lambda t: -abs(t[1]))
print('top coefs:')
for f, c in imp[:12]:
    print(f'  {f}: {c:+.3f}')

export = {
    'feats': FEATS,
    'mu': mu.tolist(),
    'sd': sd.tolist(),
    'coef': clf.coef_[0].tolist(),
    'intercept': float(clf.intercept_[0]),
    'auc_valid': float(auc_va),
}
with open('/kaggle/working/value_model.json', 'w') as f:
    json.dump(export, f)
print('wrote /kaggle/working/value_model.json')
