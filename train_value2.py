"""Train stronger value functions on the 400-game data: logistic vs LightGBM."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

df = pd.read_parquet('/kaggle/working/value_train_data.parquet')
print('rows:', len(df), 'won_rate:', round(df['won'].mean(), 3))

FEATS = ['prize_diff', 'my_prize', 'op_prize', 'my_active_hp', 'op_active_hp',
         'my_total_hp', 'op_total_hp', 'my_hand', 'op_hand', 'my_deck', 'op_deck',
         'my_bench', 'op_bench', 'my_active_energy', 'my_alakazam', 'my_kadabra',
         'my_abra', 'turn']
FEATS = [f for f in FEATS if f in df.columns]
X = df[FEATS].astype(float).values
y = df['won'].astype(int).values
n = len(X); idx = np.arange(n); np.random.RandomState(0).shuffle(idx)
cut = int(n * 0.8); tr, va = idx[:cut], idx[cut:]

# logistic
mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1.0
clf = LogisticRegression(max_iter=3000, C=1.0).fit((X[tr]-mu)/sd, y[tr])
auc_lr = roc_auc_score(y[va], clf.predict_proba((X[va]-mu)/sd)[:, 1])
print(f'logistic AUC valid={auc_lr:.4f}')
json.dump({'feats': FEATS, 'mu': mu.tolist(), 'sd': sd.tolist(),
           'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]),
           'auc_valid': float(auc_lr)}, open('/kaggle/working/value_model.json', 'w'))

# lightgbm
dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=FEATS)
dva = lgb.Dataset(X[va], label=y[va], reference=dtr)
params = {'objective': 'binary', 'metric': 'auc', 'num_leaves': 31,
          'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 1, 'min_child_samples': 30, 'verbose': -1}
bst = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                callbacks=[lgb.early_stopping(40, verbose=False)])
auc_gbm = roc_auc_score(y[va], bst.predict(X[va]))
print(f'lightgbm AUC valid={auc_gbm:.4f} best_iter={bst.best_iteration}')
bst.save_model('/kaggle/working/value_gbm.txt', num_iteration=bst.best_iteration)
json.dump({'feats': FEATS}, open('/kaggle/working/value_gbm_feats.json', 'w'))
print('saved value_model.json (logistic) and value_gbm.txt (lightgbm)')
print('winner:', 'GBM' if auc_gbm > auc_lr + 0.01 else 'logistic')
