"""Build pokemon-20260625-v0-07d10-remote-pc.ipynb from v07d9-remote-pc notebook.

v07d10: fix v07d9 bugs + re-run dim96 removal + online PPO lambda_il=0.5.

Bug fixes vs v07d9:
  Bug 1 (game collection): _v07d2_live_features used hardcoded np.zeros((n,97)) and
    np.zeros((0,97)) instead of _V07D2_FEAT_DIM=96. This caused a shape mismatch
    (97-dim array vs 96-dim model) in every game, silently caught, 0 valid games.
    Fix: replace hardcoded 97 with _V07D2_FEAT_DIM in both zeros() calls.

  Bug 2 (Part B loader): pipeline generates main.py with _V06D15MainScorer using
    Linear(97, 512) and X[:,96]=1.0. The 96-dim trained model could not load into
    the 97-dim class → CellExecutionError in Part B.
    Fix: add a main.py patch block at the END of Cell[23], after RL training completes,
    to update the model class and feature construction in main.py to 96-dim.

Hypothesis (same as v07d9): dim96 removal + honest 96-dim training enables RL to find
genuine winner-selective features without the return-condition shortcut.
Primary metric: winner_margin (stored == inference now; no dim96 distinction).
"""
import json, re

SRC = '/kaggle/working/pokemon-20260625-v0-07d9-remote-pc.ipynb'
DST = '/kaggle/working/pokemon-20260625-v0-07d10-remote-pc.ipynb'

with open(SRC) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell[1]: Config ───────────────────────────────────────────────────────────
src1 = ''.join(cells[1]['source'])

src1 = src1.replace(
    "EXPERIMENT_NAME   = 'v0_07d9_dim96_removal_online_ppo'",
    "EXPERIMENT_NAME   = 'v0_07d10_dim96_removal_fixed'"
)
src1 = src1.replace(
    "RUN_PREFIX = os.environ.get('V07D9_RUN_PREFIX', 'pokemon-20260625-v0-07d9-remote-pc')",
    "RUN_PREFIX = os.environ.get('V07D10_RUN_PREFIX', 'pokemon-20260625-v0-07d10-remote-pc')"
)

assert 'v0_07d10_dim96_removal_fixed' in src1,   'EXPERIMENT_NAME'
assert 'v0-07d10-remote-pc' in src1,             'RUN_PREFIX'
cells[1]['source'] = src1.splitlines(keepends=True)

# ── Cell[23]: RL training ─────────────────────────────────────────────────────
src23 = ''.join(cells[23]['source'])

# ── Fix Bug 1: live_features hardcoded 97 → _V07D2_FEAT_DIM ──────────────────
# Early-return for empty options
assert "return np.zeros((0, 97), dtype=np.float32)" in src23, 'zeros(0,97) not found'
src23 = src23.replace(
    "return np.zeros((0, 97), dtype=np.float32)",
    "return np.zeros((0, _V07D2_FEAT_DIM), dtype=np.float32)"
)
# Main array creation
assert "X   = np.zeros((n, 97), dtype=np.float32)" in src23, 'zeros(n,97) not found'
src23 = src23.replace(
    "X   = np.zeros((n, 97), dtype=np.float32)",
    "X   = np.zeros((n, _V07D2_FEAT_DIM), dtype=np.float32)  # v07d10: use FEAT_DIM (was hardcoded 97)"
)

# ── Fix Bug 2: patch main.py after RL training for 96-dim model ──────────────
# Find the summary print (end of RL section) and insert main.py patch before it
MAINPY_PATCH = '''
    # ── v07d10: patch pipeline-generated main.py for 96-dim model ────────────
    _mainpy_path = OUTPUT_DIR / 'main.py'
    if _mainpy_path.exists():
        _mp = _mainpy_path.read_text()
        _mp_orig = _mp
        _mp = _mp.replace('nn.Linear(97, 512)', 'nn.Linear(96, 512)')
        _mp = _mp.replace('(n_opts, 97)', '(n_opts, 96)')
        # Remove X[:,96]=1.0 lines (return-condition leakage)
        import re as _re
        _mp = _re.sub(r"[ \\t]*X\\[:,\\s*96\\]\\s*=\\s*1\\.0[^\\n]*\\n", "", _mp)
        _mp = _re.sub(r"[ \\t]*X\\[\\s*:,96\\]\\s*=\\s*1\\.0[^\\n]*\\n", "", _mp)
        _mainpy_path.write_text(_mp)
        _changed = _mp != _mp_orig
        print(f'main.py patched for 96-dim: changed={_changed}')
    else:
        print('WARNING: main.py not found for patching')

'''

OLD_SUMMARY = "    print(f'\\n=== v0-07d9 Summary ===')"
NEW_SUMMARY = MAINPY_PATCH + "    print(f'\\n=== v0-07d10 Summary ===')"
assert OLD_SUMMARY in src23, 'summary label not found'
src23 = src23.replace(OLD_SUMMARY, NEW_SUMMARY)

cells[23]['source'] = src23.splitlines(keepends=True)

# ── Clear outputs ─────────────────────────────────────────────────────────────
for c in cells:
    c['outputs'] = []
    c['execution_count'] = None

with open(DST, 'w') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written: {DST}')

# ── Sanity checks ─────────────────────────────────────────────────────────────
with open(DST) as f:
    nb2 = json.load(f)
s1  = ''.join(nb2['cells'][1]['source'])
s23 = ''.join(nb2['cells'][23]['source'])

assert 'v0_07d10_dim96_removal_fixed' in s1,               'EXPERIMENT_NAME'
assert 'v0-07d10-remote-pc' in s1,                         'RUN_PREFIX'
assert '_V07D2_FEAT_DIM      = 96' in s23,                 'FEAT_DIM=96'
assert 'np.zeros((0, _V07D2_FEAT_DIM)' in s23,             'Bug1 fix: zeros(0,FEAT_DIM)'
assert 'np.zeros((n, _V07D2_FEAT_DIM)' in s23,             'Bug1 fix: zeros(n,FEAT_DIM)'
assert 'np.zeros((0, 97)' not in s23,                      'old zeros(0,97) still present'
assert 'np.zeros((n, 97)' not in s23,                      'old zeros(n,97) still present'
assert '_mainpy_path.write_text' in s23,                   'Bug2 fix: main.py patch'
assert "nn.Linear(97, 512)" not in s23 or '_mp.replace' in s23,  'Bug2 fix present'
assert 'v0-07d10 Summary' in s23,                          'summary label'
assert 'X_inf[:, 96] = 1.0' not in s23,                   'inference dim96 removed'
assert 'X[:, 96] = 1.0' not in s23,                       'live features dim96 removed'
assert '_V07D2_LAMBDA_IL    = 0.5' in s23,                 'lambda_il=0.5'
print('All sanity checks passed.')
