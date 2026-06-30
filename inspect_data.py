import pandas as pd, glob, os
base='/kaggle/working/pokemon-20260627-v0-09d2'
# DECISION_ROWS_DF
dr=glob.glob(base+'*decision_rows*.parquet')
if dr:
    d=pd.read_parquet(dr[0])
    print('=== DECISION_ROWS_DF ===')
    print('rows:', len(d), 'cols:', list(d.columns))
    if 'player_index' in d: print('player_index dist:', d['player_index'].value_counts().to_dict())
    if 'context_name' in d: print('context_name top:', d['context_name'].value_counts().head(8).to_dict())
    if 'won' in d: print('won null/mean:', d['won'].isna().sum(), round(d['won'].dropna().mean(),3))
    if 'opponent_archetype_norm' in d: print('opp arch:', d['opponent_archetype_norm'].value_counts().to_dict())
    if 'player_archetype_norm' in d: print('player arch:', d['player_archetype_norm'].value_counts().to_dict())
# option model df: is it alakazam-player only?
om=glob.glob(base+'*alakazam_option_model_df*.parquet')
if om:
    o=pd.read_parquet(om[0])
    print('\n=== OPTION_MODEL_DF ===')
    print('rows:', len(o), 'decisions:', o['decision_id'].nunique() if 'decision_id' in o else '?')
    for c in ['player_archetype','player_index','opponent_archetype','context_name']:
        if c in o: print(f'{c}:', o[c].value_counts().head(6).to_dict())
