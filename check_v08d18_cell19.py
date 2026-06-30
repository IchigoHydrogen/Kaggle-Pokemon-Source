import json
nb = json.load(open('/kaggle/working/pokemon-20260627-v0-08d18.ipynb'))
src19 = ''.join(nb['cells'][19]['source'])

# Check the OLD_CAT string
OLD_CAT = "UNKNOWN0_CATEGORICAL_FEATURES = ['context_name', 'option_type', 'area', 'in_play_area', 'option_signature', 'opponent_archetype_norm']  # v08d10: add archetype"
print("OLD_CAT found:", OLD_CAT in src19)

# Check OLD_EXTRA
OLD_EXTRA_HEAD = "        # Extra features from ALAKAZAM_OPTION_MODEL_DF\n        _extra_numeric_candidates"
idx = src19.find("# Extra features from ALAKAZAM_OPTION_MODEL_DF")
print("extra_numeric block:", repr(src19[idx:idx+200]))

# Check injection block
OLD_INJ = "# v08d18: prize_gap"
idx2 = src19.find(OLD_INJ)
print("\ninj_prizes block:", repr(src19[idx2:idx2+500]))

# Check injection end
OLD_INJ_END = "'opponent_archetype_norm'"
idx3 = src19.find("'opponent_archetype_norm'", idx2)
print("\ninj_end:", repr(src19[idx3:idx3+200]))
