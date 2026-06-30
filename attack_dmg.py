import pandas as pd

pfx = "/kaggle/working/pokemon-20260627-v0-08d1/pokemon-20260627-v0-08d1"
ss = pd.read_parquet(pfx + "-state_summary.parquet")
opt = pd.read_parquet(pfx + "-option_rows.parquet")

atk_chosen = opt[(opt["option_type"]=="13") & (opt["is_chosen"]==1)][
    ["decision_id", "attack_id"]].copy()
print("Attack decisions:", len(atk_chosen), "unique attacks:", atk_chosen["attack_id"].nunique())

atk_ss = atk_chosen.merge(
    ss[["decision_id", "episode_id", "player_index", "turn", "op_active_id", "op_active_hp"]],
    on="decision_id")

ss_sorted = ss.sort_values(["episode_id", "player_index", "turn"]).reset_index(drop=True)
ss_sorted["next_op_hp"] = ss_sorted.groupby(["episode_id", "player_index"])["op_active_hp"].shift(-1)
ss_sorted["next_op_id"] = ss_sorted.groupby(["episode_id", "player_index"])["op_active_id"].shift(-1)

atk_with_next = atk_ss.merge(ss_sorted[["decision_id", "next_op_hp", "next_op_id"]], on="decision_id")
same_active = atk_with_next[atk_with_next["op_active_id"] == atk_with_next["next_op_id"]].copy()
same_active["damage_obs"] = (same_active["op_active_hp"] - same_active["next_op_hp"]).clip(lower=0)

# Also get KO cases (damaged + new active)
ko_cases = atk_with_next[atk_with_next["op_active_id"] != atk_with_next["next_op_id"]].copy()
ko_cases["damage_obs"] = ko_cases["op_active_hp"]  # KO = dealt at least this much

all_dmg = pd.concat([same_active, ko_cases], ignore_index=True)
dmg_lookup = all_dmg.groupby("attack_id")["damage_obs"].agg(["mean","median","count","std"])
print("Damage lookup entries:", len(dmg_lookup))
print(dmg_lookup.sort_values("count", ascending=False).head(20).to_string())
print()
# Check std deviation
print("Low-std attacks (consistent damage):")
low_std = dmg_lookup[dmg_lookup["std"] < 10].sort_values("count", ascending=False).head(20)
print(low_std.to_string())
