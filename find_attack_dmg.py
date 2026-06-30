import json, os, sys

eps_dir = "/kaggle/input/competitions/pokemon-tcg-ai-battle-episodes-2026-06-26"
files = sorted(os.listdir(eps_dir))

for fname in files[:5]:
    with open(os.path.join(eps_dir, fname)) as f:
        ep = json.load(f)
    steps = ep.get("steps", [])
    for i, step in enumerate(steps):
        if not isinstance(step, list): continue
        for agent_step in step:
            if not isinstance(agent_step, dict): continue
            obs = agent_step.get("observation", {}) or {}
            cur = obs.get("current") if isinstance(obs, dict) else None
            if not isinstance(cur, dict): continue
            logs = cur.get("logs", []) or []
            for log in logs:
                log_str = str(log)
                if "damage" in log_str.lower() or "attack" in log_str.lower():
                    print(f"Step {i}:", log_str[:200])
            # Also check if there's a dedicated damage field
            if "result" in cur:
                r = cur["result"]
                if r and str(r) not in ("None", "{}"):
                    print(f"Step {i} result:", str(r)[:100])
