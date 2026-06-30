#!/bin/bash
cd /kaggle/working || exit 9
git add -A 2>&1 | tail -2
git commit -q \
  -m "v18d: v62 forward-sim/LGBM analysis, league proxy test, Archaludon/Hop recon" \
  -m "v62 confirmed champion-level (0.500 vs champion, beats generic_fwdsim 30-0); forward-sim HURTS v62 (0.200, its multi-KO planner already does lookahead); single-pilot league non-transitive; only Alakazam(champion)/Lucario(v62) are clean strong pilots, Hop=control, Archaludon='Other' noisy. experiment-backlog updated." \
  -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -2 || echo "(nothing new to commit)"
echo "=== HEAD ==="; git log --oneline -1
echo "=== push to branch ==="
git -c credential.helper='!f(){ echo username=x; echo "password=$GHT"; };f' push origin HEAD:claude-v18d-v62-work 2>&1 | tail -6
echo "PUSH_EXIT=${PIPESTATUS[0]}"
