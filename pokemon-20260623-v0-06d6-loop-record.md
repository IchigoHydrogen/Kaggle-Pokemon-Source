# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d6`

base_notebook: `pokemon-20260623-v0-05d6.ipynb` (clean rule + leakage-free UNKNOWN_0 baseline; the v0-06 tactical/torch line is set aside as not-strength-proven)

baseline_for_comparison: `pokemon-20260623-v0-05d6`

goal: Build an outcome-weighted, rank/matchup/context-bucketed replay-agreement diagnostic. Measure how often the v0-05d6 rule agent, placed in real replay decision states, selects the same option the actual player selected, conditioned on player strength (top200) and game outcome (win). This creates the discriminating, Kaggle-correlated evaluation axis that weak-opponent eval (saturated) and same-deck mirror head-to-head (~50/50) lack, and produces a ranked disagreement backlog that becomes the roadmap for future rule improvements.

hypothesis: Real replay observations are faithful enough to run `agent(obs_dict)` directly (confirmed feasible in step-0 probe). If so, agreement between our rule agent and strong/winning alakazam players will (a) be measurable per decision, and (b) vary across rank/outcome buckets. If agreement does NOT vary across buckets, the metric (or the agent representation) is non-discriminating and must not be used as a gate.

change_scope: Diagnostics only. No change to deck, rule-agent behavior, UNKNOWN_0 thresholds, MLP, or submission policy. Add a self-contained replay-agreement stage to the notebook (copied from v0-05d6, RUN_PREFIX bumped). No submission archive promotion based on this version.

datasets:
- `pokemon-tcg-ai-battle-episodes-2026-06-22` (5062 episode replays; ACTIVE-agent observations are full obs_dicts)
- `top200-20260622-ranking` (player name -> ranking/rate)
- `pokemon-tcg-ai-battle` base card data and `cg/` simulator

success_criteria:
- Notebook (copied from v0-05d6, RUN_PREFIX `pokemon-20260623-v0-06d6`) executes fully with existing hard gates clean.
- The agent runs on real replay decision states with a low exception/fallback rate (we measure policy, not fallback).
- Agreement is reported for the slice family: `all`, `top200_actor`, `winner_actor`, `top200_winner`, each bucketed by rank (top10/top50/top200/other), matchup, and context type, restricted to alakazam-actor decisions for the headline.
- A discriminating-validity check is recorded: agreement must differ materially across rank/outcome buckets, else flagged as non-discriminating.
- A `top_disagreement_backlog` artifact ranks contexts/matchups where our agent most disagrees with top200 winners.
- All artifacts under `/kaggle/working/pokemon-20260623-v0-06d6/` with the `pokemon-20260623-v0-06d6-` prefix.

rollback_criteria:
- Replay obs cannot be fed to the agent faithfully (high exception/fallback rate) -> metric measures fallback, not policy.
- Too few alakazam-actor / top200-winner decisions for a usable sample.
- Agreement is flat across all buckets (non-discriminating) -> record the negative result; do not use as a gate.
- Any hard gate fails or the notebook does not execute.

expected_artifacts:
- `pokemon-20260623-v0-06d6-replay_agreement_by_rank.parquet`
- `pokemon-20260623-v0-06d6-replay_agreement_by_matchup.parquet`
- `pokemon-20260623-v0-06d6-replay_agreement_by_context.parquet`
- `pokemon-20260623-v0-06d6-replay_agreement_summary.json`
- `pokemon-20260623-v0-06d6-top_disagreement_backlog.parquet`
- `pokemon-20260623-v0-06d6-promotion-decision.json`

## Pre-Implementation Self-Review

experiment_plan: For each replay episode, walk steps in order; for each ACTIVE-agent decision with a non-deck `select`, reconstruct the obs_dict, run the v0-05d6 agent, and compare its chosen option indices to the recorded `action` indices. Join actor rank (top200 CSV via TeamNames) and outcome (rewards) and actor/opponent archetype (from decklists). Aggregate agreement over the slice family and buckets. Restrict the headline to alakazam-actor decisions (apples-to-apples with our archetype-specific agent).

implementation_plan: First validate logic in a standalone script on an episode subset (fast iteration): action-index semantics, deck-step exclusion, single- vs multi-select agreement, rank/outcome/archetype join, bucket aggregation, discriminating-validity check. Then integrate the validated stage into `pokemon-20260623-v0-06d6.ipynb` (copy of v0-05d6 with RUN_PREFIX bumped) reusing its `TOP200_LOOKUP`, episode discovery, `EPISODE_SPLIT_DF`, archetype/decklist machinery, and embedded agent module. Execute the notebook end-to-end and confirm hard gates.

why_this_is_the_next_best_step: Every prior loop was steered by a non-discriminating metric (weak-opponent win rate saturates; mirror head-to-head ~50/50). Replay agreement with strong winners is the cheapest available Kaggle-correlated signal and is also the improvement backlog. Building it before any rule change prevents repeating the v0-06 blind-metric regression and is the correct foundation for hypothesis C.

what_would_make_this_result_untrustworthy:
- If the agent falls into its fallback path on many replay states (feature/field mismatch), agreement would reflect the fallback, not the real policy. Mitigation: measure and report exception/fallback rate; require it low.
- If agent global state is order-dependent, isolated-decision scoring adds noise. Mitigation: process each game's decisions in order; reset agent state at game start; confirm the rule agent is effectively stateless given full obs.
- Agreement is an imitation proxy: strong players are not optimal and winning does not validate every move. The headline conditions on top200 winners to reduce this, but the conclusion must state that high agreement is a strength-correlated signal, not proof of optimality.
- If agreement is flat across rank/outcome buckets, the metric is non-discriminating; this is itself an informative (negative) result and must be reported honestly.

expected_failure_modes:
- Recorded `action` semantics differ from option indices in some contexts (e.g., deck submission, search inputs) -> handle by excluding actions whose indices fall outside `select.option`.
- Sparse alakazam-actor / top200-winner decisions -> wide CIs; report counts and Wilson CIs.
- Name-join to top200 fails for some players -> bucket as `other`/unranked and report join coverage.
- Multi-select decisions complicate "agreement" -> report single-select top-1 (headline) and multi-select exact-set/Jaccard separately.

scope_guardrails:
- No deck, rule-agent, UNKNOWN_0 threshold, MLP, or submission-policy changes.
- Diagnostics only; do not tune anything to the agreement result in this version.
- Headline restricted to alakazam-actor decisions; other archetypes reported separately, not mixed in.
- Do not promote a submission from this version.

validation_plan:
- Step-0 feasibility probe (DONE): ACTIVE-agent observations are full obs_dicts; recorded actions are option-index lists; rank/outcome/archetype are recoverable. Faithful agent replay is feasible.
- Standalone subset validation of the agreement logic, including exception/fallback rate and a sanity check that agreement varies across buckets.
- Full notebook execution with hard gates.
- Report agreement slice family with counts and Wilson CIs, the discriminating-validity check, and the disagreement backlog.
- Record a diagnostic promotion decision with implications for the next version (first rule fix or hypothesis C).

go_no_go: go. Single narrow diagnostic hypothesis (is replay agreement measurable and discriminating?), feasibility confirmed, no behavioral change, and it is the prerequisite measurement for all subsequent improvement work.

## Step-0 Feasibility Probe Result

- Episode format: `steps[i]` is a 2-element list (one entry per agent). The ACTIVE agent's `observation` is a full obs_dict with `current`, `select`, `search_begin_input`, `step`, `logs`, `remainingOverageTime` — exactly what `agent(obs_dict)` consumes via `to_observation_class`.
- Recorded `action` for normal decisions is a list of selected option indices into `select.option` (e.g. `[0]`). The deck-submission decision records a 60-card list instead, distinguishable because its values exceed `len(select.option)`; it is excluded.
- `info.TeamNames` + `rewards` (e.g. `[-1, 1]`) give per-seat player name and win/loss. Decklists give actor/opponent archetype.
- Conclusion: faithful agent-vs-replay agreement is feasible without feature-path approximation. Proceed to implementation.

## Implementation & Execution

- `pokemon-20260623-v0-06d6.ipynb` built from `pokemon-20260623-v0-05d6.ipynb` (RUN_PREFIX/EXPERIMENT bumped) with one added diagnostic cell (replay-agreement) inserted before the run-summary cell. 19 -> 20 code cells.
- Agreement logic validated standalone over all 5062 clean episodes (0 crashes, 0 agent exceptions) before notebook integration.
- Two full runs failed in the agreement cell on full-only edge cases (None rewards; non-dict top-level among the notebook's 5290 discovered files). Hardened with None-reward guard, `isinstance` checks, and per-episode try/except, then re-validated standalone at full scale and re-run.
- Final full run: `error_count=0`, `run_mode=full`, `decision_rows=1,494,391`, `option_rows=8,352,193`.
- Hard gates: submission `alakazam_rule_base`, `use_neural_for_submission=false`, `import torch`=0, no `__pycache__`/`.pyc`, archive contains `main.py`/`deck.csv`/`cg/api.py`/`cg/game.py`.
- Agreement stage: episodes 5248 (42 odd files safely skipped), eligible 488,455, alakazam-actor 142,599, agent_exceptions 0, bad_shape 0.
- Known minor: the run-summary cell writes before the appended attach line, so `v05_run_summary.json` does not embed the agreement summary; the standalone `...-replay_agreement_summary.json` is the source of truth and is recorded in `ARTIFACT_PATHS`.

## Results (full, 142,599 alakazam-actor decisions)

Overall agreement: **0.2508** [0.2486, 0.2531]. Rank-join coverage 0.677.

Outcome / strength slices (NOT discriminating):
- winner_actor 0.253, loser_actor 0.249 -> **winner-loser = +0.004**.
- top200_winner 0.251, top200_actor ~0.251.
- by rank (non-monotonic): top10 **0.227** (lowest), top50 0.263, top200 0.248, unranked 0.251.
- Interpretation: aggregate agreement is flat and non-monotonic across strength; the strongest (top10) players agree LEAST. Outcome-weighting adds essentially nothing. The chosen outcome-weighting hypothesis is empirically refuted as a discriminator.

Split stability: train 0.252, holdout 0.250, valid 0.246 (stable; holdout consistent).

Matchup: alakazam_vs_alakazam 0.235 (mirror, most divergence), hop_control 0.246, lucario 0.260, other 0.279.

Context level (the real, actionable signal) — three contexts BELOW random alignment:
- **TO_BENCH** n=3113 ours **0.049** vs rand 0.294 (lift -0.245) — high-volume systematic anti-alignment.
- **TO_DECK** n=824 ours **0.011** vs rand 0.302 (lift -0.291).
- **SETUP_BENCH** n=81 ours 0.074 vs rand 0.470 (lift -0.396).
- MAIN n=101,253 ours 0.189 vs rand 0.129 (above random, dominant volume, ~12.6 options) — long-term big target.
- Well aligned: ACTIVATE 0.750, SETUP_ACTIVE 0.650, DISCARD_ENERGY 0.593, EVOLVE 0.514, SWITCH 0.467.

## Promotion Decision

decision: **needs_followup** (diagnostic; no behavioral change, no submission promotion)

key_finding: The replay-agreement metric is feasible and stable but is a **context-level improvement roadmap, not an aggregate strength gate**. Outcome/strength weighting is flat/non-monotonic and should not be used as a strength lever. The single most actionable, high-confidence target is **TO_BENCH** (bench placement), with our agent systematically below random alignment with real players — almost certainly a real weakness, not a stylistic difference. TO_DECK and SETUP_BENCH share the pattern.

implications:
- Drop outcome-weighted imitation as a strength lever.
- Next version (v0-06d7): fix the TO_BENCH bench-placement rule on the v0-05d6 base, gated on the context-level agreement metric plus existing safety/holdout/eval gates. One context, one hypothesis.
- MAIN is a later, larger target.

artifacts:
- `pokemon-20260623-v0-06d6-replay_agreement_summary.json`
- `pokemon-20260623-v0-06d6-replay_agreement_by_rank.parquet`
- `pokemon-20260623-v0-06d6-replay_agreement_by_matchup.parquet`
- `pokemon-20260623-v0-06d6-replay_agreement_by_context.parquet`
- `pokemon-20260623-v0-06d6-top_disagreement_backlog.parquet`
- `pokemon-20260623-v0-06d6-promotion-decision.json`
- `pokemon-20260623-v0-06d6.ipynb` (executed full)
