# Pokemon Local Loop Record

## Version Goal

version: `pokemon-20260623-v0-06d5`

baseline: `pokemon-20260623-v0-06d1` (current promoted gameplay)

comparison_target: `pokemon-20260623-v0-05d6` (suspected-stronger pre-tactical-layer agent)

goal: Localize the suspected Kaggle regression. Determine, by direct same-deck head-to-head, whether the v0-06d1 phase-aware tactical scoring layer is a strength regression versus v0-05d6, with deck factor ruled out.

hypothesis: The decks of v0-05d6 and v0-06d1 are byte-identical, so the only behavioral difference is the v0-06d1 tactical scoring layer. If that layer is the regression the user observed on Kaggle, then in a seat-balanced same-deck head-to-head, v0-05d6 will beat v0-06d1 with a win rate materially above 50% (Wilson 95% CI lower bound > 0.5). If the head-to-head is statistically ~50/50, the local environment cannot reproduce the regression and hypothesis C must not be validated on this axis without a better opponent model.

change_scope: Diagnostics only. No edits to any agent, deck, model, threshold, or notebook gameplay code. Build a standalone head-to-head + saturation harness and record results. No submission archive is built or promoted in this version.

datasets:
- packaged agents already on disk: `pokemon-20260623-v0-05d6-submission.tar.gz`, `pokemon-20260623-v0-06d1-submission.tar.gz`, `pokemon-20260623-v0-06d4-submission.tar.gz`
- `cg/` simulator at `/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/cg`

success_criteria:
- Harness runs to completion with low error and zero illegal-action-type failures.
- Head-to-head `v0-05d6` vs `v0-06d1` win rate plus Wilson 95% CI is recorded over a statistically meaningful N with seat alternation.
- Both agents are also measured against `random_same_deck` and `first_valid_same_deck` to quantify weak-opponent saturation (the suspected non-discriminating axis).
- A clear diagnostic conclusion is recorded: tactical layer is a regression / is neutral / is inconclusive locally.
- All artifacts written under `/kaggle/working/pokemon-20260623-v0-06d5/` with the `pokemon-20260623-v0-06d5-` prefix.

rollback_criteria:
- Engine errors or illegal-action failures dominate the run.
- Variance is so high that no conclusion is statistically supportable at the chosen N.
- Agent load mismatch (deck or source) is detected.

expected_artifacts:
- `pokemon-20260623-v0-06d5-h2h_rows.json` (per-game rows)
- `pokemon-20260623-v0-06d5-h2h_summary.json` (head-to-head + saturation summaries with CIs)
- `pokemon-20260623-v0-06d5-regression-h2h.py` (the diagnostic harness)
- `pokemon-20260623-v0-06d5-promotion-decision.json` (diagnostic decision feeding v0-06d6/C)

## Pre-Implementation Self-Review

experiment_plan: Run `v0-05d6` vs `v0-06d1` head-to-head with strict seat alternation over N games, decks held identical (verified byte-equal), and report win rate with a Wilson 95% CI. In parallel, run each agent against `random_same_deck` and `first_valid_same_deck` to show whether the existing weak-opponent eval saturates and therefore cannot discriminate the two agents. Optionally include `v0-06d4` vs `v0-06d1` to check whether the guarded reranker moves head-to-head strength at all.

implementation_plan: Write one standalone diagnostic script that loads each agent's `main.py`/`deck.csv` directly from its packaged archive (no agent edits), reuses the proven `battle_start`/`battle_select`/`battle_finish` loop semantics from the v0-06d4 notebook `play_game`, alternates seats per game, calls `battle_finish` to avoid engine memory growth, accumulates per-game rows, and writes prefixed JSON artifacts. Seed Python `random` for the seat schedule and any sampling; the C engine RNG is not seedable from Python, so reproducibility is statistical, not exact.

why_this_is_the_next_best_step: Hypothesis C (replace the imitation teacher with replay/outcome targets) can only be validated against an opponent axis that actually discriminates strength. If the suspected regression (the tactical layer) cannot even be shown locally in direct head-to-head, then C has no trustworthy local gate and we would be repeating the v0-06 mistake. B is the cheaper, decisive prerequisite and also produces the head-to-head harness C will reuse as its gate.

what_would_make_this_result_untrustworthy:
- The C engine RNG is not seedable from Python (confirmed: identical deck order still produced different outcomes), so exact-seed reproducibility is impossible; trust depends on N, seat balancing, and the CI, not on a single deterministic match.
- N too small relative to per-game variance.
- First-player advantage not balanced (mitigated by strict seat alternation and equal seat counts).
- Agent/deck load mismatch (mitigated: decks verified byte-identical; agents loaded from their own archives).
- A local head-to-head difference that does not transfer to the Kaggle ladder, or a local ~50/50 that hides a real Kaggle gap, because local has no faithful top200 opponent.

expected_failure_modes:
- High variance hides a real but moderate gap at feasible N.
- Both agents saturate against weak opponents, giving no contrast there (this is itself an expected, informative result).
- Head-to-head is ~50/50 even though Kaggle differs, implying the local environment is not representative of the Kaggle ladder and that C needs a stronger opponent/value signal before it can be trusted.

scope_guardrails:
- No edits to any agent source, deck, model, or threshold.
- No submission archive built or promoted.
- Diagnostics only; do not tune anything to the head-to-head result in this version.
- Do not silently extend to building C in this version.

validation_plan:
- Determinism probe (DONE): engine is non-deterministic even with identical deck order, so use N + seat balance + Wilson CI.
- Run head-to-head at N large enough for a usable CI (target >= ~200 games, seat-balanced), after a small timing pilot.
- Compute win rate and Wilson 95% CI for each pairing.
- Run weak-opponent saturation for both agents.
- Sanity-check error count, illegal-action count, draw/timeout count.
- Record a diagnostic promotion decision: tactical layer regression confirmed / neutral / inconclusive, with explicit implications for v0-06d6 (C).

go_no_go: go. The version is a single, narrow diagnostic hypothesis (is the tactical layer the regression?), it builds the discriminating eval that C requires, and it makes no behavioral changes.

## Methodology Notes (determined before the main run)

- Decks are byte-identical across `v0-05d6`, `v0-06d1`, `v0-06d4` (verified by extracting `deck.csv` from each archive). So head-to-head isolates agent logic, not deck.
- Neither `v0-05d6` nor `v0-06d1` `main.py` uses `random`/`np.random`; both agents are deterministic given the observation.
- The `cg` engine is non-deterministic: identical deck order produced different outcomes (probe: results `1,1,0`). The engine RNG is not Python-seedable. Reproducibility is therefore statistical: N=600 per pairing, strict seat alternation, Wilson 95% CI.

## Results (N=600 per head-to-head, seat-balanced; saturation N=120 per opponent)

Head-to-head (zero errors, zero illegal actions across all pairings):

- `v0-05d6` vs `v0-06d1`: 278 vs 322. **v0-06d1 win rate 0.537, Wilson95 [0.497, 0.576].** v0-06d1 is marginally ahead; the CI lower bound sits at the 0.5 boundary.
- `v0-06d4` vs `v0-06d1`: 318 vs 282. v0-06d4 win rate 0.530, Wilson95 [0.490, 0.570] (CI includes 0.5 -> reranker does not meaningfully change strength vs its own teacher).
- `v0-06d4` vs `v0-05d6`: 312 vs 288. v0-06d4 win rate 0.520, Wilson95 [0.480, 0.560] (CI includes 0.5).

Saturation vs weak opponents (CIs overlap between agents -> non-discriminating):

- `v0-05d6` vs random 0.975 [0.929, 0.991]; vs first_valid 0.933 [0.874, 0.966].
- `v0-06d1` vs random 0.983 [0.941, 0.995]; vs first_valid 0.925 [0.864, 0.960].

## Promotion Decision

decision: needs_followup

promotion_type: diagnostic_regression_localization

hypothesis_B_result: **not_supported_locally**

reason: Same-deck seat-balanced head-to-head does not reproduce a v0-06d1 regression; if anything v0-06d1 is marginally stronger than v0-05d6 (0.537 [0.497, 0.576]). Deck is ruled out (byte-identical). The weak-opponent axis is saturated and non-discriminating. So the local toolchain cannot currently measure the Kaggle regression the user observed for v0-06d3 (= v0-06d1 gameplay).

key_finding: The Kaggle gap is **not** a same-deck mirror strength difference. The local environment (alakazam-mirror self-play + random/first_valid opponents) has no faithful representation of the Kaggle ladder. Validating hypothesis C on this axis would repeat the v0-06 blind-metric mistake.

implications_for_C:
- Do not validate C on the current local axis. Build a discriminating opponent axis first.
- Most likely unmeasured causes of the Kaggle gap: (a) matchup coverage - local tests only the alakazam mirror, while v0-06d1's tactical layer has archetype-specific branches (`hop_control`, `lucario`, `alakazam_mirror`) that mostly fire vs non-mirror opponents Kaggle has but local does not; (b) no faithful top200-strength opponent locally; (c) the v0-06d3 torch packaging/runtime itself (latency/overage/timeout on real Kaggle hardware).
- Recommended next diagnostic before C: diverse-matchup head-to-head using replay-derived opponent decklists to find where (if anywhere) v0-06d1 actually loses, then make that the C gate.

hard_gates:
- harness completed: yes
- errors: 0
- illegal actions: 0
- agent edits: none (diagnostics only)
- submission built: no (by design)
- artifacts prefixed under `/kaggle/working/pokemon-20260623-v0-06d5/`: yes

next_candidates:
- Diverse-matchup head-to-head (v0-05d6 vs v0-06d1 across replay-derived opponent decks).
- Replay/log-faithful opponent pool to create a discriminating local strength axis.
- Quantify v0-06d3 torch runtime cost on a realistic budget before trusting the neural path.
- Only after a discriminating axis exists: hypothesis C teacher replacement (replay action + outcome targets).

## Loop Direction Update

The original plan was v0-06d5 = B (this version), then v0-06d6 = C. B's result changes that: C cannot be trusted yet because there is no discriminating local strength axis. The corrected queue is:

1. v0-06d6 = build a discriminating, diverse-matchup opponent axis from replay/log decklists, and re-localize the regression across matchups.
2. v0-06d7 (was C) = teacher replacement with replay action + outcome targets, gated on the v0-06d6 axis.

