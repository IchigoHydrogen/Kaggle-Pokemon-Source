# Pre-Implementation Self-Review: v0-06d15

experiment_plan: >
  Add diagnostics that focus on rule/log disagreements conditioned by final game
  outcome. For valid and holdout, measure where the rule agent disagrees with the
  logged action, split by won/lost, rule family, chosen family, rank bucket, and model
  recovery. Use this to recommend the next family to study after END deferral.

implementation_plan: >
  Bump v06d14 to pokemon-20260624-v0-06d15. Keep training and feature extraction
  unchanged. After prediction table construction, compute rule_disagreement = rule_hit
  == 0. Build transition reports for rule_option_type -> chosen_option_type, family
  reports for winner-side disagreements, high-rank/rank-bucket reports, and a compact
  JSON summary. Runtime is shadow-only so no action-changing behavior is introduced.

why_this_is_the_next_best_step: >
  Stress validation is useful, but it mostly protects the current END guard. The user
  asked to push beyond local imitation toward win-rate-relevant data. Winner-conditioned
  rule disagreement is the smallest step toward outcome-aware learning that fits the
  existing replay/log pipeline without pretending to solve causal credit assignment.

what_would_make_this_result_untrustworthy:
  - Treating won=true actions as automatically optimal.
  - Ignoring rank/archetype concentration.
  - Choosing a family solely by raw count with poor model recovery.
  - Tuning future runtime thresholds on holdout.
  - Mixing ATTACK, ABILITY, and END into one broad action-changing policy.
  - Runtime action changes occur despite this being diagnostic-only.

expected_failure_modes:
  - Winner-side disagreements mostly reflect deck strength or matchup, not action quality.
  - Candidate family is high-count but unsafe, such as ABILITY or ATTACK, requiring a separate guarded formulation.
  - Model recovery is low even where disagreement signal is strong.
  - High-rank/top200 slices disagree with aggregate holdout.
  - The report confirms END is still the only clean target.

scope_guardrails:
  - No runtime action-changing policy.
  - No deck changes.
  - No architecture changes.
  - No training objective changes in this version.
  - No holdout tuning.
  - No broad hybrid promotion.
  - Treat winner-conditioning as prioritization evidence, not causal proof.

validation_plan: >
  Execute notebook end-to-end. Verify reports exist, model/crosscheck quality is stable,
  runtime smoke has zero illegal/exception and zero action changes, and promotion decision
  identifies the next candidate family with known caveats.

promotion_evidence_required: >
  learning_promote requires successful execution, holdout model_top1 >= 0.49, valid and
  holdout disagreement reports, and a next-family recommendation supported by winner-side
  count, model recovery, and no obvious high-rank contradiction. Runtime promotion is out
  of scope.

rejection_evidence: >
  Missing diagnostics, notebook failure, feature mismatch, holdout model_top1 < 0.47,
  runtime action changes, or an inconclusive report that cannot prioritize any next family.

go_no_go: go
