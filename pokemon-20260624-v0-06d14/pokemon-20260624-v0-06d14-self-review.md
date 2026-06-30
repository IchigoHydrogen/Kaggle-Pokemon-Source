# Pre-Implementation Self-Review: v0-06d14

experiment_plan: >
  Convert the v06d13 END deferral learning signal into a narrow guarded runtime probe.
  The only action-changing path is rule-selected END in MAIN, single-action decisions,
  where the model top1 is one of PLAY, ATTACH, EVOLVE, or RETREAT and passes thresholds
  selected on valid. ATTACK-selected decisions and ABILITY outputs are blocked.

implementation_plan: >
  Bump v06d13 to pokemon-20260624-v0-06d14. Keep the training pipeline unchanged.
  Add a safe END deferral grid that excludes ABILITY and ATTACK from candidates. Select
  thresholds on valid only, report holdout and bucket/stress summaries, then inject a
  guarded runtime that overrides only END to safe non-END families. Runtime counters
  must record action changes, rule type, selected type counts, and illegal/exception checks.

why_this_is_the_next_best_step: >
  v06d13 found no meaningful ATTACK within-type rerank surface, but END deferral had a
  large holdout signal. This is the narrowest action-changing step that attacks a real
  weakness while avoiding v06d12's ATTACK contamination.

what_would_make_this_result_untrustworthy:
  - Thresholds are selected using holdout.
  - Safe END report accidentally includes ABILITY or ATTACK.
  - Runtime override fires outside rule END.
  - Runtime action-change counters do not expose selected type distribution.
  - Improvements come only from weak local-agent smoke rather than replay/log holdout.
  - Model/top1 confidence is poorly calibrated and holdout benefit disappears.

expected_failure_modes:
  - END deferral causes loops or wastes actions in local play despite replay agreement.
  - Safe families alone lose much of the v06d13 END signal because ABILITY carried the gain.
  - RETREAT deferrals are contextually dangerous.
  - The generated runtime uses the wrong threshold source.
  - Loader validation selects the wrong callable.

scope_guardrails:
  - No ATTACK runtime override.
  - No ABILITY runtime override.
  - No deck changes.
  - No architecture changes.
  - No split changes.
  - No holdout tuning.
  - No broad hybrid changes outside rule-selected END.

validation_plan: >
  Execute the canonical notebook end-to-end. Verify fresh v06d14 artifacts, feature
  crosscheck, model report, safe END deferral valid/holdout reports, archive contents,
  loader validation, runtime smoke, action-change counters, and promotion decision.
  Reject if any action-changing path violates the END-only safe-family contract.

promotion_evidence_required: >
  Runtime promote requires all hard gates, holdout model_top1 >= 0.49, safe END holdout
  delta >= 0, safe END holdout benefit_minus_harm > 0, zero illegal actions, zero
  exceptions, loader OK, nonzero action changes, and runtime counters showing only END
  to PLAY/ATTACH/EVOLVE/RETREAT changes.

rejection_evidence: >
  Any illegal/exception, archive/loader failure, feature mismatch, action changes outside
  END, unsafe selected family, holdout safe END degradation, or no runtime action changes.

go_no_go: go
