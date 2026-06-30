# Pre-Implementation Self-Review: v0-06d12

experiment_plan: >
  Fix three compounding implementation errors in the MAIN option scorer in one version:
  (1) loss function from BCE (independent binary per option) to per-decision cross-entropy
      (softmax over n options within each decision, aligned with argmax evaluation metric),
  (2) feature engineering from card_id%32 one-hot (32-bucket collision, ~250 cards collapsed)
      to card metadata from all_card_data() (cardType, evolution stage, ex flags, HP, key-card
      identity flags for the Alakazam deck), feature dim stays at 96,
  (3) hybrid guard from hard veto [ATTACK, END, ABILITY] to confidence-gated override for
      ATTACK (prob>=threshold_A, margin>=threshold_A2) and END/RETREAT (prob>=threshold_B),
      with ABILITY remaining hard-vetoed. Thresholds selected by valid-split grid search.

implementation_plan:
  step_1_card_lookup: >
    At top of training cell, call all_card_data() once, build dict {cardId: card}.
    Write card_meta(card_id) helper returning 35-dim numpy array covering cardType one-hot
    (7 dims), ex/megaEx/aceSpec/tera flags (4 dims), evolution stage (3 dims), hp_norm (1),
    has_skill (1), and 19 key-card identity flags for Alakazam deck cards.
  step_2_feature_extraction: >
    Replace _v06d7_option_features reference impl (96 dims): remove card_id continuous [28]
    and card_id%32 one-hot [62-93], shift option numerics and one-hots accordingly, insert
    card_meta result at [61-95]. New layout documented in feature_spec JSON.
    Vectorized extraction updated to match. Cross-check reference vs vectorized on 100 samples.
  step_3_ce_training: >
    Build decision-indexed arrays: decision_starts, decision_n_opts, decision_chosen for each
    split. Train using padded mini-batch CE: for each batch of 256 decisions, build padded
    tensor (B, max_n_opts_in_batch, 96), forward → (B, max_n_opts), mask padding to -inf,
    apply F.cross_entropy(logits, chosen). Same early stopping on valid top-1 (patience=5,
    max_epochs=100). Same AdamW lr=0.002.
  step_4_hybrid_eval: >
    Add ATTACK/END/RETREAT to overrideable types in hybrid evaluation.
    Run threshold grid on valid split for (prob, margin) pairs, now including ATTACK/END in
    both safe_override and the danger check. Select threshold minimizing harm on danger
    bucket while maximising hybrid_top1. Report ATTACK delta explicitly in hybrid bucket df.
  step_5_runtime_injection: >
    Update runtime feature extractor (embedded in generated main.py) to match step_2 layout.
    Update guard logic to use prob/margin thresholds from step_4 for ATTACK/END/RETREAT,
    with ABILITY remaining hard-vetoed. Ensure _kaggle_submission_entrypoint defined last.
  step_6_validation: >
    Build archive, run loader validation (get_last_callable must select _kaggle_submission_entrypoint),
    run smoke eval (20 games, measure latency, action_changes, illegal=0, exception=0).
    Write promotion decision JSON.

why_this_is_the_next_best_step: >
  BCE does not align with argmax evaluation and produces scattered gradients across option rows.
  card_id%32 hash collapses ~250 cards into 32 buckets, making PLAY decisions semantically
  blind. ATTACK hard veto prevents any measurement of whether the model could improve on the
  most consequential action. All three are definitively incorrect implementations rather than
  hypotheses. Evaluating them one at a time would require interpreting results from systems
  that are still structurally broken. A single "correctly formulated base" version is the
  minimum meaningful unit of evaluation.

what_would_make_this_result_untrustworthy:
  - CE implementation bug: mask not applied before softmax → CE sees padding as live options.
  - Feature cross-check fails (vectorized ≠ reference) → training and runtime use different features.
  - card_meta helper looks up wrong card_id (e.g., uses option area index instead of card_id).
  - ATTACK override threshold tuned on holdout (not valid).
  - Runtime feature extractor not updated to match training layout → stale 96-dim layout in main.py.
  - Holdout improvement driven only by train/valid leakage in threshold selection.
  - ATTACK confidence gate too loose → interference from noisy ATTACK predictions.

expected_failure_modes:
  - CE training slower per epoch (decision-batched instead of row-batched) → may need lr warmup.
  - card metadata for unknown card_id returns zeros → harmless but worth logging hit rate.
  - ATTACK model still worse than rule after CE+card features → guard will suppress it if
    threshold too high; hybrid_top1 then equivalent to PLAY/ATTACH/EVOLVE-only baseline.
  - max_n_opts=55 → padded tensors are 55x larger per decision; with batch=256 this is
    256*55*96*4=5.4MB per batch, fine.
  - Kaggle loader validation: need _kaggle_submission_entrypoint as last callable again.

scope_guardrails:
  - No deck changes.
  - No model architecture changes (same 96→512→384→256→1 MLP).
  - No multi-select overrides.
  - No holdout-based threshold tuning; valid only.
  - ABILITY remains hard-vetoed (stateful; too risky without dedicated study).
  - No change to deterministic episode-hash split.
  - No reuse of v0-06d11 weights, predictions, or thresholds.
  - Feature dim stays at 96.

validation_plan:
  - Feature spec JSON: confirm layout_version=v06d12, feature_dim=96, card_id_hash=false.
  - Cross-check JSON: mismatches=0 between reference and vectorized extractor.
  - Model report: CE loss curve, best_epoch, best_valid_top1.
  - Holdout report: model_top1 vs v0-06d11 (0.5090), hybrid_top1 vs v0-06d11 (0.4507).
  - ATTACK bucket: model_top1 reported. If model_top1 > rule_top1 (0.504) → notable result.
  - Hybrid bucket: danger_hybrid_delta for ATTACK/END explicitly recorded.
  - Runtime probe: torch_load_ok, action_changes, latency p99, illegal=0, exception=0.
  - Loader validation: selected_callable = _kaggle_submission_entrypoint.
  - Promotion gate: all above pass before writing decision=promote.

promotion_evidence_required:
  - All hard gates pass (compile, archive, smoke, loader).
  - Holdout model_top1 >= 0.47.
  - Holdout hybrid_top1 >= 0.44.
  - ATTACK dangerous bucket hybrid_delta >= -0.03.
  - Smoke: illegal=0, exception=0.
  - Fresh training confirmed (no prior weight reuse).

rejection_evidence:
  - Any illegal action or exception.
  - Holdout model_top1 < 0.45 (severe degradation from CE+feature changes).
  - Feature cross-check mismatch (training and runtime diverged).
  - ATTACK dangerous bucket hybrid_delta < -0.05.
  - Archive or loader validation failure.
  - Evidence of holdout-based threshold tuning.

go_no_go: go
