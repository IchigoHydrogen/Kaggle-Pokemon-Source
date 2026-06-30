# Pre-Implementation Self-Review: v0-06d13

experiment_plan: >
  Run a learning-focused diagnostic version that keeps the v0-06d12 scorer formulation
  and splits, then evaluates decisions conditioned on the rule agent's selected option
  type. The main measurements are: (1) ATTACK within-type reranking among ATTACK
  options when the rule agent chose ATTACK, (2) END deferral opportunities when the
  rule agent chose END, and (3) cross-family contamination where the global model's
  top choice crosses away from ATTACK/END.

implementation_plan: >
  Bump the canonical notebook to pokemon-20260624-v0-06d13.ipynb. Add rule-selected
  option type columns to the prediction dataframe. Keep CE/card metadata/model logic
  unchanged. Add diagnostic functions that operate on valid for threshold selection
  and holdout for reporting. For ATTACK, score only ATTACK candidates inside rule-ATTACK
  decisions and compare the within-type model choice to the rule choice. For END, measure
  confidence-gated non-END deferral candidates separately from normal safe-family
  overrides. Write dedicated report artifacts and a learning promotion decision.

why_this_is_the_next_best_step: >
  v0-06d12 showed the model itself is competitive, but global hybrid selection mixes
  action-family choice with option-quality choice. Permanently vetoing ATTACK/END would
  protect the runtime baseline but cap long-term strength. A conditioned diagnostic is
  the smallest next step that can tell whether ATTACK/END are learnable without allowing
  uncontrolled cross-type overrides.

what_would_make_this_result_untrustworthy:
  - Thresholds selected on holdout instead of valid.
  - ATTACK within-type evaluation accidentally includes non-ATTACK options.
  - END deferral is judged only by aggregate top1 and not benefit/harm against rule.
  - Rule option type is inferred from model top1 instead of the actual rule-selected option.
  - Fresh v0-06d13 artifacts are not produced under the v0-06d13 prefix.
  - Holdout gains appear only in low-sample buckets.
  - Runtime action-changing behavior is enabled for ATTACK/END despite this being a diagnostic version.

expected_failure_modes:
  - ATTACK has too few multi-attack decisions for reliable within-type reranking.
  - END deferral labels are noisy because replay "chosen non-END" may reflect long-horizon strategy not captured by current features.
  - Global model logits are poorly calibrated across families, so END deferral confidence may not transfer.
  - Diagnostics reveal that separate family classifiers are needed before runtime adoption.
  - Re-executing the notebook may be slow because feature extraction remains expensive.

scope_guardrails:
  - No deck changes.
  - No architecture changes.
  - No split changes.
  - No new training objective beyond v0-06d12 CE.
  - No holdout tuning.
  - No ATTACK/END runtime promotion in this version.
  - Do not reuse v0-06d12 predictions or thresholds as current-run evidence.
  - Local weak-agent wins are smoke evidence only.

validation_plan: >
  Execute the canonical notebook end-to-end. Confirm model/feature artifacts are under
  the v0-06d13 output directory. Check feature crosscheck, model_top1, conditioned
  reports, ATTACK within-type grid, END deferral grid, and cross-family contamination.
  If runtime packaging is executed, require loader validation and zero illegal actions
  or exceptions, but treat runtime as shadow/diagnostic evidence only.

promotion_evidence_required: >
  learning_promote requires successful notebook execution, reproducible fresh v0-06d13
  artifacts, holdout model_top1 >= 0.49, and clear valid/holdout diagnostics identifying
  whether ATTACK within-type rerank or END deferral should be promoted to a future
  action-changing guarded policy. Runtime promotion is explicitly out of scope.

rejection_evidence: >
  Feature mismatch, notebook failure, holdout model_top1 < 0.47, insufficient ATTACK/END
  sample sizes with no usable diagnostic, evidence of holdout tuning, or accidental
  ATTACK/END action-changing runtime behavior.

go_no_go: go
