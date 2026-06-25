# Pokemon Loop Skills

Operational procedures for the Pokemon TCG AI Battle improvement loop.
Rules and criteria → `loop-harness.md`. Experiment queue → `experiment-backlog.md`.

---

## Before Starting Any Version

1. `git pull origin main`
2. Read `experiment-backlog.md` — identify unclaimed experiments.
3. Choose one. Move it from `Proposed` to `Claimed` (add machine name and date).
4. Immediately commit and push the claim:
   ```bash
   git add experiment-backlog.md
   git commit -m "claim: <experiment name> (<machine>)"
   git push origin main
   ```
5. If push fails (conflict = another agent claimed first): pull, re-read backlog, choose a different unclaimed experiment.

Do not begin implementation until the claim is successfully pushed.

---

## Experiment Type Policy

Each machine independently tracks its consecutive conservative experiment count.

**Definitions:**

- **conservative**: Incremental change on the current known-best approach. Examples: hyperparameter tuning, offline PPO with saved episodes, packaging fixes, IL data extensions, threshold recalibration, minor architecture tweaks.
- **aggressive**: Substantially different approach that cannot be validated by small adjustments to the current baseline. Examples: new model architecture, new RL algorithm, new feature family, new reward formulation, large structural refactor, entirely new policy strategy.

**Rule: alternate 1 conservative → 1 aggressive per machine.**

After 1 conservative experiment, the next experiment on that machine must be aggressive. After the aggressive experiment, the next may be conservative again. The cycle repeats.

- This rule applies independently per machine. One machine running conservative while the other runs aggressive is fine.
- Track with `experiment_type` and `consecutive_conservative_count` in the version goal and promotion decision.
- The count resets to 0 after any aggressive experiment, regardless of promotion outcome.
- "Aggressive" applies to the hypothesis design space, not to skipping validation. All harness gates and criteria still apply.
- An aggressive experiment that fails is still a valid `exploration_promote` result if it produces reproducible evidence for a future hypothesis.

**When choosing from backlog:** prefer the experiment tagged `[aggressive]` if your last experiment was `conservative`, and vice versa.

---

## Multi-Machine Naming Convention

| Nickname | Host | GPU |
|---|---|---|
| `ei` | ei@DESKTOP-73HS8PJ | RTX 5070 |
| `remote-pc` | fukuharaken@DESKTOP-T45CBIC | RTX 4090 |

**Prefix format:** `pokemon-YYYYMMDD-vX-YYzz-<machine>`

Examples:
```
pokemon-20260625-v0-07d5-ei.ipynb          → RUN_PREFIX = pokemon-20260625-v0-07d5-ei
pokemon-20260625-v0-07d5-remote-pc.ipynb   → RUN_PREFIX = pokemon-20260625-v0-07d5-remote-pc
```

Rules:
- Day index (`YYzz`, e.g., `07d5`) increments **independently per machine**. No cross-machine coordination needed for day numbers.
- Major version (`vX`) is **shared**. Bumping requires coordination before either machine starts a new version under the new major.
- Always include the machine suffix, even when only one machine is active.

---

## Git Workflow

**Standard workflow:**
```bash
# Start of each work session — always pull first
cd ~/wkdir/2422005/kaggle/working
git pull origin main

# After promotion decision is recorded
git add <notebook>.ipynb <RUN_PREFIX>/
git add experiment-backlog.md  # update Completed, add next candidates
git add pokemon-20260622-v0-05d1-loop-harness.md  # if baseline was updated
git commit -m "vX-YYzz-<machine>: <one-line result summary>"
git push origin main
```

**Tracked files** (see `.gitignore` for full rules):
- `*.ipynb`, `*.md`, `*.py` (excluding `tmp_v05b_*`, `tmp_*.py`)
- `deck.csv` (root only)
- `*-promotion-decision.json`, `*-mlp_submission_safety.json`

**Do not push:**
- Notebooks whose execution failed or is still in progress.
- Versions without a promotion decision record.
- Archives, model weights, parquet/npy data (covered by `.gitignore`).

**Conflict on push:** pull, rebase (`git pull --rebase`), resolve any conflicts, push again.

---

## Loop Procedure

### Step 0: Meta-Cognitive Framing (mandatory — before touching the version goal or code)

Write in plain language before step 1:

1. **Current state**: What is the runtime baseline? Research baseline? Known gaps?
2. **Formal requirement**: What does the harness require for the target promotion type (winner_margin gate, guarded mode gate, etc.)?
3. **Gap being closed**: What specific gap does this experiment close? What gaps remain after it?
4. **What is actually being tested**: Distinguish what the experiment tests from what it merely changes.
5. **Causal assumptions**: Which assumptions underlie the hypothesis? Where could they be wrong?
6. **Own bias**: What bias do you have toward proposing this experiment? How does the design resist it?
7. **Negative result shape**: What would a negative result look like? Can you distinguish it from a positive one?

Do not skip this even if the next experiment feels obvious. Write it before proceeding to step 1.

---

### Steps 1–14

1. Read runtime and research baseline summaries from `loop-harness.md`. Record `baseline_winner_margin`.
2. Write the version goal using the template in `loop-harness.md`. Fill in `experiment_type` and `consecutive_conservative_count`.
3. Write the pre-implementation self-review. **Do not proceed unless it produces an explicit `go`.**
4. Copy or bump the notebook name and `RUN_PREFIX`. Include machine suffix.
5. Make the smallest implementation change that tests the hypothesis.
6. Run a smoke check.
7. Execute the canonical notebook end-to-end.
8. Verify artifacts were regenerated under `/kaggle/working/<RUN_PREFIX>/`.
9. Verify artifact filenames use `<RUN_PREFIX>-` prefix.
10. Compute winner_top1, loser_top1, winner_margin on holdout. Compare against baseline.
11. Compare holdout_model_top1 against baseline (regression guard).
12. Run local smoke/durability checks: legality, exceptions, packaging, runtime.
13. Record the promotion decision using the template in `loop-harness.md`. Write `<RUN_PREFIX>-promotion-decision.json`.
14. Update `experiment-backlog.md`: move this experiment to `Completed`; add `next_candidates` from the promotion decision as new `Proposed` entries.
15. Git: add, commit, push (see Git Workflow above).

**Do not continue automatically into the next version without instruction.**

**If a run fails:** fix only when the fix is necessary to validate the current hypothesis and does not broaden scope. If the failure reveals a flawed hypothesis or validation design, record the failure and stop.

---

## Long-Running Execution Monitoring

Notebook execution can take several minutes (feature extraction, RL training, game simulation). Do not poll excessively.

**Recommended cadence:**
- Check promptly after starting to confirm the process began.
- During long cells: check every 2–5 min. Use 5–10 min intervals for known slow phases (game simulation, offline RL epochs).
- Prefer targeted checks: process still running (`ps aux | grep nbconvert`), latest artifact timestamp (`ls -lt <dir>/ | head -3`), final report file tail.
- Avoid repeatedly dumping large JSON files, full directory listings, or full notebook output while running.
- Increase monitoring detail only on: error, timeout, no CPU activity, no file changes for an unexpectedly long time.

**After execution finishes:** perform full verification once — notebook exit status, required artifacts, all reports, archive contents, loader gates, smoke gates, promotion decision.

**For agent-managed execution:** schedule `ScheduleWakeup(300)` immediately after launching background execution. At each wakeup:
1. Check if nbconvert process is running: `ps aux | grep nbconvert | grep -v grep`
2. Check log tail: `tail -5 <logfile>`
3. Check newest artifact: `ls -lt <output_dir>/ | head -3`
4. Still running → `ScheduleWakeup(300)` and stop.
5. Finished → proceed to full verification.

---

## Experiment Backlog Management

- Read `experiment-backlog.md` before starting any version (after `git pull`).
- Claim one experiment per session (pull → edit → push immediately, before implementation).
- Tag every proposed experiment `[conservative]` or `[aggressive]`.
- After completing a version, move it to `Completed` and copy `next_candidates` from the promotion decision as new `Proposed` entries.
- Keep `Completed` to the most recent 8–10 entries (archive older ones or trim).
- When the backlog `Proposed` list is empty, propose new experiments based on the latest promotion decision's `next_candidates` and current research gaps.

---

## Notebook Inter-Cell Contract

Cell [23] (training/RL cell) must define these notebook-global variables before it exits, or Cell [25] (runtime probe and promotion decision) will fail with NameError:

- `MAIN_HYBRID_REPORT` — dict with at minimum: `quality_ok` (bool), `all_gates_ok` (bool), `winner_margin` (float), `model` (dict: `model_path`, `best_epoch`, `epochs_run`, `best_valid_top1`, `param_count`), `holdout_summary` (dict: `model_top1`), `holdout_hybrid_summary` (dict: `hybrid_top1`).
- `MAIN_LEARNING_REPORT` — dict with training log and final metric values.
- `_card_table` — dict mapping `int(card_id)` → card object, needed by Cell [25]'s feature cross-check.

**When `SKIP_PIPELINE=True`:** any variable normally defined in skipped cells (4–21) that is needed by Cell [23] or Cell [25] must be defined in Cell [3] (always executed) or in the offline branch of Cell [23].

**Known issue (as of v07d4):** `import_agent_from_source` is defined in Cell [19] and used in Cell [25]. Fix: move the definition to Cell [3]. This is the pending fix for v07d5.

---

## Operational Checklist

Before each version:
- [ ] `git pull` done
- [ ] Experiment claimed in backlog and pushed
- [ ] Experiment type matches alternation rule (conservative if last was aggressive; aggressive if last was conservative)
- [ ] Meta-cognitive framing written
- [ ] Version goal written with all template fields filled
- [ ] Self-review written with explicit `go`

During each version:
- [ ] Only one hypothesis being tested
- [ ] Notebook named with correct prefix including machine suffix
- [ ] Artifacts writing to `/kaggle/working/<RUN_PREFIX>/`

After each version:
- [ ] winner_margin computed on holdout (not train/valid)
- [ ] winner_margin compared against baseline_winner_margin
- [ ] Promotion decision written and saved as `<RUN_PREFIX>-promotion-decision.json`
- [ ] Harness Baseline Handling updated if research baseline improved
- [ ] Backlog updated (move to Completed, add next candidates)
- [ ] `git push` done
