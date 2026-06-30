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


---

## Strategic Pre-Loop Audit (Step 0 の前に必ず実施)

ループを始める前に、以下の問いに答える。Step 0 の meta-cognitive framing より一段上のレイヤー。
この問いに答えずに次は何を試すかを決めるのは禁止。

### A. 資産棚卸し（何が使われていないか）

訓練済み・計算済みだが submission や評価で使われていないものを列挙する：

- 訓練済みモデルで submission 未使用のもの（MLP、BC、value model など）
- 計算済み特徴量・スコアで活用されていないもの
- 診断で正シグナルが見えたが、まだ攻めていないコンテキスト・フェーズ・スライス
- 用意されているデータセットで使っていないもの

問い：今 submission に使われているのは全資産の何割か？
この割合が低ければ、新しい手法を作るより資産を活用する方が先。

### B. 構造的仮定の洗い出し（何を当たり前にしているか）

暗黙のうちに受け入れている仮定を言語化する：

- データ: すべての決定を等重みで扱っていないか？時間軸を無視していないか？
- モデル: 単一の汎用モデルに頼っていないか？コンテキスト専用の方が良くないか？
- 評価: aggregate 指標だけ見ていて、スライス別の分析を怠っていないか？
- 対象: 特定のコンテキストに執着しすぎていないか？

問い：この仮定が間違っていたら、次は何を試すか？
これが答えられるなら、その実験の方が今の計画より高価値かもしれない。

### C. 直近 N バージョンの振り返り

直近 3-5 バージョンを見て：

- 同じ種類の変更（パラメータ調整）が続いていないか？
- winner_margin の改善が止まっていないか？
- バックログの [aggressive] 実験を先送りし続けていないか？

ルール：パラメータ系実験が 2 回続いたら、次は必ず構造実験かアルゴリズム実験。

---

## 実験設計の情報密度最大化

1 回の実験から得られる情報量を最大にする。run コストは固定、情報量は設計次第。

### 1 run = 最低 2 つの問いに答える

悪い例：MLP を MAIN に有効化するだけ → 成否の 1 bit しか得られない。

良い例：MLP を MAIN に有効化する（B）＋全コンテキストの winner_margin を診断する（A）
→ B の成否 ＋ 次に攻めるべきコンテキストの地図 が同時に得られる。

設計チェック：この run が失敗した場合、次の仮説は何になるか？それも同じ run で計測できないか？

### Phase / スライス分析を標準化

aggregate の winner_margin だけでなく、常に以下の breakdown を取る：

- ゲームフェーズ別：早盤 (step <= T/3)、中盤、終盤
- コンテキスト別：UNKNOWN_0、MAIN、TO_BENCH など
- 対象コンテキストの hit/miss 別（policy が発動した決定 vs しなかった決定）

これを入れるだけで「何が効いて何が効かなかったか」が同一 run で見える。
早盤が最強シグナルという発見も phase 分析で初めて見えた（v0-05d9）。

### 意外な発見を事前に想定する

バージョンゴールに以下を書く：

  unexpected_finding_protocol:
    仮説が逆だった場合: ○○を確認して次の仮説を□□に切り替える
    効果がなかった場合: △△の診断データを見て次のターゲットを絞る

---

## 意外な発見の活用プロトコル

実験結果が仮説と異なる方向を示した場合、それ自体が次の最良の仮説になることが多い。

### 発見の記録テンプレート

  仮説:    ○○だと思っていた
  実際:    △△だった
  逆転の程度: 軽微 / 顕著 / 完全に逆
  理由の仮説: これは△△を意味するかもしれない。なぜなら...
  次の仮説: だから、□□を試すべき

例（v0-05d9）:
  仮説:    終盤の決定ほど勝敗因果が強い → MC down-weight で終盤優先学習が効く
  実際:    早盤の winner_margin (0.102) > 終盤 (0.076) — 逆だった
  理由仮説: UNKNOWN_0 の早盤はプレイヤーの判断力の差が最も出る局面（手札運の影響が小さい）
  次の仮説: 早盤 UNKNOWN_0 専用 policy か、逆方向の重みを試す

### 意外な発見 = キャンセルではなく昇格

仮説が外れても exploration_promote にできる条件：
- 発見が再現可能（同一 holdout split 上で数値が出ている）
- 次の仮説が具体的に生まれた
- バックログに「意外な発見から派生した実験」として追加した

---

## データ戦略の判断フレーム

「最新データのみ vs 複数データセット統合」を毎回アドホックに決めず、判断基準を持つ。

### メタ変動仮説 → 最新のみ

- 上位プレイヤーのデッキ構成が変わっている
- 新カード・環境変化がある
- ランキングデータが 1 週間以上古い

最新データセットのみを使う。古いデータを混ぜると現在のメタのシグナルが薄まる。

### メタ安定仮説 → 統合

- 同じデッキ・環境が続いている証拠がある
- 最新データだけではサンプル不足でシグナルが弱い（n < 1000 episodes など）

複数エポックを結合して使う。ただし episode_id で dedup を必ず行う。

バージョンゴールに data_strategy: latest_only か combined と理由を書く義務。

---

## Bold 仮説の生成プロンプト

次は何を試すかを考えるとき、以下の問いで発想を広げる。

### A. 別の ML 原理からの転用

- 今は IL（教師あり）で学んでいる。RL の考え方（リターン、アドバンテージ、方策勾配）を持ち込んだら？
- 今は point-wise にオプションをスコアリングしている。pair-wise や list-wise 学習に変えたら？
- 今は全決定を独立に扱っている。系列情報（前の手、ゲーム状態の遷移）を使ったら？
- 今は単一モデルで全コンテキストをカバーしている。コンテキスト専用の小モデルに分けたら？

### B. フレーミングの転換

- winner を模倣するではなく、loser との差分を直接最小化するに変えたら？
- action を予測するではなく、action の value を推定するに変えたら？
- 単一ターンの決定ではなく、次の N ターンのシナリオを評価したら？

### C. 既存資産の大胆な活用

- 今使っていない訓練済みモデルを submission に入れたら、最悪何が起きるか？
  その最悪ケースが許容範囲なら、試す価値がある。
- 小さな alpha で blending するのではなく、全面的に切り替えたら？

ルール：怖いから入れないは理由にならない。gates を設けて確認できるならやる。

---

## Cross-Track 学習（v0-05d と v0-07d）

2 つのトラックが独立して走るときの情報共有プロトコル。

### 各トラックの役割

  v0-05d | 解釈可能・ノイズ少・高速 | policy table hit/correct、per-context winner_margin
  v0-07d | スケーラブル・end-to-end  | RL winner_margin、guarded action_changes

### 共有すべき知見

v0-05d が v0-07d に渡せるもの：
- どのコンテキストで winner_margin が高いか（特徴量設計・報酬設計のヒント）
- どのゲームフェーズで policy の差が大きいか（RL の episode 収集戦略）

v0-07d が v0-05d に渡せるもの：
- RL で学んだ行動傾向（policy table の signature 設計のヒント）
- winner_margin が高かった対戦相手・デッキ（評価スライスの絞り込み）

### Cross-Track 同期のタイミング

- promotion decision を書いたら、もう一方のトラックへの示唆を 1 行追加する
- 4 バージョンに 1 回（または major な発見があった時）、両トラックの知見を統合して experiment-backlog.md を更新する

---

## 実験の「レバレッジ」評価

次の実験候補が複数あるとき、以下の観点でスコアを比較して優先度を決める。

  情報量          : 成否の 1 bit だけか、失敗でも次の仮説が生まれるか
  資産活用度      : 新しいコードを書くか、既存の訓練済みモデルを活用するか
  仮説の独自性    : 同じアプローチのパラメータ調整か、別の ML 原理や別フレーミングか
  結果の解釈容易性: 何が効いたかわからなくなるか、変更 1 つで因果が追えるか
  影響面積        : 1 コンテキストのみか、全決定に影響するか

情報量が高く、資産活用度が高く、仮説が独自な実験を優先する。
情報量が低く、新しいコードが多い実験は後回し。
