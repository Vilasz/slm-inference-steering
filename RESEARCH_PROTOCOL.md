# Early correctness signals in code generation — protocol v0.2

**Version:** 0.2  
**Date:** 2026-10-12  
**Status:** draft for advisor approval  
**Scope:** Qwen2.5-Coder-1.5B-Instruct, HumanEval+, task-grouped prediction and early abort.

This revision implements the methodological corrections in the supplied manuscript.
The user authorized implementation on 2026-10-12. This authorization is not a
record of advisor approval or a retrospective preregistration. The historical
[v0.1](docs/protocols/RESEARCH_PROTOCOL_v0.1.md) and its candidate bank are preserved.

| Role | Name | Decision | Date |
|---|---|---|---|
| Student | João Felipe Vilas Boas | pending scientific freeze | — |
| Advisor | Prof. Dário Oliveira | pending | — |

## 1. Questions and claims

Q1 asks whether prefix representations predict eventual functional correctness
on tasks excluded from training. Q2 asks whether H+B improves on a specified
observable baseline B within the same task. Q3 asks whether this improves a
paired cost–resolution trade-off over both verifier stopping and an observable
abort policy. These are distinct questions. Prediction does not establish a
causal correctness mechanism or information beyond the full prefix and model.

Primary Q2 endpoint: 32 generated tokens. Positive incremental evidence requires
ΔWT-AUC(H+B−B)>0, a paired task-bootstrap 95% interval entirely above zero, and
Δlog loss(H+B−B)<0. Report its interval too; if it includes zero, probabilistic
improvement remains uncertain. Association permutation is a separate diagnostic,
not a test of conditional incremental information. Other checkpoints and layers
are secondary; do not replace the primary endpoint after seeing its result.

## 2. Prior exposure and evidence status

Before this revision, v0.1 contained 39/960 pilot trajectories, with no fitted
pilot probes and no selected temperature. Earlier exploratory artifacts include
150 final-token activation vectors across 30 HumanEval tasks. Those are late
representations with the historical verifier, not 32-token HumanEval+ evidence.
The legacy inventory contains 30 tasks: four in the current pilot and 26 in the
132-task remainder. File presence establishes prior generation, not which
results were inspected. Keep the fixed split, disclose this history, and do not
claim an entirely researcher-unseen HumanEval holdout. Do not exclude tasks
because of their outcomes. Audit external-benchmark exposure as well.

Historical reanalysis is explicitly exploratory and cannot choose the new
temperature, checkpoint, layer grid, threshold rule or primary endpoint.

## 3. Generation and verifier

Exact revisions, dtype and hyperparameters are in `configs/protocol_v0.2.yaml`.
Use immutable model/tokenizer revision 2e1fd397ee46e1388853d2af2c993145b0f1098a,
temperature selected by the pilot, top-p=0.95, top-k disabled, and at most 512
new tokens. One candidate at a time; no verifier feedback between attempts.
Seeds depend on protocol version, benchmark version, task ID and attempt ID.
Aborted attempts consume attempt budget and never change subsequent seeds.

Use official EvalPlus 0.3.1, full HumanEval+ base and plus tests, fixed checker
limits (minimum 1 second; ground-truth factor 4), and record dataset versions
and hashes. Canonical and deliberately wrong control programs precede collection.
The worker runs under POSIX/WSL; generation runs on the local GPU.

Primary label is functional approval by both suites for the extracted program.
Truncation is a finish attribute, not automatic functional failure. Also record
`passed_nontruncated`. Syntax/runtime/assertion/program-timeout/extraction errors
are program failures. Worker crashes, malformed RPC and infrastructure failures
halt collection without appending a fabricated negative label. The official
checker may not distinguish assertion from runtime failure; preserve its status.

Store prompt and completion token IDs, sampled log probabilities/entropies,
pre-top-p margins, extracted source, label/status, finish reason, timings,
seeds, revision/template hashes, activation paths and hashes. JSONL is append-only,
with an exclusive writer lock. Never mix v0.1 and v0.2 records or restart a
partial bank under altered scientific configuration.

## 4. Pilot and sample size

Choose 32 HumanEval tasks by ascending SHA256 of
`protocol-v0.1-pilot:` plus task ID (preserved to avoid exposing a new development
set). Generate ten attempts per task at each temperature 0.6, 0.8 and 1.0: 960
trajectories. Attempt IDs include the temperature index; interleave temperatures.
No pilot probe AUC may guide generation choices.

After the complete grid, eligible temperatures must have aggregate pass rate
strictly between 0.1 and 0.9 and combined syntax/empty/truncation rate at most
five percentage points above the minimum across temperatures. Maximize the
number of mixed tasks; break ties by pass rate closest to 0.5, lower mean length,
then lower temperature. If none qualify, amend prospectively on development
data and repeat; never silently default to a favorable temperature.

For the main bank, use the remaining 132 tasks with 20 complete attempts each
(2,640 total), even after an early success. This complete bank supports paired
replay. A session time limit stops between attempts without changing sample size.
Generalization units are tasks, not candidates or candidate pairs.

## 5. Numerical and temporal validity

Local reference: RTX 3050 Laptop 6 GB; BF16 weights, FP32 linear projection
compute, eager attention, TF32 and reduced-precision GEMM reductions disabled.
Record versions, GPU/driver, peak allocated/reserved VRAM and wall time. Native
FP16/BF16 failed earlier technical checks; their failures are documented separately.
Save selected vectors as FP16 and account for storage rounding in validation.

Capture block outputs before final model normalization at zero-based blocks
6,13,20,27. Checkpoints are prompt-last (0), 8,16,32,64 and final. At t>0 feed
the t-th sampled token before reading its activation. Compare prefix-only,
full-forward and cached-generation states at each available checkpoint:
per-layer relative L2 <=0.01 and max absolute difference <=0.02+0.01*max|reference|.
Repeated seed must reproduce token IDs. Non-finite values halt collection.

Check natural termination and the token limit before invoking the abort rule.
Operational/predictive prefix risk set is T>t; completed-at-t states may exist
in storage but do not enter the continuation analysis. Never pad short sequences.
Final states are descriptive late references, not guaranteed upper bounds.
Report counts, mixed tasks, prevalence and truncation by checkpoint. Canonicalize
numerically equivalent prompt vectors within task for the deterministic control.

## 6. Features and learning

B uses prompt/signature token counts and fixed format indicators, plus mean
NLL, last-token NLL, mean entropy, maximum entropy and mean logit margin of the
observed prefix. NLL/entropy use the actually sampled, temperature/top-p
distribution. Margin uses the two largest temperature-scaled logits before
top-p; it remains finite for a singleton nucleus. No future length or verifier
label is a feature. H is one selected block vector; H+B concatenates that vector
with the same B features. Include prior, prompt-only, prompt-state and
confidence-only baselines, random direction, shuffled-training-label and
prefix stopping/length diagnostics. Inverted scores are orientation diagnostics.

Use L2 logistic regression with lbfgs and an unpenalized intercept. Imputation,
standardization and model fitting happen in the appropriate training partition.
Sample weights are n/(m*n_i), assigning equal total weight to each task and
mean weight one; preserve sklearn's C convention, not an assumed lambda=1/C.
C grid: 0.001,0.01,0.1,1,10. Select task-balanced log loss; within one estimated
standard error of the best, choose stronger regularization then shallower layer.
The one-SE rule is a selection heuristic, not a confidence interval.

Use five fixed outer GroupKFold partitions reused across checkpoints and
families; three inner grouped folds for selection. Produce calibration raw scores
by cross-fitting the *whole hyperparameter selection* on the outer-training
tasks, with another grouped selection confined to each cross-fitting training
partition. Fit Platt calibration to these scores. The calibrator and threshold
remain development fits; their apparent retention is not independently certified.
All final performance is evaluated on untouched outer tasks. Record fit/score
task IDs, selected hyperparameters, models and calibrators. Insufficient class
or task support must be reported rather than replaced by a favorable split.

## 7. Estimation and inference

WT-AUC is the unweighted mean of tie-aware AUCs over mixed eligible tasks.
Report the number of such tasks; single-class tasks remain in probabilistic
and operational metrics. Aggregate ROC-AUC, average precision (named explicitly),
log loss and Brier are task-balanced. Calibration uses ten fixed equal-width bins.

Use 2,000 task-bootstrap replications with seed 20260921, paired across methods.
For ΔWT-AUC sample mixed tasks; for losses and policy outcomes sample all
contributing tasks. Report conditional population and undefined replicates.
These intervals condition on fitted OOF scorers and do not capture full
retraining instability or all dependence caused by overlapping training folds.

For association, freeze each outer-fold scorer and permute labels within each
eligible task in that fold 10,000 times. Compute one-sided WT-AUC p-values with
the plus-one correction. Empty/nonmixed folds receive p=1. Combine by
min(1,5*min(p_k)); independence between folds is not required by Bonferroni.
State the within-task exchangeability assumption and limited power. This is
not a conditional test of adding H given B. Fixed-effects regression is an
associative secondary diagnostic; fitting task intercepts to evaluation labels
does not yield an out-of-sample predictor. Its conventional SE omits learned-score
uncertainty. Secondary families require declared multiplicity treatment.

## 8. Policy and operational evaluation

One checkpoint per fitted policy, chosen from 8,16,32,64. Abort iff active and
score<q. Search no-abort q=0 and thresholds immediately above each distinct
development probability, covering all strict-below decision boundaries/ties.
Retain only pairs with empirical recall >=0.97 among correct active development
candidates. With no active positives, allow no-abort only. Minimize mean
completion tokens per task until verified success or N attempted candidates.
Numerical cost ties (1e-9 tokens) prefer higher resolution, earlier checkpoint,
then smaller threshold. B and H+B receive the same search and data budget.

Replay fixed-N, verifier stopping, B-abort and H+B-abort on identical independent
candidate streams. Abort charges exactly t completion tokens. Natural finishes
charge full length. Stop a sequential policy at its first verified success.
Report per-decision identities, scores, active status and charged tokens, plus
false-abort denominators among started correct candidates, active and global.

N=20 is primary; N=1,5,10 are secondary. For each comparison H+B minus verifier
and H+B minus B, require the lower two-sided 95% CI for resolution difference
to exceed -0.03 and the upper CI for token difference to be below zero. Report
baseline savings and incremental savings separately. No final incremental
system-benefit claim without Q2 support and online replay agreement.

Online validation uses preselected task IDs and saved models, checks the exact
prefixes/decisions/costs and verifies completions. A one-task smoke test is not
a latency estimate. Measure paired generation/score/prefill/verification time
against verifier stopping on a declared subset; report research preparation
cost separately. Token savings do not imply wall-clock savings.

## 9. External robustness

Secondary MBPP+ frozen transfer: choose 32 tasks by SHA256 of
`protocol-v0.2-mbpp-transfer:` plus ID, 20 attempts each, with the frozen
HumanEval generation regime. Fit final B/H/H+B and calibrators/policies on the
allowed HumanEval main-bank development data using the same internal selection.
Freeze the model bundle and source hashes before generating/scoring MBPP+.
Do not tune on MBPP+ labels. Recalibrating or refitting there is a separately
named replication. Audit prior MBPP exposure and report limited external scope.

## 10. Gates, reproducibility and amendments

Gate 0: actual student/advisor approval, frozen status, committed configuration
and sampling record, immutable tag `research-protocol-v0.2` matching this file.
Gate 1: numerical, seed, verifier, split-isolation and accounting checks.
Gate 2: complete valid pilot and deterministic temperature selection.
Gate 3: complete audited main bank. Gate 4: external predictions and Q1/Q2.
Gate 5: replay plus online validation. Gate 6: secondary robustness and writing.
Pilot, tests and historical exploratory analyses can proceed before Gate 0.
Never manufacture approval or alter the status merely to bypass a gate.

The notebook and manuscript must label each result as technical/pilot,
exploratory historical, or confirmatory. Every reported number/figure must trace
to an artifact, input hash and command. A null or inconclusive result is a valid
outcome. The next amendment must record date, rationale, affected estimands and
all relevant results already known.
