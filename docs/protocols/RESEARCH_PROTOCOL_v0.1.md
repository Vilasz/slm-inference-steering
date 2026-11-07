# Research Protocol

## Early trajectory-specific correctness signals for efficient Best-of-N code generation

**Author:** João Felipe Vilas Boas  
**Advisor:** Prof. Dário Oliveira  
**Institution:** Escola de Matemática Aplicada — FGV  
**Protocol version:** 0.1  
**Status:** draft for advisor approval  
**Last updated:** 2026-09-21

---

## 1. Purpose of this document

This document freezes the central scientific question, confirmatory claims,
experimental design, evaluation rules, and decision gates of the project before
the definitive experiments are run.

The protocol has four purposes:

1. prevent the project from expanding into several weakly connected research
   agendas;
2. prevent test-set leakage and retrospective selection of favorable layers,
   checkpoints, thresholds, or metrics;
3. distinguish confirmatory analyses from exploratory extensions;
4. make negative, partial, and positive results equally interpretable.

The protocol must be committed and tagged before generation of the definitive
Candidate Bank. Any later change that can affect a confirmatory result must be
recorded in the amendment log at the end of this file.

---

## 2. Frozen research scope

### 2.1 Central research question

> Do the early hidden states of a Small Language Model contain
> trajectory-specific information about the future functional correctness of a
> code generation, beyond task difficulty and observable decoding-confidence
> signals, and can that information reduce the cost of Best-of-N inference?

The expression **trajectory-specific** is essential. The project does not ask
only whether the model can recognize difficult prompts. It asks whether, among
multiple stochastic generations of the **same prompt**, its internal state can
distinguish trajectories that will eventually pass the verifier from those that
will fail.

### 2.2 Three subquestions

| ID | Question | Target conclusion |
|---|---|---|
| Q1 | When does future correctness become predictable? | Measure how predictive performance changes from the prompt to early completion prefixes. |
| Q2 | Is the signal specific to the current trajectory? | Test discrimination within the same task and incremental value over observable prefix statistics. |
| Q3 | Is the signal operationally useful? | Test whether recall-controlled early abort reduces generated tokens without a material loss in task resolution. |

The logical structure is:

```text
predictable → trajectory-specific → actionable
```

Q2 is the central scientific contribution. Q3 is attempted only after Q2 is
measured; a null Q2 result does not invalidate the thesis.

### 2.3 Primary claim

The primary claim under evaluation is:

> A linear score derived from early hidden states provides out-of-sample,
> within-task information about future functional correctness beyond information
> contained in observable prefix statistics.

This claim will be evaluated using predictions for held-out tasks, task-balanced
within-task AUC, task-clustered bootstrap confidence intervals, and label
permutations restricted to each task.

The confirmatory checkpoint for this claim is fixed at **32 generated
completion tokens**. Checkpoints 8, 16, and 64 characterize the temporal curve;
they do not replace checkpoint 32 as the primary endpoint after results are
observed.

### 2.4 Secondary claim

The secondary claim is:

> A policy that aborts low-scoring trajectories at a validation-selected prefix
> can preserve the resolution rate of verifier-based Best-of-N within a
> predeclared non-inferiority margin while reducing the expected number of
> generated completion tokens.

### 2.5 Explicitly out of the main scope

The following topics are not part of the confirmatory core:

- causal Activation Steering;
- Average Treatment Effect estimation for activation interventions;
- sparse autoencoders and feature-level interpretability;
- cross-model latent transfer;
- optimal stopping for deciding whether an additional independent attempt is
  worthwhile;
- survival analysis as a primary contribution;
- broad model benchmarking;
- DS-1000;
- extensive hyperparameter sweeps after test results are known.

Activation Steering may be included only as a short exploratory appendix after
Q1–Q3 and the robustness experiment are complete. It must not alter the main
claims.

---

## 3. Definitions and estimands

Let task \(i\) have prompt \(x_i\). Attempt \(j\) produces a stochastic
completion

\[
Y_{ij} \sim p_\theta(\cdot \mid x_i, c),
\]

where \(c\) is the frozen decoding configuration. Functional correctness is

\[
S_{ij}=V(x_i,Y_{ij})\in\{0,1\},
\]

with \(V\) denoting the EvalPlus verifier.

For transformer layer \(\ell\) and completion offset \(t\), the extracted hidden
state is

\[
h_{ij\ell t}\in\mathbb{R}^d.
\]

A linear probe fitted only on training tasks produces a latent score

\[
z_{ij\ell t}=f_{\ell t}(h_{ij\ell t}).
\]

The primary statistical estimand is the out-of-sample improvement in
task-balanced within-task ranking obtained by adding latent information to the
observable baseline:

\[
\Delta\operatorname{WT\!\!-\!AUC}
=
\operatorname{WT\!\!-\!AUC}_{H+B}
-
\operatorname{WT\!\!-\!AUC}_{B}.
\]

For every mixed task \(i\), define

\[
\operatorname{AUC}_i
=
P(\hat p_{ij}^{+}>\hat p_{ik}^{-}\mid x_i),
\]

where \(j\) indexes a correct attempt and \(k\) an incorrect attempt. Ties count
as \(0.5\). Then

\[
\operatorname{WT\!\!-\!AUC}
=
\frac{1}{|\mathcal M|}
\sum_{i\in\mathcal M}\operatorname{AUC}_i,
\]

where \(\mathcal M\) is the set of tasks containing at least one correct and one
incorrect trajectory in the evaluated risk set.

This gives equal weight to each informative task, regardless of its number or
proportion of correct attempts.

---

## 4. Confirmatory hypotheses and decision rules

### H1 — Temporal predictability

Early hidden states predict eventual functional correctness on held-out tasks.
Performance is evaluated over completion offsets, against observable and
prompt-only baselines.

H1 is descriptive and establishes when useful information appears. It is not,
by itself, the novel claim.

### H2 — Trajectory specificity and incremental information

On held-out mixed tasks, the combined hidden-plus-observable model ranks correct
above incorrect trajectories from the same prompt better than the observable
model alone at the predeclared 32-token checkpoint.

Evidence for H2 requires all of the following:

1. \(\Delta\operatorname{WT\!\!-\!AUC}>0\);
2. the 95% task-clustered bootstrap interval for the delta excludes zero;
3. the one-sided within-task permutation test gives \(p<0.05\);
4. combined-model out-of-fold log loss is lower than observable-model log loss.

Effect sizes and confidence intervals remain primary; the p-value is not
reported in isolation.

### H3 — Actionability under controlled recall

Relative to verifier-based Best-of-N with the same maximum attempt budget, the
latent early-abort policy must satisfy both:

1. **resolution non-inferiority:** the lower endpoint of the 95% task-bootstrap
   interval for
   \(\operatorname{Resolution}_{latent}-\operatorname{Resolution}_{verifier}\)
   is greater than \(-3\) percentage points;
2. **compute improvement:** the upper endpoint of the 95% task-bootstrap
   interval for
   \(\operatorname{Tokens}_{latent}-\operatorname{Tokens}_{verifier}\)
   is below zero.

The abort threshold is selected without test data to retain at least 97% of
eventually correct trajectories on validation data.

If H2 is not supported, H3 may still be reported as a predeclared diagnostic,
but no operational benefit will be claimed based on an isolated favorable
threshold.

---

## 5. Experimental environment

### 5.1 Main model

- **Model family:** Qwen2.5-Coder-1.5B-Instruct
- **Number of main models:** one
- **Inference mode:** `eval()` with dropout disabled
- **Weights and tokenizer revision:** exact immutable revision/commit to be
  recorded in the run manifest before the Candidate Bank run
- **Precision, attention backend, package versions, CUDA version, and hardware:**
  recorded automatically in the run manifest

The project studies a phenomenon, not a leaderboard of models. Additional
models are outside the confirmatory scope.

### 5.2 Main benchmark

- **Benchmark:** HumanEval+
- **Evaluation implementation:** EvalPlus
- **Unit of grouping and resampling:** `task_id`
- **Unit of generation:** `(task_id, attempt_id, seed)`

The benchmark version and dataset hash must be stored in the manifest.

### 5.3 Planned robustness benchmark

- **Benchmark:** MBPP+
- **Purpose:** one external check of whether the phenomenon extends beyond
  HumanEval+
- **Status:** secondary robustness analysis, not required to establish the main
  HumanEval+ result

If the full MBPP+ run exceeds the compute budget, the subset must be selected by
a deterministic rule based only on `task_id`, before any MBPP+ outcomes are
inspected.

---

## 6. Pilot and freezing of the sampling regime

The pilot exists only to select a sampling regime that creates enough
within-task variation. It must not select the configuration with the best probe
performance.

### 6.1 Pilot task set

- Select 32 HumanEval+ tasks using the smallest hexadecimal values of
  `SHA256("protocol-v0.1-pilot:" + task_id)`.
- Mark these tasks as `development_only=true`.
- They may be used to choose the decoding regime and debug the pipeline.
- They must never appear in an outer test fold or in confirmatory metrics.

### 6.2 Candidate pilot grid

Unless hardware validation reveals a technical incompatibility, compare:

| Parameter | Values |
|---|---|
| temperature | 0.6, 0.8, 1.0 |
| top-p | 0.95 |
| attempts per task | 10 |
| max new tokens | 512 |

All other generation parameters remain identical.

### 6.3 Predeclared selection rule

Choose the temperature that maximizes the number of mixed pilot tasks, subject
to:

- aggregate pass rate strictly between 10% and 90%;
- no material increase in invalid/truncated outputs relative to the other
  candidates;
- the same verifier and prompt template used for all candidates.

Tie-breakers, in order:

1. pass rate closest to 50%;
2. lower mean completion-token count;
3. lower temperature.

The selected temperature, exact prompt template, stopping rules, and package
versions are then frozen in the run manifest. Probe AUC, latent scores, or early
abort performance must not be computed during regime selection.

### 6.4 Default expected final regime

The anticipated configuration, to be confirmed by the rule above, is:

```yaml
temperature: 0.8
top_p: 0.95
attempts_per_task: 20
max_new_tokens: 512
do_sample: true
```

Any deviation is determined by the pilot rule, not discretionary inspection.

---

## 7. Definitive Candidate Bank

### 7.1 Sample

For every non-pilot HumanEval+ task, generate 20 attempts under the frozen
regime. With 132 confirmatory tasks, the expected maximum is 2,640 definitive
trajectories.

Pilot trajectories are stored separately and are never silently merged into
confirmatory evaluation.

### 7.2 Attempt-specific seeds

Every attempt receives its own deterministic seed derived from:

```text
(protocol_version, benchmark_version, task_id, attempt_id)
```

Random-number state must be reset at the start of each attempt. This makes later
attempts invariant to whether an earlier attempt is fully generated or aborted,
which is necessary for valid offline policy replay.

### 7.3 Required fields

Each trajectory record must include at least:

```text
protocol_version
run_id
benchmark_name
benchmark_version
task_id
attempt_id
seed
model_id
model_revision
tokenizer_revision
prompt_template_hash
decoding_config
prompt
raw_completion
extracted_code
verifier_version
passed
error_type
generated_tokens
finish_reason
truncated
generation_time_seconds
verification_time_seconds
token_ids
token_logprobs
token_entropies
artifact_hashes
```

The bank is append-only after generation. Corrections create a new version; they
do not overwrite past outcomes without an audit trail.

### 7.4 Functional label

The primary label is passing all HumanEval+ tests under EvalPlus. Syntax errors,
runtime errors, timeouts, failed assertions, empty extractions, and truncations
are failures for the primary analysis. `error_type` is retained for secondary
error analysis.

No manual relabeling is allowed based on whether a completion appears
conceptually correct.

---

## 8. Temporal activation extraction

### 8.1 Checkpoints

Extract activations at:

\[
t\in\{0,8,16,32,64,T\},
\]

where:

- \(t=0\): hidden state of the last prompt token (`prompt_last`);
- \(t\in\{8,16,32,64\}\): hidden state at that offset within the generated
  completion;
- \(t=T\): final generated token, used only as a late upper bound.

Offsets are relative to the completion, never absolute positions in the full
prompt-plus-completion sequence.

### 8.2 Interpretation of `prompt_last`

For repeated attempts of the same prompt, `prompt_last` is expected to be
identical up to numerical tolerance because stochastic sampling has not yet
caused the trajectories to diverge. Therefore:

- it may encode task difficulty;
- it cannot provide genuine within-task trajectory discrimination;
- its expected within-task AUC is 0.5 after tie handling;
- it is a task-level baseline and negative control for trajectory specificity.

### 8.3 Risk sets and short completions

At checkpoint \(t\), the prediction risk set contains only trajectories that
actually reach \(t\) generated tokens. The number of trajectories, tasks, mixed
tasks, pass rate, and truncation rate in every risk set must be reported.

Trajectories ending before \(t\) are not imputed. In policy replay, they finish
naturally and are verified; they are not treated as aborted at \(t\).

### 8.4 Layers

Candidate layers are fixed at the transformer blocks nearest to:

\[
25\%,\ 50\%,\ 75\%,\ 100\%
\]

of model depth. Exact integer indices are computed once from the loaded model
configuration and written to the manifest.

No exhaustive layer scan is used for a confirmatory claim.

### 8.5 Causal-prefix invariance test

For a fixed stored completion, compare the state at offset \(t\) obtained from:

1. a forward pass over `prompt + completion[:t]`;
2. a forward pass over the entire `prompt + completion` and selection of the
   same causal position.

The states must agree within a predeclared numerical tolerance appropriate to
the inference precision. This test must pass before definitive probes run. It
demonstrates that prefix features do not use future tokens.

---

## 9. Predictors and baselines

All features at checkpoint \(t\) may use only the prompt and the first \(t\)
completion tokens.

### B0 — Global prior

Training-fold prevalence only. No per-task empirical pass rate is available for
an unseen test task.

### B1 — Prompt observables

- prompt token count;
- function-signature token count, when reliably parseable;
- prompt formatting indicators fixed before outcome inspection.

### B2 — Prompt representation

A linear probe on `prompt_last`. This is a strong task-difficulty baseline but
not a trajectory-specific predictor.

### B3 — Observable prefix confidence

At completion checkpoint \(t\):

- cumulative mean token negative log-likelihood;
- last-token negative log-likelihood;
- cumulative mean predictive entropy;
- maximum predictive entropy so far;
- cumulative mean top-1/top-2 logit margin.

For the primary analysis, these quantities are computed from the actual
sampling distribution after temperature scaling and top-p filtering, with
renormalization. Raw-model-distribution variants may be exploratory. The exact
numerical implementation is frozen before the definitive run.

### B — Combined observable baseline

Prompt observables plus observable prefix-confidence features.

### H — Hidden-only model

An L2-regularized linear logistic probe on the selected hidden state.

### H+B — Hidden plus observable model

The hidden state plus the exact features used by B. H+B versus B is the relevant
incremental comparison.

An empirical per-task pass rate calculated from other attempts is allowed only
as a labeled **oracle diagnostic**. It cannot be used in the deployed policy or
the main out-of-sample comparison.

---

## 10. Model fitting and leakage controls

### 10.1 Outer evaluation

Use five-fold `GroupKFold`, grouping exclusively by `task_id`. No task may occur
in both outer training and outer test data.

The final reported predictions are concatenated outer-test predictions, so each
confirmatory task is evaluated exactly once by a model that did not train on
that task.

### 10.2 Inner selection

Within each outer-training partition, use grouped inner cross-validation to
select:

- layer from the four frozen depth fractions;
- logistic regularization
  \(C\in\{10^{-3},10^{-2},10^{-1},1,10\}\).

For the confirmatory H2 analysis, the checkpoint is fixed at 32 completion
tokens. The primary selection objective is mean task-balanced validation log
loss for H+B. Ties within one standard error are resolved by:

1. stronger regularization, meaning smaller \(C\);
2. shallower layer.

The full temporal curves remain descriptive. For H3, the abort checkpoint may
be selected from \(\{8,16,32,64\}\) using policy replay entirely within the
outer-training data under the rule in Section 12.2. The outer test fold never
selects a checkpoint, layer, regularization, calibration method, or threshold.

### 10.3 Preprocessing

- Feature scaling parameters are fitted on training data only.
- Missing observable features are handled by a rule fitted on training data
  only and accompanied by missingness indicators.
- Each task receives equal total fitting weight: if task \(i\) contributes
  \(n_i\) eligible trajectories, each receives weight proportional to
  \(1/n_i\).
- No dimensionality reduction is planned for the primary linear probe.
- Any convergence failure is logged; increasing iteration limits is allowed,
  but changing the estimator after seeing test performance is not.

### 10.4 Correctness directions

Any mean-difference, Fisher/LDA, or other correctness direction is constructed
using outer-training labels only. Test labels must never enter direction
construction, feature selection, or sign orientation.

### 10.5 Calibration

Probability calibration uses Platt scaling fitted to grouped inner
out-of-fold predictions from the outer-training data. The fitted calibrator is
then applied to the outer-test predictions.

Isotonic regression may be reported as a robustness analysis but cannot replace
the primary result after test inspection.

---

## 11. Statistical analysis

### 11.1 Q1 metrics

At every checkpoint, report:

- ROC-AUC;
- PR-AUC;
- log loss;
- Brier score;
- calibration plot;
- size and composition of the checkpoint risk set.

These are computed from outer-test predictions. Confidence intervals resample
tasks, never individual trajectories.

### 11.2 Q2 primary metric

Report task-balanced WT-AUC for B, H, and H+B, plus the paired deltas H+B minus
B and H minus B.

Only mixed tasks contribute to WT-AUC. The number and identity distribution of
always-correct, mixed, and always-wrong tasks must be reported. Because “mixed”
is defined using observed outcomes, conclusions are explicitly conditional on
tasks for which the sampling regime produces both outcomes.

### 11.3 Fixed-effects analysis

As a complementary inferential analysis, fit a task-fixed-effects logistic
model to out-of-fold scores:

\[
\operatorname{logit}P(S_{ij}=1)
=
\alpha_i+\beta_B^\top B_{ij,t}+\beta_z z_{ij,t}.
\]

Here \(\alpha_i\) absorbs all task-constant information. The coefficient
\(\beta_z\), its clustered uncertainty, and predictive deltas are reported.
Tasks with no within-task label variation do not identify the fixed-effects
coefficient and are excluded from this particular regression. This regression
complements but does not replace out-of-sample WT-AUC.

### 11.4 Clustered bootstrap

- Resampling unit: task.
- Default replications: 2,000.
- The same resampled task indices are used for paired model comparisons.
- Report percentile 95% intervals and the random seed.

If computational cost makes 2,000 replications infeasible for an intermediate
analysis, fewer may be used for debugging, but final tables use 2,000.

### 11.5 Within-task permutation test

For the primary test:

1. keep outer-test predictions fixed;
2. shuffle correctness labels only among attempts belonging to the same task
   and checkpoint risk set;
3. preserve the number of correct and incorrect attempts within every task;
4. recompute the paired WT-AUC delta;
5. repeat 10,000 times;
6. compute the one-sided finite-sample p-value

\[
p=\frac{1+\sum_b\mathbb{1}
\{\Delta_b^{null}\geq\Delta^{obs}\}}{B+1}.
\]

This destroys trajectory-label alignment while preserving task difficulty
exactly.

### 11.6 Multiplicity

Only the nested-CV-selected H+B versus B comparison is confirmatory for H2.
Layer-by-layer, checkpoint-by-checkpoint, error-type, and subgroup results are
descriptive unless explicitly amended before test evaluation. This avoids
selecting the best result from a large matrix.

---

## 12. Early-abort policy

### 12.1 Policy definition

For a validation-selected checkpoint \(t^*\), compute calibrated probability

\[
\hat p_{ij}=P(S_{ij}=1\mid h_{ij,t^*},B_{ij,t^*}).
\]

The policy is:

```text
if a trajectory ends before t*:
    verify it normally
elif calibrated_probability < q:
    abort at t*
    continue to the next attempt
else:
    finish generation
    verify it
    stop the task if correct
```

### 12.2 Threshold selection

For each candidate checkpoint in \(\{8,16,32,64\}\), use grouped inner
out-of-fold predictions from outer-training data and select the largest
threshold \(q\) that retains at least 97% of eventually correct trajectories.
Among all checkpoint-threshold pairs satisfying the recall constraint, choose
the pair with the lowest mean replayed completion-token cost. If pairs are tied
within one standard error, choose the earlier checkpoint and then the lower
threshold. Freeze the pair and apply it to the outer test fold.

No test-fold recall, resolution, or token saving is used to tune \(q\).

### 12.3 Attempt order and maximum budgets

- Attempt order is increasing `attempt_id`.
- Maximum budgets reported: \(N\in\{1,5,10,20\}\).
- The primary policy comparison uses \(N=20\).
- Verifier early stopping ends a task immediately after the first verified
  correct completion.

### 12.4 Token accounting

For an aborted attempt, completion cost is exactly \(t^*\) generated tokens.
For a retained attempt, completion cost is its stored full generated-token
length. Prompt-prefill tokens and verifier time are reported separately in a
secondary accounting because implementation-level KV-cache reuse can change
their real cost.

Primary compute metric:

\[
\text{generated completion tokens per task}.
\]

Secondary metrics:

- generated completion tokens per solved task;
- attempts started per task;
- fully generated attempts per task;
- wall-clock generation time;
- verification time;
- correct-trajectory false-abort rate;
- task resolution rate.

### 12.5 Offline replay validity

Offline replay is valid only because each attempt has an independent,
preassigned seed. Aborting attempt \(j\) therefore does not change the sampled
completion for attempt \(j+1\).

At least one small online run must reproduce offline replay decisions and token
accounting before the system result is finalized.

---

## 13. Operational baselines

Compare:

1. **Fixed N:** generate exactly \(N\) attempts, reported for
   \(N\in\{1,5,10,20\}\);
2. **Verifier Best-of-N:** generate attempts sequentially and stop after the
   first verified success;
3. **Observable adaptive:** same abort procedure and recall constraint using B
   only;
4. **Latent adaptive:** same procedure using H+B.

Observable and latent adaptive policies must have identical candidate streams,
budgets, checkpoint eligibility, calibration procedure, threshold-selection
rule, and token accounting.

The core system comparison is latent adaptive versus both verifier Best-of-N and
observable adaptive.

---

## 14. Mandatory negative controls

The following controls are required:

1. `prompt_last` for within-task prediction;
2. within-task label permutation;
3. random normalized hidden-state direction;
4. direction fitted using permuted training labels;
5. sign-inverted correctness direction, when a direction-based analysis is
   reported;
6. length-only or stopping-pattern baseline;
7. explicit comparison with observable entropy/log-probability features.

A control is generated independently in each training fold when it depends on
data. No control may use test labels.

---

## 15. Required automated tests

The definitive pipeline must fail if any of these invariants are violated:

1. a `task_id` appears in both train and test;
2. a pilot task appears in a confirmatory outer test fold;
3. test labels are used to construct or orient a latent direction;
4. test performance selects a checkpoint, layer, hyperparameter, calibrator, or
   abort threshold;
5. an activation labeled offset \(t\) is not actually the \(t\)-th completion
   token;
6. prefix activation changes when future tokens are appended beyond numerical
   tolerance;
7. two definitive attempts share the same `(task_id, attempt_id)` or seed;
8. an attempt lacks its generation or verification provenance;
9. offline replay changes later attempt identity after an abort;
10. an aborted attempt is charged more or fewer than \(t^*\) completion tokens;
11. a naturally finished trajectory shorter than \(t^*\) is treated as an
    abort;
12. bootstrap or split logic resamples trajectories independently instead of
    grouping by task.

The test report and run manifest are artifacts of the final results.

---

## 16. Predeclared interpretation of outcomes

### Outcome A — No early predictive signal

Hidden and observable models remain near chance or fail to improve predictive
loss. Conclusion: no robust early correctness signal was detected in this
model-and-benchmark regime.

### Outcome B — Task-level signal only

Overall prediction is strong, but WT-AUC is near 0.5 and H+B does not improve B
within tasks. Conclusion: apparent self-knowledge is primarily task-difficulty
information, not knowledge of the fate of the sampled trajectory.

### Outcome C — Trajectory-specific but not actionable

H2 is supported, but the signal appears too late, is insufficiently calibrated,
or causes too many false aborts to reduce compute under the non-inferiority
constraint. Conclusion: trajectory information is decodable but not useful for
this early-abort policy.

### Outcome D — Trajectory-specific and actionable

H2 and H3 are supported. Conclusion: early hidden states contain incremental
information about the current trajectory and can improve the compute–resolution
frontier of Best-of-N under the tested regime.

All four are valid thesis outcomes. No new primary endpoint will be introduced
to rescue an unfavorable result.

---

## 17. Planned final artifacts

### Figures

1. experimental design;
2. predictive performance over completion time;
3. within-task discrimination and paired model deltas;
4. correct-trajectory recall versus token saving;
5. compute–resolution frontier.

### Tables

1. sample composition and risk sets;
2. observable, hidden, and combined predictive performance;
3. within-task tests and incremental value;
4. adaptive-policy performance;
5. HumanEval+ versus MBPP+ robustness.

### Repository outputs

```text
RESEARCH_PROTOCOL.md
RESULTS.md
configs/
    protocol_v0.1.yaml
scripts/
    build_candidate_bank.py
    extract_prefix_activations.py
    run_grouped_probes.py
    analyze_within_task_signal.py
    calibrate_abort_policy.py
    replay_abort_policy.py
    run_robustness.py
    generate_tcc_artifacts.py
runs/
    manifests/
    candidate_bank/
    activations/
    probes/
    adaptive_policy/
    robustness/
tests/
```

Large generated artifacts need not be committed to Git, but their manifests,
hashes, schemas, configs, and reproduction commands must be versioned.

---

## 18. Execution gates

### Gate 0 — Protocol approval

- [ ] advisor accepts the central question and main scope;
- [ ] pilot rule is accepted;
- [ ] primary and secondary endpoints are accepted;
- [ ] non-inferiority margin and 97% recall constraint are accepted;
- [ ] protocol committed and tagged.

### Gate 1 — Pipeline validity

- [ ] grouped split tests pass;
- [ ] causal-prefix invariance test passes;
- [ ] seed determinism and independent-attempt tests pass;
- [ ] verifier and token accounting are validated.

### Gate 2 — Sampling freeze

- [ ] pilot completed without probe inspection;
- [ ] regime selected by the predeclared rule;
- [ ] exact config and environment manifest committed.

### Gate 3 — Candidate Bank freeze

- [ ] all definitive trajectories generated;
- [ ] completeness and duplicate checks pass;
- [ ] artifact hashes recorded;
- [ ] bank marked read-only/versioned.

### Gate 4 — Confirmatory prediction

- [ ] nested grouped predictions produced;
- [ ] Q1 metrics and risk sets produced;
- [ ] Q2 bootstrap and permutations produced;
- [ ] results recorded before exploratory analysis.

### Gate 5 — Adaptive inference

- [ ] thresholds selected from training/validation only;
- [ ] offline replay validated online on a small subset;
- [ ] H3 non-inferiority and compute tests completed.

### Gate 6 — Robustness and writing

- [ ] one MBPP+ robustness experiment completed or resource limitation
  documented;
- [ ] final figures and tables generated from scripts;
- [ ] thesis claims match the predeclared outcome category;
- [ ] deviations listed in the amendment log.

---

## 19. Reproducibility and provenance

Every definitive run must record:

- Git commit hash and dirty-worktree status;
- protocol version;
- config file hash;
- Python, PyTorch, Transformers, EvalPlus, CUDA, and driver versions;
- model and tokenizer immutable revisions;
- benchmark version/hash;
- GPU model and precision;
- all random seeds;
- start/end timestamps;
- input and output artifact hashes;
- command used to execute the run;
- failures, retries, and exclusions.

Exclusions after generation are allowed only for documented pipeline corruption,
not for poor model outcomes. The unfiltered and filtered counts must both be
reported.

---

## 20. Protocol amendments and exploratory work

Any change after the protocol tag must be appended below before the affected
result is evaluated. Never rewrite the original decision silently.

Required amendment fields:

```text
date:
author:
change:
reason:
made before or after inspecting affected outcomes:
analyses affected:
confirmatory or exploratory status after change:
commit:
```

### Amendment log

### 2026-10-03 — Implementation specifications while protocol is still a draft

date: 2026-10-03  
author: Codex (implementation assistant; approval remains with student/advisor)  
change: Draft implementation fixes the inner grouped CV at three folds, transformer
block-output indexing at `[6, 13, 20, 27]`, `top_k=0`, EOS-inclusive token accounting,
and the treatment of a one-token sampling-support margin as missing with a
training-fitted missingness indicator. The pilot's "material increase" criterion
is operationalized as at most 5 percentage points above the lowest combined
syntax/empty/truncation rate among the three candidate temperatures. Causal
prefix validation uses per-layer relative L2 error at most 0.01 and maximum
absolute error at most `0.02 + 0.01 * max(abs(reference_layer_state))`.
This vector-level criterion replaces elementwise allclose after a bfloat16
preflight exposed GEMM-shape rounding around near-zero coordinates; it is
specified before generating any labeled protocol-pilot trajectory or fitting
probes. Both the full forward and cached generation are compared with the
prefix-only reference, with every layer's errors saved.
The official POSIX EvalPlus checker runs in WSL while generation runs in Windows.  
precision validation: The cached Qwen 1.5B model produced non-finite logits with
float16/eager on the local CUDA environment. bfloat16/eager produced finite
logits at the same memory footprint and is therefore the proposed inference
precision, chosen before new pilot trajectories were generated.  
reason: Make unspecified implementation choices reproducible and executable on
the local 6 GB GPU without weakening the task grouping or confirmatory endpoints.  
made before or after inspecting affected outcomes: Before generating any new
protocol-pilot outcome; earlier legacy exploratory outputs were already present.  
analyses affected: Pilot sampling selection, prefix features, nested grouped
evaluation, verifier provenance and causal-prefix validation.  
confirmatory or exploratory status after change: Proposed specifications for
advisor approval; protocol remains draft, and definitive generation is gated.  
commit: Pending; the assistant has not committed or tagged on behalf of the author.

---

## 21. Approval record

| Role | Name | Decision | Date | Notes |
|---|---|---|---|---|
| Student | João Felipe Vilas Boas | Pending |  |  |
| Advisor | Prof. Dário Oliveira | Pending |  |  |

Once approved, update the status at the top of this document to `frozen`, commit
the file, and create an immutable Git tag such as:

```bash
git tag -a research-protocol-v0.1 -m "Freeze research protocol before definitive experiments"
```

Do not move the tag if the protocol changes. Create a new protocol version and a
new tag.
