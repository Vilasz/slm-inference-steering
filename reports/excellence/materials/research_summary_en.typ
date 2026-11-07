#set page(paper: "a4", margin: 1.9cm, numbering: "1")
#set text(font: "Times New Roman", size: 10.4pt, lang: "en")
#set par(justify: true, leading: 0.45em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 15pt)
#show heading.where(level: 2): set text(size: 11.5pt)
= Early correctness signals for efficient code generation
*João Felipe Vilas Boas — FGV Escola de Matemática Aplicada*\
Undergraduate research summary · October 2026

== Research question and contribution
Can early hidden states distinguish successful and unsuccessful code-generation
attempts for the same problem, improve on observable decoding statistics, and
support economical interruption? These are separate claims. A predictor that
recognizes easy prompts may have high pooled AUC while failing to distinguish
attempts within a prompt. We therefore separate task difficulty, within-task
discrimination, probabilistic prediction, and sequential operational value.

The contribution is a bounded, auditable empirical study and an explanatory
analysis of a negative result. It does not claim to invent hidden-state probing,
PCA, early abort, or geometric stopping-cost identities. The study examines when
readout capacity and threshold transport obstruct practical use of predictive signals.

== Design and validation
Qwen2.5-Coder-1.5B-Instruct generates stochastic Python candidates using immutable
weights/tokenizer, temperature 1.0 selected by a separate 960-trajectory pilot,
top-p 0.95 and a 512-token limit. The source bank contains 2,640 attempts across
132 HumanEval+ tasks; the frozen transfer bank contains 640 attempts across 32
MBPP+ tasks. Official EvalPlus 0.3.1 base and plus tests define functional success.
The complete bank retains twenty attempts per task, including after early successes.

Five outer task-grouped folds evaluate held-out tasks. Three inner grouped folds
select layer and L2 regularization using task-balanced log loss. Calibration
scores cross-fit the entire selection procedure. At the fixed 32-token endpoint,
eligibility is strictly T>32: 2,158 candidates remain, across 132 tasks, of which
92 contain both outcomes. Within-task AUC gives each mixed task equal weight;
proper losses retain homogeneous tasks. Paired task bootstrap intervals condition
on fitted scorers and do not capture all retraining uncertainty.

== Original results
#table(columns: (1fr, 1fr, 1fr, 1fr), inset: 4pt,
 table.header([*Predictor*], [*Pooled AUC*], [*Within-task AUC*], [*Log loss*]),
 [Observables B], [0.701], [0.628], [0.630],
 [Hidden H], [0.592], [0.568], [0.687],
 [H+B], [0.646], [0.581], [0.668])

The H+B minus B within-task AUC difference is −0.0470, 95% interval
[−0.0994, 0.0056]. The log-loss difference is +0.0379 [0.0067, 0.0706].
The original incremental and joint operational criteria are not met. This
delimits the tested procedure; it does not establish that hidden states contain
no correctness information. Independent pair counting and reconstruction of all
three primary scorers reproduce the archived probabilities to numerical precision.

#pagebreak()
== Explanatory follow-up: capacity, decisions, and economics
Diagnostics on the original data were defined after its results and remain
exploratory. Observable ablations retain grouped internal selection. Mean NLL
and margin alone preserve much of the within-task ranking, but their log loss
is worse than the full observable baseline. PCA fitted exclusively within
training partitions and an anchored residual readout reduce the original
combined-model deficit. PCA64 achieves within-task AUC 0.630 and log loss 0.632;
the residual achieves 0.635 and 0.629. Their paired intervals against B include
zero. Stronger joint regularization alone does not resolve the deficit.

The residual uses logit(p_B) as a fixed offset and an L2-penalized hidden-state
correction. Training offsets exclude their own labels from first-stage fitting,
selection, and calibration; this separation is repeated inside residual selection.
These controls support sensitivity to parametrization without isolating dimensionality
as a unique cause or proving complete signal redundancy.

Frozen threshold transfer exposes a different failure mode. H+B retains only
276/366 active correct MBPP+ candidates: 75.41% [61.87%, 87.74%], despite achieving
the 97% empirical development target. Ninety correct candidates are discarded
across twelve tasks; five tasks account for sixty discards. Two tasks lose all
their available successes. Rates among all potential candidates and initiated
candidates differ because earlier decisions determine which later attempts start.
Source OOF, source gate development, and full-source-fitted target scoring are
reported separately; the comparison does not identify a pure causal domain effect.

An idealized label-aware oracle aborts incorrect active candidates and preserves
every correct one on the same ordered stream. At eight tokens and N=20, its
token savings against verifier stopping are 77.75% on HumanEval+ and 76.25% on
MBPP+, with identical observed resolution. This measures conditional economic
headroom, not an implementable predictor or a universal lower bound for lossy gates.
With iid attempts conditional on a task, mean attempt cost d, success probability
p and global correct retention r, expected cost is d times the truncated geometric
sum with ratio 1−pr. For an unlimited budget and positive p,r, lower expected cost
requires $d/E[T]<r$. Within-attempt cost and success may depend. Task heterogeneity,
finite budgets, coverage losses and gate overhead limit direct use of pooled formulas.

== Limitations, independent validation, and research skills
The evidence concerns one small model and two function-level benchmarks. Twenty-six
source tasks appeared in earlier local artifacts; researcher novelty and absence
of pretraining contamination are not claimed. Execution was prospectively authorized
by the student; the historical record contains no advisor approval. The extension
records student approval and student-reported advisor review, without independent preregistration. New analyses are post-hoc and their intervals are pointwise. Passing
tests is operational correctness, not formal verification; the checker is not a
complete security sandbox. Token savings are not latency, energy, or FLOPs savings.

A subsequent prospectively approved study completed twenty disjoint technical
tasks and 120 locally unexposed MBPP+ tasks, with twenty attempts per task.
The sole endpoint compares source-frozen B and H+B gates at the same eight-token
checkpoint: H+B minus B correct retention is -0.1636 [-0.2402, -0.0943]. This
meets the criterion of an upper two-sided 95% task-bootstrap bound below zero.
The protocol, IDs and sources were frozen before new trajectories. The student
explicitly approved and reported advisor review. No target fitting occurred;
this endpoint does not revise the original incremental or operational decisions.

The project demonstrates task-grouped experimental design, nested model selection,
strict cross-fitting, PyTorch instrumentation, verifier integration, paired
statistical estimation, stopping-cost analysis, and reproducible software with
hashes, regression tests and auditable artifacts. It provides concrete methods
and limitations that can be examined in a research interview and thesis defense.

#set text(size: 8.3pt)
Selected context: Hewitt & Liang (2019), probing controls;
Guo et al. (2017), calibration; Ovadia et al. (2019), uncertainty under shift;
Ruan et al. (2026), independently certified early abort in agent episodes.
Full references, source metadata, data provenance and commands accompany the thesis.
