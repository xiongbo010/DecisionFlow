# DecisionFlow Experiment Ledger

Last updated: 2026-10-06

This file is the canonical record of completed, partial, superseded, and planned
DecisionFlow experiments. It separates evidence about circuit correctness,
logical consistency, task performance, scalability, and generality. Generated
outputs remain under `pilot/results/` and `experiments/results/generated/`.

## Status vocabulary

- **Complete:** the intended run finished and a result artifact exists.
- **Pilot:** useful evidence with limited data, a development split, an
  independently reconstructed environment, or exploratory rules.
- **Infrastructure only:** adapter/tests exist, while the target benchmark has
  not been run end to end.
- **Superseded:** numerically valid for its original protocol, but a later run
  uses a fairer or better specified comparison.
- **Planned:** discussed and not yet run.

## Current evidence in one paragraph

DecisionFlow has strong evidence for exactness: across Boolean, categorical,
ordinal, soft-constraint, and finite-horizon experiments, its normalization
mass, marginals, and MAP outputs agree with enumeration, analytic formulas, or
independent dynamic programming to numerical precision. It consistently removes
violations when the constraint specification matches the task. Accuracy gains
depend on the dataset and the quality of the local probabilities: ToxicChat and
OpenAI Moderation improve, GoEmotions becomes worse under the tested ontology,
and some SOP/planning experiments are solved mainly because complete rules
supply information missing from the local model. Exact search and task-specific
DP can match DecisionFlow on MAP or even the full finite-state recursion.
DecisionFlow's present advantage is a reusable probabilistic interface providing
valid mass, marginals, conditionals, and MAP, rather than universal superiority
in task success or latency.

## 1. Jev benchmark: five SDD pilots

All five experiments use released Jev 1.13.0 probabilities. The same formulas
were compiled to SDDs and checked sample by sample against enumeration or an
independent analytic oracle.

| Dataset | Size and variables | Constraint | Main result | Exactness/cost | Status |
|---|---|---|---|---|---|
| OpenAI Moderation | 1,680; 8 Noul | Three category implications | 68 raw conflicts removed; exact match 0.6679 to 0.6738 with marginals and 0.6923 with joint MAP; mean Z 0.8975 | SDD agrees with enumeration; published validation artifact | Complete pilot |
| ToxiGen | 940; 1 Noul + 1 Score(5) | Benign implies non-toxic; extremely offensive implies toxic | 4/4 conflicts removed; toxic accuracy 0.8777 to 0.8809; changes are small and bootstrap intervals include zero; mean Z 0.9595 | Max error 3.33e-16; 0.037 ms/sample | Complete pilot |
| GoEmotions | 5,427; 28 Noul | Neutral excludes 27 specific emotions | 635/635 conflicts removed, but marginal Micro-F1 falls 0.2390 to 0.2216 and Brier/NLL worsen; joint MAP exact match rises 0.0321 to 0.0540; mean Z 0.8014 | 49-node SDD represents 134,217,729 valid worlds; 0.172 ms/sample | Complete negative/mixed pilot |
| ToxicChat | 5,083; 2 Noul | Jailbreak implies toxic | 43/43 conflicts removed; Micro-F1 0.7692 to 0.7947; exact match 0.9587 to 0.9636, joint MAP 0.9650; calibration improves; mean Z 0.9652 | Max error 2.22e-16; 0.0039 ms/sample | Complete positive pilot |
| HelpSteer2 | 1,038; 5 Score(5) | Three hard and three learned soft preference rules | Hard rules remove 2/2 conflicts with tiny MAE improvement and worse Brier; soft rules remove 112 marginal-argmax violations and improve Spearman slightly while worsening Brier/NLL; hard mean Z 0.9863 | Hard: 0.283 ms/sample; hard+soft: 0.853 ms/sample; both agree with 3,125-world enumeration | Complete mixed pilot |

Canonical reports:

- `pilot/results/openai_moderation_pilot.md`
- `pilot/results/toxigen_sdd_pilot.md`
- `pilot/results/goemotions_sdd_pilot.md`
- `pilot/results/toxicchat_sdd_pilot.md`
- `pilot/results/helpsteer2_sdd_pilot.md`
- `pilot/results/five_dataset_results_table.tex`

The DecisionFlow experiment package can rerun all five cached-score evaluations
through the public library API. Those integration outputs are in
`experiments/results/generated/full/` and should not be counted as five new
experiments.

### Interpretation retained for the paper

- Constraint satisfaction is guaranteed only relative to the encoded formula
  and represented candidate space.
- Conflict removal does not imply better labels or better calibration.
- GoEmotions is the principal counterexample: a plausible rule can be
  misaligned with multi-annotator gold and degrade prediction quality.
- Soft penalties trade consistency against probability quality and require
  sensitivity analysis.

## 2. LocalLLaMA Typed Decisions

**Complete pilot.** JevAny-Qwen3.5-4B-LoRA predictions were generated for the
full 400-example test set across four five-variable workflows. Frozen
cross-question rules were audited on training data and compiled once per
workflow.

| Workflow | Raw conflicts | After marginals/MAP | Mean Z | Raw accuracy | Marginal accuracy | MAP accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Agent trace observability | 0 | 0 / 0 | 0.9690 | 0.588 | 0.590 | 0.588 |
| Customer service | 9 | 0 / 0 | 0.8714 | 0.672 | 0.668 | 0.666 |
| Invoice processing | 19 | 0 / 0 | 0.8600 | 0.602 | 0.590 | 0.594 |
| Security incidents | 0 | 0 / 0 | 0.8577 | 0.682 | 0.682 | 0.682 |
| Overall | 28 | 0 / 0 | 0.8895 | 0.6360 | 0.6325 | 0.6325 |

All SDD queries agree with exhaustive enumeration. This run demonstrates
cross-type consistency and also shows a small aggregate accuracy decrease.
Canonical report: `pilot/results/typed_decisions_jevany_4b_full.md`.

## 3. SOP-Bench

### 3.1 Dangerous goods, single-step six-head formulation

**Complete pilot.** Four component Score variables, product validity, and final
hazard class are joined by an SDD.

- Qwen3-0.6B, 20 cases: all 20 local tuples conflict; conditioning removes all
  conflicts, but hazard accuracy only rises from 0.15 to 0.20. Mean Z is
  0.000902, exposing weak local potentials.
- Llama-3-8B, all 274 cases: local hazard accuracy 0.3577, conditioned marginal
  accuracy 0.3905, joint-MAP accuracy 0.4781. Joint MAP removes all 158 local
  conflicts; marginal argmax still leaves 49. Mean Z is 0.2997. Brier worsens
  after conditioning.
- SDD inference agrees with enumeration. Exact inference is about 6.96 ms/case
  versus about 601 ms for the six model heads on the 8B MPS run.

Canonical report: `pilot/results/sop_dangerous_goods_probjev.md`.

### 3.2 Dangerous goods, multi-step workflow

**Complete 50-case pilot.** Each step predicts next action, readiness, and final
class.

| Method | Task success | Policy violations | Inconsistent typed tuples |
|---|---:|---:|---:|
| Local Jev | 0% | 100% | 51.0% |
| Action-masked workflow | 4% | 0% | 38.8% |
| Procedural SOP | 100% | 0% | 19.0% |
| DecisionFlow/ProbJev | 100% | 0% | 0% |

The procedural baseline matches final task success because the SOP is complete
and deterministic. DecisionFlow's additional result is a consistent joint at
every step. Circuit compilation plus inference is about 1.16 ms/step versus
329 ms/step for the model.

### 3.3 Referral abuse v2, multi-step workflow

**Complete 50-case pilot.** Six tools and five coupled typed outputs per state.

| Method | Task success | Policy violations | Inconsistent typed tuples |
|---|---:|---:|---:|
| Local Jev | 0% | 100% | 100% |
| Action-masked workflow | 12% | 0% | 95.7% |
| Procedural SOP | 100% | 0% | 81.4% |
| DecisionFlow/ProbJev | 100% | 0% | 0% |

Mean Z is 0.0364. Circuit work is about 1.83 ms/step versus 634 ms/step for the
model. The formalized SOP reproduces all 200 released final decisions before
the 50-case model run.

### 3.4 Traffic spoofing, non-deterministic policy

**Complete 200-case pilot.** The policy preserves two valid actions in medium
risk cases instead of encoding a unique gold output.

- Local action accuracy 69%; conditioned marginals and MAP 71%.
- All five observed policy conflicts are removed.
- Medium-risk action accuracy rises from 51.7% to 56.2%.
- All 19 low-risk cases remain wrong because the local risk head is wrong.
- Mean Z is 0.799. SDD queries agree with enumeration.

This is the strongest SOP example for preserving genuine discretion. Canonical
summary: `pilot/results/sop_cross_task_summary.md`.

## 4. Multi-step planning and control

### 4.1 JevAny control panel

**Complete offline-policy pilot.** An empirical symbolic policy was reconstructed
from 11 archived JevAny attempts (167 decisions); this was not a fresh model run
or a replay of the original 3D simulator.

- Horizon 8: 22 valid trajectories, Z = 0.026614, inference 0.148 ms.
- Raw greedy and one-step constrained argmax loop and fail.
- Receding-horizon DecisionFlow executes a six-action safe sequence and succeeds.
- Z, first-action marginals, MAP, and trajectory count match full enumeration.
- Horizon sensitivity was run for H=1 through 10; feasibility begins at H=6.

Canonical report: `pilot/results/jevany_control_panel_multistep.md`.

### 4.2 AutoPlanBench Logistics

**Complete one-instance pilot.** Instance 0, horizon 5, 26 grounded actions, 45
reachable states, four valid trajectories.

- Qwen3-0.6B/OpenJev: raw greedy immediately proposes an invalid action;
  executable-action greedy loops; DecisionFlow succeeds. Z = 6.28e-10 and exact
  inference is about 0.060 ms.
- Laya typed-decisions: raw greedy is invalid; executable-action greedy loops;
  DecisionFlow succeeds. Z = 1.86e-8 and exact inference is about 0.073 ms.
- Both variants agree with exhaustive trajectory enumeration.

These are single-instance mechanism results, not benchmark-level success rates.

### 4.3 AutoPlanBench Depot

**Complete one-instance pilot.** Instance 2, horizon 12, 996 grounded actions,
115,105 reachable layered states, 11,544 valid trajectories.

- Laya local greedy and constrained beam widths 4, 16, and 64 all fail.
- The official plan is valid and succeeds.
- DecisionFlow succeeds with Z = 1.63e-10; inference takes about 2.04 ms on the
  232 goal-reaching states.
- Z, marginals, and MAP agree with independent backward DP.

The result illustrates delayed feasibility, while it lacks exact A*/DP runtime
baselines and additional instances.

### 4.4 CLM-8B T-Rex

Several distinct protocols were run and must remain separate.

1. **Fixed-snapshot pilot, 15 scenarios (3 seeds x 5):** local greedy survival
   0%; constrained beam widths 1/4/16 and DecisionFlow survival 100%; mean Z
   0.1233; mean DecisionFlow inference 0.028 ms. The official planner-labeled
   prompt agrees with the planner on these snapshots.
2. **Official real-time harness, five 60-second seeds:** without shield, 0/5
   seeds survive and 27 deaths occur; with the official runtime shield, 4/5
   survive, with 4,820 shield interventions. Latency changes the environment,
   so this is not directly comparable to the fixed-snapshot experiment.
3. **Official lockstep-6 harness, five seeds:** 5/5 survive, mean best score
   697, and 598 shield interventions.
4. **DecisionFlow full-course latency-free controller, five seeds:** 5/5
   survive, mean best score 697, 2,597 structured decisions, no infeasible
   decisions, mean Z 0.6637. Its closest comparison is lockstep-6, not the
   real-time run.

These runs validate integration and temporal inference. They do not yet isolate
DecisionFlow's effect under one identical end-to-end timing protocol.

### 4.5 FleetFlow tight-budget stress test

FleetFlow is an independent parameterized reimplementation inspired by the
JevAny Fleet archived replay. It is not the official JevAny benchmark.

- Controlled myopic scorer, 10 instances, B=7: local and one-step shield 0%;
  beam width 16 and DecisionFlow 100%; mean Z 0.0599.
- JevAny-Qwen3.5-4B, 10 instances, B=7: local, shield, and beam through width 16
  are 0%; DecisionFlow 100%; mean Z 0.0146.
- CLM-8B, 10 instances, B=7: local, shield, and beam through width 256 are 0%;
  DecisionFlow 100%; mean Z 3.95e-13. The tiny Z shows that constraints, rather
  than the local model, supply most of the solution.

#### Fair budget sweep

**Complete three-instance pilot** for B in {7, 9, 12, 16, 32}. One budget-blind
CLM probability table and one transition graph are shared by all methods;
model scoring and graph construction are excluded from inference time.

| B | Greedy/shield/beam<=256 | Exact best-first | Backward DP | DecisionFlow | Mean Z | A* / DP / DF ms |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0% | 100% | 100% | 100% | 3.16e-13 | 3.87 / 0.68 / 0.16 |
| 9 | 0% | 100% | 100% | 100% | 7.51e-12 | 5.83 / 1.30 / 1.54 |
| 12 | 0% | 100% | 100% | 100% | 5.62e-11 | 9.11 / 2.67 / 5.94 |
| 16 | 0% | 100% | 100% | 100% | 2.92e-10 | 14.68 / 4.91 / 9.21 |
| 32 | 0% | 100% | 100% | 100% | 7.81e-9 | 29.75 / 14.83 / 32.72 |

Task-specific DP is fastest at larger budgets. Best-first matches the MAP.
DecisionFlow provides the complete query set through the reusable trajectory
API. Figure: `experiments/results/generated/fleetflow-clm8b-budget-curve-fair.svg`.

The earlier files named `fleetflow-clm8b-budget-B*.json` and the figure without
`fair` are **superseded** for method comparison because they did not include the
shared full probability table, exact best-first, or DP as peer baselines.

## 5. Dataset and adapter investigations

### 5.1 GLiNER2.5 / Fast Decisions

**Exploratory development pilot; not a formal benchmark result.** Eight public
development domains with 100 rows each were audited for candidate rules.

- Only four domains retained any zero-gold-violation rule.
- Several retained rules are weak (for example, at least one label), and the
  agent-handoff auxiliary label is deterministically derived from the target.
- Independent decoding already had zero conflicts; DecisionFlow did not change
  accuracy or joint exact match.
- The official test split is not public, and rules were selected using the full
  development gold.

Conclusion: Fast Decisions is not a useful primary evaluation of the claimed
gap. Keep the run as a documented negative dataset-suitability result.

### 5.2 Tau2 Retail

**Infrastructure only.** A `HalfDuplexAgent` adapter, finite candidate builder,
authentication/confirmation/handoff memory, local/procedural/circuit ablations,
and circuit diagnostics are implemented inside the local Tau2 checkout. Unit
tests cover authentication, confirmation, write authorization, entity memory,
and exact circuit queries. A two-turn local smoke script exists.

No complete Tau2 task suite with the user simulator has been run, and no valid
benchmark success rate should be reported. Relevant files:

- `third_party/tau2-bench/src/tau2/agent/probjev_retail_agent.py`
- `third_party/tau2-bench/tests/test_probjev_retail_agent.py`
- `openjev_local/tau2_retail_smoke.py`

### 5.3 Jev/JevAny dataset audits

- The 37-dataset Jev suitability audit exists at
  `pilot/results/jev37_dataset_suitability_audit.md`.
- The JevAny case library was inspected. Most gallery items are archived replay
  media, not downloadable counterfactual environments.
- The Fleet gallery uses a 27B SFT checkpoint, recovery after partial dispatch,
  and a 32-operation budget. FleetFlow uses an independent environment and must
  retain the stress-test label.

## 6. What has not been run

### Priority A: needed for the current paper claim

1. **General scaling and crossover experiment.** Vary variable count, domain
   size, constraint density, and structural width over chains, trees, grids,
   cardinality constraints, and cross-variable implications. Compare exhaustive
   enumeration, closed-form/task-specific DP, exact best-first, SAT/ILP/MaxSAT,
   direct WMC, reusable compiled circuits, rejection sampling, importance
   sampling, and MCMC. Report grounding time, compilation time, circuit size,
   per-query time, memory, query count, and the amortization crossover.
2. **Independent versus directed or learned base joint.** On HelpSteer2,
   ToxiGen, Typed Decisions, and a multi-step task, compare the product of local
   heads with directed conditional scoring and a learned correlation model.
   Measure joint NLL, marginal calibration, ordering sensitivity, Z, and task
   quality. This directly addresses the surrogate-joint limitation.
3. **Z risk-coverage study across datasets and models.** Test whether low Z
   predicts errors, interventions, or ontology mismatch. Precommit to treating Z
   only as a consistency statistic if it lacks predictive value. Typed Decisions
   contains preliminary risk-coverage code, but no consolidated cross-dataset
   result has been run.
4. **Stronger model sweep.** Run at least three local-model strengths with the
   same prompts and constraints. Plot raw violation rate, correction rate,
   accuracy, calibration, and Z as model strength increases. The expected
   possibility that gains vanish for strong models must be reported.
5. **Constraint provenance and leakage audit.** For every main result, record
   rule source, derivation split, freeze time, gold support, counterexamples, and
   whether the rule is normative, empirical, or exploratory. Apply multiple
   comparison correction to predeclared primary metrics.

### Priority B: needed for broader generality

6. **Real DL ontology front end.** Ground an actual TBox/ABox or OWL/DL fragment
   into the common constraint IR, measure reasoner/grounding cost, compilation,
   and queries. Current experiments use manually formalized propositional,
   ordinal, SOP, or transition constraints; none validates a general DL
   compilation pipeline.
7. **Official or external multi-step benchmark at scale.** Run multiple
   AutoPlanBench Logistics and Depot instances with exact-search and DP
   baselines. A full official JevAny Fleet reproduction requires the released
   27B checkpoint or a fresh compatible scorer, the official 32-step recovery
   semantics, and counterfactual state access that the gallery does not provide.
8. **Tau2 Retail end-to-end evaluation.** Run the complete user simulator and
   compare the stock agent, local typed decisions, procedural policy masking,
   runtime shield, and DecisionFlow under identical APIs, tasks, and model
   budgets. Report policy violations separately from reward and task success.
9. **Sampling baselines in the low-Z regime.** Empirically plot Z against
   rejection acceptance, effective sample size, marginal error, MAP recovery,
   and wall time. FleetFlow predicts an extreme disadvantage for rejection
   sampling, but this has not been measured.
10. **Candidate recall and dynamic grounding.** Vary retriever recall and
    candidate-set size; quantify cases where the correct action has zero support.
    Measure recompilation rate when entities or relational structure change.

### Priority C: extensions discussed but not evaluated

11. **Structured training.** Fine-tune or regularize the typed decision model
    with semantic loss, circuit likelihood, constrained distillation, or
    structured NLL. Compare no-training post-hoc conditioning, constraint-aware
    fine-tuning, and a native structured model.
12. **Global trajectory normalization and label-bias controls.** Compare local
    conditional products with globally normalized trajectory potentials,
    explicit Stop actions, and length normalization on variable-length tasks.
13. **Chance variables and control-as-inference.** Separate controllable action
    constraints from stochastic environment outcomes and evaluate expectimax to
    avoid conditioning the environment into favorable outcomes.
14. **Multi-agent DecisionFlow.** Add agent identity, communication actions,
    shared resources, ownership constraints, and joint/team marginals. No
    multi-agent benchmark has been implemented.
15. **BPI 2019/process-mining evaluation.** Discussed as a possible source of
    business-process constraints; no adapter or run exists.
16. **Robustness to wrong or incomplete rules.** Inject missing rules,
    over-restrictive rules, exceptions, and noisy evidence; measure Z=0 rate,
    abstention, softening, and accuracy degradation.

## 7. Recommended next execution order

1. Run the synthetic scaling/crossover matrix with exact best-first, DP, direct
   WMC, compiled DecisionFlow, and sampling baselines.
2. Consolidate Z risk-coverage across the five Jev datasets, Typed Decisions,
   SOP-Bench, and FleetFlow.
3. Run independent versus directed/learned joint ablations using existing
   cached data.
4. Expand AutoPlanBench to multiple Logistics and Depot instances under the
   shared-probability fair protocol.
5. Run Tau2 Retail end to end only after freezing the task subset, simulator,
   model, and policy-information parity across baselines.
6. Add a real DL grounding experiment if ontology remains part of the paper's
   central claim; otherwise present DL as one possible constraint source.

## 8. Maintenance rules

- Add every new run here when its result file is created.
- Never replace a result silently; mark the earlier protocol as superseded.
- Distinguish model scoring, grounding, compilation, and query time.
- State whether all methods receive the same probabilities, candidates,
  transition model, constraints, and budget.
- Record whether rules were frozen before evaluating the reported split.
- Label reconstructed environments, archived-policy replays, development-set
  pilots, and official benchmark runs separately.
- Treat 100% constrained success as a feasibility guarantee when complete rules
  determine the answer; do not present it alone as evidence of model reasoning.
