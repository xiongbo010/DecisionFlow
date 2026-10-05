# OpenAI Moderation pilot

## Typed Decisions multi-workflow pilot

`typed_decisions_sdd_pilot.py` evaluates five heterogeneous typed variables in
each of the four `LocalLLaMA/typed-decisions` workflows.  Frozen cross-question
rules are compiled into one reusable SDD per workflow.  The script verifies
valid mass, every typed marginal, and joint MAP against complete enumeration.

First run the soft-gold diagnostic.  This audits the rules and circuit; it is
not a model result:

```bash
PYTHONPATH=pilot .venv-openjev/bin/python pilot/typed_decisions_sdd_pilot.py
```

Generate resumable JevAny predictions and rerun the pilot with those local
distributions:

```bash
PYTHONPATH=pilot .venv-openjev/bin/python \
  pilot/typed_decisions_generate_jevany.py \
  --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA \
  --device mps --dtype fp16

PYTHONPATH=pilot .venv-openjev/bin/python \
  pilot/typed_decisions_sdd_pilot.py \
  --predictions pilot/results/typed_decisions_jevany_4b_predictions.jsonl
```

The primary model evaluation reports per-workflow local conflicts, conflicts
remaining after marginal decoding, zero-conflict joint MAP, consistency mass
`Z`, accuracy, KL from soft gold, Brier score, and circuit cost.  Rules are
audited on the training split before test metrics are read.

This pilot tests ontology conditioning on the OpenAI Moderation portion of the public Jev benchmark without training or modifying Jev.

## Inputs

- Dataset: `mmathys/openai-moderation-api-evaluation`, pinned by the benchmark to its public evaluation split.
- Model responses: released `jev-1.13.0` response cache from Zenodo record `10.5281/zenodo.23039006`.
- Questions and label handling: reconstructed from `AppliedMachineLearning-Lab/jev-benchmarking`.

The Jev response data are used only for research and evaluation under the Jev Responses License included in the benchmark repository. They are not used for training or distillation.

## Ontology

The pilot uses three category implications:

- `sexual/minors -> sexual`
- `hate/threatening -> hate`
- `violence/graphic -> violence`

All three have zero violations in the available gold annotations. The plausible rule `hate/threatening -> violence` is excluded because the available gold labels contain four counterexamples among 41 known positive antecedents.

## Inference

The eight Noul variables define 256 Boolean worlds. The script enumerates them exactly, removes worlds that violate an implication, and computes:

- valid mass `Z`;
- ontology-conditioned marginals;
- constrained joint MAP.

Enumeration serves as a transparent correctness oracle for the pilot. `validate_sdd.py` compiles the same formula into an SDD. Weighted model counting computes `Z` and all eight marginals; max-product traversal over the deterministic, decomposable circuit computes joint MAP. The validation compares every quantity with enumeration for every sample.

## Reproduction

The script requires Python and NumPy:

```bash
python pilot/openai_moderation_pilot.py \
  --data tmp/jev-benchmarking/samples-1680.jsonl.gz \
  --responses tmp/jev-benchmarking/cache-responses.db \
  --json-out pilot/results/openai_moderation_pilot.json \
  --report-out pilot/results/openai_moderation_pilot.md
```

The generated Markdown report contains the headline metrics and paired bootstrap intervals. The JSON file contains the complete metric set, rule audit, valid-mass quantiles, and corrections by category.

Install the circuit dependency and reproduce the SDD validation with:

```bash
python -m pip install -r pilot/requirements-sdd.txt
python pilot/validate_sdd.py \
  --data tmp/jev-benchmarking/samples-1680.jsonl.gz \
  --responses tmp/jev-benchmarking/cache-responses.db \
  --artifact-dir pilot/artifacts \
  --json-out pilot/results/openai_moderation_sdd_validation.json \
  --report-out pilot/results/openai_moderation_sdd_validation.md
```

This command writes the compiled `.sdd` and `.vtree` files, reloads them as a serialization check, and fails immediately if any sample exceeds the stated numerical tolerances or returns a different MAP assignment.

## ToxiGen mixed-type pilot

`toxigen_sdd_pilot.py` evaluates one Noul (`toxic`) jointly with one five-level Score (`toxicity`). The primary ontology uses two high-precision endpoint constraints:

- `Benign -> not Toxic`
- `Extremely offensive -> Toxic`

The script audits these rules against the gold data, represents Score with exactly-one indicators and categorical literal weights, compiles the formula to an SDD, and validates `Z`, all six marginals, and joint MAP against direct enumeration for every sample. It also runs the stronger `Toxic iff Score is non-Benign` formulation as a sensitivity analysis because that rule conflicts with the benchmark's operational binary-label construction.

```bash
python pilot/toxigen_sdd_pilot.py \
  --data tmp/toxigen-test.parquet \
  --responses tmp/jev-benchmarking/cache-responses.db \
  --artifact-dir pilot/artifacts \
  --json-out pilot/results/toxigen_sdd_pilot.json \
  --report-out pilot/results/toxigen_sdd_pilot.md
```

The released Jev Score probabilities are serialized to two decimals. Six of the 940 distributions sum to `0.99`; the script records and renormalizes these rows before constructing the categorical factor.

## GoEmotions ontology pilot

`goemotions_sdd_pilot.py` evaluates all 28 Noul heads jointly under `Neutral -> not Emotion` for all 27 specific emotions. Because the released labels aggregate multiple annotators and can contain both neutral and a specific emotion, the primary evaluation first canonicalizes each gold set by removing neutral whenever a specific emotion is present. This minimum-deletion repair is fixed by the ontology and independent of model outputs. The untouched aggregated labels are retained as a sensitivity analysis. An independent analytic oracle validates SDD `Z`, all leaf marginals, ontology-derived positive/negative/ambiguous parent marginals, and joint MAP.

```bash
python pilot/goemotions_sdd_pilot.py \
  --eval-data tmp/go-emotions-test.parquet \
  --dev-data tmp/go-emotions-validation.parquet \
  --responses tmp/jev-benchmarking/cache-responses.db \
  --artifact-dir pilot/artifacts \
  --json-out pilot/results/goemotions_sdd_pilot.json \
  --report-out pilot/results/goemotions_sdd_pilot.md
```

The report contrasts independently thresholded marginals with joint MAP. Marginal thresholding can sacrifice joint decision quality even when every world in the conditioned distribution is valid; joint MAP directly returns an ontology-consistent label set.

## ToxicChat hard-implication pilot

`toxicchat_sdd_pilot.py` evaluates the two native Noul heads (`toxic` and `jailbreak`) under the hard ontology rule:

- `Jailbreak -> Toxic`

The rule has zero violations among jailbreak-positive examples in both the public train and test splits. The script reconstructs all 5,083 test requests, reads the released Jev responses, compiles the implication to an SDD, and validates valid mass, both marginals, and joint MAP against direct enumeration for every sample. It reports per-head and joint metrics, calibration, conflict elimination, valid-mass diagnostics, and paired bootstrap intervals.

```bash
python pilot/toxicchat_sdd_pilot.py \
  --eval-data tmp/toxicchat-test.parquet \
  --train-data tmp/toxicchat-train.parquet \
  --responses tmp/jev-benchmarking/cache-responses.db \
  --artifact-dir pilot/artifacts \
  --json-out pilot/results/toxicchat_sdd_pilot.json \
  --report-out pilot/results/toxicchat_sdd_pilot.md
```

The primary result eliminates all 43 inconsistent local decisions. Conditioned marginals improve Micro-F1 from 0.7692 to 0.7947 and exact match from 0.9587 to 0.9636; constrained joint MAP reaches 0.9650 exact match.

## HelpSteer2 hard/soft Score pilot

`helpsteer2_sdd_pilot.py` constructs a joint over five Score(5) variables: Helpfulness, Correctness, Coherence, Complexity, and Verbosity. The hard circuit uses three conservative admissibility rules with zero violations in both the public train and validation splits. The hard+soft circuit adds three stronger preference rules through functionally determined violation indicators. Each violation literal receives a penalty estimated exclusively from the training annotations using Laplace-smoothed violation odds.

```bash
python pilot/helpsteer2_sdd_pilot.py \
  --eval-data tmp/helpsteer2-validation.parquet \
  --train-data tmp/helpsteer2-train.parquet \
  --responses tmp/jev-benchmarking/cache-responses.db \
  --artifact-dir pilot/artifacts \
  --json-out pilot/results/helpsteer2_sdd_pilot.json \
  --report-out pilot/results/helpsteer2_sdd_pilot.md
```

Both SDDs validate against enumeration over all 3,125 typed assignments on all 1,038 evaluation examples. The hard circuit uses 25 Boolean indicators; the soft circuit adds three violation indicators and retains the same 2,575 admissible typed assignments. The experiment reports ranking, calibration, argmax, joint-MAP, rule-conflict, and bootstrap results separately because stronger soft penalties can improve rank correlation while worsening calibrated probability scores.

## Important evaluation caveat

The source JSON omits some category keys when their labels are unknown. The published benchmark loader treats `row[code] == 1` as positive and all other values as non-positive. The main table reproduces that policy for direct comparability. The report also evaluates explicitly known labels separately.

## SOP-Bench dangerous-goods pilot

`sop_dangerous_goods_probjev.py` converts one industrial SOP into six OpenJev
questions: product-ID validity, four ordered component scores, and final hazard
class. A reusable SDD encodes exact-one typed domains, ID validity evidence,
single-missing-score imputation, the two-missing-score failure rule, and the
mapping from the completed score to the final class.

The public `sop.txt` orders classes A through D but omits their numeric cutoffs.
The pilot records the cutoff completion separately from explicit SOP rules and
audits it against all 274 released rows before inference. The released rows do
not contain totals 13 or 14.

```bash
PYTHONPATH=pilot .venv-openjev/bin/pytest -q \
  pilot/test_sop_dangerous_goods_probjev.py

.venv-openjev/bin/python pilot/sop_dangerous_goods_probjev.py \
  --limit 20 --device cpu
```

The exactness test compares SDD valid mass, every typed marginal, and joint MAP
with exhaustive enumeration over all 1,296 component-score combinations. The
pilot writes the circuit to `pilot/artifacts/` and complete per-example results
to `pilot/results/sop_dangerous_goods_probjev.json`.

### Multi-step SOP policy pilot

`sop_dangerous_goods_multistep.py` turns the same SOP into a tool-use process.
For a valid product ID, the agent must call each of four scoring tools exactly
once and may finalize only after all results are available. An invalid ID must
finalize immediately as `Unable to Decide`. At every step, OpenJev scores a
next-action Choice, an evidence-sufficient Noul, and a hazard-class Choice.

The experiment separates four systems:

1. `local`: independent local argmax decisions;
2. `masked`: a workflow mask restricts the next action, while OpenJev predicts
   the final class;
3. `procedural`: the workflow mask plus the deterministic SOP class formula;
4. `probjev`: one conditioned joint over action, readiness, and class at every
   step, with exact marginals, valid mass, and joint MAP.

```bash
PYTHONPATH=pilot .venv-openjev/bin/pytest -q \
  pilot/test_sop_dangerous_goods_multistep.py

HF_HUB_OFFLINE=1 .venv-openjev/bin/python \
  pilot/sop_dangerous_goods_multistep.py \
  --limit 10 --device mps --dtype float16 \
  --model /path/to/Meta-Llama-3-8B-Instruct
```

On the stored 50-task run, local argmax violates the workflow on every task.
Action masking removes those violations but succeeds on only 4% because the
model usually fails to apply the final arithmetic rule. Both the procedural
baseline and ProbJev reach 100% task success. The procedural baseline leaves
19.0% of the separately scored intermediate typed tuples inconsistent; ProbJev
returns a consistent joint tuple at every step. This result establishes the
value of joint consistency, while also showing that a deterministic workflow
is an equally strong task-level solution when all rules are complete and
hand-coded. See `pilot/results/sop_dangerous_goods_multistep.md`.

## SOP-Bench referral-abuse v2 pilot

`sop_referral_abuse_v2_multistep.py` implements a seven-step investigation with
six tools and five typed outputs per state: action, evidence sufficiency,
violation type, risk severity, and enforcement action. Its grounded circuit
represents the score thresholds, tie-breaking policy, risk hierarchy, and
cross-type enforcement matrix. The formalized public SOP reproduces all 200
released gold decisions before any model evaluation.

In the 50-case Llama-3-8B run, local Jev succeeds on 0%, an action-only workflow
mask succeeds on 12%, and both the procedural SOP and ProbJev succeed on 100%.
ProbJev is the only system whose full typed tuple is consistent at every step.
The circuit takes about 1.83 ms per step, compared with about 634 ms for the
model. Full traces are in
`pilot/results/sop_referral_abuse_v2_multistep_llama3_8b_50.json`.

## SOP-Bench traffic-spoofing pilot

`sop_traffic_spoofing_probjev.py` tests a non-deterministic policy. High risk
requires account closure, low risk without a confirmed violation requires no
action, and medium risk admits either temporary suspension or a warning. The
compiled circuit therefore preserves genuine decision discretion instead of
encoding a unique gold output.

Across all 200 released cases, conditioning improves action accuracy from 69%
to 71% and removes all five inconsistent local tuples. Medium-risk action
accuracy rises from 51.7% to 56.2%. Low-risk accuracy remains zero because the
local risk head misclassifies these examples; the constraint layer cannot
recover evidence absent from its input probabilities. See
`pilot/results/sop_traffic_spoofing_probjev.md` for the complete interpretation.

## JevAny control-panel multi-step Choice pilot

`jevany_control_panel_multistep.py` reconstructs an empirical Markov policy
from the 11 archived JevAny control-panel attempts (167 decisions). The task has
explicit temporal preconditions: isolate power before bleeding, bleed before
clamping, restore power before pumping, and accept only a clamped fixture at
20--30 kPa. A layered sum-product/max-product circuit conditions action
sequences on reaching acceptance without violating these rules.

```bash
python3 pilot/jevany_control_panel_multistep.py

PYTHONPATH=pilot .venv-openjev/bin/python -m pytest -q \
  pilot/test_jevany_control_panel_multistep.py
```

At horizon eight, the circuit represents 22 valid trajectories with valid mass
0.026614. It agrees exactly with exhaustive enumeration for valid mass, all
first-action marginals, trajectory MAP, and trajectory count. Raw Jev greedy
and a one-step constraint mask repeatedly toggle power and fail within 12
steps. Receding-horizon inference selects the six-action safe sequence and
reaches acceptance. Full results are in
`pilot/results/jevany_control_panel_multistep.md`.
