# DecisionFlow experiments

The living record of completed, partial, superseded, and planned evaluations is
maintained in [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md).

This directory is the research and paper-reproduction layer. It is deliberately
separate from the installable `decisionflow` library:

- `src/decisionflow/` implements reusable typed decisions, constraints,
  probabilistic inference, and trajectory APIs;
- this directory owns dataset acquisition, label transformations, prompts,
  model checkpoints, cached scores, metrics, statistical tests, and tables.

Experiment code may import the public `decisionflow` API. The library never
imports this package and contains no dataset names or paper-result commands.

## Reproduction levels

1. **Core verification** runs the library test suite and needs no benchmark
   data or model.
2. **Cached-score evaluation** reads standardized JSONL records containing
   model probabilities and gold annotations. It deterministically reruns
   DecisionFlow inference without a GPU or model API.
3. **Score regeneration** is dataset- and model-specific. It recreates the
   standardized records from pinned raw data and model revisions.

The cached-score path is the canonical paper reproduction path. Score
regeneration records model revision, prompt hash, software environment, and
hardware because those runs can vary across inference stacks.

## Install

From this directory:

```bash
python -m pip install -e '..[sdd]'
python -m pip install -e '.[data]'
```

Validate the experiment registry and its DecisionFlow programs:

```bash
decisionflow-experiments verify
```

Evaluate standardized cached scores:

```bash
decisionflow-experiments run \
  --experiment toxicchat \
  --input data/cached/toxicchat.jsonl \
  --output results/generated/toxicchat.jsonl

decisionflow-experiments summarize \
  --input results/generated/toxicchat.jsonl \
  --output results/generated/toxicchat-summary.json
```

Grounded multi-step records use a separate experimental command while still
calling the reusable library trajectory API:

```bash
decisionflow-experiments run-trajectories \
  --input data/cached/multistep.jsonl \
  --output results/generated/multistep.jsonl
```

This command belongs to the experiment package, not the DecisionFlow CLI.

## FleetFlow

`FleetFlow` is a parameterized, executable reimplementation inspired by the
archived Fleet Dispatch case in JevAny. It models vehicle assignment, cold-chain
preparation, charging, route clearance, operation budgets, dispatch, and final
verification. The experiment compares local argmax, a one-step runtime shield,
constrained beam search, and exact DecisionFlow inference under identical local
probabilities and transition rules.

Run the deterministic mechanism check:

```bash
decisionflow-experiments fleetflow \
  --scorer myopic \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --output experiments/results/generated/fleetflow-myopic.json
```

For a real JevAny model, start any compatible `/v1/systemone` server and use:

```bash
decisionflow-experiments fleetflow \
  --scorer jevany \
  --base-url http://127.0.0.1:8008 \
  --model jevany-latest \
  --cache-dir experiments/artifacts/fleetflow-jevany \
  --output experiments/results/generated/fleetflow-jevany.json
```

The released CLM-8B encoder and projection head can be evaluated through the
same FleetFlow protocol on Apple Silicon:

```bash
PYTHONPATH=src:experiments/src \
.venv-openjev/bin/python -m decisionflow_experiments.fleetflow \
  --seeds 0 1 2 3 4 5 6 7 8 9 --horizon 7 --scorer clm \
  --clm-cache experiments/results/generated/cache/fleetflow-clm8b.json \
  --output experiments/results/generated/fleetflow-clm8b-10.json
```

The CLM adapter uses neutral `Choice` prompts and the same candidate action
descriptions supplied to every comparison method. Model scoring remains in the
experiment package; DecisionFlow-Core only receives the resulting local
probabilities and finite transition graph.

For inference comparisons, FleetFlow first materializes one shared probability
table over every reachable nonterminal state. Local argmax, the one-step shield,
constrained beam search, exact best-first search, task-specific backward dynamic
programming, and DecisionFlow then read the same scores and transition graph.
Model scoring and graph construction are reported separately from inference
time. The budget-sweep figure can be regenerated with
`experiments/scripts/plot_fleetflow_budget.py`.

The JevAny gallery publishes archived replays instead of the original
counterfactual simulator. Each result therefore records both the source of the
task motif and the independent environment implementation.

## Standard cached-score record

Each JSONL row has four top-level fields:

```json
{
  "id": "example-1",
  "state": {"text": "..."},
  "probabilities": {
    "toxic": {"false": 0.8, "true": 0.2},
    "jailbreak": {"false": 0.3, "true": 0.7}
  },
  "gold": {"toxic": true, "jailbreak": true},
  "provenance": {"model": "...", "revision": "...", "prompt_sha256": "..."}
}
```

Raw text can be omitted from redistributable score caches when licensing or
privacy requires it. The `id`, probabilities, gold values, and provenance are
sufficient for deterministic inference and metric recomputation.

## Experiment coverage

The registry currently defines the decision schemas and constraint packs used
for OpenAI Moderation, ToxiGen, GoEmotions, ToxicChat, and HelpSteer2. Typed
Decisions and SOP-Bench use per-example finite relations and therefore provide
their grounded request and constraint pack directly in each cached record.
Multi-step legacy experiments provide a finite trajectory object accepted by
the internal `decisionflow.trajectory.TrajectoryEngine` baseline.

Legacy exploratory scripts and locally downloaded data remain in the ignored
`pilot/` artifacts and data directories. The pilot Python sources are tracked
as the reference definitions of data loading, label handling, full metrics,
bootstrap intervals, and audits. The `full` runner replaces their inference
functions with DecisionFlow internal reference-engine calls and writes complete
JSON results. These adapters are being migrated to public declarative workflows:

```bash
decisionflow-experiments full --experiment all --repo-root .
```

Generated JSON files are written under
`experiments/results/generated/full/`. LaTeX table generation is intentionally
outside this pipeline.

## GLiNER2.5 / Fast Decisions constraint pilot

Fast Decisions publishes typed labels but no machine-readable constraint pack.
The GLiNER2.5 pilot therefore keeps a small, reviewable registry of semantic
candidate rules, audits every rule against the public development gold, and
uses only zero-violation rules for inference. Rejected candidates and their
counterexample counts remain in the JSON report.

```bash
python -m pip install -e '.[gliner25]'
decisionflow-experiments gliner25-fast-decisions \
  --dataset all \
  --output experiments/results/generated/gliner25-fast-decisions.json
```

The three conditions share one GLiNER2.5 score tensor per example: independent
decoding, GLiNER2.5 exact constrained MAP, and DecisionFlow exact conditioning.
For multi-label heads, DecisionFlow represents each label as a Boolean variable.
The report includes accuracy, joint exact match, violations, NLL, valid mass,
MAP agreement, and separate scoring/decoding/inference times. The official
300-example-per-domain test split is not public, so this remains a development
pilot rather than a reproduction of the model-card benchmark.

## AutoPlanBench Logistics pilot

The Logistics pilot parses the released STRIPS domain and problem, grounds the
finite action inventory, scores every reachable nonterminal state with a local
OpenJev masked-logit model, and calls DecisionFlow for exact finite-horizon
trajectory inference. Benchmark parsing, prompts, caches, and evaluation remain
in this experiment package.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src:experiments/src:third_party/openjev \
.venv-openjev/bin/python -m decisionflow_experiments.logistics \
  --dataset-root third_party/autoplanbench/data \
  --instance 0 --horizon 5 \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
  --cache experiments/artifacts/logistics-instance-0-openjev.json \
  --output experiments/results/generated/logistics-instance-0.json
```

The report compares raw greedy decoding, greedy decoding with immediate action
masking, and DecisionFlow trajectory inference. It includes valid mass,
first-action marginals, trajectory MAP, scoring and inference latency, and an
independent exhaustive-enumeration check. The score cache records the model,
prompt hash, action inventory, and local distributions; changing the model or
prompt invalidates it automatically.

Use the trained open-source Laya typed-decisions checkpoint with the same
states, action inventory, prompt semantics, and DecisionFlow inference:

```bash
PYTHONPATH=src:experiments/src:third_party/laya \
.venv-openjev/bin/python -m decisionflow_experiments.logistics \
  --dataset-root third_party/autoplanbench/data \
  --instance 0 --horizon 5 --scorer laya \
  --model convaiinnovations/laya-typed-decisions \
  --device cpu --batch-size 4 \
  --cache experiments/artifacts/logistics-instance-0-laya.json \
  --output experiments/results/generated/logistics-instance-0-laya.json
```

The Laya adapter raises the option-head budget to 512 tokens and scores all 26
actions. It does not shortlist or pre-mask candidates, keeping the candidate
space identical to the OpenJev condition.

## AutoPlanBench Depot multi-step experiment

The Depot adapter evaluates a harder typed planning domain that combines truck
transport with crate stacking and hoist-resource rules. At each step, Laya
scores a state-specific `Choice` over every currently executable action. The
same cached distributions feed local greedy decoding, finite-width constrained
beam search, and DecisionFlow exact trajectory inference. The official plan is
validated as a reference solution.

```bash
PYTHONPATH=src:experiments/src:third_party/laya \
.venv-openjev/bin/python -m decisionflow_experiments.depot \
  --dataset-root third_party/autoplanbench/data \
  --instance 2 --horizon 12 \
  --model convaiinnovations/laya-typed-decisions \
  --device cpu --batch-size 8 --beam-widths 4 16 64 \
  --cache experiments/artifacts/depot-instance-2-laya.json \
  --output experiments/results/generated/depot-instance-2-laya.json
```

The result reports reachable and goal-reaching graph sizes, the number of valid
complete trajectories, each baseline's success and selected actions,
DecisionFlow valid mass, first-action marginals, joint MAP trajectory, and an
independent backward-DP exactness check. Only the benchmark adapter knows about
PDDL; the internal reference engine receives a finite trajectory specification.

## CLM-8B T-Rex multi-step experiment

This adapter uses the deterministic T-Rex physics from the CLM repository as a
temporal decision environment. CLM receives a neutral `Choice` question over
`jump`, `duck`, and `run`; planner safety labels and recommended actions are
excluded from the prompt. The same local action distributions feed an
unconstrained greedy rollout, constrained beam search, and DecisionFlow exact
finite-horizon inference.

```bash
python -m pip install -e third_party/clm-tune-mlx
PYTHONPATH=src:experiments/src \
.venv-openjev/bin/python -m decisionflow_experiments.trex \
  --clm-root third_party/clm \
  --seeds 0 1 2 --scenarios-per-seed 5 \
  --horizon 4 --decision-frames 6 \
  --scorer clm \
  --cache experiments/results/generated/cache/trex-clm8b.json \
  --output experiments/results/generated/trex-clm8b-3seeds-15scenarios.json
```

The experiment downloads the official CLM projection head and uses the
8-bit MLX Qwen3-8B encoder on Apple Silicon. Model files, probability caches,
and generated reports remain outside the reusable package and are ignored by
Git. The report contains local rollout and beam success, valid mass,
first-action marginals, trajectory MAP, inference latency, and per-scenario
physics metadata. For the same captured snapshots it also evaluates the
official planner-labeled prompt, including the proposed action, shielded
action, planner agreement, and intervention count.

The unmodified official real-time harness can be run against the same local
checkpoint through the MLX-compatible `/v1/systemone` service:

```bash
clm-tune-mlx-serve --port 8700 --checkpoint /path/to/CLM_v0.1-8B.safetensors
PYTHONPATH=third_party/clm/src \
.venv-openjev/bin/python third_party/clm/examples/t_rex/run.py \
  --model clm --seeds 5 --duration 60 --inflight 6 --prompt labeled \
  --out experiments/results/generated/trex-official-clm-realtime.json
```

Add `--no-shield` for the official model-only ablation. These end-to-end
real-time reports and the fixed-snapshot DecisionFlow report answer different
questions and are retained separately.

Run the matching full-course DecisionFlow controller over the same five seeds
and 60-second duration:

```bash
PYTHONPATH=src:experiments/src \
.venv-openjev/bin/python -m decisionflow_experiments.trex_full \
  --clm-root third_party/clm --seeds 0 1 2 3 4 --duration 60 \
  --horizon 4 --decision-frames 6 --capture-distance 140 \
  --cache experiments/results/generated/cache/trex-clm8b.json \
  --output experiments/results/generated/trex-decisionflow-full-5x60.json
```

The controller uses neutral action descriptions and selects the largest
constraint-conditioned first-action marginal at each threat-relevant control
point. Obstacle-free control points deterministically continue running. This is
a latency-free physics evaluation, so its closest official comparison is the
runner's `--lockstep 6` condition. It is not directly comparable to the
real-time condition, where inference latency changes the game state before an
answer arrives.

The full runner currently expects the locally acquired inputs documented in
`pilot/README.md`: the released Jev response SQLite cache, the moderation JSONL
archive, and the four public parquet datasets under `tmp/`. These inputs remain
outside Git because of size and upstream licensing; their paths are resolved by
the runner and missing inputs fail explicitly.
