# DecisionFlow experiments

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
Multi-step experiments provide a `trajectory` object accepted by
`decisionflow.TrajectoryEngine`.

Legacy exploratory scripts and locally downloaded data remain in the ignored
`pilot/` artifacts and data directories. The pilot Python sources are tracked
as the reference definitions of data loading, label handling, full metrics,
bootstrap intervals, and audits. The `full` runner replaces their inference
functions with DecisionFlow public-API calls and writes complete JSON results:

```bash
decisionflow-experiments full --experiment all --repo-root .
```

Generated JSON files are written under
`experiments/results/generated/full/`. LaTeX table generation is intentionally
outside this pipeline.

The full runner currently expects the locally acquired inputs documented in
`pilot/README.md`: the released Jev response SQLite cache, the moderation JSONL
archive, and the four public parquet datasets under `tmp/`. These inputs remain
outside Git because of size and upstream licensing; their paths are resolved by
the runner and missing inputs fail explicitly.
