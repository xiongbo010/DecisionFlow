# Inference backends

The workflow compiler produces one `StructuredDecisionFlow` IR regardless of
the selected inference method. Backends declare the queries they support, so a
point-prediction method does not need to fabricate marginals or valid mass.

## Public workflow methods

| Backend | Exact | Valid mass | Marginals | Complete-flow MAP | Main use |
|---|---:|---:|---:|---:|---|
| `greedy` | no | no | no | no | local decision-model baseline and policy diagnostics |
| `a_star` / `astar` | yes | no | no | yes | exact best-flow search with an admissible zero heuristic |
| `beam_search` / `beam` | no | no | no | approximate | finite-width prefix search |
| `dynamic_programming` / `dp` | yes | yes | yes | yes | exact sum-product and max-product on the flow DAG |
| `sat` / `dpll` | feasibility | no | no | no | return any policy-consistent terminal flow |
| `enumeration` / `auto` | yes | yes | yes | yes | reference inference over finite compiled flows |
| `pc` / `sdd` | yes | yes | yes | yes | SDD inference over the compiled world distribution |
| `top_k` | no | no | truncated | truncated | budgeted high-mass approximation |
| `sampling` | no | estimate | estimate | observed | Monte Carlo approximation |

`None` or an empty mapping means that the backend does not support that query.
Use `result.supports(query)`, `decisionflow backends`, or
`decisionflow_backends()` to inspect capabilities.

## Selection

```python
from decisionflow import DecisionFlow

flow = DecisionFlow(
    model=model,
    workflow="workflow.yaml",
    policies="policies.yaml",
    backend="pc",
)

exact = flow.infer(state)
greedy = flow.infer(state, backend="greedy")
sampled = flow.infer(
    state,
    backend="sampling",
    backend_options={"samples": 20_000, "seed": 0},
)
```

## Backend contract

Workflow backends implement:

```python
infer(program: StructuredDecisionFlow) -> FlowResult
```

The registry stores lazy factories, aliases, exactness, and query capabilities.
An application can extend the internal registry with SAT/ILP, direct WMC,
alternative probabilistic circuits, MCMC, heuristic search, or learned
inference without changing the workflow compiler or public result format.

### Search semantics

A* assigns every transition the cost `-log p`. The current heuristic is zero,
so the method is uniform-cost search and returns the exact weighted MAP flow.
Soft penalties are added when a terminal flow is reached. Beam search uses the
same prefix probabilities but keeps only `width` prefixes at each expansion.

### Dynamic programming

DP recursively computes hard valid mass, soft-normalized mass, and max-product
weight. A forward/backward pass obtains typed marginals from edge posterior
mass. The current compiler produces an acyclic prefix graph by bounding loops
with `max_steps`.

### SAT

SAT requires at least one compiled world to be true and forbids incomplete or
hard-policy-violating worlds. A built-in DPLL solver returns one satisfying flow.
The backend intentionally ignores probabilities; probability-aware
optimization belongs to A*, DP, PC, WMC, or a future weighted MaxSAT backend.

## Current PC lowering

The v0.1 compiler first materializes finite complete flows. The PC backend then
encodes the flow identity as a weighted categorical variable, compiles hard and
soft policies into an SDD, and aggregates the world posterior back into typed
marginals. This is exact but does not yet exploit repeated workflow structure.

A later structural lowering can translate decision nodes, activation variables,
guards, and transitions directly into circuit factors. The public workflow and
backend interfaces remain unchanged.

## Internal reference engines

The package retains low-level grounded typed-decision and finite-state
trajectory engines for regression tests and research baselines. They live in
`decisionflow.engine` and `decisionflow.trajectory`; they are not part of the
top-level user API. The public `DecisionFlow` compiler owns question generation
and decision-structure construction.
