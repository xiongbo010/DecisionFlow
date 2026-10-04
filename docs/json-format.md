# DecisionFlow v0.0 JSON format

## Typed decision request

```json
{
  "id": "case-1",
  "state": {"text": "...", "account": {"age_days": 3}},
  "questions": [
    {"id": "route", "type": "choice", "options": ["billing", "security"]},
    {"id": "fraud", "type": "noul"},
    {"id": "urgency", "type": "score", "options": [0, 1, 2, 3]}
  ],
  "probabilities": {
    "route": {"billing": 0.6, "security": 0.4},
    "fraud": 0.7,
    "urgency": [0.1, 0.2, 0.3, 0.4]
  },
  "evidence": {"optional_variable": "fixed_value"}
}
```

`choice` and `score` accept an option list or a criteria object. `noul`
always has the domain `[false, true]`. Probabilities may be maps, vectors in
option order, or a single probability of `true` for `noul`.

When a scorer adapter is configured, the `probabilities` field is omitted.

## Constraint pack

```json
{
  "name": "ticket-policy",
  "version": "1.0",
  "hard": [
    {
      "name": "fraud-routing",
      "expr": {
        "implies": [
          {"eq": [{"var": "fraud"}, true]},
          {"in": [{"var": "route"}, ["security"]]}
        ]
      }
    }
  ],
  "soft": [
    {
      "name": "prefer-review-for-high-urgency",
      "penalty": 1.5,
      "expr": {
        "implies": [
          {"ge": [{"var": "urgency"}, 2]},
          {"ne": [{"var": "route"}, "billing"]}
        ]
      }
    }
  ]
}
```

### Expressions

- operands: `{"var": "name"}`, `{"state": "path.to.value"}`, literals,
  or `{"const": value}`;
- comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`;
- membership: `in`, `not_in`;
- Boolean: `not`, `all`, `any`, `implies`, `iff`;
- cardinality: `at_most`, `at_least`, `exactly` with `{k, items}`;
- arbitrary finite relations: `allowed_table` and `forbidden_table` with
  `{variables, rows}`.

The table expression allows an ontology reasoner, SOP compiler, workflow
engine, or external policy system to pass any grounded finite relation without
adding a dependency to DecisionFlow core.

Hard constraints remove assignments. A soft constraint multiplies the weight
of each violating assignment by `exp(-penalty)`. Results report hard valid mass
separately from the soft-factor normalizer.

## Finite trajectory format

```json
{
  "initial_state": "start",
  "actions": ["inspect", "finish"],
  "horizon": 2,
  "terminal_states": ["done"],
  "policy": {
    "start": {"inspect": 0.4, "finish": 0.6},
    "inspected": {"inspect": 0.1, "finish": 0.9}
  },
  "transitions": {
    "start": {"inspect": "inspected"},
    "inspected": {"finish": "done"}
  }
}
```

Transitions may also contain multiple `{next_state, probability}` outcomes.
The trajectory engine sums over environment outcomes and reports the valid
trajectory mass, first-action marginals, and trajectory MAP.

## Tool registration

`DecisionFlowTools.schemas` exposes provider-neutral JSON Schemas for three tools:

- `decisionflow_infer(request, constraints)`;
- `decisionflow_evaluate(requests, constraints)`;
- `decisionflow_trajectory(spec)`.

Create `DecisionFlowTools` with a `DecisionFlow` instance whose scorer is already
configured. This keeps model credentials and SDK objects outside tool
arguments while allowing an agent runtime to call DecisionFlow with ordinary JSON.
