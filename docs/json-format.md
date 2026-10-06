# DecisionFlow v0.1 declarative format

The public input consists of a workflow document, optional policies, an initial
state, a configured decision model, and inference options.

## Workflow

```json
{
  "name": "service-ticket",
  "version": "1",
  "start": "classify",
  "max_steps": 4,
  "steps": {
    "classify": {
      "questions": [
        {"id": "route", "type": "choice", "options": ["billing", "security"]},
        {"id": "fraud", "type": "noul"},
        {"id": "urgency", "type": "score", "options": [0, 1, 2]}
      ],
      "transitions": [
        {
          "name": "possible-fraud",
          "when": {"eq": [{"var": "fraud"}, true]},
          "goto": "review",
          "set": {"queue": {"const": "security"}}
        },
        {"otherwise": true, "goto": "resolved"}
      ]
    },
    "review": {
      "questions": [
        {"id": "action", "type": "choice", "options": ["freeze", "dismiss"]}
      ],
      "transitions": [{"otherwise": true, "goto": "resolved"}]
    },
    "resolved": {"terminal": true}
  },
  "constraints": {
    "hard": [
      {
        "name": "fraud-routes-security",
        "expr": {
          "implies": [
            {"eq": [{"var": "classify.fraud"}, true]},
            {"eq": [{"var": "classify.route"}, "security"]}
          ]
        }
      }
    ]
  }
}
```

### Steps and questions

Each non-terminal step contains zero or more typed questions and one or more
ordered transitions. Question types are `choice`, `noul`, and `score`.
Transitions use first-match semantics. `otherwise: true` is an unconditional
fallback.

### State updates

`set` maps dotted state paths to values. Values may be literals,
`{"const": value}`, `{"var": "question"}`, or `{"state": "path"}`.

### Variables

Transition conditions can use unqualified current-step variables such as
`fraud`. Policies use stable names such as `classify.fraud`. Inference outputs
use occurrence names such as `classify@0.fraud`. Variables belonging to a branch
that was not visited receive `__inactive__` in exact marginals.

### Expressions

- operands: `{"var": "name"}`, `{"state": "path.to.value"}`, literals,
  or `{"const": value}`;
- comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`;
- membership: `in`, `not_in`;
- Boolean composition: `not`, `all`, `any`, `implies`, `iff`;
- cardinality: `at_most`, `at_least`, `exactly` with `{k, items}`;
- finite relations: `allowed_table` and `forbidden_table`.

Hard policies remove complete flows. Soft policies multiply violating-flow
weights by `exp(-penalty)`.

## Separate policy document

The optional policy document uses the same `hard` and `soft` format. Its rules
are appended to inline workflow constraints.

## Initial state

The state is any JSON value, although declarative state updates require an
object. State facts are accessed through `{"state": "path"}`.

## Tool registration

`DecisionFlowTools.schemas` exposes two provider-neutral tools:

- `decisionflow_run(state, workflow, policies, backend, backend_options)`;
- `decisionflow_backends()`.

The model is configured on `DecisionFlowTools` and therefore credentials and
SDK objects never appear in tool arguments.
