# WorldJev v0.0

WorldJev turns local probability distributions from typed decision models into
a normalized distribution over consistent decision assignments. It separates
constraint frontends from inference backends and works with hosted APIs, local
models, Python callables, or precomputed probabilities.

The v0.0 pipeline is:

```text
typed request ──> scorer adapter ──> local probabilities
      │                                      │
JSON constraints ──> JSON frontend ──> decision program
                                             │
                         enumeration / SDD backend
                                             │
                       marginals · joint MAP · valid mass Z
```

```bash
pip install -e .
pip install -e '.[sdd]'   # optional scalable SDD backend
```

```python
from worldjev import WorldJev

request = {
    "state": {"channel": "card"},
    "questions": [
        {"id": "route", "type": "choice", "options": ["billing", "security"]},
        {"id": "fraud", "type": "noul"},
    ],
    "probabilities": {
        "route": {"billing": 0.6, "security": 0.4},
        "fraud": {"false": 0.3, "true": 0.7},
    },
}
constraints = {
    "hard": [{
        "name": "fraud-routes-to-security",
        "expr": {"implies": [
            {"eq": [{"var": "fraud"}, True]},
            {"eq": [{"var": "route"}, "security"]}
        ]}
    }]
}

result = WorldJev(backend="auto").infer(request, constraints)
print(result.marginals)
print(result.joint_map)
print(result.valid_mass)
```

`backend="auto"` uses the transparent enumeration oracle for small spaces and
the optional SDD backend for larger spaces. The SDD backend compiles the rule
structure once, caches it, and changes only the model-provided leaf weights on
later requests with the same grounded structure.

## Model adapters

WorldJev does not import a particular model SDK. Configure one of these
provider-neutral adapters and pass it to `WorldJev(scorer=...)`:

- `PrecomputedScorer` for cached experiment outputs;
- `CallableScorer` for a local Python model;
- `TypedResponseScorer` for Jev-style `answers` returned by any SDK;
- `HttpJsonScorer` for a local server or hosted endpoint.

Custom scorers implement one method: `score(DecisionRequest) -> LocalPotentials`.
This boundary supports OpenJev, JevAny, locally hosted models, and closed APIs
without provider code in the inference core.

## Agent tool calls

```python
from worldjev import WorldJev, WorldJevTools
from worldjev.scorers import TypedResponseScorer

scorer = TypedResponseScorer(call_your_model_sdk)
tools = WorldJevTools(WorldJev(scorer=scorer, backend="auto"))

# Register tools.schemas with the agent provider, then dispatch its tool call:
output = tools.call(tool_name, tool_arguments)
```

The registered tools are `worldjev_infer`, `worldjev_evaluate`, and
`worldjev_trajectory`. Their inputs and outputs contain only JSON-compatible
objects.

The JSON CLI accepts one request or JSONL batches:

```bash
worldjev infer request.json --constraints policy.json
worldjev evaluate requests.jsonl --constraints policy.json --output results.jsonl
worldjev trajectory trajectory.json
```

See `docs/json-format.md` for the v0.0 interchange format.

## Constraint coverage

The JSON frontend supports Boolean composition, implications, equivalence,
comparisons, set membership, cardinality, hard rules, and weighted soft rules.
An `allowed_table` or `forbidden_table` can represent any grounded finite
relation. Consequently, a workflow engine, SOP parser, policy system, ontology
reasoner, or temporal-rule compiler can remain outside the core and submit its
grounded relation through the same interface.

## Regression coverage

Core regression tests cover every decision structure used in the current
pilots:

| Experiment family | Typed structure exercised |
|---|---|
| OpenAI Moderation | eight Boolean decisions and hierarchical implications |
| ToxiGen | Boolean plus ordinal Score |
| GoEmotions | 28 Boolean decisions and mutual exclusion |
| ToxicChat | two Boolean decisions and implication |
| HelpSteer2 | five ordinal Scores with hard and soft rules |
| Typed Decisions | heterogeneous Choice, Noul, and Score requests |
| SOP-Bench | finite SOP relations, state evidence, and multi-step policies |
| JevAny control panel | finite-horizon action/transition constraints |

Dataset acquisition, prompting, and benchmark metrics stay in `pilot/`; the
library contains no dataset-specific branches or label names.

## v0.0 boundaries

- All candidate domains are finite.
- The SDD backend currently accepts the independent local-potential joint. The
  `JointBuilder` interface reserves directed and learned joint models; arbitrary
  joint builders already work with the enumeration backend.
- Multi-step inference uses an explicit finite state graph and separates action
  probabilities from stochastic environment transitions.
- Ontologies, SOPs, and policy languages require an external grounding step to
  WorldJev JSON expressions or finite tables.
- The cache is process-local; persistent circuit artifacts and a network
  service are reserved interfaces for later releases.
