import pytest

from decisionflow.backends import EnumerationBackend, create_backend_registry
from decisionflow.engine import DecisionEngine
from decisionflow.trajectory import TrajectoryEngine, parse_trajectory


def typed_request():
    return {
        "state": {},
        "questions": [
            {"id": "route", "type": "choice", "options": ["billing", "security"]},
            {"id": "fraud", "type": "noul"},
        ],
        "probabilities": {
            "route": {"billing": 0.7, "security": 0.3},
            "fraud": 0.8,
        },
    }


def typed_constraints():
    return {
        "hard": [
            {
                "expr": {
                    "implies": [
                        {"eq": [{"var": "fraud"}, True]},
                        {"eq": [{"var": "route"}, "security"]},
                    ]
                }
            }
        ]
    }


def trajectory_payload():
    return {
        "initial_state": "start",
        "actions": ["inspect", "finish"],
        "horizon": 2,
        "terminal_states": ["done"],
        "policy": {
            "start": {"inspect": 0.4, "finish": 0.6},
            "inspected": {"inspect": 0.1, "finish": 0.9},
        },
        "transitions": {
            "start": {"inspect": "inspected"},
            "inspected": {"finish": "done"},
        },
    }


def test_typed_backend_can_be_selected_per_call():
    engine = DecisionEngine(backend="enumeration")
    exact = engine.infer(typed_request(), typed_constraints())
    sampled = engine.infer(
        typed_request(),
        typed_constraints(),
        backend="rejection_sampling",
        backend_options={"samples": 20_000, "seed": 7},
    )

    assert exact.inference.backend == "enumeration"
    assert sampled.inference.backend == "rejection_sampling"
    assert sampled.inference.exact is False
    assert sampled.valid_mass == pytest.approx(exact.valid_mass, abs=0.02)
    assert sampled.marginals["fraud"][True] == pytest.approx(
        exact.marginals["fraud"][True], abs=0.03
    )


def test_greedy_returns_a_point_prediction_without_joint_queries():
    result = DecisionEngine(backend="greedy").infer(
        typed_request(), typed_constraints()
    )

    assert result.prediction == {"route": "billing", "fraud": True}
    assert result.prediction_kind == "local_greedy"
    assert result.joint_map == {}
    assert result.marginals == {}
    assert result.valid_mass is None
    assert result.supports("point_prediction")
    assert not result.supports("marginals")
    assert result.diagnostics["prediction_feasible"] is False


def test_custom_backend_registration():
    registry = create_backend_registry()
    registry.register(
        "reference",
        lambda **options: EnumerationBackend(max_worlds=options.get("limit", 100)),
        capabilities=("valid_mass", "marginals", "joint_map"),
        exact=True,
        aliases=("ref",),
    )
    engine = DecisionEngine(backend="ref", backend_registry=registry)

    result = engine.infer(typed_request(), typed_constraints())

    assert result.inference.backend == "enumeration"
    assert registry.descriptor("ref").name == "reference"


def test_trajectory_backends_share_one_interface():
    spec = parse_trajectory(trajectory_payload())
    engine = TrajectoryEngine()
    dynamic = engine.infer(spec)
    best_first = engine.infer(spec, backend="best_first")
    beam = engine.infer(spec, backend="beam", backend_options={"width": 2})
    sampled = engine.infer(
        spec,
        backend="monte_carlo",
        backend_options={"samples": 20_000, "seed": 11},
    )

    assert dynamic.valid_mass == pytest.approx(0.36)
    assert dynamic.trajectory_map == best_first.trajectory_map == beam.trajectory_map
    assert best_first.supports("joint_map")
    assert not best_first.supports("valid_mass")
    assert sampled.valid_mass == pytest.approx(dynamic.valid_mass, abs=0.02)
    assert sampled.first_action_marginals["inspect"] == pytest.approx(1.0)


def test_trajectory_greedy_can_expose_a_local_dead_end():
    spec = parse_trajectory(trajectory_payload())
    result = TrajectoryEngine(backend="greedy").infer(spec)

    assert result.prediction_kind == "stepwise_greedy"
    assert result.trajectory_map == ()
    assert result.valid_mass is None
    assert result.supports("point_prediction")
    assert result.diagnostics["terminal_reached"] is False
    assert result.diagnostics["failed_action"] == "finish"


def test_zero_horizon_terminal_state():
    payload = trajectory_payload()
    payload.update(
        {
            "initial_state": "done",
            "horizon": 0,
        }
    )
    result = TrajectoryEngine().infer(parse_trajectory(payload))

    assert result.valid_mass == 1.0
    assert result.trajectory_map == ()
    assert result.first_action_marginals == {"inspect": 0.0, "finish": 0.0}
