"""Local greedy baseline for typed decision requests."""

from __future__ import annotations

import time
from typing import Any, Dict

from ..core import DecisionProgram, DecisionResult, InferenceInfo, LocalPotentials
from ..joints import JointBuilder


class GreedyBackend:
    """Select each typed value independently by its local model probability."""

    name = "greedy"
    exact = False
    capabilities = ("point_prediction", "feasibility_check")

    def infer(
        self,
        program: DecisionProgram,
        potentials: LocalPotentials,
        joint: JointBuilder,
    ) -> DecisionResult:
        started = time.perf_counter()
        prediction: Dict[str, Any] = {}
        for question in program.request.questions:
            row = potentials.values[question.id]
            prediction[question.id] = max(
                question.options,
                key=lambda value: row[value],
            )
        hard_violations = [
            constraint.name
            for constraint in program.hard_constraints
            if not constraint.expr.evaluate(prediction, program.request.state)
        ]
        soft_violations = [
            constraint.name
            for constraint in program.soft_constraints
            if not constraint.expr.evaluate(prediction, program.request.state)
        ]
        elapsed = 1000.0 * (time.perf_counter() - started)
        return DecisionResult(
            prediction=prediction,
            prediction_kind="local_greedy",
            marginals={},
            joint_map={},
            valid_mass=None,
            map_probability=None,
            local_potentials=potentials.values,
            inference=InferenceInfo(
                backend=self.name,
                exact=False,
                scope="one independent local argmax per typed question",
                inference_ms=elapsed,
                capabilities=self.capabilities,
                notes=(
                    "The point prediction can violate grounded constraints and is not a joint MAP.",
                ),
            ),
            diagnostics={
                "program": {"name": program.name, "version": program.version},
                "joint_builder": getattr(joint, "name", type(joint).__name__),
                "prediction_feasible": not hard_violations,
                "hard_constraint_violations": hard_violations,
                "soft_constraint_violations": soft_violations,
            },
        )
