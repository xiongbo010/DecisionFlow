"""Transparent exact reference backend over finite typed assignments."""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Dict, Mapping, Tuple

from ..core import DecisionProgram, DecisionResult, InferenceInfo, LocalPotentials
from ..errors import UnsatisfiableError
from ..joints import JointBuilder


class EnumerationBackend:
    name = "enumeration"
    exact = True

    def __init__(self, max_worlds: int = 1_000_000):
        self.max_worlds = max_worlds

    def infer(
        self,
        program: DecisionProgram,
        potentials: LocalPotentials,
        joint: JointBuilder,
    ) -> DecisionResult:
        request = program.request
        total_worlds = math.prod(len(question.options) for question in request.questions)
        if total_worlds > self.max_worlds:
            raise ValueError(
                "enumeration needs %d worlds, above max_worlds=%d; install worldjev[sdd]"
                % (total_worlds, self.max_worlds)
            )
        started = time.perf_counter()
        marginals: Dict[str, Dict[Any, float]] = {
            question.id: {option: 0.0 for option in question.options}
            for question in request.questions
        }
        hard_z = 0.0
        normalizer = 0.0
        best_weight = -1.0
        best_assignment: Dict[str, Any] = {}
        valid_count = 0
        raw_map: Dict[str, Any] = {
            question.id: max(
                question.options,
                key=lambda value: potentials.values[question.id][value],
            )
            for question in request.questions
        }
        raw_violations = [
            constraint.name
            for constraint in program.hard_constraints
            if not constraint.expr.evaluate(raw_map, request.state)
        ]
        domains = [question.options for question in request.questions]
        names = [question.id for question in request.questions]
        for values in itertools.product(*domains):
            assignment = dict(zip(names, values))
            if any(
                not constraint.expr.evaluate(assignment, request.state)
                for constraint in program.hard_constraints
            ):
                continue
            valid_count += 1
            base_weight = joint.weight(assignment, request, potentials)
            hard_z += base_weight
            weight = base_weight
            for constraint in program.soft_constraints:
                if not constraint.expr.evaluate(assignment, request.state):
                    weight *= math.exp(-constraint.penalty)
            if weight <= 0:
                continue
            normalizer += weight
            for name, value in assignment.items():
                marginals[name][value] += weight
            if weight > best_weight:
                best_weight = weight
                best_assignment = assignment
        if normalizer <= 0:
            raise UnsatisfiableError(
                "no positive-mass assignment satisfies the grounded constraints"
            )
        for row in marginals.values():
            for value in row:
                row[value] /= normalizer
        elapsed = 1000.0 * (time.perf_counter() - started)
        return DecisionResult(
            marginals=marginals,
            joint_map=best_assignment,
            valid_mass=hard_z,
            map_probability=best_weight / normalizer,
            local_potentials=potentials.values,
            inference=InferenceInfo(
                backend=self.name,
                exact=True,
                scope="finite candidate space specified by the request",
                inference_ms=elapsed,
                world_count=total_worlds,
                valid_world_count=valid_count,
            ),
            diagnostics={
                "program": {"name": program.name, "version": program.version},
                "joint_builder": getattr(joint, "name", type(joint).__name__),
                "raw_local_map": raw_map,
                "raw_local_map_violations": raw_violations,
                "hard_constraints": [item.name for item in program.hard_constraints],
                "soft_constraints": [item.name for item in program.soft_constraints],
                "normalizer": normalizer,
                "hard_valid_mass": hard_z,
            },
        )
