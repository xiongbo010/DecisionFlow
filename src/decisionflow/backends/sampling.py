"""Approximate rejection/importance sampling over finite typed decisions."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, Optional

from ..core import DecisionProgram, DecisionResult, InferenceInfo, LocalPotentials
from ..errors import UnsatisfiableError
from ..joints import IndependentJoint, JointBuilder


class RejectionSamplingBackend:
    """Estimate constrained queries by sampling from the independent base joint."""

    name = "rejection_sampling"
    exact = False
    capabilities = ("valid_mass", "marginals", "joint_map")

    def __init__(self, samples: int = 10_000, seed: Optional[int] = 0):
        if samples <= 0:
            raise ValueError("samples must be positive")
        self.samples = int(samples)
        self.seed = seed

    @staticmethod
    def _draw(row, options, rng: random.Random):
        threshold = rng.random()
        cumulative = 0.0
        for option in options:
            cumulative += row[option]
            if threshold <= cumulative:
                return option
        return options[-1]

    def infer(
        self,
        program: DecisionProgram,
        potentials: LocalPotentials,
        joint: JointBuilder,
    ) -> DecisionResult:
        if not isinstance(joint, IndependentJoint):
            raise NotImplementedError(
                "rejection sampling currently requires IndependentJoint"
            )
        started = time.perf_counter()
        rng = random.Random(self.seed)
        request = program.request
        marginals: Dict[str, Dict[Any, float]] = {
            question.id: {option: 0.0 for option in question.options}
            for question in request.questions
        }
        accepted = 0
        weighted_total = 0.0
        best_weight = -1.0
        best_assignment: Dict[str, Any] = {}
        for _ in range(self.samples):
            assignment = {
                question.id: self._draw(
                    potentials.values[question.id], question.options, rng
                )
                for question in request.questions
            }
            if any(
                not constraint.expr.evaluate(assignment, request.state)
                for constraint in program.hard_constraints
            ):
                continue
            accepted += 1
            soft_weight = 1.0
            for constraint in program.soft_constraints:
                if not constraint.expr.evaluate(assignment, request.state):
                    soft_weight *= math.exp(-constraint.penalty)
            weighted_total += soft_weight
            for name, value in assignment.items():
                marginals[name][value] += soft_weight
            base_weight = joint.weight(assignment, request, potentials)
            candidate_weight = base_weight * soft_weight
            if candidate_weight > best_weight:
                best_weight = candidate_weight
                best_assignment = dict(assignment)
        if weighted_total <= 0:
            raise UnsatisfiableError(
                "sampling found no positive-weight satisfying assignment; "
                "increase samples or use an exact backend"
            )
        for row in marginals.values():
            for value in row:
                row[value] /= weighted_total
        hard_z = accepted / self.samples
        soft_normalizer = weighted_total / self.samples
        marginal_map = {name: max(row, key=row.get) for name, row in marginals.items()}
        elapsed = 1000.0 * (time.perf_counter() - started)
        return DecisionResult(
            marginals=marginals,
            joint_map=best_assignment,
            valid_mass=hard_z,
            map_probability=min(1.0, best_weight / soft_normalizer),
            local_potentials=potentials.values,
            inference=InferenceInfo(
                backend=self.name,
                exact=False,
                scope="Monte Carlo estimate over the finite candidate space",
                inference_ms=elapsed,
                world_count=self.samples,
                valid_world_count=accepted,
                capabilities=self.capabilities,
                notes=("MAP is the highest-weight assignment observed in the sample",),
            ),
            diagnostics={
                "program": {"name": program.name, "version": program.version},
                "joint_builder": getattr(joint, "name", type(joint).__name__),
                "samples": self.samples,
                "accepted_samples": accepted,
                "acceptance_rate": hard_z,
                "estimated_soft_normalizer": soft_normalizer,
                "marginal_map": marginal_map,
            },
        )
