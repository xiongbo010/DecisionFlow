#!/usr/bin/env python3
"""Compile the moderation constraints to an SDD and validate exact inference.

The existing 2^8 enumeration is retained only as a correctness oracle.  The
SDD backend performs weighted model counting for Z and literal marginals, and
max-product inference over the circuit for joint MAP.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".deps"))

from pysdd.sdd import SddManager, SddNode, Vtree  # noqa: E402

from openai_moderation_pilot import (  # noqa: E402
    HEADS,
    IMPLICATIONS,
    INDEX,
    enumerate_worlds,
    exact_condition,
    load_rows,
)


@dataclass(frozen=True)
class MapResult:
    weight: float
    assignment: dict[int, int]


def assignment_mask(assignment: dict[int, int]) -> int:
    """Map an assignment to the same integer ordering used by the oracle."""
    # Internal max-product calls contain partial assignments for a vtree
    # subtree.  Missing variables contribute zero to the global bit mask.
    return sum(int(value) << (var - 1) for var, value in assignment.items())


def better(left: MapResult | None, right: MapResult, atol: float = 1e-15) -> MapResult:
    """Select higher weight, breaking numerical ties by the oracle's world order."""
    if left is None or right.weight > left.weight + atol:
        return right
    if math.isclose(right.weight, left.weight, rel_tol=0.0, abs_tol=atol):
        return right if assignment_mask(right.assignment) < assignment_mask(left.assignment) else left
    return left


def merge(left: MapResult, right: MapResult) -> MapResult:
    if left.assignment.keys() & right.assignment.keys():
        raise RuntimeError("Non-decomposable SDD element encountered")
    assignment = dict(left.assignment)
    assignment.update(right.assignment)
    return MapResult(left.weight * right.weight, assignment)


def max_product(
    node: SddNode,
    vtree: Vtree,
    positive: np.ndarray,
    negative: np.ndarray,
    true_node: SddNode,
) -> MapResult:
    """Exact MPE/MAP on an SDD, including variables omitted by trimming.

    The recursion mirrors ``SddNode.models``.  Determinism turns decision-node
    disjunctions into max operations, while decomposability permits products of
    prime and sub results.  A true node chooses the heavier literal in its vtree
    scope.  Ties select false, matching the oracle's ascending bit-mask order.
    """
    if node.is_false():
        return MapResult(-math.inf, {})

    if vtree.is_leaf():
        var = int(vtree.var())
        if node.is_true():
            if positive[var - 1] > negative[var - 1]:
                return MapResult(float(positive[var - 1]), {var: 1})
            return MapResult(float(negative[var - 1]), {var: 0})
        if not node.is_literal() or abs(int(node.literal)) != var:
            raise RuntimeError("Unexpected terminal/vtree pairing")
        value = int(node.literal > 0)
        weight = positive[var - 1] if value else negative[var - 1]
        return MapResult(float(weight), {var: value})

    if node.is_true():
        return merge(
            max_product(true_node, vtree.left(), positive, negative, true_node),
            max_product(true_node, vtree.right(), positive, negative, true_node),
        )

    node_vtree = node.vtree()
    if node_vtree == vtree:
        best: MapResult | None = None
        for prime, sub in node.elements():
            if sub.is_false():
                continue
            candidate = merge(
                max_product(prime, vtree.left(), positive, negative, true_node),
                max_product(sub, vtree.right(), positive, negative, true_node),
            )
            best = better(best, candidate)
        if best is None:
            return MapResult(-math.inf, {})
        return best

    # A trimmed SDD may live strictly below the requested vtree.  Complete the
    # missing sibling scope with its locally most probable assignment.
    if Vtree.is_sub(node_vtree, vtree.left()):
        return merge(
            max_product(node, vtree.left(), positive, negative, true_node),
            max_product(true_node, vtree.right(), positive, negative, true_node),
        )
    return merge(
        max_product(true_node, vtree.left(), positive, negative, true_node),
        max_product(node, vtree.right(), positive, negative, true_node),
    )


def compile_constraints(artifact_dir: Path) -> tuple[SddManager, SddNode, float]:
    start = time.perf_counter()
    vtree = Vtree(var_count=len(HEADS), var_order=list(range(1, len(HEADS) + 1)), vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    root = manager.true()
    for child, parent in IMPLICATIONS:
        child_lit = manager.literal(INDEX[child] + 1)
        parent_lit = manager.literal(INDEX[parent] + 1)
        root &= (~child_lit) | parent_lit
    root.ref()
    compile_seconds = time.perf_counter() - start

    artifact_dir.mkdir(parents=True, exist_ok=True)
    root.save(str(artifact_dir / "openai_moderation_constraints.sdd").encode())
    manager.vtree().save(str(artifact_dir / "openai_moderation_constraints.vtree").encode())
    return manager, root, compile_seconds


def reload_model_count(artifact_dir: Path) -> int:
    """Reload the serialized circuit and vtree to verify the saved artifacts."""
    vtree_path = str(artifact_dir / "openai_moderation_constraints.vtree").encode()
    sdd_path = str(artifact_dir / "openai_moderation_constraints.sdd").encode()
    loaded_vtree = Vtree.from_file(vtree_path)
    loaded_manager = SddManager.from_vtree(loaded_vtree)
    loaded_root = loaded_manager.read_sdd_file(sdd_path)
    if loaded_root is None:
        raise RuntimeError("The serialized SDD could not be reloaded")
    return int(loaded_root.global_model_count())


def oracle_details(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    worlds, valid = enumerate_worlds()
    valid_worlds = worlds[valid]
    map_weights = np.empty(len(probs), dtype=float)
    map_ties = np.empty(len(probs), dtype=np.int64)
    for row, p in enumerate(probs):
        weights = np.prod(np.where(valid_worlds == 1, p, 1.0 - p), axis=1)
        best = float(np.max(weights))
        map_weights[row] = best
        map_ties[row] = int(np.sum(np.isclose(weights, best, rtol=0.0, atol=1e-15)))
    return map_weights, map_ties


def validate(probs: np.ndarray, artifact_dir: Path) -> dict:
    oracle_marginals, oracle_z, oracle_map = exact_condition(probs)
    oracle_map_weights, oracle_map_ties = oracle_details(probs)

    manager, root, compile_seconds = compile_constraints(artifact_dir)
    circuit_model_count = int(root.global_model_count())
    reloaded_model_count = reload_model_count(artifact_dir)
    expected_model_count = int(enumerate_worlds()[1].sum())
    if circuit_model_count != expected_model_count:
        raise AssertionError(f"SDD model count {circuit_model_count} != oracle {expected_model_count}")
    if reloaded_model_count != circuit_model_count:
        raise AssertionError(f"Reloaded SDD model count {reloaded_model_count} != compiled SDD {circuit_model_count}")

    wmc = root.wmc(log_mode=False)
    circuit_z = np.empty(len(probs), dtype=float)
    circuit_marginals = np.empty_like(probs)
    circuit_map = np.empty_like(probs, dtype=np.int8)
    circuit_map_weights = np.empty(len(probs), dtype=float)

    start = time.perf_counter()
    for row, p in enumerate(probs):
        for var, probability in enumerate(p, start=1):
            wmc.set_literal_weight(var, float(probability))
            wmc.set_literal_weight(-var, float(1.0 - probability))
        circuit_z[row] = float(wmc.propagate())
        circuit_marginals[row] = [float(wmc.literal_pr(var)) for var in range(1, len(HEADS) + 1)]

        mpe = max_product(root, manager.vtree(), p, 1.0 - p, manager.true())
        circuit_map_weights[row] = mpe.weight
        circuit_map[row] = [mpe.assignment[var] for var in range(1, len(HEADS) + 1)]
    inference_seconds = time.perf_counter() - start

    z_error = np.abs(circuit_z - oracle_z)
    marginal_error = np.abs(circuit_marginals - oracle_marginals)
    map_weight_error = np.abs(circuit_map_weights - oracle_map_weights)
    map_probability_error = np.abs(circuit_map_weights / circuit_z - oracle_map_weights / oracle_z)
    map_assignment_match = np.all(circuit_map == oracle_map, axis=1)

    tolerances = {"z": 1e-12, "marginal": 1e-12, "map_weight": 1e-12, "map_probability": 1e-12}
    failures = {
        "z": np.flatnonzero(z_error > tolerances["z"]).tolist(),
        "marginal": np.flatnonzero(np.max(marginal_error, axis=1) > tolerances["marginal"]).tolist(),
        "map_weight": np.flatnonzero(map_weight_error > tolerances["map_weight"]).tolist(),
        "map_probability": np.flatnonzero(map_probability_error > tolerances["map_probability"]).tolist(),
        "map_assignment": np.flatnonzero(~map_assignment_match).tolist(),
    }
    if any(failures.values()):
        first = {key: value[:10] for key, value in failures.items() if value}
        raise AssertionError(f"SDD/oracle mismatch: {first}")

    return {
        "status": "pass",
        "backend": {"library": "PySDD", "version": "1.0.6", "vtree": "balanced"},
        "formula": {
            "variables": HEADS,
            "implications": [f"{child} -> {parent}" for child, parent in IMPLICATIONS],
            "total_assignments": 1 << len(HEADS),
            "satisfying_assignments_oracle": expected_model_count,
            "satisfying_assignments_sdd": circuit_model_count,
        },
        "circuit": {
            "size_elements": int(root.size()),
            "node_count": int(root.count()),
            "compile_seconds": compile_seconds,
            "serialized_reload_model_count": reloaded_model_count,
            "serialized_reload_check": "pass",
            "sdd_path": str(artifact_dir / "openai_moderation_constraints.sdd"),
            "vtree_path": str(artifact_dir / "openai_moderation_constraints.vtree"),
        },
        "validation": {
            "n_samples": int(len(probs)),
            "z_samples_passed": int(np.sum(z_error <= tolerances["z"])),
            "marginal_values_checked": int(marginal_error.size),
            "marginal_values_passed": int(np.sum(marginal_error <= tolerances["marginal"])),
            "map_weights_passed": int(np.sum(map_weight_error <= tolerances["map_weight"])),
            "map_probabilities_passed": int(np.sum(map_probability_error <= tolerances["map_probability"])),
            "map_assignments_passed": int(map_assignment_match.sum()),
            "samples_with_tied_map": int(np.sum(oracle_map_ties > 1)),
            "max_map_tie_count": int(np.max(oracle_map_ties)),
            "max_abs_error_z": float(np.max(z_error)),
            "max_abs_error_marginal": float(np.max(marginal_error)),
            "max_abs_error_map_weight": float(np.max(map_weight_error)),
            "max_abs_error_map_probability": float(np.max(map_probability_error)),
            "tolerances": tolerances,
        },
        "timing": {
            "total_inference_seconds": inference_seconds,
            "mean_inference_milliseconds_per_sample": 1000.0 * inference_seconds / len(probs),
        },
    }


def markdown(result: dict) -> str:
    formula = result["formula"]
    circuit = result["circuit"]
    valid = result["validation"]
    timing = result["timing"]
    return f"""# SDD exact-inference validation

**Status: PASS.** The compiled SDD and exhaustive enumeration agree on every sample.

## Compiled formula

- Variables: {', '.join(formula['variables'])}
- Constraints: {', '.join(formula['implications'])}
- Satisfying assignments: {formula['satisfying_assignments_sdd']} (SDD) = {formula['satisfying_assignments_oracle']} (enumeration), out of {formula['total_assignments']}
- SDD size: {circuit['size_elements']} elements, {circuit['node_count']} nodes
- Compile time: {circuit['compile_seconds'] * 1000:.3f} ms
- Serialized SDD reload: PASS ({circuit['serialized_reload_model_count']} satisfying assignments)

## Sample-wise checks

| Quantity | Checks passed | Maximum absolute error | Tolerance |
|---|---:|---:|---:|
| Valid mass $Z$ | {valid['z_samples_passed']} / {valid['n_samples']} | {valid['max_abs_error_z']:.3e} | {valid['tolerances']['z']:.0e} |
| Typed marginals | {valid['marginal_values_passed']} / {valid['marginal_values_checked']} | {valid['max_abs_error_marginal']:.3e} | {valid['tolerances']['marginal']:.0e} |
| MAP unnormalized weight | {valid['map_weights_passed']} / {valid['n_samples']} | {valid['max_abs_error_map_weight']:.3e} | {valid['tolerances']['map_weight']:.0e} |
| MAP normalized probability | {valid['map_probabilities_passed']} / {valid['n_samples']} | {valid['max_abs_error_map_probability']:.3e} | {valid['tolerances']['map_probability']:.0e} |
| MAP assignment | {valid['map_assignments_passed']} / {valid['n_samples']} | exact match | — |

The validation covers {valid['n_samples']} samples and {valid['marginal_values_checked']} marginal values. There were {valid['samples_with_tied_map']} samples with tied MAP worlds (maximum tie count {valid['max_map_tie_count']}); both backends use the same ascending-world tie-break.

Total circuit inference time was {timing['total_inference_seconds']:.3f} s, or {timing['mean_inference_milliseconds_per_sample']:.3f} ms per sample, including $Z$, eight marginals, and MAP.

## Artifacts

- SDD: `{circuit['sdd_path']}`
- Vtree: `{circuit['vtree_path']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    _, probs, _, _ = load_rows(args.data, args.responses)
    result = validate(probs, args.artifact_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report_out.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
