#!/usr/bin/env python3
"""OpenJev + exact SDD pilot for SOP-Bench dangerous goods.

The pilot asks OpenJev for six local distributions from the same product state:
product-ID validity, four component severity scores, and the final hazard class.
An SDD then conditions their product distribution on the SOP rules.  The script
reports local, constraint-conditioned marginal, and joint-MAP predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openjev import Choice, LocalJev, Noul, Score
from pysdd.sdd import SddManager, SddNode, Vtree


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT
    / "third_party"
    / "sop-bench"
    / "src"
    / "amazon_sop_bench"
    / "benchmarks"
    / "data"
    / "dangerous_goods"
)
DATA_PATH = DATA_DIR / "test_set_with_outputs.csv"
RESULT_DIR = ROOT / "pilot" / "results"
ARTIFACT_DIR = ROOT / "pilot" / "artifacts"

MODEL_ID = "Qwen/Qwen3-0.6B"
SCORE_NAMES = ("sds", "handling", "transportation", "disposal")
SCORE_VALUES = tuple(range(6))
CLASS_VALUES = (
    "Unable to Decide",
    "Hazard Class A",
    "Hazard Class B",
    "Hazard Class C",
    "Hazard Class D",
)
FLOOR = 1e-8

# The public SOP specifies ordering but omits numeric class cutoffs.  These
# bands are the unique monotone completion consistent with every released row;
# the released data contain no totals of 13 or 14.
CLASS_BANDS = (
    (4, 7, "Hazard Class A"),
    (8, 12, "Hazard Class B"),
    (13, 16, "Hazard Class C"),
    (17, 20, "Hazard Class D"),
)


def valid_product_id(value: str) -> bool:
    return re.fullmatch(r"P_\d{5}", value) is not None


def completed_score(values: tuple[int, int, int, int], product_valid: bool) -> int:
    """Apply the SOP's validation and single-missing-value imputation rule."""
    if not product_valid:
        return 0
    missing = sum(value == 0 for value in values)
    if missing >= 2:
        return 0
    if missing == 1:
        replacement = max(values)
        values = tuple(replacement if value == 0 else value for value in values)
    return sum(values)


def class_for(values: tuple[int, int, int, int], product_valid: bool) -> str:
    total = completed_score(values, product_valid)
    if total == 0:
        return "Unable to Decide"
    for lower, upper, label in CLASS_BANDS:
        if lower <= total <= upper:
            return label
    raise ValueError(f"No hazard-class band for score {total}")


@dataclass(frozen=True)
class MapResult:
    weight: float
    assignment: dict[int, int]


def assignment_mask(assignment: dict[int, int]) -> int:
    return sum(int(value) << (var - 1) for var, value in assignment.items())


def better(left: MapResult | None, right: MapResult) -> MapResult:
    if left is None or right.weight > left.weight + 1e-15:
        return right
    if math.isclose(right.weight, left.weight, rel_tol=0.0, abs_tol=1e-15):
        if assignment_mask(right.assignment) < assignment_mask(left.assignment):
            return right
    return left


def merge(left: MapResult, right: MapResult) -> MapResult:
    if left.assignment.keys() & right.assignment.keys():
        raise RuntimeError("Non-decomposable SDD element")
    assignment = dict(left.assignment)
    assignment.update(right.assignment)
    return MapResult(left.weight * right.weight, assignment)


def max_product(
    node: SddNode,
    vtree: Vtree,
    positive: list[float],
    negative: list[float],
    true_node: SddNode,
) -> MapResult:
    """Exact semantic MAP on a deterministic, decomposable SDD."""
    if node.is_false():
        return MapResult(-math.inf, {})
    if vtree.is_leaf():
        var = int(vtree.var())
        if node.is_true():
            if positive[var - 1] > negative[var - 1]:
                return MapResult(positive[var - 1], {var: 1})
            return MapResult(negative[var - 1], {var: 0})
        if not node.is_literal() or abs(int(node.literal)) != var:
            raise RuntimeError("Unexpected terminal/vtree pairing")
        value = int(node.literal > 0)
        weight = positive[var - 1] if value else negative[var - 1]
        return MapResult(weight, {var: value})
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
    if Vtree.is_sub(node_vtree, vtree.left()):
        return merge(
            max_product(node, vtree.left(), positive, negative, true_node),
            max_product(true_node, vtree.right(), positive, negative, true_node),
        )
    return merge(
        max_product(true_node, vtree.left(), positive, negative, true_node),
        max_product(node, vtree.right(), positive, negative, true_node),
    )


class DangerousGoodsCircuit:
    """Reusable SDD for the dangerous-goods SOP decision relation."""

    def __init__(self) -> None:
        self.groups: dict[str, dict[Any, int]] = {}
        next_var = 1
        self.groups["product_valid"] = {False: next_var, True: next_var + 1}
        next_var += 2
        for name in SCORE_NAMES:
            self.groups[name] = {value: next_var + value for value in SCORE_VALUES}
            next_var += len(SCORE_VALUES)
        self.groups["hazard_class"] = {
            value: next_var + index for index, value in enumerate(CLASS_VALUES)
        }
        next_var += len(CLASS_VALUES)
        self.var_count = next_var - 1

        start = time.perf_counter()
        self.vtree = Vtree(
            var_count=self.var_count,
            var_order=list(range(1, self.var_count + 1)),
            vtree_type="balanced",
        )
        self.manager = SddManager.from_vtree(self.vtree)
        root = self.manager.true()
        for group in self.groups.values():
            literals = [self.manager.literal(var) for var in group.values()]
            at_least_one = self.manager.false()
            for literal in literals:
                at_least_one |= literal
            root &= at_least_one
            for index, left in enumerate(literals):
                for right in literals[index + 1 :]:
                    root &= (~left) | (~right)

        for product_valid in (False, True):
            valid_lit = self.manager.literal(
                self.groups["product_valid"][product_valid]
            )
            for sds in SCORE_VALUES:
                for handling in SCORE_VALUES:
                    for transportation in SCORE_VALUES:
                        for disposal in SCORE_VALUES:
                            values = (sds, handling, transportation, disposal)
                            antecedent = valid_lit
                            for name, value in zip(SCORE_NAMES, values, strict=True):
                                antecedent &= self.manager.literal(
                                    self.groups[name][value]
                                )
                            label = class_for(values, product_valid)
                            consequent = self.manager.literal(
                                self.groups["hazard_class"][label]
                            )
                            root &= (~antecedent) | consequent

        if root.is_false():
            raise RuntimeError("Dangerous-goods constraints are unsatisfiable")
        root.ref()
        self.root = root
        self.compile_seconds = time.perf_counter() - start

    def save(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        self.root.save(str(ARTIFACT_DIR / "sop_dangerous_goods.sdd").encode())
        self.manager.vtree().save(
            str(ARTIFACT_DIR / "sop_dangerous_goods.vtree").encode()
        )

    def infer(
        self,
        local: dict[str, dict[Any, float]],
        *,
        product_valid_evidence: bool,
    ) -> dict[str, Any]:
        positive = [1.0] * self.var_count
        negative = [1.0] * self.var_count
        for name, group in self.groups.items():
            probabilities = normalize(local[name])
            for value, var in group.items():
                weight = probabilities[value]
                if name == "product_valid" and value != product_valid_evidence:
                    weight = 0.0
                positive[var - 1] = weight
                # Exactly-one categorical encoding: false literals have unit
                # weight, so a world receives only its selected category mass.
                negative[var - 1] = 1.0

        wmc = self.root.wmc(log_mode=False)
        for var in range(1, self.var_count + 1):
            wmc.set_literal_weight(var, positive[var - 1])
            wmc.set_literal_weight(-var, negative[var - 1])
        z_value = float(wmc.propagate())
        if z_value <= 0.0:
            raise RuntimeError("OpenJev assigns zero mass to the feasible set")

        marginals: dict[str, dict[Any, float]] = {}
        for name, group in self.groups.items():
            values = {
                value: float(wmc.literal_pr(var)) for value, var in group.items()
            }
            total = sum(values.values())
            marginals[name] = {value: mass / total for value, mass in values.items()}

        mpe = max_product(
            self.root,
            self.manager.vtree(),
            positive,
            negative,
            self.manager.true(),
        )
        joint_map: dict[str, Any] = {}
        for name, group in self.groups.items():
            selected = [value for value, var in group.items() if mpe.assignment[var] == 1]
            if len(selected) != 1:
                raise RuntimeError(f"Invalid MAP assignment for {name}: {selected}")
            joint_map[name] = selected[0]

        return {
            "z": z_value,
            "marginals": marginals,
            "joint_map": joint_map,
            "joint_map_unnormalized_probability": mpe.weight,
            "joint_map_probability": mpe.weight / z_value,
        }


def normalize(values: dict[Any, float]) -> dict[Any, float]:
    floored = {key: max(float(value), FLOOR) for key, value in values.items()}
    total = sum(floored.values())
    return {key: value / total for key, value in floored.items()}


def score_distribution(answer: dict[str, Any]) -> dict[int, float]:
    probabilities = answer["probabilities"]
    return normalize({int(index): probability for index, probability in probabilities.items()})


def local_distributions(result: dict[str, Any]) -> dict[str, dict[Any, float]]:
    answers = result["answers"]
    return {
        "product_valid": normalize(
            {False: 1.0 - float(answers["product_valid"]["noul"]),
             True: float(answers["product_valid"]["noul"])}
        ),
        "sds": score_distribution(answers["sds"]),
        "handling": score_distribution(answers["handling"]),
        "transportation": score_distribution(answers["transportation"]),
        "disposal": score_distribution(answers["disposal"]),
        "hazard_class": normalize(
            {
                value: float(answers["hazard_class"]["probabilities"][value])
                for value in CLASS_VALUES
            }
        ),
    }


def make_state(row: dict[str, str]) -> str:
    return (
        "Dangerous-goods classification record:\n"
        f"Product ID: {row['product_id']}\n"
        f"SDS Section 2: {row['sds_label_text']}\n"
        "Handling and storage guidelines: "
        f"{row['handling_and_storage_guidelines']}\n"
        f"Transportation requirements: {row['transportation_requirements']}\n"
        f"Disposal guidelines: {row['disposal_guidelines']}\n\n"
        "Domain policy: each component receives an integer severity score from "
        "1 (lowest) to 5 (highest); use 0 only when a component is missing. "
        "A valid product ID has format P_ followed by exactly five digits. "
        "One missing component is imputed with the maximum available component "
        "score; two or more missing components make the case Unable to Decide. "
        "The four class bands are A=4--7, B=8--12, C=13--16, and D=17--20."
    )


def questions() -> dict[str, Any]:
    scale = [
        "missing or unavailable",
        "severity 1 (lowest)",
        "severity 2",
        "severity 3",
        "severity 4",
        "severity 5 (highest)",
    ]
    return {
        "product_valid": Noul(
            instructions=(
                "Does the product ID exactly match P_ followed by five digits?"
            )
        ),
        "sds": Score(
            instructions="What is the SDS hazard-statement severity score?",
            criteria=scale,
        ),
        "handling": Score(
            instructions="What is the handling-and-storage severity score?",
            criteria=scale,
        ),
        "transportation": Score(
            instructions="What is the transportation severity score?",
            criteria=scale,
        ),
        "disposal": Score(
            instructions="What is the disposal severity score?",
            criteria=scale,
        ),
        "hazard_class": Choice(
            instructions="What is the final hazard class under the domain policy?",
            criteria={
                value: (
                    "Insufficient or invalid evidence"
                    if value == "Unable to Decide"
                    else f"Final classification {value[-1]}"
                )
                for value in CLASS_VALUES
            },
        ),
    }


def argmax(distribution: dict[Any, float]) -> Any:
    return max(distribution, key=distribution.__getitem__)


def assignment_from_distributions(
    distributions: dict[str, dict[Any, float]],
) -> dict[str, Any]:
    return {name: argmax(values) for name, values in distributions.items()}


def assignment_consistent(
    assignment: dict[str, Any],
    product_valid_evidence: bool | None = None,
) -> bool:
    product_valid = bool(assignment["product_valid"])
    if product_valid_evidence is not None and product_valid != product_valid_evidence:
        return False
    values = tuple(int(assignment[name]) for name in SCORE_NAMES)
    return assignment["hazard_class"] == class_for(
        values,
        product_valid if product_valid_evidence is None else product_valid_evidence,
    )


def gold_assignment(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"product_valid": valid_product_id(row["product_id"])}
    for name, column in zip(
        SCORE_NAMES,
        ("sds_label_score", "handling_score", "transportation_score", "disposal_score"),
        strict=True,
    ):
        text = row[column].strip()
        result[name] = 0 if not text else int(float(text))
    result["hazard_class"] = row["hazard_class"]
    return result


def multiclass_brier(distribution: dict[Any, float], gold: Any) -> float:
    return sum((probability - float(value == gold)) ** 2 for value, probability in distribution.items())


def audit_dataset(rows: list[dict[str, str]]) -> dict[str, Any]:
    mismatches = []
    observed_totals: dict[str, set[int]] = {label: set() for label in CLASS_VALUES}
    for index, row in enumerate(rows):
        gold = gold_assignment(row)
        values = tuple(gold[name] for name in SCORE_NAMES)
        derived_total = completed_score(values, gold["product_valid"])
        derived_class = class_for(values, gold["product_valid"])
        observed_totals[gold["hazard_class"]].add(derived_total)
        if derived_total != int(row["hazard_score"]) or derived_class != gold["hazard_class"]:
            mismatches.append(index)
    return {
        "rows": len(rows),
        "rule_mismatches": mismatches,
        "observed_totals_by_class": {
            label: sorted(values) for label, values in observed_totals.items()
        },
        "invalid_product_ids": sum(
            not valid_product_id(row["product_id"]) for row in rows
        ),
    }


def summarize(
    records: list[dict[str, Any]],
    circuit: DangerousGoodsCircuit,
    model_id: str,
) -> dict[str, Any]:
    n = len(records)
    local_conflicts = sum(not record["local_consistent"] for record in records)
    marginal_conflicts = sum(not record["marginal_consistent"] for record in records)
    summary: dict[str, Any] = {
        "samples": n,
        "model": model_id,
        "local_conflicts": local_conflicts,
        "conditioned_marginal_argmax_conflicts": marginal_conflicts,
        "joint_map_conflicts": 0,
        "local_conflicts_removed_by_joint_map_percent": (
            100.0 if local_conflicts else 0.0
        ),
        "mean_valid_mass_z": float(np.mean([r["z"] for r in records])),
        "median_valid_mass_z": float(np.median([r["z"] for r in records])),
        "min_valid_mass_z": float(np.min([r["z"] for r in records])),
        "circuit": {
            "variables": circuit.var_count,
            "satisfying_assignments": int(circuit.root.global_model_count()),
            "nodes": int(circuit.root.count()),
            "elements": int(circuit.root.size()),
            "compile_seconds": circuit.compile_seconds,
        },
    }
    for prediction in ("local", "marginal", "joint_map"):
        summary[f"{prediction}_hazard_class_accuracy"] = sum(
            record[prediction]["hazard_class"] == record["gold"]["hazard_class"]
            for record in records
        ) / n
        summary[f"{prediction}_joint_exact_accuracy"] = sum(
            all(record[prediction][name] == record["gold"][name] for name in record["gold"])
            for record in records
        ) / n
    summary["local_hazard_class_brier"] = float(
        np.mean([record["local_class_brier"] for record in records])
    )
    summary["conditioned_hazard_class_brier"] = float(
        np.mean([record["conditioned_class_brier"] for record in records])
    )
    summary["mean_openjev_latency_ms"] = float(
        np.mean([record["openjev_latency_ms"] for record in records])
    )
    summary["mean_circuit_inference_ms"] = float(
        np.mean([record["circuit_inference_ms"] for record in records])
    )
    return summary


def run(
    limit: int | None,
    offset: int,
    device: str | None,
    model_id: str = MODEL_ID,
    dtype: str = "float32",
) -> dict[str, Any]:
    with DATA_PATH.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    audit = audit_dataset(all_rows)
    if audit["rule_mismatches"]:
        raise AssertionError(f"SOP rule audit failed: {audit['rule_mismatches'][:10]}")
    rows = all_rows[offset : None if limit is None else offset + limit]
    if not rows:
        raise ValueError("No rows selected")

    circuit = DangerousGoodsCircuit()
    circuit.save()
    engine = LocalJev(model_id, device=device, dtype=dtype)
    typed_questions = questions()
    records: list[dict[str, Any]] = []
    for sample_index, row in enumerate(rows, start=offset):
        openjev_start = time.perf_counter()
        result = engine.system_one(make_state(row), typed_questions)
        openjev_ms = 1000.0 * (time.perf_counter() - openjev_start)
        local = local_distributions(result)
        local_prediction = assignment_from_distributions(local)

        circuit_start = time.perf_counter()
        conditioned = circuit.infer(
            local,
            product_valid_evidence=valid_product_id(row["product_id"]),
        )
        circuit_ms = 1000.0 * (time.perf_counter() - circuit_start)
        marginal_prediction = assignment_from_distributions(conditioned["marginals"])
        gold = gold_assignment(row)
        product_valid_evidence = valid_product_id(row["product_id"])
        records.append(
            {
                "sample_index": sample_index,
                "product_id": row["product_id"],
                "gold": gold,
                "local": local_prediction,
                "marginal": marginal_prediction,
                "joint_map": conditioned["joint_map"],
                "local_consistent": assignment_consistent(
                    local_prediction, product_valid_evidence
                ),
                "marginal_consistent": assignment_consistent(
                    marginal_prediction, product_valid_evidence
                ),
                "z": conditioned["z"],
                "local_class_brier": multiclass_brier(
                    local["hazard_class"], gold["hazard_class"]
                ),
                "conditioned_class_brier": multiclass_brier(
                    conditioned["marginals"]["hazard_class"], gold["hazard_class"]
                ),
                "local_distributions": local,
                "conditioned_marginals": conditioned["marginals"],
                "joint_map_probability": conditioned["joint_map_probability"],
                "openjev_latency_ms": openjev_ms,
                "circuit_inference_ms": circuit_ms,
            }
        )
        print(
            f"[{len(records):>3}/{len(rows)}] {row['product_id']}: "
            f"local={local_prediction['hazard_class']} "
            f"map={conditioned['joint_map']['hazard_class']} "
            f"gold={gold['hazard_class']} Z={conditioned['z']:.4f}",
            flush=True,
        )

    report = {
        "dataset": "Amazon SOP-Bench / dangerous_goods",
        "data_path": str(DATA_PATH),
        "dataset_audit": audit,
        "rule_provenance": {
            "sop_explicit": [
                "product ID must match P_ followed by five digits",
                "each component score is 1--5; zero denotes missing",
                "one missing component is replaced by the maximum available score",
                "two or more missing components imply score 0 and Unable to Decide",
                "hazard severity increases monotonically from class A to D",
            ],
            "benchmark_completion": [
                "A=4--7, B=8--12, C=13--16, D=17--20",
            ],
            "completion_note": (
                "Numeric class cutoffs are absent from sop.txt. The completion is "
                "monotone and reproduces every released row; totals 13 and 14 do "
                "not occur in the released dataset."
            ),
        },
        "summary": summarize(records, circuit, model_id),
        "records": records,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "sop_dangerous_goods_probjev.json",
    )
    args = parser.parse_args()
    report = run(args.limit, args.offset, args.device, args.model, args.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
