#!/usr/bin/env python3
"""Typed Decisions pilot for an exact probabilistic consistency layer.

The script accepts either JevAny prediction JSONL or the benchmark's soft-gold
distributions as a diagnostic source.  For each workflow it constructs the
independent base joint from the five typed marginals, conditions that joint on
frozen cross-question rules, and validates SDD Z, marginals, and MAP against
complete enumeration.

Soft gold is useful for auditing whether the proposed rules agree with the
benchmark.  It is not a model result; use ``--predictions`` for the actual
JevAny experiment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".deps"))

import pyarrow.parquet as pq
from pysdd.sdd import SddManager, SddNode, Vtree
from validate_sdd import max_product


@dataclass(frozen=True)
class Condition:
    variable: str
    allowed: frozenset[str]


@dataclass(frozen=True)
class Rule:
    name: str
    text: str
    antecedent: tuple[Condition, ...]
    consequent: Condition


def cond(variable: str, *allowed: str) -> Condition:
    return Condition(variable, frozenset(allowed))


# These rules are declared before test evaluation.  They encode contradictions
# between the meanings of the typed outputs.  Broader workflow preferences are
# deliberately excluded from the hard circuit.
RULES: dict[str, tuple[Rule, ...]] = {
    "agent_trace_observability": (
        Rule(
            "harmful_requires_review",
            "outcome=harmful -> needs_review=true",
            (cond("outcome", "harmful"),),
            cond("needs_review", "true"),
        ),
        Rule(
            "harmful_requires_intervention",
            "outcome=harmful -> action in {human_review, stop}",
            (cond("outcome", "harmful"),),
            cond("action", "human_review", "stop"),
        ),
        Rule(
            "stop_requires_review",
            "action=stop -> needs_review=true",
            (cond("action", "stop"),),
            cond("needs_review", "true"),
        ),
    ),
    "customer_service": (
        Rule(
            "escalation_requires_human",
            "action=escalate_to_human -> needs_human=true",
            (cond("action", "escalate_to_human"),),
            cond("needs_human", "true"),
        ),
    ),
    "invoice_processing": (
        Rule(
            "unreconciled_not_approved",
            "matches_order=false -> disposition != approve",
            (cond("matches_order", "false"),),
            cond("disposition", "hold", "manual_review", "reject"),
        ),
        Rule(
            "material_discrepancy_not_approved",
            "discrepancy_severity=3 -> disposition != approve",
            (cond("discrepancy_severity", "3"),),
            cond("disposition", "hold", "manual_review", "reject"),
        ),
    ),
    "security_incidents": (
        Rule(
            "compromise_is_positive",
            "credential_compromise=true -> true_positive=true",
            (cond("credential_compromise", "true"),),
            cond("true_positive", "true"),
        ),
        Rule(
            "contain_is_positive",
            "disposition=contain -> true_positive=true",
            (cond("disposition", "contain"),),
            cond("true_positive", "true"),
        ),
        Rule(
            "benign_close_is_negative",
            "disposition=close_benign -> true_positive=false",
            (cond("disposition", "close_benign"),),
            cond("true_positive", "false"),
        ),
        Rule(
            "contain_is_urgent",
            "disposition=contain -> urgency=3",
            (cond("disposition", "contain"),),
            cond("urgency", "3"),
        ),
    ),
}

EXCLUDED_CANDIDATES = {
    "invoice_processing": {
        "duplicate_not_approved": {
            "rule": "duplicate=true -> disposition != approve",
            "reason": (
                "Excluded from the hard circuit: the train soft-gold argmax violates it in "
                "8/28 duplicate-positive cases, so 'appears to duplicate' is not treated as a confirmed duplicate."
            ),
        }
    }
}


@dataclass
class Workflow:
    name: str
    variables: tuple[str, ...]
    domains: dict[str, tuple[str, ...]]
    offsets: dict[str, int]
    worlds: np.ndarray
    valid: np.ndarray
    rules: tuple[Rule, ...]

    @property
    def boolean_variables(self) -> int:
        return sum(len(self.domains[name]) for name in self.variables)

    def literal(self, variable: str, value: str) -> int:
        return self.offsets[variable] + self.domains[variable].index(value) + 1

    def labels(self, assignment: np.ndarray) -> dict[str, str]:
        return {
            variable: self.domains[variable][int(assignment[index])]
            for index, variable in enumerate(self.variables)
        }


def read_parquet(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return [
        {name: table[name][row].as_py() for name in table.column_names}
        for row in range(table.num_rows)
    ]


def question_domain(question: dict) -> tuple[str, ...]:
    if question["type"] == "choice":
        return tuple(question["criteria"])
    if question["type"] == "noul":
        return ("false", "true")
    if question["type"] == "score":
        return tuple(str(index) for index in range(len(question["criteria"])))
    raise ValueError(f"Unknown question type: {question['type']}")


def violates(labels: dict[str, str], rule: Rule) -> bool:
    antecedent = all(labels[item.variable] in item.allowed for item in rule.antecedent)
    return (
        antecedent and labels[rule.consequent.variable] not in rule.consequent.allowed
    )


def build_workflow(row: dict) -> Workflow:
    questions = json.loads(row["questions"])
    variables = tuple(questions)
    domains = {name: question_domain(questions[name]) for name in variables}
    offsets: dict[str, int] = {}
    offset = 0
    for name in variables:
        offsets[name] = offset
        offset += len(domains[name])
    worlds = np.asarray(
        list(itertools.product(*(range(len(domains[name])) for name in variables))),
        dtype=np.int8,
    )
    rules = RULES[row["workflow"]]
    valid = np.ones(len(worlds), dtype=bool)
    provisional = Workflow(
        row["workflow"], variables, domains, offsets, worlds, valid, rules
    )
    for index, world in enumerate(worlds):
        labels = provisional.labels(world)
        valid[index] = not any(violates(labels, rule) for rule in rules)
    return provisional


def gold_probabilities(row: dict, workflow: Workflow) -> list[np.ndarray]:
    gold = json.loads(row["gold"])
    return [
        np.asarray(
            [
                float(gold[name]["probabilities"][value])
                for value in workflow.domains[name]
            ],
            dtype=float,
        )
        for name in workflow.variables
    ]


def answer_probabilities(answer: dict, domain: tuple[str, ...]) -> np.ndarray:
    if answer["type"] == "noul":
        positive = float(answer["noul"])
        result = np.asarray([1.0 - positive, positive], dtype=float)
    else:
        result = np.asarray(
            [float(answer["probabilities"][value]) for value in domain], dtype=float
        )
    total = float(result.sum())
    if total <= 0:
        raise ValueError("zero-mass prediction")
    return result / total


def load_predictions(path: Path) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("status", "ok") != "ok":
                continue
            row_id = entry.get("id") or entry.get("record_id")
            response = entry.get("response") or entry.get("prediction") or entry
            answers = response.get("answers")
            if row_id is None or answers is None:
                raise ValueError("prediction JSONL needs an id and response.answers")
            predictions[str(row_id)] = answers
    return predictions


def prediction_probabilities(answers: dict, workflow: Workflow) -> list[np.ndarray]:
    return [
        answer_probabilities(answers[name], workflow.domains[name])
        for name in workflow.variables
    ]


def normalize(probabilities: Iterable[np.ndarray]) -> list[np.ndarray]:
    result = []
    for probability in probabilities:
        probability = np.asarray(probability, dtype=float)
        total = float(probability.sum())
        if total <= 0 or np.any(probability < 0) or np.any(~np.isfinite(probability)):
            raise ValueError("invalid local distribution")
        result.append(probability / total)
    return result


def world_weights(workflow: Workflow, probabilities: list[np.ndarray]) -> np.ndarray:
    weights = np.ones(len(workflow.worlds), dtype=float)
    for index, probability in enumerate(probabilities):
        weights *= probability[workflow.worlds[:, index]]
    return weights


def enumerate_inference(
    workflow: Workflow, probabilities: list[np.ndarray]
) -> tuple[float, list[np.ndarray], np.ndarray, float, int]:
    weights = world_weights(workflow, probabilities)
    constrained = np.where(workflow.valid, weights, 0.0)
    z = float(constrained.sum())
    if z <= 0:
        raise ValueError(f"zero valid mass for {workflow.name}")
    q = constrained / z
    marginals = [
        np.asarray(
            [
                q[workflow.worlds[:, index] == value].sum()
                for value in range(len(probability))
            ]
        )
        for index, probability in enumerate(probabilities)
    ]
    best = float(constrained.max())
    tied = np.flatnonzero(np.isclose(constrained, best, rtol=0.0, atol=1e-15))
    return z, marginals, workflow.worlds[int(tied[0])], best, len(tied)


def event_node(
    manager: SddManager, workflow: Workflow, condition: Condition
) -> SddNode:
    node = manager.false()
    for value in workflow.domains[condition.variable]:
        if value in condition.allowed:
            node |= manager.literal(workflow.literal(condition.variable, value))
    return node


def compile_sdd(
    workflow: Workflow, artifact_dir: Path
) -> tuple[SddManager, SddNode, dict]:
    started = time.perf_counter()
    variables = workflow.boolean_variables
    vtree = Vtree(
        var_count=variables,
        var_order=list(range(1, variables + 1)),
        vtree_type="balanced",
    )
    manager = SddManager.from_vtree(vtree)
    root = manager.true()

    for name in workflow.variables:
        literals = [
            manager.literal(workflow.literal(name, value))
            for value in workflow.domains[name]
        ]
        at_least_one = manager.false()
        for literal in literals:
            at_least_one |= literal
        root &= at_least_one
        for left_index, left in enumerate(literals):
            for right in literals[left_index + 1 :]:
                root &= (~left) | (~right)

    for rule in workflow.rules:
        antecedent = manager.true()
        for condition in rule.antecedent:
            antecedent &= event_node(manager, workflow, condition)
        consequent = event_node(manager, workflow, rule.consequent)
        root &= (~antecedent) | consequent

    root.ref()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sdd_path = artifact_dir / f"typed_decisions_{workflow.name}.sdd"
    vtree_path = artifact_dir / f"typed_decisions_{workflow.name}.vtree"
    root.save(str(sdd_path).encode())
    manager.vtree().save(str(vtree_path).encode())
    info = {
        "boolean_variables": variables,
        "typed_assignments": len(workflow.worlds),
        "valid_assignments": int(workflow.valid.sum()),
        "node_count": int(root.count()),
        "size_elements": int(root.size()),
        "compile_seconds": time.perf_counter() - started,
        "sdd_path": str(sdd_path),
        "vtree_path": str(vtree_path),
    }
    return manager, root, info


def circuit_inference(
    workflow: Workflow,
    manager: SddManager,
    root: SddNode,
    probabilities: list[np.ndarray],
) -> tuple[float, list[np.ndarray], np.ndarray, float]:
    positive = np.ones(workflow.boolean_variables, dtype=float)
    negative = np.ones(workflow.boolean_variables, dtype=float)
    wmc = root.wmc(log_mode=False)
    for name, probability in zip(workflow.variables, probabilities):
        for value, weight in zip(workflow.domains[name], probability):
            literal = workflow.literal(name, value)
            positive[literal - 1] = float(weight)
            wmc.set_literal_weight(literal, float(weight))
            wmc.set_literal_weight(-literal, 1.0)
    z = float(wmc.propagate())
    marginals = [
        np.asarray(
            [
                float(wmc.literal_pr(workflow.literal(name, value)))
                for value in workflow.domains[name]
            ]
        )
        for name in workflow.variables
    ]
    mpe = max_product(root, manager.vtree(), positive, negative, manager.true())
    assignment = np.empty(len(workflow.variables), dtype=np.int8)
    for index, name in enumerate(workflow.variables):
        active = [
            value_index
            for value_index, value in enumerate(workflow.domains[name])
            if mpe.assignment[workflow.literal(name, value)] == 1
        ]
        if len(active) != 1:
            raise AssertionError(
                f"non-one-hot circuit MAP for {workflow.name}/{name}: {active}"
            )
        assignment[index] = active[0]
    return z, marginals, assignment, float(mpe.weight)


def argmax_assignment(probabilities: list[np.ndarray]) -> np.ndarray:
    return np.asarray(
        [int(np.argmax(probability)) for probability in probabilities], dtype=np.int8
    )


def assignment_violations(workflow: Workflow, assignment: np.ndarray) -> list[str]:
    labels = workflow.labels(assignment)
    return [rule.name for rule in workflow.rules if violates(labels, rule)]


def gold_labels(row: dict, workflow: Workflow) -> np.ndarray:
    gold = json.loads(row["gold"])
    return np.asarray(
        [
            workflow.domains[name].index(str(gold[name]["label"]).lower())
            for name in workflow.variables
        ],
        dtype=np.int8,
    )


def distribution_metrics(
    gold: list[np.ndarray], predicted: list[np.ndarray], labels: np.ndarray
) -> dict[str, float]:
    accuracy = np.mean(
        [int(np.argmax(p)) == int(label) for p, label in zip(predicted, labels)]
    )
    kl = np.mean(
        [
            float(
                np.sum(
                    g
                    * (np.log(np.clip(g, 1e-12, 1.0)) - np.log(np.clip(p, 1e-12, 1.0)))
                )
            )
            for g, p in zip(gold, predicted)
        ]
    )
    brier = np.mean([float(np.sum((p - g) ** 2)) for g, p in zip(gold, predicted)])
    return {
        "accuracy": float(accuracy),
        "kl_from_soft_gold": float(kl),
        "brier": float(brier),
    }


def audit_rules(rows: list[dict], workflow: Workflow) -> dict:
    audit = {
        rule.name: {"rule": rule.text, "violations": 0, "antecedent": 0}
        for rule in workflow.rules
    }
    for row in rows:
        labels = workflow.labels(gold_labels(row, workflow))
        for rule in workflow.rules:
            antecedent = all(
                labels[item.variable] in item.allowed for item in rule.antecedent
            )
            audit[rule.name]["antecedent"] += int(antecedent)
            audit[rule.name]["violations"] += int(violates(labels, rule))
    for details in audit.values():
        details["conditional_violation_rate"] = (
            details["violations"] / details["antecedent"]
            if details["antecedent"]
            else 0.0
        )
    return audit


def summarize_workflow(records: list[dict], workflow: Workflow) -> dict:
    result: dict[str, float | int] = {}
    for key in (
        "raw_accuracy",
        "conditioned_accuracy",
        "map_accuracy",
        "raw_kl",
        "conditioned_kl",
        "raw_brier",
        "conditioned_brier",
    ):
        result[key] = float(np.mean([record[key] for record in records]))
    result.update(
        {
            "examples": len(records),
            "raw_conflicting_examples": int(
                sum(record["raw_violations"] > 0 for record in records)
            ),
            "conditioned_conflicting_examples": int(
                sum(record["conditioned_violations"] > 0 for record in records)
            ),
            "map_conflicting_examples": int(
                sum(record["map_violations"] > 0 for record in records)
            ),
            "conflicts_removed_percent": float(
                100.0
                * sum(
                    record["raw_violations"] > 0 and record["map_violations"] == 0
                    for record in records
                )
                / max(1, sum(record["raw_violations"] > 0 for record in records))
            ),
            "mean_valid_mass": float(np.mean([record["z"] for record in records])),
            "min_valid_mass": float(np.min([record["z"] for record in records])),
            "valid_mass_quantiles": {
                str(quantile): float(
                    np.quantile([record["z"] for record in records], quantile)
                )
                for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
            "joint_exact_match": {
                "raw": float(
                    np.mean([record["raw_joint_correct"] for record in records])
                ),
                "conditioned_marginal": float(
                    np.mean([record["conditioned_joint_correct"] for record in records])
                ),
                "joint_map": float(
                    np.mean([record["map_joint_correct"] for record in records])
                ),
            },
            "changed_heads": {
                "conditioned_marginal": int(
                    sum(record["conditioned_changed_heads"] for record in records)
                ),
                "joint_map": int(
                    sum(record["map_changed_heads"] for record in records)
                ),
            },
        }
    )
    result["per_head"] = {}
    for index, name in enumerate(workflow.variables):
        raw = np.asarray([record["raw_assignment"][index] for record in records])
        conditioned = np.asarray(
            [record["conditioned_assignment"][index] for record in records]
        )
        joint_map = np.asarray([record["map_assignment"][index] for record in records])
        gold = np.asarray([record["gold_assignment"][index] for record in records])
        result["per_head"][name] = {
            "raw_accuracy": float(np.mean(raw == gold)),
            "conditioned_accuracy": float(np.mean(conditioned == gold)),
            "map_accuracy": float(np.mean(joint_map == gold)),
            "conditioned_fixes": int(np.sum((raw != gold) & (conditioned == gold))),
            "conditioned_regressions": int(
                np.sum((raw == gold) & (conditioned != gold))
            ),
            "map_fixes": int(np.sum((raw != gold) & (joint_map == gold))),
            "map_regressions": int(np.sum((raw == gold) & (joint_map != gold))),
        }
    return result


def aggregate_metrics(workflows: dict[str, dict]) -> dict:
    records = [
        record for details in workflows.values() for record in details["records"]
    ]
    total = len(records)
    scalar_keys = (
        "raw_accuracy",
        "conditioned_accuracy",
        "map_accuracy",
        "raw_kl",
        "conditioned_kl",
        "raw_brier",
        "conditioned_brier",
    )
    result = {
        key: float(np.mean([record[key] for record in records])) for key in scalar_keys
    }
    raw_conflicts = int(sum(record["raw_violations"] > 0 for record in records))
    result.update(
        {
            "examples": total,
            "decisions": int(total * 5),
            "raw_conflicting_examples": raw_conflicts,
            "conditioned_conflicting_examples": int(
                sum(record["conditioned_violations"] > 0 for record in records)
            ),
            "map_conflicting_examples": int(
                sum(record["map_violations"] > 0 for record in records)
            ),
            "conflicts_removed_percent": float(
                100.0
                * sum(
                    record["raw_violations"] > 0 and record["map_violations"] == 0
                    for record in records
                )
                / max(1, raw_conflicts)
            ),
            "mean_valid_mass": float(np.mean([record["z"] for record in records])),
            "min_valid_mass": float(np.min([record["z"] for record in records])),
            "joint_exact_match": {
                "raw": float(
                    np.mean([record["raw_joint_correct"] for record in records])
                ),
                "conditioned_marginal": float(
                    np.mean([record["conditioned_joint_correct"] for record in records])
                ),
                "joint_map": float(
                    np.mean([record["map_joint_correct"] for record in records])
                ),
            },
        }
    )
    z = np.asarray([record["z"] for record in records])
    raw_accuracy = np.asarray([record["raw_accuracy"] for record in records])
    raw_joint = np.asarray(
        [record["raw_joint_correct"] for record in records], dtype=float
    )
    result["valid_mass_diagnostic"] = {
        "pearson_with_per_example_head_accuracy": float(
            np.corrcoef(z, raw_accuracy)[0, 1]
        ),
        "pearson_with_joint_exact_match": float(np.corrcoef(z, raw_joint)[0, 1]),
        "risk_coverage": {},
    }
    order = np.argsort(-z)
    for coverage in (0.25, 0.5, 0.75, 1.0):
        selected = order[: max(1, int(total * coverage))]
        result["valid_mass_diagnostic"]["risk_coverage"][str(coverage)] = {
            "head_accuracy": float(raw_accuracy[selected].mean()),
            "joint_exact_match": float(raw_joint[selected].mean()),
            "minimum_z": float(z[selected].min()),
        }
    return result


def run(
    train_path: Path,
    test_path: Path,
    predictions_path: Path | None,
    artifact_dir: Path,
    allow_partial_predictions: bool = False,
) -> dict:
    train_rows = read_parquet(train_path)
    test_rows = read_parquet(test_path)
    predictions = (
        None if predictions_path is None else load_predictions(predictions_path)
    )
    if predictions is not None:
        test_ids = {row["id"] for row in test_rows}
        unexpected = sorted(set(predictions) - test_ids)
        if unexpected:
            raise KeyError(f"predictions contain unknown IDs, first: {unexpected[0]}")
        missing = sorted(test_ids - set(predictions))
        if missing and not allow_partial_predictions:
            raise KeyError(
                f"missing {len(missing)} predictions, first: {missing[0]}; "
                "use --allow-partial-predictions only for a pilot subset"
            )
        if allow_partial_predictions:
            test_rows = [row for row in test_rows if row["id"] in predictions]
    source = (
        "teacher_soft_gold_diagnostic" if predictions is None else str(predictions_path)
    )
    output = {
        "setup": {
            "dataset": "LocalLLaMA/typed-decisions",
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "partial_predictions": bool(
                predictions is not None and allow_partial_predictions
            ),
            "probability_source": source,
            "base_joint": "product of the five local typed distributions",
        },
        "workflows": {},
    }

    for workflow_name in sorted(RULES):
        train = [row for row in train_rows if row["workflow"] == workflow_name]
        test = [row for row in test_rows if row["workflow"] == workflow_name]
        if not test:
            continue
        workflow = build_workflow(test[0])
        manager, root, circuit = compile_sdd(workflow, artifact_dir)
        records = []
        errors = {
            "z": 0.0,
            "marginal": 0.0,
            "map_weight": 0.0,
            "unique_map_mismatch": 0,
        }
        started = time.perf_counter()

        for row in test:
            gold = normalize(gold_probabilities(row, workflow))
            if predictions is None:
                local = [probability.copy() for probability in gold]
            else:
                if row["id"] not in predictions:
                    raise KeyError(f"missing prediction for {row['id']}")
                local = normalize(
                    prediction_probabilities(predictions[row["id"]], workflow)
                )

            enum_z, enum_marginals, enum_map, enum_weight, ties = enumerate_inference(
                workflow, local
            )
            circuit_z, marginals, joint_map, circuit_weight = circuit_inference(
                workflow, manager, root, local
            )
            errors["z"] = max(errors["z"], abs(circuit_z - enum_z))
            errors["marginal"] = max(
                errors["marginal"],
                max(
                    float(np.max(np.abs(left - right)))
                    for left, right in zip(marginals, enum_marginals)
                ),
            )
            errors["map_weight"] = max(
                errors["map_weight"], abs(circuit_weight - enum_weight)
            )
            if ties == 1 and not np.array_equal(joint_map, enum_map):
                errors["unique_map_mismatch"] += 1

            labels = gold_labels(row, workflow)
            raw = argmax_assignment(local)
            conditioned = argmax_assignment(marginals)
            raw_metrics = distribution_metrics(gold, local, labels)
            conditioned_metrics = distribution_metrics(gold, marginals, labels)
            records.append(
                {
                    "id": row["id"],
                    "z": circuit_z,
                    "raw_accuracy": raw_metrics["accuracy"],
                    "conditioned_accuracy": conditioned_metrics["accuracy"],
                    "map_accuracy": float(np.mean(joint_map == labels)),
                    "raw_kl": raw_metrics["kl_from_soft_gold"],
                    "conditioned_kl": conditioned_metrics["kl_from_soft_gold"],
                    "raw_brier": raw_metrics["brier"],
                    "conditioned_brier": conditioned_metrics["brier"],
                    "raw_violations": len(assignment_violations(workflow, raw)),
                    "conditioned_violations": len(
                        assignment_violations(workflow, conditioned)
                    ),
                    "map_violations": len(assignment_violations(workflow, joint_map)),
                    "gold_assignment": labels.tolist(),
                    "raw_assignment": raw.tolist(),
                    "conditioned_assignment": conditioned.tolist(),
                    "map_assignment": joint_map.tolist(),
                    "raw_joint_correct": bool(np.all(raw == labels)),
                    "conditioned_joint_correct": bool(np.all(conditioned == labels)),
                    "map_joint_correct": bool(np.all(joint_map == labels)),
                    "conditioned_changed_heads": int(np.sum(raw != conditioned)),
                    "map_changed_heads": int(np.sum(raw != joint_map)),
                }
            )

        inference_seconds = time.perf_counter() - started
        tolerance = 1e-11
        if max(errors["z"], errors["marginal"], errors["map_weight"]) > tolerance:
            raise AssertionError(
                f"SDD/enumeration mismatch for {workflow_name}: {errors}"
            )
        if errors["unique_map_mismatch"]:
            raise AssertionError(f"unique MAP mismatch for {workflow_name}: {errors}")

        output["workflows"][workflow_name] = {
            "variables": list(workflow.variables),
            "domains": {
                name: list(values) for name, values in workflow.domains.items()
            },
            "rules": [rule.text for rule in workflow.rules],
            "excluded_candidate_rules": EXCLUDED_CANDIDATES.get(workflow_name, {}),
            "train_argmax_rule_audit": audit_rules(train, workflow),
            "test_argmax_rule_audit": audit_rules(test, workflow),
            "circuit": {
                **circuit,
                "milliseconds_per_sample": 1000.0 * inference_seconds / len(test),
            },
            "exactness": {"status": "pass", **errors, "tolerance": tolerance},
            "metrics": summarize_workflow(records, workflow),
            "records": records,
        }
        root.deref()

    output["aggregate"] = aggregate_metrics(output["workflows"])

    return output


def markdown_report(result: dict) -> str:
    lines = [
        "# Typed Decisions SDD pilot",
        "",
        f"Probability source: `{result['setup']['probability_source']}`.",
        "",
        "| Workflow | Raw conflicts | Marginal conflicts | MAP conflicts | Removed | Mean $Z$ | Raw acc. | Marg. acc. | MAP acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, details in result["workflows"].items():
        metric = details["metrics"]
        lines.append(
            f"| {name} | {metric['raw_conflicting_examples']} | "
            f"{metric['conditioned_conflicting_examples']} | {metric['map_conflicting_examples']} | "
            f"{metric['conflicts_removed_percent']:.1f}\\% | {metric['mean_valid_mass']:.4f} | "
            f"{metric['raw_accuracy']:.4f} | {metric['conditioned_accuracy']:.4f} | {metric['map_accuracy']:.4f} |"
        )
    aggregate = result["aggregate"]
    lines.append(
        f"| **All** | **{aggregate['raw_conflicting_examples']}** | "
        f"**{aggregate['conditioned_conflicting_examples']}** | **{aggregate['map_conflicting_examples']}** | "
        f"**{aggregate['conflicts_removed_percent']:.1f}\\%** | **{aggregate['mean_valid_mass']:.4f}** | "
        f"**{aggregate['raw_accuracy']:.4f}** | **{aggregate['conditioned_accuracy']:.4f}** | "
        f"**{aggregate['map_accuracy']:.4f}** |"
    )
    lines.append("")
    if result["setup"]["probability_source"] == "teacher_soft_gold_diagnostic":
        lines.append(
            "The soft-gold mode is a constraint audit and circuit correctness test, not a model evaluation."
        )
    elif result["setup"]["partial_predictions"]:
        lines.append(
            "This is a partial model pilot; it is not the final 400-case benchmark result."
        )
    lines.extend(
        [
            "Every workflow passed sample-wise agreement between SDD inference and exhaustive enumeration.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, default=Path("tmp/typed-decisions-review/train.parquet")
    )
    parser.add_argument(
        "--test", type=Path, default=Path("tmp/typed-decisions-review/test.parquet")
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--allow-partial-predictions", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=Path("pilot/artifacts"))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("pilot/results/typed_decisions_sdd_pilot.json"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("pilot/results/typed_decisions_sdd_pilot.md"),
    )
    args = parser.parse_args()
    result = run(
        args.train,
        args.test,
        args.predictions,
        args.artifact_dir,
        allow_partial_predictions=args.allow_partial_predictions,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(markdown_report(result), encoding="utf-8")
    print(markdown_report(result))


if __name__ == "__main__":
    main()
