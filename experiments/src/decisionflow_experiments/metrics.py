"""Model-agnostic summaries for cached DecisionFlow experiment outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence


def _lookup(row: Mapping[Any, float], value: Any) -> float:
    if value in row:
        return float(row[value])
    variants = [str(value), str(value).lower()]
    if isinstance(value, bool):
        variants.extend(["true" if value else "false", "True" if value else "False"])
    for key in variants:
        if key in row:
            return float(row[key])
    raise KeyError("probability row has no value %r" % (value,))


def _coerce(value: Any, options: Sequence[Any]) -> Any:
    if value in options:
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true" and True in options:
            return True
        if lowered == "false" and False in options:
            return False
        for option in options:
            if str(option) == value:
                return option
    return value


def _argmax(row: Mapping[Any, float], options: Sequence[Any]) -> Any:
    return max(options, key=lambda option: _lookup(row, option))


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2.0 * tp / denominator


def summarize(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate common probability, prediction, and consistency metrics.

    Dataset-specific papers may add metrics on top of this report. The common
    report is intentionally derived only from question types, probabilities,
    gold values, and DecisionFlow diagnostics.
    """
    records = list(rows)
    if not records:
        raise ValueError("cannot summarize an empty result set")
    decisions = 0
    samples_with_gold = 0
    raw_correct = marginal_correct = joint_correct = 0
    raw_joint = marginal_joint = constrained_joint = 0
    nll = brier = 0.0
    score_abs_error = 0.0
    score_count = 0
    boolean_counts = {
        "raw": defaultdict(lambda: [0, 0, 0]),
        "marginal": defaultdict(lambda: [0, 0, 0]),
        "joint": defaultdict(lambda: [0, 0, 0]),
    }
    raw_conflict_samples = marginal_conflict_samples = 0
    valid_mass = []

    for record in records:
        valid_mass.append(float(record["valid_mass"]))
        diagnostics = record.get("diagnostics", {})
        raw_map = diagnostics.get("raw_local_map", {})
        marginal_map = diagnostics.get("marginal_map", {})
        joint_map = record["joint_map"]
        raw_conflict_samples += bool(diagnostics.get("raw_local_map_violations"))
        marginal_conflict_samples += bool(diagnostics.get("marginal_map_violations"))
        gold = record.get("gold")
        if not isinstance(gold, Mapping):
            continue
        samples_with_gold += 1
        raw_all = marginal_all = joint_all = True
        for question in record["questions"]:
            name = question["id"]
            if name not in gold:
                continue
            options = question["options"]
            truth = _coerce(gold[name], options)
            raw = _coerce(raw_map[name], options)
            marginal = _coerce(marginal_map[name], options)
            joint = _coerce(joint_map[name], options)
            raw_hit, marginal_hit, joint_hit = raw == truth, marginal == truth, joint == truth
            raw_correct += raw_hit
            marginal_correct += marginal_hit
            joint_correct += joint_hit
            raw_all &= raw_hit
            marginal_all &= marginal_hit
            joint_all &= joint_hit
            decisions += 1

            posterior = record["marginals"][name]
            gold_probability = max(_lookup(posterior, truth), 1e-15)
            nll -= math.log(gold_probability)
            brier += sum(
                (_lookup(posterior, option) - float(option == truth)) ** 2
                for option in options
            )

            if question["type"] in {"score", "ordinal"} and all(
                isinstance(option, (int, float)) and not isinstance(option, bool)
                for option in options
            ):
                expectation = sum(float(option) * _lookup(posterior, option) for option in options)
                score_abs_error += abs(expectation - float(truth))
                score_count += 1

            if question["type"] in {"noul", "boolean"}:
                for label, prediction in (("raw", raw), ("marginal", marginal), ("joint", joint)):
                    counts = boolean_counts[label][name]
                    counts[0] += bool(prediction is True and truth is True)
                    counts[1] += bool(prediction is True and truth is False)
                    counts[2] += bool(prediction is False and truth is True)
        raw_joint += raw_all
        marginal_joint += marginal_all
        constrained_joint += joint_all

    def accuracy(correct: int) -> float | None:
        return None if decisions == 0 else correct / decisions

    def macro_f1(kind: str) -> float | None:
        values = [_f1(*counts) for counts in boolean_counts[kind].values()]
        return None if not values else sum(values) / len(values)

    return {
        "records": len(records),
        "records_with_gold": samples_with_gold,
        "decisions_with_gold": decisions,
        "prediction": {
            "raw_local_accuracy": accuracy(raw_correct),
            "marginal_accuracy": accuracy(marginal_correct),
            "joint_map_accuracy": accuracy(joint_correct),
            "raw_joint_exact_match": None if not samples_with_gold else raw_joint / samples_with_gold,
            "marginal_joint_exact_match": None if not samples_with_gold else marginal_joint / samples_with_gold,
            "joint_map_exact_match": None if not samples_with_gold else constrained_joint / samples_with_gold,
            "raw_boolean_macro_f1": macro_f1("raw"),
            "marginal_boolean_macro_f1": macro_f1("marginal"),
            "joint_map_boolean_macro_f1": macro_f1("joint"),
        },
        "probability": {
            "marginal_nll": None if decisions == 0 else nll / decisions,
            "multiclass_brier": None if decisions == 0 else brier / decisions,
            "score_mae_from_expectation": None if score_count == 0 else score_abs_error / score_count,
        },
        "consistency": {
            "raw_conflicting_records": raw_conflict_samples,
            "marginal_conflicting_records": marginal_conflict_samples,
            "joint_map_conflicting_records": 0,
            "raw_conflicts_removed_fraction": (
                None
                if raw_conflict_samples == 0
                else (raw_conflict_samples - marginal_conflict_samples) / raw_conflict_samples
            ),
            "mean_valid_mass": sum(valid_mass) / len(valid_mass),
            "min_valid_mass": min(valid_mass),
            "max_valid_mass": max(valid_mass),
        },
    }
