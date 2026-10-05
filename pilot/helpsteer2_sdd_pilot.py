#!/usr/bin/env python3
"""Exact hard- and soft-constraint SDD pilot for HelpSteer2's five Scores."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".deps"))

from pysdd.sdd import SddManager, SddNode, Vtree  # noqa: E402

from validate_sdd import max_product  # noqa: E402


MODEL = "jev-1.13.0"
HEADS = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
H, C, K, X, V = range(5)
LEVELS = np.arange(5)

ATTRIBUTES = {
    "helpfulness": (
        "Overall helpfulness of `response` to `prompt`.",
        [
            "Not helpful at all.",
            "Slightly helpful.",
            "Moderately helpful.",
            "Very helpful.",
            "Extremely helpful.",
        ],
    ),
    "correctness": (
        "Inclusion of all pertinent facts in `response`, without errors.",
        [
            "Mostly incorrect or missing key facts.",
            "Several errors or omissions.",
            "Some errors or omissions.",
            "Minor errors or omissions.",
            "Fully correct and complete.",
        ],
    ),
    "coherence": (
        "Consistency and clarity of expression in `response`.",
        [
            "Incoherent.",
            "Often unclear or inconsistent.",
            "Somewhat clear.",
            "Mostly clear and consistent.",
            "Perfectly clear and consistent.",
        ],
    ),
    "complexity": (
        "Intellectual depth required to write `response`.",
        [
            "Basic: anyone who speaks the language could write it.",
            "Simple: needs only everyday knowledge.",
            "Intermediate: needs some education in the topic.",
            "Advanced: needs expertise in the field.",
            "Expert: needs deep domain expertise.",
        ],
    ),
    "verbosity": (
        "Amount of detail in `response`, relative to what `prompt` asks for.",
        ["Very succinct.", "Succinct.", "Moderate length.", "Verbose.", "Very verbose."],
    ),
}


@dataclass(frozen=True)
class Rule:
    name: str
    text: str
    antecedent: Callable[[np.ndarray], np.ndarray]
    consequent: Callable[[np.ndarray], np.ndarray]

    def violation(self, values: np.ndarray) -> np.ndarray:
        return self.antecedent(values) & ~self.consequent(values)


HARD_RULES = (
    Rule("h3_c1", "Helpfulness >= 3 -> Correctness >= 1", lambda y: y[..., H] >= 3, lambda y: y[..., C] >= 1),
    Rule("h3_k1", "Helpfulness >= 3 -> Coherence >= 1", lambda y: y[..., H] >= 3, lambda y: y[..., K] >= 1),
    Rule("h4_k2", "Helpfulness >= 4 -> Coherence >= 2", lambda y: y[..., H] >= 4, lambda y: y[..., K] >= 2),
)

SOFT_RULES = (
    Rule("h3_c2", "Helpfulness >= 3 -> Correctness >= 2", lambda y: y[..., H] >= 3, lambda y: y[..., C] >= 2),
    Rule("h3_k2", "Helpfulness >= 3 -> Coherence >= 2", lambda y: y[..., H] >= 3, lambda y: y[..., K] >= 2),
    Rule("c4_h3", "Correctness >= 4 -> Helpfulness >= 3", lambda y: y[..., C] >= 4, lambda y: y[..., H] >= 3),
)


def questions() -> dict[str, dict]:
    return {
        head: {"type": "score", "instructions": instruction, "criteria": levels}
        for head, (instruction, levels) in ATTRIBUTES.items()
    }


def request_key(state: dict, qs: dict) -> str:
    payload = json.dumps({"model": MODEL, "state": state, "questions": qs}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_gold(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_parquet(path)
    return frame, frame[list(HEADS)].to_numpy(dtype=np.int8)


def load_eval(path: Path, db_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    frame, gold = load_gold(path)
    qs = questions()
    probs: list[list[list[float]]] = []
    db = sqlite3.connect(db_path)
    missing: list[str] = []
    try:
        for row in frame.itertuples(index=False):
            state = {"prompt": row.prompt, "response": row.response}
            key = request_key(state, qs)
            cached = db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            if cached is None:
                missing.append(key)
                continue
            answers = json.loads(cached[0])["answers"]
            probs.append(
                [
                    [float(answers[head]["probabilities"][str(level)]) for level in range(5)]
                    for head in HEADS
                ]
            )
    finally:
        db.close()
    if missing:
        raise RuntimeError(f"Missing {len(missing)} cached responses; first key: {missing[0]}")
    out = np.asarray(probs, dtype=float)
    sums = out.sum(axis=2)
    if np.any(sums <= 0):
        raise AssertionError("Jev returned a zero-mass Score distribution")
    normalization = {
        "distributions_renormalized": int(np.sum(np.abs(sums - 1.0) > 1e-12)),
        "max_pre_normalization_sum_error": float(np.max(np.abs(sums - 1.0))),
    }
    out /= sums[:, :, None]
    return out, gold, normalization


def audit_rules(values: np.ndarray, rules: tuple[Rule, ...]) -> dict:
    audit = {}
    for rule in rules:
        antecedent = rule.antecedent(values)
        violations = rule.violation(values)
        audit[rule.name] = {
            "rule": rule.text,
            "antecedent_count": int(antecedent.sum()),
            "violation_count": int(violations.sum()),
            "conditional_violation_rate": float(
                violations.sum() / antecedent.sum() if antecedent.any() else 0.0
            ),
        }
    return audit


def estimate_penalties(train_gold: np.ndarray) -> tuple[np.ndarray, dict]:
    """Laplace-smoothed violation odds, estimated only from the training split."""
    penalties, details = [], {}
    for rule in SOFT_RULES:
        antecedent = rule.antecedent(train_gold)
        violations = rule.violation(train_gold)
        n = int(antecedent.sum())
        v = int(violations.sum())
        penalty = min(1.0, (v + 1.0) / (n - v + 1.0))
        penalties.append(penalty)
        details[rule.name] = {
            "rule": rule.text,
            "antecedent_count": n,
            "violation_count": v,
            "laplace_smoothed_violation_odds": penalty,
            "penalty": penalty,
            "lambda": float(-math.log(penalty)),
        }
    return np.asarray(penalties, dtype=float), details


def typed_worlds() -> np.ndarray:
    return np.asarray(list(itertools.product(range(5), repeat=5)), dtype=np.int8)


def hard_valid(worlds_: np.ndarray) -> np.ndarray:
    mask = np.ones(len(worlds_), dtype=bool)
    for rule in HARD_RULES:
        mask &= ~rule.violation(worlds_)
    return mask


def soft_violations(worlds_: np.ndarray) -> np.ndarray:
    return np.column_stack([rule.violation(worlds_) for rule in SOFT_RULES]).astype(np.int8)


def score_var(head: int, level: int) -> int:
    return head * 5 + level + 1


def selector_var(rule_index: int) -> int:
    return 26 + rule_index


def assignment_mask(levels: np.ndarray, violations: np.ndarray | None = None) -> int:
    mask = 0
    for head, level in enumerate(levels):
        mask |= 1 << (score_var(head, int(level)) - 1)
    if violations is not None:
        for index, value in enumerate(violations):
            mask |= int(value) << (selector_var(index) - 1)
    return mask


def ordered_valid_worlds(soft: bool) -> tuple[np.ndarray, np.ndarray | None]:
    all_worlds = typed_worlds()
    valid_worlds = all_worlds[hard_valid(all_worlds)]
    violations = soft_violations(valid_worlds) if soft else None
    masks = np.asarray(
        [assignment_mask(world, None if violations is None else violations[i]) for i, world in enumerate(valid_worlds)],
        dtype=np.int64,
    )
    order = np.argsort(masks, kind="stable")
    return valid_worlds[order], None if violations is None else violations[order]


def enumerate_inference(
    probs: np.ndarray, penalties: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    worlds_, violations = ordered_valid_worlds(soft=penalties is not None)
    marginals = np.empty_like(probs)
    z_values = np.empty(len(probs), dtype=float)
    maps = np.empty((len(probs), 5), dtype=np.int8)
    map_weights = np.empty(len(probs), dtype=float)
    if violations is None:
        penalty_weight = np.ones(len(worlds_), dtype=float)
    else:
        penalty_weight = np.prod(np.where(violations == 1, penalties, 1.0), axis=1)

    for row, p in enumerate(probs):
        weights = np.ones(len(worlds_), dtype=float)
        for head in range(5):
            weights *= p[head, worlds_[:, head]]
        weights *= penalty_weight
        z = float(weights.sum())
        if z <= 0:
            raise RuntimeError(f"Zero valid mass at row {row}")
        q = weights / z
        z_values[row] = z
        for head in range(5):
            marginals[row, head] = [q[worlds_[:, head] == level].sum() for level in range(5)]
        best = float(np.max(weights))
        best_index = int(np.flatnonzero(np.isclose(weights, best, rtol=0.0, atol=1e-15))[0])
        maps[row] = worlds_[best_index]
        map_weights[row] = best
    return marginals, z_values, maps, map_weights


def event_node(manager: SddManager, head: int, allowed: range) -> SddNode:
    node = manager.false()
    for level in allowed:
        node |= manager.literal(score_var(head, level))
    return node


def rule_nodes(manager: SddManager, rule_name: str) -> tuple[SddNode, SddNode]:
    if rule_name == "h3_c1":
        return event_node(manager, H, range(3, 5)), event_node(manager, C, range(1, 5))
    if rule_name == "h3_k1":
        return event_node(manager, H, range(3, 5)), event_node(manager, K, range(1, 5))
    if rule_name == "h4_k2":
        return event_node(manager, H, range(4, 5)), event_node(manager, K, range(2, 5))
    if rule_name == "h3_c2":
        return event_node(manager, H, range(3, 5)), event_node(manager, C, range(2, 5))
    if rule_name == "h3_k2":
        return event_node(manager, H, range(3, 5)), event_node(manager, K, range(2, 5))
    if rule_name == "c4_h3":
        return event_node(manager, C, range(4, 5)), event_node(manager, H, range(3, 5))
    raise ValueError(rule_name)


def compile_sdd(soft: bool, artifact_dir: Path) -> tuple[SddManager, SddNode, float, str]:
    tag = "hard_soft" if soft else "hard"
    var_count = 25 + (len(SOFT_RULES) if soft else 0)
    start = time.perf_counter()
    vtree = Vtree(var_count=var_count, var_order=list(range(1, var_count + 1)), vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    root = manager.true()

    for head in range(5):
        literals = [manager.literal(score_var(head, level)) for level in range(5)]
        at_least_one = manager.false()
        for literal in literals:
            at_least_one |= literal
        root &= at_least_one
        for left_index, left in enumerate(literals):
            for right in literals[left_index + 1 :]:
                root &= (~left) | (~right)

    for rule in HARD_RULES:
        antecedent, consequent = rule_nodes(manager, rule.name)
        root &= (~antecedent) | consequent

    if soft:
        for index, rule in enumerate(SOFT_RULES):
            antecedent, consequent = rule_nodes(manager, rule.name)
            violation = antecedent & (~consequent)
            selector = manager.literal(selector_var(index))
            root &= ((~selector) | violation) & ((~violation) | selector)

    root.ref()
    compile_seconds = time.perf_counter() - start
    artifact_dir.mkdir(parents=True, exist_ok=True)
    root.save(str(artifact_dir / f"helpsteer2_{tag}.sdd").encode())
    manager.vtree().save(str(artifact_dir / f"helpsteer2_{tag}.vtree").encode())
    return manager, root, compile_seconds, tag


def reload_model_count(tag: str, artifact_dir: Path) -> int:
    vtree = Vtree.from_file(str(artifact_dir / f"helpsteer2_{tag}.vtree").encode())
    manager = SddManager.from_vtree(vtree)
    root = manager.read_sdd_file(str(artifact_dir / f"helpsteer2_{tag}.sdd").encode())
    if root is None:
        raise RuntimeError(f"Serialized HelpSteer2 {tag} SDD could not be reloaded")
    return int(root.global_model_count())


def circuit_inference(
    probs: np.ndarray, penalties: np.ndarray | None, artifact_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    soft = penalties is not None
    manager, root, compile_seconds, tag = compile_sdd(soft, artifact_dir)
    wmc = root.wmc(log_mode=False)
    marginals = np.empty_like(probs)
    z_values = np.empty(len(probs), dtype=float)
    maps = np.empty((len(probs), 5), dtype=np.int8)
    map_weights = np.empty(len(probs), dtype=float)
    var_count = 25 + (len(SOFT_RULES) if soft else 0)

    start = time.perf_counter()
    for row, p in enumerate(probs):
        positive = np.ones(var_count, dtype=float)
        negative = np.ones(var_count, dtype=float)
        for head in range(5):
            for level in range(5):
                var = score_var(head, level)
                positive[var - 1] = p[head, level]
                negative[var - 1] = 1.0
                wmc.set_literal_weight(var, float(p[head, level]))
                wmc.set_literal_weight(-var, 1.0)
        if soft:
            for index, penalty in enumerate(penalties):
                var = selector_var(index)
                positive[var - 1] = penalty
                negative[var - 1] = 1.0
                wmc.set_literal_weight(var, float(penalty))
                wmc.set_literal_weight(-var, 1.0)

        z_values[row] = float(wmc.propagate())
        for head in range(5):
            marginals[row, head] = [
                float(wmc.literal_pr(score_var(head, level))) for level in range(5)
            ]

        mpe = max_product(root, manager.vtree(), positive, negative, manager.true())
        map_weights[row] = mpe.weight
        for head in range(5):
            active = [level for level in range(5) if mpe.assignment[score_var(head, level)] == 1]
            if len(active) != 1:
                raise AssertionError(f"Non-one-hot MAP at row {row}, head {head}: {active}")
            maps[row, head] = active[0]
    inference_seconds = time.perf_counter() - start

    info = {
        "tag": tag,
        "library": "PySDD",
        "version": "1.0.6",
        "vtree": "balanced",
        "boolean_variables": var_count,
        "node_count": int(root.count()),
        "size_elements": int(root.size()),
        "model_count": int(root.global_model_count()),
        "reloaded_model_count": reload_model_count(tag, artifact_dir),
        "compile_seconds": compile_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_sample": inference_seconds * 1000.0 / len(probs),
        "sdd_path": str(artifact_dir / f"helpsteer2_{tag}.sdd"),
        "vtree_path": str(artifact_dir / f"helpsteer2_{tag}.vtree"),
    }
    return marginals, z_values, maps, map_weights, info


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        # Circuit and enumeration results can differ at ~1e-16. Treat those
        # numerical representations of the same score as ties, as Spearman's
        # definition requires.
        while end < len(values) and math.isclose(
            float(values[order[end]]), float(values[order[start]]), rel_tol=0.0, abs_tol=1e-12
        ):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = rankdata(left), rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def score_metrics(gold: np.ndarray, probs: np.ndarray, pred: np.ndarray | None = None) -> dict:
    expected = np.sum(probs * LEVELS[None, None, :], axis=2)
    argmax = np.argmax(probs, axis=2).astype(np.int8) if pred is None else pred
    one_hot = np.eye(5, dtype=float)[gold]
    head_metrics = {}
    for head_index, head in enumerate(HEADS):
        head_metrics[head] = {
            "spearman": spearman(gold[:, head_index], expected[:, head_index]),
            "mae": float(np.mean(np.abs(expected[:, head_index] - gold[:, head_index]))),
            "argmax_accuracy": float(np.mean(argmax[:, head_index] == gold[:, head_index])),
            "categorical_brier": float(
                np.mean(np.sum((probs[:, head_index] - one_hot[:, head_index]) ** 2, axis=1))
            ),
            "nll": float(
                np.mean(-np.log(np.clip(probs[np.arange(len(gold)), head_index, gold[:, head_index]], 1e-12, 1.0)))
            ),
        }
    return {
        "heads": head_metrics,
        "mean_spearman": float(np.nanmean([head_metrics[h]["spearman"] for h in HEADS])),
        "mean_mae": float(np.mean(np.abs(expected - gold))),
        "mean_argmax_accuracy": float(np.mean(argmax == gold)),
        "joint_exact_match": float(np.mean(np.all(argmax == gold, axis=1))),
        "mean_categorical_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=2))),
        "mean_nll": float(
            np.mean(-np.log(np.clip(probs[np.arange(len(gold))[:, None], np.arange(5)[None, :], gold], 1e-12, 1.0)))
        ),
    }


def map_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    per_head = {
        head: float(np.mean(pred[:, index] == gold[:, index]))
        for index, head in enumerate(HEADS)
    }
    return {
        "head_accuracy": per_head,
        "mean_accuracy": float(np.mean(pred == gold)),
        "mean_absolute_error": float(np.mean(np.abs(pred - gold))),
        "joint_exact_match": float(np.mean(np.all(pred == gold, axis=1))),
    }


def constraint_counts(assignments: np.ndarray) -> dict:
    hard_matrix = np.column_stack([rule.violation(assignments) for rule in HARD_RULES])
    soft_matrix = np.column_stack([rule.violation(assignments) for rule in SOFT_RULES])
    return {
        "hard_violations_total": int(hard_matrix.sum()),
        "hard_conflicting_examples": int(np.any(hard_matrix, axis=1).sum()),
        "hard_by_rule": {rule.name: int(hard_matrix[:, i].sum()) for i, rule in enumerate(HARD_RULES)},
        "soft_violations_total": int(soft_matrix.sum()),
        "soft_conflicting_examples": int(np.any(soft_matrix, axis=1).sum()),
        "soft_by_rule": {rule.name: int(soft_matrix[:, i].sum()) for i, rule in enumerate(SOFT_RULES)},
    }


def validate_backend(
    probs: np.ndarray, penalties: np.ndarray | None, artifact_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    oracle_m, oracle_z, oracle_map, oracle_w = enumerate_inference(probs, penalties)
    circuit_m, circuit_z, circuit_map, circuit_w, circuit = circuit_inference(probs, penalties, artifact_dir)
    exact = {
        "status": "pass",
        "z_max_abs_error": float(np.max(np.abs(circuit_z - oracle_z))),
        "marginal_max_abs_error": float(np.max(np.abs(circuit_m - oracle_m))),
        "map_weight_max_abs_error": float(np.max(np.abs(circuit_w - oracle_w))),
        "map_assignment_mismatches": int(np.sum(np.any(circuit_map != oracle_map, axis=1))),
        "z_values_checked": int(len(probs)),
        "marginal_values_checked": int(circuit_m.size),
        "map_assignments_checked": int(len(probs)),
        "tolerance": 1e-11,
    }
    if max(exact["z_max_abs_error"], exact["marginal_max_abs_error"], exact["map_weight_max_abs_error"]) > exact["tolerance"]:
        raise AssertionError(f"SDD/enumeration numerical mismatch: {exact}")
    if exact["map_assignment_mismatches"]:
        raise AssertionError(f"SDD/enumeration MAP mismatch: {exact}")
    expected_models = int(hard_valid(typed_worlds()).sum())
    if circuit["model_count"] != expected_models or circuit["reloaded_model_count"] != expected_models:
        raise AssertionError(f"Unexpected circuit model count: {circuit}, expected {expected_models}")
    return circuit_m, circuit_z, circuit_map, circuit, exact


def bootstrap(
    gold: np.ndarray,
    raw_probs: np.ndarray,
    new_probs: np.ndarray,
    n_boot: int = 1000,
) -> dict:
    rng = np.random.default_rng(0)
    samples = {"mean_spearman": [], "mean_mae": [], "mean_brier": [], "joint_exact_match": []}
    raw_expected = np.sum(raw_probs * LEVELS[None, None, :], axis=2)
    new_expected = np.sum(new_probs * LEVELS[None, None, :], axis=2)
    raw_pred = np.argmax(raw_probs, axis=2)
    new_pred = np.argmax(new_probs, axis=2)
    one_hot = np.eye(5, dtype=float)[gold]
    for _ in range(n_boot):
        idx = rng.integers(0, len(gold), len(gold))
        g = gold[idx]
        raw_s = np.nanmean([spearman(g[:, h], raw_expected[idx, h]) for h in range(5)])
        new_s = np.nanmean([spearman(g[:, h], new_expected[idx, h]) for h in range(5)])
        samples["mean_spearman"].append(new_s - raw_s)
        samples["mean_mae"].append(
            np.mean(np.abs(new_expected[idx] - g)) - np.mean(np.abs(raw_expected[idx] - g))
        )
        samples["mean_brier"].append(
            np.mean(np.sum((new_probs[idx] - one_hot[idx]) ** 2, axis=2))
            - np.mean(np.sum((raw_probs[idx] - one_hot[idx]) ** 2, axis=2))
        )
        samples["joint_exact_match"].append(
            np.mean(np.all(new_pred[idx] == g, axis=1)) - np.mean(np.all(raw_pred[idx] == g, axis=1))
        )
    return {
        name: {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
        for name, values in samples.items()
    }


def soft_strength_sweep(
    probs: np.ndarray, gold: np.ndarray, base_penalties: np.ndarray
) -> dict:
    """Predeclared sensitivity grid; alpha=1 is the primary soft model."""
    result = {}
    for alpha in (0.125, 0.25, 0.5, 0.75, 1.0):
        tempered = base_penalties**alpha
        marginals, partition, _, _ = enumerate_inference(probs, tempered)
        pred = np.argmax(marginals, axis=2).astype(np.int8)
        metric = score_metrics(gold, marginals)
        result[str(alpha)] = {
            "alpha": alpha,
            "penalties": tempered.tolist(),
            "mean_partition": float(partition.mean()),
            "mean_spearman": metric["mean_spearman"],
            "mean_mae": metric["mean_mae"],
            "mean_categorical_brier": metric["mean_categorical_brier"],
            "mean_nll": metric["mean_nll"],
            "mean_argmax_accuracy": metric["mean_argmax_accuracy"],
            "joint_exact_match": metric["joint_exact_match"],
            "soft_conflicting_examples": constraint_counts(pred)["soft_conflicting_examples"],
        }
    return result


def run(
    eval_path: Path, train_path: Path, db_path: Path, artifact_dir: Path
) -> dict:
    probs, gold, normalization = load_eval(eval_path, db_path)
    _, train_gold = load_gold(train_path)
    penalties, penalty_details = estimate_penalties(train_gold)

    hard_m, hard_z, hard_map, hard_circuit, hard_exact = validate_backend(probs, None, artifact_dir)
    soft_m, soft_z, soft_map, soft_circuit, soft_exact = validate_backend(probs, penalties, artifact_dir)

    untouched_error = {
        "hard_complexity": float(np.max(np.abs(hard_m[:, X] - probs[:, X]))),
        "hard_verbosity": float(np.max(np.abs(hard_m[:, V] - probs[:, V]))),
        "soft_complexity": float(np.max(np.abs(soft_m[:, X] - probs[:, X]))),
        "soft_verbosity": float(np.max(np.abs(soft_m[:, V] - probs[:, V]))),
    }
    if max(untouched_error.values()) > 1e-12:
        raise AssertionError(f"Unconstrained marginals changed: {untouched_error}")

    raw_pred = np.argmax(probs, axis=2).astype(np.int8)
    hard_pred = np.argmax(hard_m, axis=2).astype(np.int8)
    soft_pred = np.argmax(soft_m, axis=2).astype(np.int8)

    return {
        "setup": {
            "model": MODEL,
            "dataset": "nvidia/HelpSteer2",
            "n_train": int(len(train_gold)),
            "n_eval": int(len(gold)),
            "score_variables": list(HEADS),
            "levels_per_score": 5,
            "typed_assignments": 5**5,
            "hard_valid_assignments": int(hard_valid(typed_worlds()).sum()),
            "score_probability_normalization": normalization,
        },
        "rules": {
            "hard_train_audit": audit_rules(train_gold, HARD_RULES),
            "hard_eval_audit": audit_rules(gold, HARD_RULES),
            "soft_train_estimates": penalty_details,
            "soft_eval_audit": audit_rules(gold, SOFT_RULES),
            "penalty_definition": "Laplace-smoothed training violation odds: (v+1)/(n-v+1)",
        },
        "hard": {
            "circuit": hard_circuit,
            "exact_validation": hard_exact,
            "mean_z": float(hard_z.mean()),
            "mean_invalid_mass": float(np.mean(1.0 - hard_z)),
            "min_z": float(hard_z.min()),
            "metrics": score_metrics(gold, hard_m),
            "map_metrics": map_metrics(gold, hard_map),
            "argmax_constraints": constraint_counts(hard_pred),
            "map_constraints": constraint_counts(hard_map),
            "bootstrap_minus_raw": bootstrap(gold, probs, hard_m),
        },
        "soft": {
            "circuit": soft_circuit,
            "exact_validation": soft_exact,
            "mean_partition": float(soft_z.mean()),
            "min_partition": float(soft_z.min()),
            "metrics": score_metrics(gold, soft_m),
            "map_metrics": map_metrics(gold, soft_map),
            "argmax_constraints": constraint_counts(soft_pred),
            "map_constraints": constraint_counts(soft_map),
            "bootstrap_minus_raw": bootstrap(gold, probs, soft_m),
        },
        "raw": {
            "metrics": score_metrics(gold, probs),
            "argmax_constraints": constraint_counts(raw_pred),
        },
        "changes": {
            "hard_argmax_rows_vs_raw": int(np.any(hard_pred != raw_pred, axis=1).sum()),
            "soft_argmax_rows_vs_raw": int(np.any(soft_pred != raw_pred, axis=1).sum()),
            "hard_map_rows_vs_raw": int(np.any(hard_map != raw_pred, axis=1).sum()),
            "soft_map_rows_vs_raw": int(np.any(soft_map != raw_pred, axis=1).sum()),
            "unconstrained_marginal_max_abs_error": untouched_error,
        },
        "soft_strength_sensitivity": soft_strength_sweep(probs, gold, penalties),
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def markdown(result: dict) -> str:
    setup = result["setup"]
    rules = result["rules"]
    raw = result["raw"]["metrics"]
    hard = result["hard"]
    soft = result["soft"]
    hm, sm = hard["metrics"], soft["metrics"]
    hmap, smap = hard["map_metrics"], soft["map_metrics"]
    hb, sb = hard["bootstrap_minus_raw"], soft["bootstrap_minus_raw"]
    sweep = result["soft_strength_sensitivity"]

    hard_rule_lines = "\n".join(
        f"- `{entry['rule']}`: train {entry['violation_count']}/{entry['antecedent_count']} violations; "
        f"validation {rules['hard_eval_audit'][name]['violation_count']}/{rules['hard_eval_audit'][name]['antecedent_count']}."
        for name, entry in rules["hard_train_audit"].items()
    )
    soft_rule_lines = "\n".join(
        f"- `{entry['rule']}`: train {entry['violation_count']}/{entry['antecedent_count']}; "
        f"penalty {entry['penalty']:.6f}; validation "
        f"{rules['soft_eval_audit'][name]['violation_count']}/{rules['soft_eval_audit'][name]['antecedent_count']}."
        for name, entry in rules["soft_train_estimates"].items()
    )
    sweep_rows = "\n".join(
        f"| {entry['alpha']:.3f} | {entry['mean_spearman']:.6f} | {entry['mean_mae']:.6f} | "
        f"{entry['mean_categorical_brier']:.6f} | {entry['mean_nll']:.6f} | "
        f"{entry['soft_conflicting_examples']} |"
        for entry in sweep.values()
    )

    return f"""# HelpSteer2 hard- and soft-constraint SDD pilot

## Setup

- Model: `{setup['model']}`
- Training examples used only for rule audit and penalty estimation: {setup['n_train']}
- Evaluation examples: {setup['n_eval']}
- Typed variables: five Score(5) variables
- Typed assignment space: {setup['typed_assignments']}
- Assignments satisfying the hard ontology: {setup['hard_valid_assignments']}

### Hard admissibility rules

{hard_rule_lines}

### Soft preference rules

{soft_rule_lines}

Soft-rule penalties use {rules['penalty_definition']}. A violation indicator is logically equivalent to each rule violation inside the SDD; its positive literal receives the learned penalty, so weighted model counting remains exact.

## Exact circuit validation

Both circuits pass complete validation against enumeration over all {setup['typed_assignments']} typed assignments for every evaluation example.

| Circuit | Boolean variables | SDD nodes | Elements | Models | Max $Z$ error | Max marginal error | MAP mismatches | ms/sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard | {hard['circuit']['boolean_variables']} | {hard['circuit']['node_count']} | {hard['circuit']['size_elements']} | {hard['circuit']['model_count']} | {hard['exact_validation']['z_max_abs_error']:.3e} | {hard['exact_validation']['marginal_max_abs_error']:.3e} | {hard['exact_validation']['map_assignment_mismatches']} | {hard['circuit']['milliseconds_per_sample']:.4f} |
| Hard + soft | {soft['circuit']['boolean_variables']} | {soft['circuit']['node_count']} | {soft['circuit']['size_elements']} | {soft['circuit']['model_count']} | {soft['exact_validation']['z_max_abs_error']:.3e} | {soft['exact_validation']['marginal_max_abs_error']:.3e} | {soft['exact_validation']['map_assignment_mismatches']} | {soft['circuit']['milliseconds_per_sample']:.4f} |

Each circuit was serialized, reloaded, and recovered {setup['hard_valid_assignments']} models.
The Complexity and Verbosity factors occur in both five-Score joints while remaining unconstrained; their conditioned marginals match the raw marginals to a maximum error of {max(result['changes']['unconstrained_marginal_max_abs_error'].values()):.3e}.

## Marginal inference results

| Metric | Raw | Hard constraints | Hard + soft constraints |
|---|---:|---:|---:|
| Mean Spearman | {fmt(raw['mean_spearman'])} | {fmt(hm['mean_spearman'])} | {fmt(sm['mean_spearman'])} |
| Mean MAE | {fmt(raw['mean_mae'])} | {fmt(hm['mean_mae'])} | {fmt(sm['mean_mae'])} |
| Mean categorical Brier | {fmt(raw['mean_categorical_brier'])} | {fmt(hm['mean_categorical_brier'])} | {fmt(sm['mean_categorical_brier'])} |
| Mean NLL | {fmt(raw['mean_nll'])} | {fmt(hm['mean_nll'])} | {fmt(sm['mean_nll'])} |
| Mean argmax accuracy | {fmt(raw['mean_argmax_accuracy'])} | {fmt(hm['mean_argmax_accuracy'])} | {fmt(sm['mean_argmax_accuracy'])} |
| Five-Score exact match | {fmt(raw['joint_exact_match'])} | {fmt(hm['joint_exact_match'])} | {fmt(sm['joint_exact_match'])} |

Hard valid mass has mean $Z={hard['mean_z']:.6f}$, mean invalid mass {hard['mean_invalid_mass']:.6f}, and minimum $Z={hard['min_z']:.6f}$. The hard+soft partition has mean {soft['mean_partition']:.6f} and minimum {soft['min_partition']:.6f}.

## Joint MAP results

| Metric | Hard MAP | Hard + soft MAP |
|---|---:|---:|
| Mean level accuracy | {hmap['mean_accuracy']:.6f} | {smap['mean_accuracy']:.6f} |
| Mean absolute level error | {hmap['mean_absolute_error']:.6f} | {smap['mean_absolute_error']:.6f} |
| Five-Score exact match | {hmap['joint_exact_match']:.6f} | {smap['joint_exact_match']:.6f} |

Raw local argmax produces {result['raw']['argmax_constraints']['hard_conflicting_examples']} hard-conflicting examples. Hard-conditioned marginals leave {hard['argmax_constraints']['hard_conflicting_examples']}, hard MAP leaves {hard['map_constraints']['hard_conflicting_examples']}, soft-conditioned marginals leave {soft['argmax_constraints']['hard_conflicting_examples']}, and soft MAP leaves {soft['map_constraints']['hard_conflicting_examples']}.

For the stronger soft rules, raw local argmax has {result['raw']['argmax_constraints']['soft_conflicting_examples']} examples with at least one violation; hard-only marginals have {hard['argmax_constraints']['soft_conflicting_examples']}; hard+soft marginals have {soft['argmax_constraints']['soft_conflicting_examples']}; hard+soft MAP has {soft['map_constraints']['soft_conflicting_examples']}.

## Paired bootstrap deltas versus raw marginals

| Inference | Metric | Delta | 95% CI |
|---|---|---:|---:|
| Hard | Mean Spearman | {hb['mean_spearman']['mean']:+.6f} | [{hb['mean_spearman']['ci95'][0]:+.6f}, {hb['mean_spearman']['ci95'][1]:+.6f}] |
| Hard | Mean MAE | {hb['mean_mae']['mean']:+.6f} | [{hb['mean_mae']['ci95'][0]:+.6f}, {hb['mean_mae']['ci95'][1]:+.6f}] |
| Hard | Mean Brier | {hb['mean_brier']['mean']:+.6f} | [{hb['mean_brier']['ci95'][0]:+.6f}, {hb['mean_brier']['ci95'][1]:+.6f}] |
| Hard | Exact match | {hb['joint_exact_match']['mean']:+.6f} | [{hb['joint_exact_match']['ci95'][0]:+.6f}, {hb['joint_exact_match']['ci95'][1]:+.6f}] |
| Hard + soft | Mean Spearman | {sb['mean_spearman']['mean']:+.6f} | [{sb['mean_spearman']['ci95'][0]:+.6f}, {sb['mean_spearman']['ci95'][1]:+.6f}] |
| Hard + soft | Mean MAE | {sb['mean_mae']['mean']:+.6f} | [{sb['mean_mae']['ci95'][0]:+.6f}, {sb['mean_mae']['ci95'][1]:+.6f}] |
| Hard + soft | Mean Brier | {sb['mean_brier']['mean']:+.6f} | [{sb['mean_brier']['ci95'][0]:+.6f}, {sb['mean_brier']['ci95'][1]:+.6f}] |
| Hard + soft | Exact match | {sb['joint_exact_match']['mean']:+.6f} | [{sb['joint_exact_match']['ci95'][0]:+.6f}, {sb['joint_exact_match']['ci95'][1]:+.6f}] |

## Soft-strength sensitivity

The primary soft model uses the full training-derived penalty, `rho_r^alpha` with `alpha=1`. The following predeclared grid shows the trade-off without selecting a value on validation performance.

| alpha | Mean Spearman | Mean MAE | Mean Brier | Mean NLL | Soft-conflicting argmax examples |
|---:|---:|---:|---:|---:|---:|
{sweep_rows}

This experiment separates logical admissibility from defeasible quality preferences while retaining exact inference for both.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.eval_data, args.train_data, args.responses, args.artifact_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = markdown(result)
    args.report_out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
