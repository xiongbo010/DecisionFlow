#!/usr/bin/env python3
"""Ontology-conditioned SDD pilot for Jev's 28-head GoEmotions task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".deps"))

from pysdd.sdd import SddManager, SddNode, Vtree  # noqa: E402

from openai_moderation_pilot import (  # noqa: E402
    average_precision,
    bootstrap_decision_delta,
    ece,
    f1_binary,
    roc_auc,
)
from validate_sdd import assignment_mask, max_product  # noqa: E402


MODEL = "jev-1.13.0"
LABELS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
)
INDEX = {label: i for i, label in enumerate(LABELS)}
NEUTRAL = INDEX["neutral"]
SENTIMENT_GROUPS = {
    "positive": (
        "amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration",
        "gratitude", "relief", "approval",
    ),
    "negative": (
        "fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust",
        "anger", "annoyance", "disapproval",
    ),
    "ambiguous": ("realization", "surprise", "curiosity", "confusion"),
}
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def questions() -> dict[str, dict]:
    result = {
        label: {"type": "noul", "instructions": f"Does the author of `comment` express {label}?"}
        for label in LABELS
        if label != "neutral"
    }
    result["neutral"] = {
        "type": "noul",
        "instructions": "Is `comment` emotionally neutral, expressing no particular emotion?",
    }
    return result


def request_key(text: str, qs: dict) -> str:
    payload = json.dumps({"model": MODEL, "state": {"comment": text}, "questions": qs}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def dev_order(size: int, limit: int) -> list[int]:
    return sorted(
        range(size),
        key=lambda i: hashlib.sha256(f"go_emotions/simplified/{i}".encode()).hexdigest(),
    )[:limit]


def load_rows(
    data_path: Path, db_path: Path, indices: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(data_path)
    selected = list(range(len(frame))) if indices is None else indices
    qs = questions()
    probabilities, gold = [], []
    db = sqlite3.connect(db_path)
    try:
        for row_index in selected:
            row = frame.iloc[row_index]
            key = request_key(row["text"], qs)
            cached = db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            if cached is None:
                raise RuntimeError(f"Missing cached response for row {row_index}, key {key}")
            answers = json.loads(cached[0])["answers"]
            probabilities.append([float(answers[label]["noul"]) for label in LABELS])
            target = np.zeros(len(LABELS), dtype=np.int8)
            target[np.asarray(row["labels"], dtype=int)] = 1
            gold.append(target)
    finally:
        db.close()
    return np.asarray(probabilities), np.asarray(gold)


def gold_from_frame(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    gold = np.zeros((len(frame), len(LABELS)), dtype=np.int8)
    for row, labels in enumerate(frame["labels"]):
        gold[row, np.asarray(labels, dtype=int)] = 1
    return gold


def select_dev_audited_constraints(dev_gold: np.ndarray) -> list[int]:
    return [i for i in range(NEUTRAL) if not np.any((dev_gold[:, NEUTRAL] == 1) & (dev_gold[:, i] == 1))]


def canonicalize_neutral(gold: np.ndarray) -> tuple[np.ndarray, int]:
    """Project aggregated labels onto Neutral -> no Emotion by minimal deletion.

    When neutral co-occurs with one or more specific emotions, removing neutral
    changes one bit and preserves all specific annotations.  The transformation
    depends only on the ontology and never on model outputs.
    """
    repaired = gold.copy()
    conflicts = (repaired[:, NEUTRAL] == 1) & np.any(repaired[:, :NEUTRAL] == 1, axis=1)
    repaired[conflicts, NEUTRAL] = 0
    return repaired, int(conflicts.sum())


def analytic_condition(
    probabilities: np.ndarray, constrained: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    conditioned = probabilities.copy()
    z_values = np.empty(len(probabilities))
    map_assignments = np.empty_like(probabilities, dtype=np.int8)
    map_weights = np.empty(len(probabilities))
    parent_marginals = {name: np.empty(len(probabilities)) for name in SENTIMENT_GROUPS}
    constrained_set = set(constrained)

    for row, p in enumerate(probabilities):
        p_neutral = p[NEUTRAL]
        all_constrained_false = float(np.prod(1.0 - p[constrained])) if constrained else 1.0
        z = (1.0 - p_neutral) + p_neutral * all_constrained_false
        if z <= 0:
            raise RuntimeError(f"Zero valid mass at row {row}")
        z_values[row] = z
        conditioned[row, NEUTRAL] = p_neutral * all_constrained_false / z
        conditioned[row, constrained] = (1.0 - p_neutral) * p[constrained] / z

        local = (p[:NEUTRAL] > 0.5).astype(np.int8)
        nonneutral_assignment = np.concatenate((local, [0])).astype(np.int8)
        nonneutral_weight = float((1.0 - p_neutral) * np.prod(np.maximum(p[:NEUTRAL], 1.0 - p[:NEUTRAL])))
        neutral_assignment = np.concatenate((local, [1])).astype(np.int8)
        neutral_assignment[constrained] = 0
        neutral_weight = float(
            p_neutral
            * np.prod(1.0 - p[constrained])
            * np.prod(
                [max(p[i], 1.0 - p[i]) for i in range(NEUTRAL) if i not in constrained_set]
            )
        )
        if neutral_weight > nonneutral_weight + 1e-15:
            best_assignment, best_weight = neutral_assignment, neutral_weight
        elif nonneutral_weight > neutral_weight + 1e-15:
            best_assignment, best_weight = nonneutral_assignment, nonneutral_weight
        else:
            left = {i + 1: int(value) for i, value in enumerate(nonneutral_assignment)}
            right = {i + 1: int(value) for i, value in enumerate(neutral_assignment)}
            if assignment_mask(right) < assignment_mask(left):
                best_assignment, best_weight = neutral_assignment, neutral_weight
            else:
                best_assignment, best_weight = nonneutral_assignment, nonneutral_weight
        map_assignments[row] = best_assignment
        map_weights[row] = best_weight

        for group_name, labels in SENTIMENT_GROUPS.items():
            group = {INDEX[label] for label in labels}
            p_no_group = float(np.prod([1.0 - p[i] for i in group]))
            remaining_constraints = [i for i in constrained if i not in group]
            valid_and_no_group = p_no_group * (
                (1.0 - p_neutral)
                + p_neutral * float(np.prod(1.0 - p[remaining_constraints]))
            )
            parent_marginals[group_name][row] = 1.0 - valid_and_no_group / z

    return conditioned, z_values, map_assignments, map_weights, parent_marginals


def compile_sdd(name: str, constrained: list[int], artifact_dir: Path) -> tuple[SddManager, SddNode, float]:
    start = time.perf_counter()
    vtree = Vtree(var_count=len(LABELS), var_order=list(range(1, len(LABELS) + 1)), vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    root = manager.true()
    neutral = manager.literal(NEUTRAL + 1)
    for emotion in constrained:
        root &= (~neutral) | (~manager.literal(emotion + 1))
    root.ref()
    elapsed = time.perf_counter() - start
    artifact_dir.mkdir(parents=True, exist_ok=True)
    root.save(str(artifact_dir / f"goemotions_{name}.sdd").encode())
    manager.vtree().save(str(artifact_dir / f"goemotions_{name}.vtree").encode())
    return manager, root, elapsed


def reload_count(name: str, artifact_dir: Path) -> int:
    vtree = Vtree.from_file(str(artifact_dir / f"goemotions_{name}.vtree").encode())
    manager = SddManager.from_vtree(vtree)
    root = manager.read_sdd_file(str(artifact_dir / f"goemotions_{name}.sdd").encode())
    return int(root.global_model_count())


def circuit_condition(
    probabilities: np.ndarray, name: str, constrained: list[int], artifact_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    manager, root, compile_seconds = compile_sdd(name, constrained, artifact_dir)
    group_nodes: dict[str, SddNode] = {}
    for group_name, labels in SENTIMENT_GROUPS.items():
        query = manager.false()
        for label in labels:
            query |= manager.literal(INDEX[label] + 1)
        group_nodes[group_name] = root & query
        group_nodes[group_name].ref()

    wmc = root.wmc(log_mode=False)
    group_wmc = {group: node.wmc(log_mode=False) for group, node in group_nodes.items()}
    conditioned = np.empty_like(probabilities)
    z_values = np.empty(len(probabilities))
    maps = np.empty_like(probabilities, dtype=np.int8)
    map_weights = np.empty(len(probabilities))
    parent_marginals = {group: np.empty(len(probabilities)) for group in SENTIMENT_GROUPS}

    def set_weights(weighted_counter, p: np.ndarray) -> None:
        for var, probability in enumerate(p, start=1):
            weighted_counter.set_literal_weight(var, float(probability))
            weighted_counter.set_literal_weight(-var, float(1.0 - probability))

    start = time.perf_counter()
    for row, p in enumerate(probabilities):
        set_weights(wmc, p)
        z_values[row] = float(wmc.propagate())
        conditioned[row] = [float(wmc.literal_pr(var)) for var in range(1, len(LABELS) + 1)]
        for group_name, counter in group_wmc.items():
            set_weights(counter, p)
            parent_marginals[group_name][row] = float(counter.propagate()) / z_values[row]

        mpe = max_product(root, manager.vtree(), p, 1.0 - p, manager.true())
        maps[row] = [mpe.assignment[var] for var in range(1, len(LABELS) + 1)]
        map_weights[row] = mpe.weight
    inference_seconds = time.perf_counter() - start

    circuit = {
        "node_count": int(root.count()),
        "size_elements": int(root.size()),
        "model_count": int(root.global_model_count()),
        "reloaded_model_count": reload_count(name, artifact_dir),
        "compile_seconds": compile_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_sample": 1000 * inference_seconds / len(probabilities),
    }
    return conditioned, z_values, maps, map_weights, parent_marginals, circuit


def best_threshold(gold: np.ndarray, probabilities: np.ndarray) -> float:
    if gold.sum() == 0:
        return 0.5
    scores = np.asarray([f1_binary(gold, probabilities >= threshold) for threshold in GRID])
    best = np.flatnonzero(scores == scores.max())
    return float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]])


def tune_thresholds(gold: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return np.asarray([best_threshold(gold[:, i], probabilities[:, i]) for i in range(gold.shape[1])])


def prediction_metrics(gold: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict:
    predictions = probabilities >= thresholds
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return {
        "micro_f1": f1_binary(gold.ravel(), predictions.ravel()),
        "macro_f1": float(np.mean([f1_binary(gold[:, i], predictions[:, i]) for i in range(gold.shape[1])])),
        "exact_match": float(np.mean(np.all(predictions == gold, axis=1))),
        "mean_brier": float(np.mean((probabilities - gold) ** 2)),
        "mean_ece": float(np.mean([ece(gold[:, i], probabilities[:, i]) for i in range(gold.shape[1])])),
        "mean_auprc": float(np.mean([average_precision(gold[:, i], probabilities[:, i]) for i in range(gold.shape[1])])),
        "marginal_nll": float(
            np.mean(-(gold * np.log(clipped) + (1 - gold) * np.log(1 - clipped)))
        ),
    }


def decision_metrics(gold: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "micro_f1": f1_binary(gold.ravel(), predictions.ravel()),
        "macro_f1": float(np.mean([f1_binary(gold[:, i], predictions[:, i]) for i in range(gold.shape[1])])),
        "exact_match": float(np.mean(np.all(predictions == gold, axis=1))),
    }


def conflict_mask(predictions: np.ndarray, constrained: list[int]) -> np.ndarray:
    return (predictions[:, NEUTRAL] == 1) & np.any(predictions[:, constrained] == 1, axis=1)


def conflict_summary(raw: np.ndarray, conditioned: np.ndarray, maps: np.ndarray, constrained: list[int]) -> dict:
    raw_count = int(conflict_mask(raw, constrained).sum())
    conditioned_count = int(conflict_mask(conditioned, constrained).sum())
    map_count = int(conflict_mask(maps, constrained).sum())

    def eliminated(remaining: int) -> float | None:
        return None if raw_count == 0 else float((raw_count - remaining) / raw_count)

    return {
        "raw_conflicts": raw_count,
        "conditioned_conflicts": conditioned_count,
        "joint_map_conflicts": map_count,
        "conditioned_conflict_elimination_rate": eliminated(conditioned_count),
        "joint_map_conflict_elimination_rate": eliminated(map_count),
    }


def parent_metrics(gold: np.ndarray, marginals: dict[str, np.ndarray]) -> dict:
    output = {}
    for name, labels in SENTIMENT_GROUPS.items():
        indices = [INDEX[label] for label in labels]
        target = np.any(gold[:, indices] == 1, axis=1).astype(np.int8)
        probability = marginals[name]
        output[name] = {
            "auroc": roc_auc(target, probability),
            "auprc": average_precision(target, probability),
            "brier": float(np.mean((probability - target) ** 2)),
        }
    return output


def bootstrap_delta(
    gold: np.ndarray,
    raw_probabilities: np.ndarray,
    conditioned_probabilities: np.ndarray,
    raw_thresholds: np.ndarray,
    conditioned_thresholds: np.ndarray,
    n_boot: int = 1000,
) -> dict:
    rng = np.random.default_rng(0)
    values = {"micro_f1": [], "exact_match": [], "mean_brier": []}
    for _ in range(n_boot):
        indices = rng.integers(0, len(gold), len(gold))
        target = gold[indices]
        raw_p = raw_probabilities[indices]
        conditioned_p = conditioned_probabilities[indices]
        raw_pred = raw_p >= raw_thresholds
        conditioned_pred = conditioned_p >= conditioned_thresholds
        values["micro_f1"].append(
            f1_binary(target.ravel(), conditioned_pred.ravel())
            - f1_binary(target.ravel(), raw_pred.ravel())
        )
        values["exact_match"].append(
            np.mean(np.all(conditioned_pred == target, axis=1))
            - np.mean(np.all(raw_pred == target, axis=1))
        )
        values["mean_brier"].append(
            np.mean((conditioned_p - target) ** 2) - np.mean((raw_p - target) ** 2)
        )
    return {
        name: {
            "mean": float(np.mean(sample)),
            "ci95": [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))],
        }
        for name, sample in values.items()
    }


def evaluate_policy(
    name: str,
    constrained: list[int],
    eval_p: np.ndarray,
    eval_gold: np.ndarray,
    dev_p: np.ndarray,
    dev_gold: np.ndarray,
    artifact_dir: Path,
) -> dict:
    oracle = analytic_condition(eval_p, constrained)
    circuit = circuit_condition(eval_p, name, constrained, artifact_dir)
    oracle_p, oracle_z, oracle_map, oracle_map_w, oracle_parents = oracle
    circ_p, circ_z, circ_map, circ_map_w, circ_parents, circuit_info = circuit
    errors = {
        "z": float(np.max(np.abs(circ_z - oracle_z))),
        "marginal": float(np.max(np.abs(circ_p - oracle_p))),
        "map_weight": float(np.max(np.abs(circ_map_w - oracle_map_w))),
        "map_assignment_mismatches": int(np.sum(np.any(circ_map != oracle_map, axis=1))),
        "parent_marginal": float(
            max(np.max(np.abs(circ_parents[group] - oracle_parents[group])) for group in SENTIMENT_GROUPS)
        ),
    }
    if max(errors[key] for key in ("z", "marginal", "map_weight", "parent_marginal")) > 1e-12:
        raise AssertionError(f"SDD/analytic mismatch for {name}: {errors}")
    if errors["map_assignment_mismatches"]:
        raise AssertionError(f"SDD/analytic MAP mismatch for {name}: {errors}")

    dev_conditioned = analytic_condition(dev_p, constrained)[0]
    raw_thresholds = tune_thresholds(dev_gold, dev_p)
    conditioned_thresholds = tune_thresholds(dev_gold, dev_conditioned)
    fixed = np.full(len(LABELS), 0.5)
    raw_fixed = eval_p >= fixed
    cond_fixed = circ_p >= fixed
    raw_tuned = eval_p >= raw_thresholds
    cond_tuned = circ_p >= conditioned_thresholds

    return {
        "name": name,
        "constrained_emotions": [LABELS[i] for i in constrained],
        "n_constraints": len(constrained),
        "circuit": circuit_info,
        "exact_validation": errors,
        "valid_mass": {
            "mean_z": float(circ_z.mean()),
            "mean_invalid_mass": float(np.mean(1.0 - circ_z)),
            "min_z": float(circ_z.min()),
        },
        "fixed_0_5": {
            "raw": prediction_metrics(eval_gold, eval_p, fixed),
            "conditioned": prediction_metrics(eval_gold, circ_p, fixed),
            "conflicts": conflict_summary(raw_fixed, cond_fixed, circ_map, constrained),
            "paired_bootstrap_conditioned_minus_raw": bootstrap_delta(
                eval_gold, eval_p, circ_p, fixed, fixed
            ),
        },
        "dev_tuned": {
            "raw": prediction_metrics(eval_gold, eval_p, raw_thresholds),
            "conditioned": prediction_metrics(eval_gold, circ_p, conditioned_thresholds),
            "thresholds_changed": int(np.sum(raw_thresholds != conditioned_thresholds)),
            "conflicts": conflict_summary(raw_tuned, cond_tuned, circ_map, constrained),
        },
        "joint_map": decision_metrics(eval_gold, circ_map),
        "paired_bootstrap_joint_map_minus_raw": bootstrap_decision_delta(
            eval_gold, raw_fixed.astype(np.int8), circ_map
        ),
        "parent_queries": parent_metrics(eval_gold, circ_parents),
    }


def run(eval_path: Path, dev_path: Path, db_path: Path, artifact_dir: Path) -> dict:
    eval_p, original_eval_gold = load_rows(eval_path, db_path)
    dev_frame = pd.read_parquet(dev_path)
    dev_indices = dev_order(len(dev_frame), 1000)
    dev_p, original_dev_gold_sample = load_rows(dev_path, db_path, dev_indices)
    original_full_dev_gold = gold_from_frame(dev_path)
    repaired_eval_gold, eval_repairs = canonicalize_neutral(original_eval_gold)
    repaired_dev_gold_sample, dev_sample_repairs = canonicalize_neutral(original_dev_gold_sample)
    _, full_dev_repairs = canonicalize_neutral(original_full_dev_gold)
    all_emotions = list(range(NEUTRAL))

    return {
        "setup": {
            "model": MODEL,
            "n_eval": int(len(eval_p)),
            "n_dev_for_thresholds": int(len(dev_p)),
            "variables": list(LABELS),
            "base_worlds": 1 << len(LABELS),
        },
        "gold_repair": {
            "ontology": "Neutral -> not Emotion for all 27 specific emotions.",
            "operation": "If neutral co-occurs with any specific emotion, remove neutral and retain all specific labels.",
            "model_independent": True,
            "full_validation_examples_repaired": full_dev_repairs,
            "threshold_dev_examples_repaired": dev_sample_repairs,
            "test_examples_repaired": eval_repairs,
            "post_repair_violations": int(
                np.sum(
                    (repaired_eval_gold[:, NEUTRAL] == 1)
                    & np.any(repaired_eval_gold[:, :NEUTRAL] == 1, axis=1)
                )
            ),
        },
        "primary": evaluate_policy(
            "full_neutral_disjoint", all_emotions, eval_p, repaired_eval_gold,
            dev_p, repaired_dev_gold_sample, artifact_dir
        ),
        "original_gold_sensitivity": evaluate_policy(
            "full_neutral_disjoint_original_gold", all_emotions, eval_p, original_eval_gold,
            dev_p, original_dev_gold_sample, artifact_dir
        ),
    }


def row(label: str, raw: float, conditioned: float) -> str:
    return f"| {label} | {raw:.6f} | {conditioned:.6f} | {conditioned - raw:+.6f} |"


def markdown(result: dict) -> str:
    primary = result["primary"]
    sensitivity = result["original_gold_sensitivity"]
    repair = result["gold_repair"]
    exact = primary["exact_validation"]
    fixed = primary["fixed_0_5"]
    tuned = primary["dev_tuned"]
    conflict = fixed["conflicts"]
    boot = fixed["paired_bootstrap_conditioned_minus_raw"]
    tuned_conflict = tuned["conflicts"]
    map_boot = primary["paired_bootstrap_joint_map_minus_raw"]
    metric_rows = "\n".join(
        row(name, fixed["raw"][key], fixed["conditioned"][key])
        for name, key in (
            ("Micro-F1", "micro_f1"), ("Macro-F1", "macro_f1"), ("Exact match", "exact_match"),
            ("Mean AUPRC", "mean_auprc"), ("Mean Brier", "mean_brier"), ("Mean ECE", "mean_ece"),
            ("Marginal NLL", "marginal_nll"),
        )
    )
    tuned_rows = "\n".join(
        row(name, tuned["raw"][key], tuned["conditioned"][key])
        for name, key in (("Micro-F1", "micro_f1"), ("Macro-F1", "macro_f1"), ("Exact match", "exact_match"))
    )
    parent_rows = "\n".join(
        f"| {name} | {metrics['auroc']:.6f} | {metrics['auprc']:.6f} | {metrics['brier']:.6f} |"
        for name, metrics in primary["parent_queries"].items()
    )
    return f"""# GoEmotions ontology-conditioned SDD pilot

## Setup and ontology audit

- Model: `{result['setup']['model']}`
- Evaluation samples: {result['setup']['n_eval']}
- Variables: 28 Noul heads; base space: $2^{{28}}={result['setup']['base_worlds']:,}$ worlds
- Primary ontology: `Neutral -> not Emotion` for all 27 specific emotions
- Gold canonicalization: remove `neutral` when any specific emotion is present
- Repaired examples: {repair['full_validation_examples_repaired']} on full validation; {repair['test_examples_repaired']} on test
- Post-repair ontology violations: {repair['post_repair_violations']}

## Exact circuit validation

**Status: PASS.** The SDD agrees with an independent analytic oracle on every sample.

| Quantity | Maximum absolute error |
|---|---:|
| Valid mass $Z$ | {exact['z']:.3e} |
| 28 typed marginals | {exact['marginal']:.3e} |
| Three parent-concept marginals | {exact['parent_marginal']:.3e} |
| MAP weight | {exact['map_weight']:.3e} |
| MAP assignments | {exact['map_assignment_mismatches']} mismatches |

The primary SDD has {primary['circuit']['node_count']} nodes and {primary['circuit']['size_elements']} elements, represents {primary['circuit']['model_count']:,} valid worlds, and reloads with the same model count. Mean inference time is {primary['circuit']['milliseconds_per_sample']:.3f} ms per sample, including $Z$, 28 leaf marginals, three ontology-derived parent marginals, and joint MAP.

## Primary results at threshold 0.5

| Metric | Raw | Conditioned | Delta |
|---|---:|---:|---:|
{metric_rows}

Mean valid mass $Z$: {primary['valid_mass']['mean_z']:.6f}
Mean invalid mass $1-Z$: {primary['valid_mass']['mean_invalid_mass']:.6f}
Raw conflicts: {conflict['raw_conflicts']}
Conditioned-marginal conflicts: {conflict['conditioned_conflicts']}
Conflict elimination by conditioned marginals: {conflict['conditioned_conflict_elimination_rate']:.1%}
Conflict elimination by joint MAP: {conflict['joint_map_conflict_elimination_rate']:.1%}

Joint-MAP decisions have Micro-F1 {primary['joint_map']['micro_f1']:.6f}, Macro-F1 {primary['joint_map']['macro_f1']:.6f}, and exact match {primary['joint_map']['exact_match']:.6f}.

Paired bootstrap deltas for joint MAP minus raw local decisions:

- Micro-F1: {map_boot['micro_f1_delta']['mean']:+.6f}, 95% CI [{map_boot['micro_f1_delta']['ci95'][0]:+.6f}, {map_boot['micro_f1_delta']['ci95'][1]:+.6f}]
- Exact match: {map_boot['exact_match_delta']['mean']:+.6f}, 95% CI [{map_boot['exact_match_delta']['ci95'][0]:+.6f}, {map_boot['exact_match_delta']['ci95'][1]:+.6f}]

Paired bootstrap deltas for conditioned marginals minus raw:

- Micro-F1: {boot['micro_f1']['mean']:+.6f}, 95% CI [{boot['micro_f1']['ci95'][0]:+.6f}, {boot['micro_f1']['ci95'][1]:+.6f}]
- Exact match: {boot['exact_match']['mean']:+.6f}, 95% CI [{boot['exact_match']['ci95'][0]:+.6f}, {boot['exact_match']['ci95'][1]:+.6f}]
- Mean Brier: {boot['mean_brier']['mean']:+.6f}, 95% CI [{boot['mean_brier']['ci95'][0]:+.6f}, {boot['mean_brier']['ci95'][1]:+.6f}]

## Results with independently retuned development thresholds

Raw and conditioned thresholds are tuned separately on the same 1,000-example development sample, then frozen for test.

| Metric | Raw | Conditioned | Delta |
|---|---:|---:|---:|
{tuned_rows}

Conditioned thresholds differ on {tuned['thresholds_changed']} / 28 heads. Conflict elimination under the separately tuned thresholds is {tuned_conflict['conditioned_conflict_elimination_rate']:.1%}; joint MAP remains 100% consistent.

## Ontology-derived parent queries

The official GoEmotions sentiment mapping defines positive, negative, and ambiguous parent concepts. Their probabilities are computed as union queries over the conditioned joint, without additional Jev calls.

| Parent concept | AUROC | AUPRC | Brier |
|---|---:|---:|---:|
{parent_rows}

## Original aggregated-gold sensitivity

Using the untouched aggregated labels leaves {repair['test_examples_repaired']} ontology-conflicting test examples. Against those original labels, Micro-F1 changes from {sensitivity['fixed_0_5']['raw']['micro_f1']:.6f} to {sensitivity['fixed_0_5']['conditioned']['micro_f1']:.6f}, and mean Brier changes from {sensitivity['fixed_0_5']['raw']['mean_brier']:.6f} to {sensitivity['fixed_0_5']['conditioned']['mean_brier']:.6f}. This comparison isolates the effect of evaluating a logically constrained model against uncanonicalized multi-annotator labels.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.eval_data, args.dev_data, args.responses, args.artifact_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report_out.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
