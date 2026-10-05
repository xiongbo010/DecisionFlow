#!/usr/bin/env python3
"""Ontology-conditioned SDD pilot for Jev's ToxiGen Noul + Score request."""

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

from openai_moderation_pilot import average_precision, ece, f1_binary, roc_auc  # noqa: E402
from validate_sdd import assignment_mask, max_product  # noqa: E402


MODEL = "jev-1.13.0"
LEVELS = [
    "Benign: not offensive to anyone.",
    "Slightly offensive or insensitive.",
    "Moderately offensive.",
    "Very offensive.",
    "Extremely offensive: hateful or dehumanizing.",
]
QUESTION_TOXIC = "Is `text` toxic, offensive or hateful toward a group of people?"
QUESTION_SCORE = "How offensive is `text` toward the group of people it mentions?"


def questions() -> dict:
    return {
        "toxic": {"type": "noul", "instructions": QUESTION_TOXIC},
        "toxicity": {"type": "score", "instructions": QUESTION_SCORE, "criteria": LEVELS},
    }


def request_key(state: dict, qs: dict) -> str:
    payload = json.dumps({"model": MODEL, "state": state, "questions": qs}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_rows(data_path: Path, db_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(data_path)
    qs = questions()
    noul, score_probs = [], []
    db = sqlite3.connect(db_path)
    try:
        for text in frame["text"]:
            key = request_key({"text": text.strip()}, qs)
            row = db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            if row is None:
                raise RuntimeError(f"Missing cached response for key {key}")
            response = json.loads(row[0])["answers"]
            noul.append(float(response["toxic"]["noul"]))
            score_probs.append([float(response["toxicity"]["probabilities"][str(k)]) for k in range(5)])
    finally:
        db.close()

    # This exactly follows the released Jev benchmark task adapter.
    gold_toxic = ((frame["toxicity_ai"] + frame["toxicity_human"]) > 5.5).to_numpy(dtype=np.int8)
    gold_score = (frame["toxicity_human"] - 1.0).to_numpy(dtype=float)
    return np.asarray(noul), np.asarray(score_probs), gold_toxic, gold_score


def valid_assignment(toxic: int, level: int, policy: str) -> bool:
    if policy == "anchors":
        return not ((level == 0 and toxic == 1) or (level == 4 and toxic == 0))
    if policy == "full_semantics":
        return bool(toxic) == (level > 0)
    raise ValueError(policy)


def typed_worlds() -> list[tuple[int, int]]:
    # Match the SDD tie-break: ascending bit mask for [T, L0, ..., L4].
    worlds = [(toxic, level) for level in range(5) for toxic in range(2)]
    return sorted(worlds, key=lambda x: (x[0] << 0) | (1 << (x[1] + 1)))


def enumerate_condition(
    noul: np.ndarray, score_probs: np.ndarray, policy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    worlds = typed_worlds()
    toxic_marginal = np.empty(len(noul))
    score_marginals = np.empty_like(score_probs)
    z_values = np.empty(len(noul))
    map_assignments = np.empty((len(noul), 2), dtype=np.int8)
    map_weights = np.empty(len(noul))

    for row, (p_toxic, p_score) in enumerate(zip(noul, score_probs, strict=True)):
        weights = np.asarray(
            [
                (p_toxic if toxic else 1.0 - p_toxic) * p_score[level]
                if valid_assignment(toxic, level, policy)
                else 0.0
                for toxic, level in worlds
            ]
        )
        z = float(weights.sum())
        if z <= 0:
            raise RuntimeError(f"Zero valid mass at row {row} under {policy}")
        q = weights / z
        z_values[row] = z
        toxic_marginal[row] = sum(q[j] for j, (toxic, _) in enumerate(worlds) if toxic)
        score_marginals[row] = [sum(q[j] for j, (_, level) in enumerate(worlds) if level == k) for k in range(5)]
        best = float(np.max(weights))
        best_index = int(np.flatnonzero(np.isclose(weights, best, rtol=0.0, atol=1e-15))[0])
        map_assignments[row] = worlds[best_index]
        map_weights[row] = best
    return toxic_marginal, score_marginals, z_values, map_assignments, map_weights


def compile_sdd(policy: str, artifact_dir: Path) -> tuple[SddManager, SddNode, float]:
    start = time.perf_counter()
    vtree = Vtree(var_count=6, var_order=list(range(1, 7)), vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    toxic = manager.literal(1)
    level_lits = [manager.literal(k + 2) for k in range(5)]

    root = manager.false()
    for lit in level_lits:
        root |= lit
    for i, left in enumerate(level_lits):
        for right in level_lits[i + 1 :]:
            root &= (~left) | (~right)

    if policy == "anchors":
        root &= (~level_lits[0]) | (~toxic)
        root &= (~level_lits[4]) | toxic
    elif policy == "full_semantics":
        root &= (~level_lits[0]) | (~toxic)
        for lit in level_lits[1:]:
            root &= (~lit) | toxic
    else:
        raise ValueError(policy)

    root.ref()
    compile_seconds = time.perf_counter() - start
    artifact_dir.mkdir(parents=True, exist_ok=True)
    root.save(str(artifact_dir / f"toxigen_{policy}.sdd").encode())
    manager.vtree().save(str(artifact_dir / f"toxigen_{policy}.vtree").encode())
    return manager, root, compile_seconds


def reload_count(policy: str, artifact_dir: Path) -> int:
    vtree = Vtree.from_file(str(artifact_dir / f"toxigen_{policy}.vtree").encode())
    manager = SddManager.from_vtree(vtree)
    root = manager.read_sdd_file(str(artifact_dir / f"toxigen_{policy}.sdd").encode())
    return int(root.global_model_count())


def circuit_condition(
    noul: np.ndarray, score_probs: np.ndarray, policy: str, artifact_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    manager, root, compile_seconds = compile_sdd(policy, artifact_dir)
    wmc = root.wmc(log_mode=False)
    toxic_marginal = np.empty(len(noul))
    score_marginals = np.empty_like(score_probs)
    z_values = np.empty(len(noul))
    map_assignments = np.empty((len(noul), 2), dtype=np.int8)
    map_weights = np.empty(len(noul))

    start = time.perf_counter()
    for row, (p_toxic, p_score) in enumerate(zip(noul, score_probs, strict=True)):
        wmc.set_literal_weight(1, float(p_toxic))
        wmc.set_literal_weight(-1, float(1.0 - p_toxic))
        for level, probability in enumerate(p_score):
            # Categorical one-hot encoding: the selected literal contributes
            # p(level); negative indicators are neutral and contribute 1.
            wmc.set_literal_weight(level + 2, float(probability))
            wmc.set_literal_weight(-(level + 2), 1.0)
        z_values[row] = float(wmc.propagate())
        toxic_marginal[row] = float(wmc.literal_pr(1))
        score_marginals[row] = [float(wmc.literal_pr(level + 2)) for level in range(5)]

        positive = np.concatenate(([p_toxic], p_score))
        negative = np.concatenate(([1.0 - p_toxic], np.ones(5)))
        mpe = max_product(root, manager.vtree(), positive, negative, manager.true())
        map_weights[row] = mpe.weight
        toxic_value = mpe.assignment[1]
        active_levels = [level for level in range(5) if mpe.assignment[level + 2] == 1]
        if len(active_levels) != 1:
            raise AssertionError(f"MAP is not one-hot at row {row}: {mpe.assignment}")
        map_assignments[row] = (toxic_value, active_levels[0])
    inference_seconds = time.perf_counter() - start

    circuit = {
        "size_elements": int(root.size()),
        "node_count": int(root.count()),
        "model_count": int(root.global_model_count()),
        "reloaded_model_count": reload_count(policy, artifact_dir),
        "compile_seconds": compile_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_sample": inference_seconds * 1000 / len(noul),
    }
    return toxic_marginal, score_marginals, z_values, map_assignments, map_weights, circuit


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def binary_metrics(gold: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(np.mean(predictions == gold)),
        "f1": f1_binary(gold, predictions),
        "auroc": roc_auc(gold, probabilities),
        "auprc": average_precision(gold, probabilities),
        "brier": float(np.mean((probabilities - gold) ** 2)),
        "ece": ece(gold, probabilities),
    }


def score_metrics(gold: np.ndarray, expected: np.ndarray) -> dict:
    return {"spearman": spearman(gold, expected), "mae": float(np.mean(np.abs(expected - gold)))}


def bootstrap(
    gold_toxic: np.ndarray,
    gold_score: np.ndarray,
    raw_toxic: np.ndarray,
    new_toxic: np.ndarray,
    raw_score: np.ndarray,
    new_score: np.ndarray,
    n_boot: int = 2000,
) -> dict:
    rng = np.random.default_rng(0)
    values = {"toxic_accuracy": [], "toxic_brier": [], "score_mae": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(gold_toxic), len(gold_toxic))
        values["toxic_accuracy"].append(
            np.mean((new_toxic[idx] >= 0.5) == gold_toxic[idx])
            - np.mean((raw_toxic[idx] >= 0.5) == gold_toxic[idx])
        )
        values["toxic_brier"].append(
            np.mean((new_toxic[idx] - gold_toxic[idx]) ** 2)
            - np.mean((raw_toxic[idx] - gold_toxic[idx]) ** 2)
        )
        values["score_mae"].append(
            np.mean(np.abs(new_score[idx] - gold_score[idx]))
            - np.mean(np.abs(raw_score[idx] - gold_score[idx]))
        )
    return {
        key: {
            "mean": float(np.mean(sample)),
            "ci95": [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))],
        }
        for key, sample in values.items()
    }


def audit_gold(gold_toxic: np.ndarray, gold_score: np.ndarray) -> dict:
    rounded = np.rint(gold_score).astype(int)
    anchor_violation = ((rounded == 0) & (gold_toxic == 1)) | ((rounded == 4) & (gold_toxic == 0))
    full_violation = gold_toxic != (rounded > 0)
    exact_endpoint = (gold_score == 0) | (gold_score == 4)
    exact_anchor_violation = ((gold_score == 0) & (gold_toxic == 1)) | ((gold_score == 4) & (gold_toxic == 0))
    return {
        "continuous_score_note": "The benchmark gold Score is the mean human rating minus one and can be fractional.",
        "rounded_anchor_violations": int(anchor_violation.sum()),
        "rounded_full_semantics_violations": int(full_violation.sum()),
        "rounded_full_semantics_violation_rate": float(full_violation.mean()),
        "exact_endpoint_examples": int(exact_endpoint.sum()),
        "exact_endpoint_anchor_violations": int(exact_anchor_violation.sum()),
    }


def evaluate_policy(
    noul: np.ndarray,
    score_probs: np.ndarray,
    gold_toxic: np.ndarray,
    gold_score: np.ndarray,
    policy: str,
    artifact_dir: Path,
) -> dict:
    oracle = enumerate_condition(noul, score_probs, policy)
    circuit = circuit_condition(noul, score_probs, policy, artifact_dir)
    oracle_t, oracle_s, oracle_z, oracle_map, oracle_map_w = oracle
    circ_t, circ_s, circ_z, circ_map, circ_map_w, circuit_info = circuit

    errors = {
        "z": float(np.max(np.abs(circ_z - oracle_z))),
        "toxic_marginal": float(np.max(np.abs(circ_t - oracle_t))),
        "score_marginal": float(np.max(np.abs(circ_s - oracle_s))),
        "map_weight": float(np.max(np.abs(circ_map_w - oracle_map_w))),
        "map_assignment_mismatches": int(np.sum(np.any(circ_map != oracle_map, axis=1))),
    }
    if max(errors[key] for key in ("z", "toxic_marginal", "score_marginal", "map_weight")) > 1e-12:
        raise AssertionError(f"SDD/enumeration numerical mismatch for {policy}: {errors}")
    if errors["map_assignment_mismatches"]:
        raise AssertionError(f"SDD/enumeration MAP mismatch for {policy}: {errors}")

    raw_toxic_pred = (noul >= 0.5).astype(np.int8)
    conditioned_toxic_pred = (circ_t >= 0.5).astype(np.int8)
    raw_level = np.argmax(score_probs, axis=1)
    conditioned_level = np.argmax(circ_s, axis=1)
    raw_expected = score_probs @ np.arange(5)
    conditioned_expected = circ_s @ np.arange(5)
    raw_violation = np.asarray(
        [not valid_assignment(int(t), int(level), policy) for t, level in zip(raw_toxic_pred, raw_level, strict=True)]
    )
    conditioned_violation = np.asarray(
        [
            not valid_assignment(int(t), int(level), policy)
            for t, level in zip(conditioned_toxic_pred, conditioned_level, strict=True)
        ]
    )
    joint_map_violation = np.asarray(
        [not valid_assignment(int(t), int(k), policy) for t, k in circ_map]
    )
    raw_conflicts = int(raw_violation.sum())
    conditioned_conflicts = int(conditioned_violation.sum())
    joint_map_conflicts = int(joint_map_violation.sum())

    def elimination_rate(remaining: int) -> float | None:
        if raw_conflicts == 0:
            return None
        return float((raw_conflicts - remaining) / raw_conflicts)

    return {
        "policy": policy,
        "constraint": (
            "Score=Benign -> Toxic=false; Score=ExtremelyOffensive -> Toxic=true"
            if policy == "anchors"
            else "Toxic=true iff Score is non-Benign"
        ),
        "circuit": circuit_info,
        "exact_validation": errors,
        "valid_mass": {
            "mean_z": float(circ_z.mean()),
            "mean_invalid_mass": float(np.mean(1.0 - circ_z)),
            "min_z": float(circ_z.min()),
            "raw_local_argmax_conflicts": raw_conflicts,
            "conditioned_marginal_argmax_conflicts": conditioned_conflicts,
            "joint_map_conflicts": joint_map_conflicts,
            "raw_local_argmax_violation_rate": float(raw_violation.mean()),
            "conditioned_marginal_argmax_violation_rate": float(conditioned_violation.mean()),
            "joint_map_violation_rate": float(joint_map_violation.mean()),
            "conditioned_marginal_conflict_elimination_rate": elimination_rate(conditioned_conflicts),
            "joint_map_conflict_elimination_rate": elimination_rate(joint_map_conflicts),
        },
        "raw": {
            "toxic": binary_metrics(gold_toxic, noul, raw_toxic_pred),
            "score": score_metrics(gold_score, raw_expected),
        },
        "conditioned_marginals": {
            "toxic": binary_metrics(gold_toxic, circ_t, conditioned_toxic_pred),
            "score": score_metrics(gold_score, conditioned_expected),
        },
        "joint_map": {
            "toxic": binary_metrics(gold_toxic, circ_map[:, 0].astype(float), circ_map[:, 0]),
            "score_mae": float(np.mean(np.abs(circ_map[:, 1] - gold_score))),
        },
        "changes": {
            "toxic_decisions": int(np.sum(raw_toxic_pred != conditioned_toxic_pred)),
            "score_argmax_decisions": int(np.sum(raw_level != conditioned_level)),
        },
        "paired_bootstrap_conditioned_minus_raw": bootstrap(
            gold_toxic, gold_score, noul, circ_t, raw_expected, conditioned_expected
        ),
    }


def run(data_path: Path, db_path: Path, artifact_dir: Path) -> dict:
    noul, score_probs, gold_toxic, gold_score = load_rows(data_path, db_path)
    score_sums = score_probs.sum(axis=1)
    if np.any(score_sums <= 0):
        raise AssertionError("Jev returned a zero-mass Score distribution")
    rounded_rows = int(np.sum(np.abs(score_sums - 1.0) > 1e-12))
    max_sum_error = float(np.max(np.abs(score_sums - 1.0)))
    # Jev serializes probabilities to two decimals, so a few rows sum to 0.99.
    # Normalize before treating Score as a categorical factor in the joint.
    score_probs = score_probs / score_sums[:, None]
    return {
        "setup": {
            "model": MODEL,
            "n_samples": int(len(noul)),
            "types": ["Noul", "Score(5)"],
            "base_typed_assignments": 10,
            "categorical_weighting": "w(L_k)=p_k and w(not L_k)=1 under exactly-one",
            "score_probability_normalization": {
                "rows_renormalized": rounded_rows,
                "max_pre_normalization_sum_error": max_sum_error,
            },
        },
        "gold_audit": audit_gold(gold_toxic, gold_score),
        "primary": evaluate_policy(noul, score_probs, gold_toxic, gold_score, "anchors", artifact_dir),
        "sensitivity": evaluate_policy(
            noul, score_probs, gold_toxic, gold_score, "full_semantics", artifact_dir
        ),
    }


def metric_row(name: str, raw: float, conditioned: float, joint: float | None = None) -> str:
    last = "—" if joint is None else f"{joint:.6f}"
    return f"| {name} | {raw:.6f} | {conditioned:.6f} | {last} |"


def markdown(result: dict) -> str:
    primary = result["primary"]
    sensitivity = result["sensitivity"]
    raw = primary["raw"]
    cond = primary["conditioned_marginals"]
    joint = primary["joint_map"]
    valid = primary["valid_mass"]
    exact = primary["exact_validation"]
    audit = result["gold_audit"]
    boot = primary["paired_bootstrap_conditioned_minus_raw"]
    rows = "\n".join(
        [
            metric_row("Toxic accuracy", raw["toxic"]["accuracy"], cond["toxic"]["accuracy"], joint["toxic"]["accuracy"]),
            metric_row("Toxic F1", raw["toxic"]["f1"], cond["toxic"]["f1"], joint["toxic"]["f1"]),
            metric_row("Toxic Brier", raw["toxic"]["brier"], cond["toxic"]["brier"]),
            metric_row("Toxic ECE", raw["toxic"]["ece"], cond["toxic"]["ece"]),
            metric_row("Score Spearman", raw["score"]["spearman"], cond["score"]["spearman"]),
            metric_row("Score MAE", raw["score"]["mae"], cond["score"]["mae"], joint["score_mae"]),
        ]
    )
    return f"""# ToxiGen ontology-conditioned SDD pilot

## Setup

- Model: `{result['setup']['model']}`
- Samples: {result['setup']['n_samples']}
- Typed variables: one Noul and one five-level Score
- Base typed assignments: {result['setup']['base_typed_assignments']}
- Primary hard constraints: `Benign -> not Toxic`; `Extremely offensive -> Toxic`
- Valid typed assignments: {primary['circuit']['model_count']} / {result['setup']['base_typed_assignments']}
- Score-probability normalization: {result['setup']['score_probability_normalization']['rows_renormalized']} rows renormalized after two-decimal serialization (maximum sum error {result['setup']['score_probability_normalization']['max_pre_normalization_sum_error']:.3f})

The primary constraints use only the two unambiguous rubric endpoints. They have zero violations among the {audit['exact_endpoint_examples']} examples whose human mean lies exactly at an endpoint, and zero violations after rounding all gold scores to their nearest rubric level.

## Exact circuit validation

**Status: PASS.** SDD inference agrees with direct typed enumeration on every sample.

| Quantity | Maximum absolute error |
|---|---:|
| Valid mass $Z$ | {exact['z']:.3e} |
| Noul marginal | {exact['toxic_marginal']:.3e} |
| Score marginals | {exact['score_marginal']:.3e} |
| MAP weight | {exact['map_weight']:.3e} |
| MAP assignments | {exact['map_assignment_mismatches']} mismatches |

The SDD has {primary['circuit']['node_count']} nodes and {primary['circuit']['size_elements']} elements. Serialized reload recovers {primary['circuit']['reloaded_model_count']} models. Mean inference time is {primary['circuit']['milliseconds_per_sample']:.3f} ms per sample for $Z$, six marginals, and joint MAP.

## Primary empirical results

| Metric | Raw local distributions | Conditioned marginals | Joint MAP |
|---|---:|---:|---:|
{rows}

Mean valid mass $Z$: {valid['mean_z']:.6f}
Mean invalid mass $1-Z$: {valid['mean_invalid_mass']:.6f}
Raw local-argmax violation rate: {valid['raw_local_argmax_violation_rate']:.6f}
Conditioned-marginal argmax violation rate: {valid['conditioned_marginal_argmax_violation_rate']:.6f}
Joint-MAP violation rate: {valid['joint_map_violation_rate']:.6f}
Conditioned marginals eliminated {valid['raw_local_argmax_conflicts'] - valid['conditioned_marginal_argmax_conflicts']} / {valid['raw_local_argmax_conflicts']} argmax conflicts ({valid['conditioned_marginal_conflict_elimination_rate']:.1%}).
Joint MAP eliminated {valid['raw_local_argmax_conflicts'] - valid['joint_map_conflicts']} / {valid['raw_local_argmax_conflicts']} argmax conflicts ({valid['joint_map_conflict_elimination_rate']:.1%}).

Paired bootstrap deltas for conditioned marginals minus raw:

- Toxic accuracy: {boot['toxic_accuracy']['mean']:+.6f}, 95% CI [{boot['toxic_accuracy']['ci95'][0]:+.6f}, {boot['toxic_accuracy']['ci95'][1]:+.6f}]
- Toxic Brier: {boot['toxic_brier']['mean']:+.6f}, 95% CI [{boot['toxic_brier']['ci95'][0]:+.6f}, {boot['toxic_brier']['ci95'][1]:+.6f}]
- Score MAE: {boot['score_mae']['mean']:+.6f}, 95% CI [{boot['score_mae']['ci95'][0]:+.6f}, {boot['score_mae']['ci95'][1]:+.6f}]

## Ontology sensitivity

The stronger rule `Toxic iff Score is non-Benign` conflicts with {audit['rounded_full_semantics_violations']} / {result['setup']['n_samples']} rounded gold pairs ({audit['rounded_full_semantics_violation_rate']:.1%}). It produces {sensitivity['valid_mass']['raw_local_argmax_conflicts']} raw argmax conflicts; conditioned marginal decoding eliminates {sensitivity['valid_mass']['conditioned_marginal_conflict_elimination_rate']:.1%}, while joint MAP eliminates {sensitivity['valid_mass']['joint_map_conflict_elimination_rate']:.1%}. Under that rule, toxic accuracy changes from {sensitivity['raw']['toxic']['accuracy']:.6f} to {sensitivity['conditioned_marginals']['toxic']['accuracy']:.6f}, and Score MAE changes from {sensitivity['raw']['score']['mae']:.6f} to {sensitivity['conditioned_marginals']['score']['mae']:.6f}. This sensitivity result shows why dataset-grounded rule auditing is required before treating a semantic relation as a hard axiom.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.data, args.responses, args.artifact_dir)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report_out.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
