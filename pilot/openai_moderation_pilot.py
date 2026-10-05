#!/usr/bin/env python3
"""No-training pilot for ontology-conditioned Jev moderation decisions.

The script reconstructs the exact requests used by the public Jev benchmark,
loads the released Jev 1.13 responses from its SQLite cache, and performs exact
inference by enumerating the 2^8 Boolean worlds. Enumeration is intentionally
used here as a transparent oracle; a circuit backend should reproduce the same
numbers before being used for larger decision spaces.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np


MODEL = "jev-1.13.0"
CATEGORIES = {
    "S": (
        "sexual",
        "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
    ),
    "H": (
        "hate",
        "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
    ),
    "V": (
        "violence",
        "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
    ),
    "HR": (
        "harassment",
        "Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.",
    ),
    "SH": (
        "self-harm",
        "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
    ),
    "S3": ("sexual/minors", "Sexual content that includes an individual who is under 18 years old."),
    "H2": (
        "hate/threatening",
        "Hateful content that also includes violence or serious harm towards the targeted group.",
    ),
    "V2": (
        "violence/graphic",
        "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    ),
}
HEADS = list(CATEGORIES)
INDEX = {head: i for i, head in enumerate(HEADS)}

# Only taxonomy-preserving implications with zero gold violations are used.
IMPLICATIONS = (("S3", "S"), ("H2", "H"), ("V2", "V"))


def questions() -> dict[str, dict]:
    return {
        code: {
            "type": "noul",
            "instructions": {
                "category": {"name": name, "definition": definition},
                "question": "Does `text` contain content that falls under `category`?",
            },
        }
        for code, (name, definition) in CATEGORIES.items()
    }


def request_key(state: dict, qs: dict) -> str:
    payload = json.dumps({"model": MODEL, "state": state, "questions": qs}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_rows(data_path: Path, db_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    with gzip.open(data_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    qs = questions()
    db = sqlite3.connect(db_path)
    states, probs, gold, known = [], [], [], []
    missing = []
    try:
        for row in rows:
            state = {"text": row["prompt"]}
            key = request_key(state, qs)
            result = db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            if result is None:
                missing.append(key)
                continue
            response = json.loads(result[0])
            states.append(state)
            probs.append([float(response["answers"][h]["noul"]) for h in HEADS])
            # This matches the benchmark loader: a missing/None category is not a positive label.
            gold.append([int(row.get(h) == 1) for h in HEADS])
            known.append([int(h in row and row[h] is not None) for h in HEADS])
    finally:
        db.close()

    if missing:
        raise RuntimeError(f"Missing {len(missing)} cached responses; first key: {missing[0]}")
    return states, np.asarray(probs), np.asarray(gold), np.asarray(known)


def enumerate_worlds() -> tuple[np.ndarray, np.ndarray]:
    worlds = np.asarray(
        [[(mask >> i) & 1 for i in range(len(HEADS))] for mask in range(1 << len(HEADS))],
        dtype=np.int8,
    )
    valid = np.ones(len(worlds), dtype=bool)
    for child, parent in IMPLICATIONS:
        valid &= ~((worlds[:, INDEX[child]] == 1) & (worlds[:, INDEX[parent]] == 0))
    return worlds, valid


def exact_condition(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    worlds, valid = enumerate_worlds()
    valid_worlds = worlds[valid]
    marginals = np.empty_like(probs)
    z_values = np.empty(len(probs))
    maps = np.empty_like(probs, dtype=np.int8)

    for i, p in enumerate(probs):
        weights = np.prod(np.where(valid_worlds == 1, p, 1.0 - p), axis=1)
        z = float(weights.sum())
        if z <= 0:
            raise RuntimeError(f"Zero valid mass at row {i}")
        z_values[i] = z
        marginals[i] = (weights[:, None] * valid_worlds).sum(axis=0) / z
        # Use a deterministic world-order tie-break.  Algebraically identical
        # products can differ at the last floating-point bit when their factors
        # are multiplied in a different order.
        best = float(np.max(weights))
        tied = np.flatnonzero(np.isclose(weights, best, rtol=0.0, atol=1e-15))
        maps[i] = valid_worlds[int(tied[0])]
    return marginals, z_values, maps


def f1_binary(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 2 * tp / den


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="stable")
    ranked = y[order]
    positives = int(ranked.sum())
    if positives == 0:
        return math.nan
    ranked_score = score[order]
    ap = 0.0
    seen = 0
    true_seen = 0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked_score[end] == ranked_score[start]:
            end += 1
        block_true = int(ranked[start:end].sum())
        seen += end - start
        true_seen += block_true
        ap += (block_true / positives) * (true_seen / seen)
        start = end
    return ap


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return math.nan
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    ranks = np.arange(1, len(y) + 1, dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2
        start = end
    rank_sum_pos = float(ranks[np.argsort(order)][y == 1].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mask = (p >= lo) & ((p < hi) if b < bins - 1 else (p <= hi))
        if np.any(mask):
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def probability_metrics(gold: np.ndarray, probs: np.ndarray) -> dict:
    clipped = np.clip(probs, 1e-12, 1 - 1e-12)
    return {
        "mean_auprc": float(np.nanmean([average_precision(gold[:, j], probs[:, j]) for j in range(gold.shape[1])])),
        "mean_brier": float(np.mean((probs - gold) ** 2)),
        "mean_ece": float(np.mean([ece(gold[:, j], probs[:, j]) for j in range(gold.shape[1])])),
        "marginal_nll": float(np.mean(-(gold * np.log(clipped) + (1 - gold) * np.log(1 - clipped)))),
    }


def decision_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "micro_f1": f1_binary(gold.ravel(), pred.ravel()),
        "macro_f1": float(np.mean([f1_binary(gold[:, j], pred[:, j]) for j in range(gold.shape[1])])),
        "exact_match": float(np.mean(np.all(gold == pred, axis=1))),
    }


def known_only_metrics(gold: np.ndarray, probs: np.ndarray, pred: np.ndarray, known: np.ndarray) -> dict:
    clipped = np.clip(probs, 1e-12, 1 - 1e-12)
    full = np.all(known == 1, axis=1)
    return {
        "n_known_labels": int(known.sum()),
        "n_fully_known_examples": int(full.sum()),
        "mean_auprc": float(
            np.nanmean(
                [average_precision(gold[known[:, j] == 1, j], probs[known[:, j] == 1, j]) for j in range(gold.shape[1])]
            )
        ),
        "mean_brier": float(np.mean(((probs - gold) ** 2)[known == 1])),
        "mean_ece": float(
            np.mean([ece(gold[known[:, j] == 1, j], probs[known[:, j] == 1, j]) for j in range(gold.shape[1])])
        ),
        "marginal_nll": float(
            np.mean((-(gold * np.log(clipped) + (1 - gold) * np.log(1 - clipped)))[known == 1])
        ),
        "micro_f1": f1_binary(gold[known == 1], pred[known == 1]),
        "macro_f1": float(
            np.mean(
                [f1_binary(gold[known[:, j] == 1, j], pred[known[:, j] == 1, j]) for j in range(gold.shape[1])]
            )
        ),
        "exact_match": float(np.mean(np.all(gold[full] == pred[full], axis=1))) if np.any(full) else math.nan,
    }


def violation_mask(assignments: np.ndarray) -> np.ndarray:
    invalid = np.zeros(len(assignments), dtype=bool)
    for child, parent in IMPLICATIONS:
        invalid |= (assignments[:, INDEX[child]] == 1) & (assignments[:, INDEX[parent]] == 0)
    return invalid


def qtile(values: np.ndarray) -> dict:
    return {str(q): float(np.quantile(values, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}


def bootstrap_delta(gold: np.ndarray, raw_pred: np.ndarray, new_pred: np.ndarray, raw_p: np.ndarray, new_p: np.ndarray, seed: int = 0, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    f1_delta, brier_delta, exact_delta = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(gold), len(gold))
        g = gold[idx]
        f1_delta.append(f1_binary(g.ravel(), new_pred[idx].ravel()) - f1_binary(g.ravel(), raw_pred[idx].ravel()))
        brier_delta.append(float(np.mean((new_p[idx] - g) ** 2) - np.mean((raw_p[idx] - g) ** 2)))
        exact_delta.append(float(np.mean(np.all(g == new_pred[idx], axis=1)) - np.mean(np.all(g == raw_pred[idx], axis=1))))

    def interval(xs: list[float]) -> dict:
        return {
            "mean": float(np.mean(xs)),
            "ci95": [float(np.quantile(xs, 0.025)), float(np.quantile(xs, 0.975))],
        }

    return {
        "micro_f1_delta": interval(f1_delta),
        "brier_delta": interval(brier_delta),
        "exact_match_delta": interval(exact_delta),
    }


def bootstrap_decision_delta(
    gold: np.ndarray, raw_pred: np.ndarray, new_pred: np.ndarray, seed: int = 1, n_boot: int = 1000
) -> dict:
    rng = np.random.default_rng(seed)
    f1_delta, exact_delta = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(gold), len(gold))
        g = gold[idx]
        f1_delta.append(f1_binary(g.ravel(), new_pred[idx].ravel()) - f1_binary(g.ravel(), raw_pred[idx].ravel()))
        exact_delta.append(float(np.mean(np.all(g == new_pred[idx], axis=1)) - np.mean(np.all(g == raw_pred[idx], axis=1))))

    def interval(xs: list[float]) -> dict:
        return {
            "mean": float(np.mean(xs)),
            "ci95": [float(np.quantile(xs, 0.025)), float(np.quantile(xs, 0.975))],
        }

    return {"micro_f1_delta": interval(f1_delta), "exact_match_delta": interval(exact_delta)}


def run(data_path: Path, db_path: Path) -> dict:
    _, raw_p, gold, known = load_rows(data_path, db_path)
    conditioned_p, z, joint_map = exact_condition(raw_p)
    # The tolerance keeps a mathematically unchanged 0.5 probability from flipping
    # because of floating-point summation during exact enumeration.
    raw_pred = (raw_p >= 0.5 - 1e-12).astype(np.int8)
    marginal_pred = (conditioned_p >= 0.5 - 1e-12).astype(np.int8)

    gold_rule_violations = {
        f"{child}_implies_{parent}": int(np.sum((gold[:, INDEX[child]] == 1) & (gold[:, INDEX[parent]] == 0)))
        for child, parent in IMPLICATIONS
    }
    gold_rule_audit = {
        f"{child}_implies_{parent}": {
            "known_pairs": int(np.sum((known[:, INDEX[child]] == 1) & (known[:, INDEX[parent]] == 1))),
            "known_child_positives": int(
                np.sum(
                    (known[:, INDEX[child]] == 1)
                    & (known[:, INDEX[parent]] == 1)
                    & (gold[:, INDEX[child]] == 1)
                )
            ),
            "known_pair_violations": int(
                np.sum(
                    (known[:, INDEX[child]] == 1)
                    & (known[:, INDEX[parent]] == 1)
                    & (gold[:, INDEX[child]] == 1)
                    & (gold[:, INDEX[parent]] == 0)
                )
            ),
        }
        for child, parent in IMPLICATIONS
    }
    raw_joint_nll = -np.sum(
        gold * np.log(np.clip(raw_p, 1e-12, 1.0))
        + (1 - gold) * np.log(np.clip(1 - raw_p, 1e-12, 1.0)),
        axis=1,
    )
    constrained_joint_nll = raw_joint_nll + np.log(z)

    changed = marginal_pred != raw_pred
    rescued = changed & (raw_pred != gold) & (marginal_pred == gold)
    harmed = changed & (raw_pred == gold) & (marginal_pred != gold)
    raw_exact_error = (~np.all(raw_pred == gold, axis=1)).astype(np.int8)

    result = {
        "setup": {
            "model": MODEL,
            "n_examples": int(len(gold)),
            "heads": HEADS,
            "implications": [f"{a} -> {b}" for a, b in IMPLICATIONS],
            "worlds_total": 1 << len(HEADS),
            "worlds_valid": int(enumerate_worlds()[1].sum()),
            "evaluation_label_policy": "Missing category keys are treated as non-positive, matching the benchmark task loader.",
            "known_label_fraction": {h: float(known[:, j].mean()) for j, h in enumerate(HEADS)},
            "gold_rule_violations": gold_rule_violations,
            "gold_rule_audit": gold_rule_audit,
        },
        "valid_mass": {
            "mean_z": float(z.mean()),
            "mean_invalid_mass": float(np.mean(1 - z)),
            "quantiles_z": qtile(z),
            "raw_argmax_violation_rate": float(violation_mask(raw_pred).mean()),
            "conditioned_marginal_violation_rate": float(violation_mask(marginal_pred).mean()),
            "joint_map_violation_rate": float(violation_mask(joint_map).mean()),
            "one_minus_z_auc_for_raw_exact_error": roc_auc(raw_exact_error, 1 - z),
        },
        "raw": {**probability_metrics(gold, raw_p), **decision_metrics(gold, raw_pred)},
        "conditioned_marginals": {**probability_metrics(gold, conditioned_p), **decision_metrics(gold, marginal_pred)},
        "known_labels_only": {
            "raw": known_only_metrics(gold, raw_p, raw_pred, known),
            "conditioned_marginals": known_only_metrics(gold, conditioned_p, marginal_pred, known),
        },
        "joint_map": decision_metrics(gold, joint_map),
        "joint_probability": {
            "raw_joint_nll_per_example": float(raw_joint_nll.mean()),
            "conditioned_joint_nll_per_example": float(constrained_joint_nll.mean()),
            "delta": float(constrained_joint_nll.mean() - raw_joint_nll.mean()),
        },
        "corrections": {
            "changed_bits": int(changed.sum()),
            "changed_bit_fraction": float(changed.mean()),
            "changed_examples": int(np.any(changed, axis=1).sum()),
            "rescued_bits": int(rescued.sum()),
            "harmed_bits": int(harmed.sum()),
            "net_correct_bits": int(rescued.sum() - harmed.sum()),
            "known_labels_only": {
                "changed_bits": int(np.sum(changed & (known == 1))),
                "rescued_bits": int(np.sum(rescued & (known == 1))),
                "harmed_bits": int(np.sum(harmed & (known == 1))),
            },
            "by_head": {
                h: {
                    "changed": int(changed[:, j].sum()),
                    "rescued": int(rescued[:, j].sum()),
                    "harmed": int(harmed[:, j].sum()),
                }
                for j, h in enumerate(HEADS)
            },
        },
        "paired_bootstrap_conditioned_minus_raw": bootstrap_delta(gold, raw_pred, marginal_pred, raw_p, conditioned_p),
        "paired_bootstrap_joint_map_minus_raw": bootstrap_decision_delta(gold, raw_pred, joint_map),
    }
    return result


def markdown(result: dict) -> str:
    raw = result["raw"]
    cond = result["conditioned_marginals"]
    jmap = result["joint_map"]
    vm = result["valid_mass"]
    corr = result["corrections"]
    joint = result["joint_probability"]
    boot = result["paired_bootstrap_conditioned_minus_raw"]
    map_boot = result["paired_bootstrap_joint_map_minus_raw"]
    known_raw = result["known_labels_only"]["raw"]
    known_cond = result["known_labels_only"]["conditioned_marginals"]

    rows = [
        ("Micro-F1", raw["micro_f1"], cond["micro_f1"], jmap["micro_f1"]),
        ("Macro-F1", raw["macro_f1"], cond["macro_f1"], jmap["macro_f1"]),
        ("Exact match", raw["exact_match"], cond["exact_match"], jmap["exact_match"]),
        ("Mean AUPRC", raw["mean_auprc"], cond["mean_auprc"], math.nan),
        ("Mean Brier", raw["mean_brier"], cond["mean_brier"], math.nan),
        ("Mean ECE", raw["mean_ece"], cond["mean_ece"], math.nan),
        ("Marginal NLL", raw["marginal_nll"], cond["marginal_nll"], math.nan),
    ]
    table = "\n".join(
        f"| {name} | {a:.6f} | {b:.6f} | {'—' if math.isnan(c) else f'{c:.6f}'} |" for name, a, b, c in rows
    )
    return f"""# OpenAI Moderation ontology-conditioning pilot

## Setup

- Model: `{result['setup']['model']}`
- Examples: {result['setup']['n_examples']}
- Variables: {len(result['setup']['heads'])} Noul heads
- Valid worlds: {result['setup']['worlds_valid']} / {result['setup']['worlds_total']}
- Constraints: {', '.join(result['setup']['implications'])}
- Gold violations: {result['setup']['gold_rule_violations']}

## Results

| Metric | Raw independent | Conditioned marginals | Constrained joint MAP |
|---|---:|---:|---:|
{table}

### Explicitly known labels only

| Metric | Raw independent | Conditioned marginals |
|---|---:|---:|
| Micro-F1 | {known_raw['micro_f1']:.6f} | {known_cond['micro_f1']:.6f} |
| Macro-F1 | {known_raw['macro_f1']:.6f} | {known_cond['macro_f1']:.6f} |
| Mean AUPRC | {known_raw['mean_auprc']:.6f} | {known_cond['mean_auprc']:.6f} |
| Mean Brier | {known_raw['mean_brier']:.6f} | {known_cond['mean_brier']:.6f} |
| Mean ECE | {known_raw['mean_ece']:.6f} | {known_cond['mean_ece']:.6f} |
| Marginal NLL | {known_raw['marginal_nll']:.6f} | {known_cond['marginal_nll']:.6f} |

Mean valid mass $Z$: {vm['mean_z']:.6f}
Mean invalid mass $1-Z$: {vm['mean_invalid_mass']:.6f}
Raw local-argmax violation rate: {vm['raw_argmax_violation_rate']:.6f}
Conditioned-marginal violation rate: {vm['conditioned_marginal_violation_rate']:.6f}
$1-Z$ AUROC for detecting a raw exact-match error: {vm['one_minus_z_auc_for_raw_exact_error']:.6f}

Raw joint NLL per example: {joint['raw_joint_nll_per_example']:.6f}
Conditioned joint NLL per example: {joint['conditioned_joint_nll_per_example']:.6f}
Joint NLL delta: {joint['delta']:+.6f}

The conditioned marginal decisions changed {corr['changed_bits']} of {result['setup']['n_examples'] * len(result['setup']['heads'])} binary outputs across {corr['changed_examples']} examples. They rescued {corr['rescued_bits']} incorrect bits and harmed {corr['harmed_bits']} correct bits, for a net change of {corr['net_correct_bits']:+d} correct bits.

Paired bootstrap deltas (conditioned minus raw):

- Micro-F1: {boot['micro_f1_delta']['mean']:+.6f}, 95% CI [{boot['micro_f1_delta']['ci95'][0]:+.6f}, {boot['micro_f1_delta']['ci95'][1]:+.6f}]
- Brier: {boot['brier_delta']['mean']:+.6f}, 95% CI [{boot['brier_delta']['ci95'][0]:+.6f}, {boot['brier_delta']['ci95'][1]:+.6f}]
- Exact match: {boot['exact_match_delta']['mean']:+.6f}, 95% CI [{boot['exact_match_delta']['ci95'][0]:+.6f}, {boot['exact_match_delta']['ci95'][1]:+.6f}]

Paired bootstrap deltas for constrained joint MAP minus raw local argmax:

- Micro-F1: {map_boot['micro_f1_delta']['mean']:+.6f}, 95% CI [{map_boot['micro_f1_delta']['ci95'][0]:+.6f}, {map_boot['micro_f1_delta']['ci95'][1]:+.6f}]
- Exact match: {map_boot['exact_match_delta']['mean']:+.6f}, 95% CI [{map_boot['exact_match_delta']['ci95'][0]:+.6f}, {map_boot['exact_match_delta']['ci95'][1]:+.6f}]

## Label-policy note

The source JSON omits some category keys to denote unknown labels. The published benchmark loader evaluates `row[code] == 1`, which makes missing values non-positive after dataset schema normalization. This pilot mirrors that policy for comparability. A confirmatory experiment should also report metrics restricted to explicitly known labels.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.data, args.responses)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report_out.write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
