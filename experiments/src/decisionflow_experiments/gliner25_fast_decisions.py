"""GLiNER2.5/Fast Decisions rule audit and three-way inference comparison.

This module intentionally lives in the experiment package.  GLiNER2.5 is one
possible local scorer for DecisionFlow, while dataset acquisition, hand-written
candidate rules, gold audits, and benchmark metrics are research artifacts.

The comparison keeps local scores fixed:

1. GLiNER2.5 independent decoding;
2. GLiNER2.5 exact constrained MAP decoding;
3. DecisionFlow exact conditioning of the same local probabilities.

Fast Decisions does not publish machine-readable constraints.  The candidate
rules below are therefore explicit hypotheses.  ``gold_zero`` admits a rule to
the comparison only when every public-development gold assignment satisfies
it.  The report retains rejected rules and their counterexample counts.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from decisionflow.core import LocalPotentials
from decisionflow.engine import DecisionEngine


DATASET_ID = "fastino/fast-decisions"
DATASET_REVISION = "1a33070cabf94ce2e29105482dd2ef6c157ad7f2"
MODEL_ID = "fastino/GLiNER2.5-Decide"


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    task: str
    label: Optional[str] = None
    other_task: Optional[str] = None
    other_labels: Tuple[str, ...] = ()
    k: Optional[int] = None
    rationale: str = ""
    provenance: str = "hand-authored semantic hypothesis"


# These are deliberately few and readable.  They represent plausible policy
# semantics, not patterns exhaustively mined from the same 100 gold examples.
CANDIDATE_RULES: Mapping[str, Tuple[Rule, ...]] = {
    "agent_handoff": (
        Rule(
            "handoff-iff-policy-requires-it",
            "iff",
            "policy_scope",
            "handoff_required",
            "should_handoff",
            ("yes",),
            rationale=(
                "The auxiliary policy-scope decision and final handoff decision have "
                "identical semantics."
            ),
            provenance=(
                "derived structured view; policy_scope gold is derived from should_handoff"
            ),
        ),
    ),
    "email_triage": (
        Rule(
            "phishing-routes-security",
            "implies_in",
            "is_phishing",
            "yes",
            "category",
            ("security",),
            "Detected phishing should enter the security category.",
        ),
        Rule(
            "phishing-escalates",
            "implies_in",
            "is_phishing",
            "yes",
            "action",
            ("escalate",),
            "Detected phishing should be escalated.",
        ),
        Rule(
            "reply-needs-reply",
            "implies_in",
            "action",
            "reply",
            "needs_reply",
            ("yes",),
            "A reply action entails that a reply is needed.",
        ),
    ),
    "ticket_route": (
        Rule(
            "fraud-review-contains-pii",
            "implies_in",
            "queue",
            "fraud_review",
            "contains_pii",
            ("yes",),
            "Fraud-review cases are expected to contain account or identity information.",
        ),
        Rule(
            "identity-contains-pii",
            "implies_in",
            "queue",
            "identity",
            "contains_pii",
            ("yes",),
            "Identity cases are expected to contain personal information.",
        ),
    ),
    "product_feedback": (
        Rule(
            "at-least-one-product-area",
            "at_least",
            "product_area",
            k=1,
            rationale="Every feedback item is assigned to at least one product area.",
            provenance="dataset schema/cardinality hypothesis",
        ),
    ),
    "clinic_request": (
        Rule(
            "urgent-is-clinical",
            "implies_in",
            "urgent",
            "yes",
            "request",
            ("appointment", "refill", "lab_results", "symptom_question", "provider_message"),
            "Urgent requests should concern a clinical action or clinical question.",
        ),
        Rule(
            "cancel-is-not-urgent",
            "implies_in",
            "request",
            "cancel_visit",
            "urgent",
            ("no",),
            "Routine visit cancellation should not be marked urgent.",
        ),
    ),
    "sports_recap": (
        Rule(
            "upcoming-is-not-upset",
            "implies_in",
            "result",
            "upcoming",
            "upset",
            ("no",),
            "An event that has not occurred cannot yet be an upset.",
        ),
        Rule(
            "draw-is-not-upset",
            "implies_in",
            "result",
            "draw",
            "upset",
            ("no",),
            "A draw is not treated as an upset win or loss.",
        ),
    ),
    "restaurant_review": (
        Rule(
            "at-least-one-aspect",
            "at_least",
            "aspects",
            k=1,
            rationale="Every review annotation names at least one discussed aspect.",
            provenance="dataset schema/cardinality hypothesis",
        ),
    ),
    "screen_tags": (
        Rule(
            "at-least-one-genre",
            "at_least",
            "genres",
            k=1,
            rationale="Every screen item has at least one genre.",
            provenance="dataset schema/cardinality hypothesis",
        ),
    ),
}


RULE_DATASETS = tuple(CANDIDATE_RULES)


def _heads(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    return row["output"]["classifications"]


def _gold(row: Mapping[str, Any], *, structured_handoff: bool = False) -> Dict[str, set[str]]:
    result = {head["task"]: set(head["true_label"]) for head in _heads(row)}
    if structured_handoff:
        handoff = next(iter(result["should_handoff"]))
        result["policy_scope"] = {
            "handoff_required" if handoff == "yes" else "self_service"
        }
    return result


def _rule_holds(rule: Rule, assignment: Mapping[str, set[str]]) -> bool:
    if rule.kind == "at_least":
        return len(assignment[rule.task]) >= int(rule.k or 0)
    antecedent = rule.label in assignment[rule.task]
    consequent = bool(assignment[rule.other_task] & set(rule.other_labels))
    if rule.kind == "implies_in":
        return (not antecedent) or consequent
    if rule.kind == "iff":
        return antecedent == consequent
    raise ValueError("unknown rule kind %r" % rule.kind)


def audit_rules(rows: Sequence[Mapping[str, Any]], dataset: str) -> list[Dict[str, Any]]:
    structured = dataset == "agent_handoff"
    assignments = [_gold(row, structured_handoff=structured) for row in rows]
    report = []
    for rule in CANDIDATE_RULES.get(dataset, ()):
        antecedent_support = None
        if rule.label is not None:
            antecedent_support = sum(rule.label in row[rule.task] for row in assignments)
        violations = [index for index, row in enumerate(assignments) if not _rule_holds(rule, row)]
        report.append({
            "name": rule.name,
            "kind": rule.kind,
            "rationale": rule.rationale,
            "provenance": rule.provenance,
            "antecedent_support": antecedent_support,
            "gold_violations": len(violations),
            "gold_violation_rate": len(violations) / len(rows),
            "counterexample_indices": violations[:10],
            "accepted_gold_zero": not violations,
        })
    return report


def accepted_rules(rows: Sequence[Mapping[str, Any]], dataset: str) -> Tuple[Rule, ...]:
    audit = {entry["name"]: entry for entry in audit_rules(rows, dataset)}
    return tuple(
        rule for rule in CANDIDATE_RULES.get(dataset, ())
        if audit[rule.name]["accepted_gold_zero"]
    )


def _schema_heads(row: Mapping[str, Any], dataset: str) -> list[Dict[str, Any]]:
    heads = [dict(head) for head in _heads(row)]
    if dataset == "agent_handoff":
        heads.append({
            "task": "policy_scope",
            "labels": ["self_service", "handoff_required"],
            "multi_label": False,
            "instruction": (
                "Given the policy and user request, decide whether the request is "
                "within self-service scope or requires a human handoff."
            ),
            "derived": True,
        })
    return heads


def _build_gliner_schema(row: Mapping[str, Any], dataset: str, rules: Sequence[Rule]):
    from gliner2.classification import ClassificationSchema
    from gliner2.classification import constraints as C

    schema = ClassificationSchema()
    for head in _schema_heads(row, dataset):
        kwargs = {}
        if head.get("instruction"):
            kwargs["instruction"] = head["instruction"]
        if head["multi_label"]:
            # Cardinality is supplied by the explicit audited rule, keeping
            # independent decoding genuinely independent.
            schema.multi(head["task"], head["labels"], min_labels=0, **kwargs)
        else:
            schema.single(head["task"], head["labels"], **kwargs)
    expressions = []
    for rule in rules:
        if rule.kind == "at_least":
            expressions.append(C.at_least(rule.task, int(rule.k)))
            continue
        left = (rule.task, rule.label)
        right_parts = [(rule.other_task, label) for label in rule.other_labels]
        right = right_parts[0] if len(right_parts) == 1 else C.any_of(*right_parts)
        expressions.append(C.implies(left, right) if rule.kind == "implies_in" else C.iff(left, right))
    if expressions:
        schema.constrain(*expressions)
    return schema


def _df_id(task: str, label: Optional[str] = None) -> str:
    return task if label is None else "%s::%s" % (task, label)


def _df_request(row: Mapping[str, Any], dataset: str) -> Dict[str, Any]:
    questions = []
    for head in _schema_heads(row, dataset):
        if head["multi_label"]:
            questions.extend(
                {"id": _df_id(head["task"], label), "type": "noul"}
                for label in head["labels"]
            )
        else:
            questions.append({
                "id": head["task"],
                "type": "choice",
                "options": list(head["labels"]),
            })
    return {"state": {"text": row["input"]}, "questions": questions}


def _df_atom(heads: Mapping[str, Mapping[str, Any]], task: str, label: str) -> Mapping[str, Any]:
    if heads[task]["multi_label"]:
        return {"eq": [{"var": _df_id(task, label)}, True]}
    return {"eq": [{"var": task}, label]}


def _df_constraints(row: Mapping[str, Any], dataset: str, rules: Sequence[Rule]) -> Dict[str, Any]:
    heads = {head["task"]: head for head in _schema_heads(row, dataset)}
    hard = []
    for rule in rules:
        if rule.kind == "at_least":
            hard.append({
                "name": rule.name,
                "source": rule.provenance,
                "expr": {
                    "at_least": {
                        "k": int(rule.k),
                        "items": [
                            _df_atom(heads, rule.task, label)
                            for label in heads[rule.task]["labels"]
                        ],
                    }
                },
            })
            continue
        left = _df_atom(heads, rule.task, str(rule.label))
        parts = [_df_atom(heads, str(rule.other_task), label) for label in rule.other_labels]
        right = parts[0] if len(parts) == 1 else {"any": parts}
        op = "implies" if rule.kind == "implies_in" else "iff"
        hard.append({
            "name": rule.name,
            "source": rule.provenance,
            "expr": {op: [left, right]},
        })
    return {"name": "%s-gold-zero-rules" % dataset, "version": "1", "hard": hard}


def _probability_rows(scores, row: Mapping[str, Any], dataset: str) -> Dict[str, Dict[Any, float]]:
    values: Dict[str, Dict[Any, float]] = {}
    for head in _schema_heads(row, dataset):
        task = head["task"]
        if head["multi_label"]:
            for label in head["labels"]:
                p = float(scores.probability(task, label))
                values[_df_id(task, label)] = {False: 1.0 - p, True: p}
        else:
            values[task] = {
                label: float(scores.probability(task, label)) for label in head["labels"]
            }
    return values


def _gliner_assignment(result, row: Mapping[str, Any], dataset: str) -> Dict[str, set[str]]:
    return {
        head["task"]: set(result.selected(head["task"]))
        for head in _schema_heads(row, dataset)
    }


def _df_assignment(mapping: Mapping[str, Any], row: Mapping[str, Any], dataset: str) -> Dict[str, set[str]]:
    result = {}
    for head in _schema_heads(row, dataset):
        task = head["task"]
        if head["multi_label"]:
            result[task] = {
                label for label in head["labels"] if bool(mapping[_df_id(task, label)])
            }
        else:
            result[task] = {str(mapping[task])}
    return result


def _df_marginal_assignment(result, row: Mapping[str, Any], dataset: str) -> Dict[str, set[str]]:
    assignment = {}
    for head in _schema_heads(row, dataset):
        task = head["task"]
        if head["multi_label"]:
            assignment[task] = {
                label
                for label in head["labels"]
                if result.marginals[_df_id(task, label)][True] >= 0.5
            }
        else:
            assignment[task] = {
                max(head["labels"], key=lambda label: result.marginals[task][label])
            }
    return assignment


def _assignment_violates(assignment: Mapping[str, set[str]], rules: Sequence[Rule]) -> bool:
    return any(not _rule_holds(rule, assignment) for rule in rules)


def _original_tasks(row: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(head["task"] for head in _heads(row))


def _hit_counts(prediction, gold, tasks):
    head_hits = sum(prediction[task] == gold[task] for task in tasks)
    return head_hits, int(head_hits == len(tasks))


def _nll(probabilities, gold, row):
    total = 0.0
    count = 0
    for head in _heads(row):
        task = head["task"]
        if head["multi_label"]:
            for label in head["labels"]:
                p = min(max(probabilities[_df_id(task, label)][True], 1e-15), 1 - 1e-15)
                total -= math.log(p if label in gold[task] else 1.0 - p)
                count += 1
        else:
            label = next(iter(gold[task]))
            total -= math.log(max(probabilities[task][label], 1e-15))
            count += 1
    return total, count


def _constrained_nll(result, gold, row):
    total = 0.0
    count = 0
    for head in _heads(row):
        task = head["task"]
        if head["multi_label"]:
            for label in head["labels"]:
                p = min(max(result.marginals[_df_id(task, label)][True], 1e-15), 1 - 1e-15)
                total -= math.log(p if label in gold[task] else 1.0 - p)
                count += 1
        else:
            label = next(iter(gold[task]))
            total -= math.log(max(result.marginals[task][label], 1e-15))
            count += 1
    return total, count


def _load_rows(dataset: str):
    from datasets import load_dataset

    return list(load_dataset(DATASET_ID, dataset, split="train", revision=DATASET_REVISION))


def evaluate_dataset(
    classifier,
    dataset: str,
    *,
    batch_size: int = 8,
    backend: str = "sdd",
) -> Dict[str, Any]:
    from gliner2.classification import ClassificationConfig

    rows = _load_rows(dataset)
    audits = audit_rules(rows, dataset)
    rules = accepted_rules(rows, dataset)
    schema = _build_gliner_schema(rows[0], dataset, rules)
    score_config = ClassificationConfig(batch_size=batch_size)
    started = time.perf_counter()
    scores = classifier.batch_score([row["input"] for row in rows], schema, config=score_config)
    scoring_seconds = time.perf_counter() - started

    independent_config = ClassificationConfig(
        decoder="independent", candidate_threshold=0.0, max_candidates_per_task=64
    )
    exact_config = ClassificationConfig(
        decoder="exact",
        exact_node_budget=2_000_000,
        candidate_threshold=0.0,
        max_candidates_per_task=64,
        on_infeasible="raise",
    )
    flow = DecisionEngine(backend=backend)
    counters = {
        name: {"head_hits": 0, "joint_hits": 0, "conflicts": 0}
        for name in ("independent", "gliner_exact", "decisionflow_marginal", "decisionflow_map")
    }
    map_mismatches = 0
    z_values = []
    raw_nll = raw_n = constrained_nll = constrained_n = 0.0
    df_seconds = gliner_decode_seconds = 0.0

    for row, score in zip(rows, scores, strict=True):
        started = time.perf_counter()
        independent = classifier.decode(score, schema, config=independent_config)
        exact = classifier.decode(score, schema, config=exact_config)
        gliner_decode_seconds += time.perf_counter() - started

        probabilities = _probability_rows(score, row, dataset)
        program = flow.compile(_df_request(row, dataset), _df_constraints(row, dataset, rules))
        started = time.perf_counter()
        result = flow.infer_program(program, LocalPotentials(probabilities))
        df_seconds += time.perf_counter() - started

        assignments = {
            "independent": _gliner_assignment(independent, row, dataset),
            "gliner_exact": _gliner_assignment(exact, row, dataset),
            "decisionflow_marginal": _df_marginal_assignment(result, row, dataset),
            "decisionflow_map": _df_assignment(result.joint_map, row, dataset),
        }
        gold = _gold(row, structured_handoff=dataset == "agent_handoff")
        tasks = _original_tasks(row)
        for name, assignment in assignments.items():
            heads, joint = _hit_counts(assignment, gold, tasks)
            counters[name]["head_hits"] += heads
            counters[name]["joint_hits"] += joint
            counters[name]["conflicts"] += _assignment_violates(assignment, rules)
        map_mismatches += assignments["gliner_exact"] != assignments["decisionflow_map"]
        z_values.append(float(result.valid_mass))
        value, count = _nll(probabilities, gold, row)
        raw_nll += value
        raw_n += count
        value, count = _constrained_nll(result, gold, row)
        constrained_nll += value
        constrained_n += count

    head_count = sum(len(_heads(row)) for row in rows)
    methods = {}
    for name, values in counters.items():
        methods[name] = {
            "head_accuracy": values["head_hits"] / head_count,
            "joint_exact_match": values["joint_hits"] / len(rows),
            "conflicting_records": values["conflicts"],
            "conflict_rate": values["conflicts"] / len(rows),
        }
    return {
        "dataset": dataset,
        "rows": len(rows),
        "heads": [
            {
                "task": head["task"],
                "labels": list(head["labels"]),
                "multi_label": bool(head["multi_label"]),
            }
            for head in _heads(rows[0])
        ],
        "candidate_rule_audit": audits,
        "accepted_rules": [rule.name for rule in rules],
        "comparison_is_nontrivial": bool(rules),
        "methods": methods,
        "probability": {
            "raw_local_nll": raw_nll / raw_n,
            "decisionflow_marginal_nll": constrained_nll / constrained_n,
            "mean_valid_mass": sum(z_values) / len(z_values),
            "min_valid_mass": min(z_values),
            "max_valid_mass": max(z_values),
        },
        "agreement": {
            "gliner_exact_vs_decisionflow_map_mismatches": map_mismatches,
        },
        "timing": {
            "model_scoring_seconds": scoring_seconds,
            "gliner_decoding_seconds": gliner_decode_seconds,
            "decisionflow_inference_seconds": df_seconds,
        },
        "notes": [
            "Public development split; the official 300-example-per-domain test split is held out.",
            "Rules were accepted using a full-development gold-zero audit; this is an exploratory pilot, not an unbiased test estimate.",
        ] + ([
            "policy_scope is an auxiliary label deterministically derived from should_handoff; it is not an independently annotated target."
        ] if dataset == "agent_handoff" else []),
    }


def run(
    datasets: Sequence[str] = RULE_DATASETS,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 8,
    backend: str = "sdd",
) -> Dict[str, Any]:
    from gliner2.classification import Classifier

    classifier = Classifier.from_pretrained(model_id, device="cpu")
    reports = [
        evaluate_dataset(classifier, dataset, batch_size=batch_size, backend=backend)
        for dataset in datasets
    ]
    return {
        "experiment": "gliner25_fast_decisions_constraints",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model_id": model_id,
        "inference_backend": backend,
        "datasets": reports,
    }


def write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
