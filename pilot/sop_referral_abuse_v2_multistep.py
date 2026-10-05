#!/usr/bin/env python3
"""OpenJev/ProbJev multi-step pilot for SOP-Bench referral abuse v2."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from openjev import Choice, LocalJev, Noul
from pysdd.sdd import SddManager, Vtree

from sop_dangerous_goods_probjev import MODEL_ID, RESULT_DIR, max_product, normalize


DATA_DIR = Path(
    "third_party/sop-bench/src/amazon_sop_bench/benchmarks/data/"
    "referral_abuse_detection_v2"
)
DATA_PATH = DATA_DIR / "test_set_with_outputs.csv"
EVIDENCE_TOOLS = (
    "investigate_account",
    "analyze_traffic_patterns",
    "analyze_temporal_patterns",
    "get_violation_history",
    "get_financial_impact",
)
GUIDELINE_TOOL = "determine_enforcement_action"
TOOLS = EVIDENCE_TOOLS + (GUIDELINE_TOOL,)
ACTIONS = TOOLS + ("finalize",)
VIOLATIONS = (
    "Temporal Fraud",
    "Abusive Account Creation",
    "Misleading Ad Copy",
    "Personal Orders",
    "No Violation",
    "Inconclusive",
)
RISKS = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "N/A")
ENFORCEMENTS = (
    "Permanent Account Closure",
    "Account Closure",
    "Temporary Suspension",
    "Warning Issued",
    "No Action",
    "Manual Review Required",
    "Inconclusive",
)
TOOL_FIELDS = {
    "investigate_account": (
        "address_validity", "email_pattern_suspicious", "website_verified",
        "connected_accounts", "login_geographic_consistency", "account_age_days",
    ),
    "analyze_traffic_patterns": (
        "revenue_amount", "click_through_rate", "referral_source_quality",
        "payment_method_shared", "order_patterns_suspicious",
    ),
    "analyze_temporal_patterns": (
        "registration_burst_detected", "off_hours_activity_percentage",
        "activity_spike_detected",
    ),
    "get_violation_history": (
        "previous_violations_count", "last_violation_date", "warning_issued",
        "account_rehabilitation_status",
    ),
    "get_financial_impact": (
        "total_lifetime_revenue", "refund_rate_percentage",
        "customer_complaint_count",
    ),
    GUIDELINE_TOOL: (),
}


def boolean(value: str) -> bool:
    return value.strip().lower() == "true"


def typed_value(field: str, value: str) -> Any:
    if field in {
        "address_validity", "email_pattern_suspicious", "website_verified",
        "login_geographic_consistency", "payment_method_shared",
        "order_patterns_suspicious", "registration_burst_detected",
        "activity_spike_detected", "warning_issued",
    }:
        return boolean(value)
    if field in {
        "connected_accounts", "account_age_days", "previous_violations_count",
        "customer_complaint_count",
    }:
        return int(value)
    if field in {
        "revenue_amount", "click_through_rate", "off_hours_activity_percentage",
        "total_lifetime_revenue", "refund_rate_percentage",
    }:
        return float(value)
    return value or None


def tool_result(row: dict[str, str], tool: str) -> dict[str, Any]:
    return {field: typed_value(field, row[field]) for field in TOOL_FIELDS[tool]}


def violation_scores(f: dict[str, Any]) -> dict[str, int]:
    return {
        "Abusive Account Creation": (
            int(not f["address_validity"])
            + int(f["email_pattern_suspicious"])
            + int(not f["website_verified"])
            + int(f["connected_accounts"] >= 15)
            + int(not f["login_geographic_consistency"])
            + 2 * int(f["registration_burst_detected"])
            + int(f["account_age_days"] < 30)
        ),
        "Misleading Ad Copy": (
            int(not f["website_verified"])
            + int(f["referral_source_quality"] in {"Low", "Medium"})
            + int(f["order_patterns_suspicious"])
            + int(f["click_through_rate"] > 0.4)
            + int(f["activity_spike_detected"])
            + int(f["customer_complaint_count"] > 5)
        ),
        "Personal Orders": (
            int(f["payment_method_shared"])
            + int(0 < f["connected_accounts"] < 15)
            + int(f["order_patterns_suspicious"])
            + int(f["referral_source_quality"] == "High")
            + int(f["off_hours_activity_percentage"] < 30)
        ),
        "Temporal Fraud": (
            2 * int(f["registration_burst_detected"])
            + int(f["off_hours_activity_percentage"] > 60)
            + int(f["activity_spike_detected"])
            + 2 * int(f["account_age_days"] < 30 and f["connected_accounts"] >= 15)
        ),
        "No Violation": (
            int(f["address_validity"])
            + int(not f["email_pattern_suspicious"])
            + int(f["website_verified"])
            + int(f["login_geographic_consistency"])
            + int(not f["payment_method_shared"])
            + int(not f["order_patterns_suspicious"])
            + int(not f["registration_burst_detected"])
            + int(f["off_hours_activity_percentage"] < 30)
            + int(f["customer_complaint_count"] <= 2)
        ),
    }


def derive_violation(f: dict[str, Any]) -> str:
    scores = violation_scores(f)
    thresholds = {
        "Abusive Account Creation": 4, "Misleading Ad Copy": 4,
        "Personal Orders": 3, "Temporal Fraud": 4, "No Violation": 6,
    }
    eligible = [name for name, score in scores.items() if score >= thresholds[name]]
    if not eligible:
        return "Inconclusive"
    priority = {
        "Temporal Fraud": 4, "Abusive Account Creation": 3,
        "Misleading Ad Copy": 2, "Personal Orders": 1, "No Violation": 0,
    }
    return max(eligible, key=lambda name: (scores[name], priority[name]))


def recent_violation(f: dict[str, Any]) -> bool:
    if not f.get("last_violation_date"):
        return False
    date = datetime.fromisoformat(str(f["last_violation_date"]).replace("Z", "+00:00"))
    # The released records were generated around 2025-12-01. The condition only
    # matters when a warned account has exactly one previous violation.
    reference = datetime(2025, 12, 1, tzinfo=timezone.utc)
    return 0 <= (reference - date).days <= 90


def derive_risk(violation: str, f: dict[str, Any]) -> str:
    if violation in {"No Violation", "Inconclusive"}:
        return "N/A"
    revenue = f["revenue_amount"]
    previous = f["previous_violations_count"]
    if (
        revenue > 5000
        or previous >= 2
        or (previous >= 1 and f["warning_issued"] and recent_violation(f))
        or f["customer_complaint_count"] > 10
    ):
        return "CRITICAL"
    if (
        1000 < revenue <= 5000
        or previous == 1
        or f["connected_accounts"] >= 25
        or f["refund_rate_percentage"] > 40
    ):
        return "HIGH"
    if (
        100 < revenue <= 1000
        or f["account_rehabilitation_status"] == "Probation"
        or 15 <= f["connected_accounts"] < 25
    ):
        return "MEDIUM"
    return "LOW"


def enforcement_for(violation: str, risk: str, previous: int | None = None) -> str:
    if violation == "No Violation":
        return "No Action"
    if violation == "Inconclusive":
        return "Manual Review Required" if previous and previous > 0 else "Inconclusive"
    if violation == "Personal Orders":
        return {
            "CRITICAL": "Account Closure", "HIGH": "Temporary Suspension",
            "MEDIUM": "Warning Issued", "LOW": "No Action",
        }[risk]
    return {
        "CRITICAL": "Permanent Account Closure", "HIGH": "Account Closure",
        "MEDIUM": "Temporary Suspension", "LOW": "Warning Issued",
    }[risk]


def derive(row: dict[str, str]) -> tuple[str, str, str]:
    facts = {field: typed_value(field, row[field]) for fields in TOOL_FIELDS.values() for field in fields}
    violation = derive_violation(facts)
    risk = derive_risk(violation, facts)
    return violation, risk, enforcement_for(violation, risk, facts["previous_violations_count"])


def semantic_triples(previous: int | None = None) -> set[tuple[str, str, str]]:
    triples = set()
    for violation in VIOLATIONS:
        risks = ("N/A",) if violation in {"No Violation", "Inconclusive"} else RISKS[:-1]
        for risk in risks:
            if violation == "Inconclusive" and previous is None:
                for action in ("Manual Review Required", "Inconclusive"):
                    triples.add((violation, risk, action))
            else:
                triples.add((violation, risk, enforcement_for(violation, risk, previous)))
    return triples


def ready(observations: dict[str, dict[str, Any]]) -> bool:
    return len(observations) == len(TOOLS)


def legal_actions(observations: dict[str, dict[str, Any]]) -> set[str]:
    return {"finalize"} if ready(observations) else set(TOOLS) - observations.keys()


def allowed_triples(row: dict[str, str], observations: dict[str, dict[str, Any]]) -> set[tuple[str, str, str]]:
    if all(tool in observations for tool in EVIDENCE_TOOLS):
        return {derive(row)}
    previous = None
    if "get_violation_history" in observations:
        previous = observations["get_violation_history"]["previous_violations_count"]
    return semantic_triples(previous)


def make_state(row: dict[str, str], observations: dict[str, dict[str, Any]]) -> str:
    observed = "none" if not observations else json.dumps(observations, ensure_ascii=False)
    return (
        "Referral-abuse SOP execution state:\n"
        f"Account ID: {row['account_id']}\n"
        f"Completed tools: {', '.join(observations) or 'none'}\n"
        f"Observed results: {observed}\n\n"
        "Policy: call each evidence tool once, obtain the enforcement guidelines, "
        "then finalize. Determine the highest qualifying violation score, break "
        "ties Temporal Fraud > Abusive Account Creation > Misleading Ad Copy > "
        "Personal Orders, determine risk severity, and map both to enforcement."
    )


def questions() -> dict[str, Any]:
    return {
        "next_action": Choice(
            instructions="Which single action should the investigator take next?",
            criteria={action: action.replace("_", " ") for action in ACTIONS},
        ),
        "evidence_sufficient": Noul(instructions="Is the evidence sufficient to finalize now?"),
        "violation_type": Choice(
            instructions="Which violation type is best supported by the evidence?",
            criteria={value: value for value in VIOLATIONS},
        ),
        "risk_severity": Choice(
            instructions="What risk severity follows from the current evidence?",
            criteria={value: value for value in RISKS},
        ),
        "enforcement_action": Choice(
            instructions="Which enforcement action follows from the violation and risk?",
            criteria={value: value for value in ENFORCEMENTS},
        ),
    }


def distributions(result: dict[str, Any]) -> dict[str, dict[Any, float]]:
    answers = result["answers"]
    p_ready = float(answers["evidence_sufficient"]["noul"])
    def choice(name: str, values: tuple[str, ...]) -> dict[str, float]:
        return normalize({v: float(answers[name]["probabilities"][v]) for v in values})
    return {
        "action": choice("next_action", ACTIONS),
        "ready": normalize({False: 1 - p_ready, True: p_ready}),
        "violation": choice("violation_type", VIOLATIONS),
        "risk": choice("risk_severity", RISKS),
        "enforcement": choice("enforcement_action", ENFORCEMENTS),
    }


@dataclass
class StepCircuit:
    groups: dict[str, dict[Any, int]]
    manager: SddManager
    root: Any
    compile_ms: float

    @classmethod
    def build(cls, actions: set[str], is_ready: bool, triples: set[tuple[str, str, str]]) -> "StepCircuit":
        domains = {
            "action": ACTIONS, "ready": (False, True), "violation": VIOLATIONS,
            "risk": RISKS, "enforcement": ENFORCEMENTS,
        }
        groups: dict[str, dict[Any, int]] = {}
        next_var = 1
        for name, values in domains.items():
            groups[name] = {value: next_var + i for i, value in enumerate(values)}
            next_var += len(values)
        start = time.perf_counter()
        manager = SddManager.from_vtree(Vtree(var_count=next_var - 1, var_order=list(range(1, next_var)), vtree_type="balanced"))
        root = manager.true()
        for group in groups.values():
            literals = [manager.literal(v) for v in group.values()]
            at_least_one = manager.false()
            for literal in literals:
                at_least_one |= literal
            root &= at_least_one
            for left, right in itertools.combinations(literals, 2):
                root &= (~left) | (~right)
        for action, var in groups["action"].items():
            if action not in actions:
                root &= ~manager.literal(var)
        root &= manager.literal(groups["ready"][is_ready])
        relation = manager.false()
        for violation, risk, enforcement in triples:
            relation |= (
                manager.literal(groups["violation"][violation])
                & manager.literal(groups["risk"][risk])
                & manager.literal(groups["enforcement"][enforcement])
            )
        root &= relation
        finalize = manager.literal(groups["action"]["finalize"])
        ready_lit = manager.literal(groups["ready"][True])
        root &= ((~finalize) | ready_lit) & ((~ready_lit) | finalize)
        if root.is_false():
            raise RuntimeError("Grounded constraints are unsatisfiable")
        root.ref()
        return cls(groups, manager, root, 1000 * (time.perf_counter() - start))

    def infer(self, local: dict[str, dict[Any, float]]) -> dict[str, Any]:
        count = sum(map(len, self.groups.values()))
        positive, negative = [1.0] * count, [1.0] * count
        for name, group in self.groups.items():
            for value, var in group.items():
                positive[var - 1] = local[name][value]
        start = time.perf_counter()
        wmc = self.root.wmc(log_mode=False)
        for var in range(1, count + 1):
            wmc.set_literal_weight(var, positive[var - 1])
            wmc.set_literal_weight(-var, negative[var - 1])
        z = float(wmc.propagate())
        marginals = {name: {v: float(wmc.literal_pr(var)) for v, var in group.items()} for name, group in self.groups.items()}
        mpe = max_product(self.root, self.manager.vtree(), positive, negative, self.manager.true())
        joint = {}
        for name, group in self.groups.items():
            selected = [value for value, var in group.items() if mpe.assignment[var] == 1]
            joint[name] = selected[0]
        return {"z": z, "marginals": marginals, "joint_map": joint, "inference_ms": 1000 * (time.perf_counter() - start)}


def argmax(values: dict[Any, float]) -> Any:
    return max(values, key=values.__getitem__)


def consistent(decision: dict[str, Any], actions: set[str], is_ready: bool, triples: set[tuple[str, str, str]]) -> bool:
    return (
        decision["action"] in actions
        and decision["ready"] == is_ready
        and (decision["action"] == "finalize") == is_ready
        and (decision["violation"], decision["risk"], decision["enforcement"]) in triples
    )


def rollout(
    engine: LocalJev,
    row: dict[str, str],
    mode: str,
    score_cache: dict[str, tuple[dict[str, dict[Any, float]], float]],
) -> dict[str, Any]:
    observations: dict[str, dict[str, Any]] = {}
    trace = []
    for step_index in range(8):
        state = make_state(row, observations)
        cache_hit = state in score_cache
        if cache_hit:
            local, model_ms = score_cache[state]
        else:
            started = time.perf_counter()
            local = distributions(engine.system_one(state, questions()))
            model_ms = 1000 * (time.perf_counter() - started)
            score_cache[state] = (local, model_ms)
        actions = legal_actions(observations)
        is_ready = ready(observations)
        triples = allowed_triples(row, observations)
        circuit = StepCircuit.build(actions, is_ready, triples)
        inferred = circuit.infer(local)
        raw = {name: argmax(values) for name, values in local.items()}
        if mode == "local":
            decision = raw
        elif mode == "masked":
            decision = dict(raw)
            decision["action"] = argmax({a: local["action"][a] for a in actions})
            decision["ready"] = is_ready
        elif mode == "probjev":
            decision = inferred["joint_map"]
        else:
            raise ValueError(mode)
        action_legal = decision["action"] in actions and (decision["action"] == "finalize") == is_ready
        trace.append({
            "step": step_index + 1, "completed_before": list(observations),
            "local": raw, "selected": decision, "action_legal": action_legal,
            "joint_consistent": consistent(decision, actions, is_ready, triples),
            "z": inferred["z"], "model_ms": model_ms,
            "model_cache_hit": cache_hit,
            "circuit_ms": circuit.compile_ms + inferred["inference_ms"],
        })
        if not action_legal:
            return {"success": False, "failure": "policy_violation", "final": None, "trace": trace}
        action = decision["action"]
        if action == "finalize":
            predicted = decision["enforcement"]
            return {"success": predicted == row["final_decision"], "failure": None if predicted == row["final_decision"] else "wrong_decision", "final": predicted, "trace": trace}
        observations[action] = tool_result(row, action)
    return {"success": False, "failure": "step_limit", "final": None, "trace": trace}


def procedural(row: dict[str, str], masked: dict[str, Any]) -> dict[str, Any]:
    run = copy.deepcopy(masked)
    if not run["trace"] or run["trace"][-1]["selected"]["action"] != "finalize":
        return run
    violation, risk, action = derive(row)
    last = run["trace"][-1]
    last["selected"].update(violation=violation, risk=risk, enforcement=action)
    last["joint_consistent"] = True
    run.update(success=action == row["final_decision"], failure=None if action == row["final_decision"] else "wrong_decision", final=action)
    return run


def summarize(records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": len(records), "model": model}
    for mode in ("local", "masked", "procedural", "probjev"):
        runs = [r[mode] for r in records]
        steps = [s for run in runs for s in run["trace"]]
        out[mode] = {
            "task_success_rate": float(np.mean([r["success"] for r in runs])),
            "policy_violation_rate": float(np.mean([r["failure"] == "policy_violation" for r in runs])),
            "wrong_decision_rate": float(np.mean([r["failure"] == "wrong_decision" for r in runs])),
            "mean_steps": float(np.mean([len(r["trace"]) for r in runs])),
            "joint_inconsistency_rate_per_step": float(np.mean([not s["joint_consistent"] for s in steps])),
            "mean_z": float(np.mean([s["z"] for s in steps])),
            "mean_model_ms_per_step": float(np.mean([s["model_ms"] for s in steps])),
            "mean_circuit_ms_per_step": float(np.mean([s["circuit_ms"] for s in steps])),
        }
    return out


def audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    mismatches = [r["account_id"] for r in rows if derive(r)[2] != r["final_decision"]]
    return {"rows": len(rows), "matches": len(rows) - len(mismatches), "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "sop_referral_abuse_v2_multistep.json")
    args = parser.parse_args()
    with DATA_PATH.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = all_rows[args.offset : args.offset + args.limit]
    engine = LocalJev(args.model, device=args.device, dtype=args.dtype)
    records = []
    for index, row in enumerate(rows, start=args.offset):
        score_cache: dict[str, tuple[dict[str, dict[Any, float]], float]] = {}
        item: dict[str, Any] = {"sample_index": index, "account_id": row["account_id"]}
        for mode in ("local", "masked", "probjev"):
            item[mode] = rollout(engine, row, mode, score_cache)
        item["procedural"] = procedural(row, item["masked"])
        records.append(item)
        print(f"[{len(records):>3}/{len(rows)}] {row['account_id']} " + " ".join(f"{m}={'OK' if item[m]['success'] else item[m]['failure']}" for m in ("local", "masked", "procedural", "probjev")), flush=True)
    report = {"dataset": "SOP-Bench referral_abuse_detection_v2", "rule_audit": audit(all_rows), "summary": summarize(records, args.model), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"rule_audit": report["rule_audit"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
