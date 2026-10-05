#!/usr/bin/env python3
"""ProbJev pilot for the non-deterministic SOP-Bench traffic policy."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openjev import Choice, LocalJev
from pysdd.sdd import SddManager, Vtree

from sop_dangerous_goods_probjev import MODEL_ID, RESULT_DIR, max_product, normalize


DATA_PATH = Path("third_party/sop-bench/src/amazon_sop_bench/benchmarks/data/traffic_spoofing_detection/test_set_with_outputs.csv")
RISKS = ("High", "Medium", "Low")
VIOLATIONS = (
    "Traffic/URL Cloaking", "Spoofing Traffic", "Redirecting Traffic",
    "Blank Source Traffic", "Click Funneling", "Pop Ads Masking",
    "Traffic Stuffing", "Plugin/Software Usage", "None",
)
ACTIONS = ("Account Closure", "Temporary Suspension", "Warning Issued", "No Action")
NON_NONE = set(VIOLATIONS) - {"None"}


def valid_triples() -> set[tuple[str, str, str]]:
    triples = {("Low", "None", "No Action")}
    for violation in NON_NONE:
        triples.add(("High", violation, "Account Closure"))
        triples.add(("Medium", violation, "Temporary Suspension"))
        triples.add(("Medium", violation, "Warning Issued"))
    return triples


def state(row: dict[str, str]) -> str:
    fields = (
        "partner_id", "registered_websites", "earnings_amount", "unique_users",
        "total_orders", "total_clicks", "click_spike", "engagement_score",
        "session_count", "conversion_rate", "computer_traffic_percent",
        "phone_traffic_percent", "unattributed_clicks", "bounce_rate",
        "visit_duration_seconds", "direct_traffic_percent", "top_referral_source",
        "traffic_analysis_result", "source_verification_result",
    )
    evidence = "\n".join(f"{field}: {row[field]}" for field in fields)
    return (
        "Traffic-spoofing investigation:\n" + evidence + "\n\n"
        "Policy: high-risk violations require Account Closure. Medium-risk "
        "violations permit Temporary Suspension (default) or Warning Issued "
        "when contextual evidence supports discretion. Low risk without a "
        "confirmed violation requires No Action."
    )


def questions() -> dict[str, Any]:
    return {
        "risk": Choice(instructions="What is the traffic-spoofing risk level?", criteria={v: v for v in RISKS}),
        "violation": Choice(instructions="What violation type is supported?", criteria={v: v for v in VIOLATIONS}),
        "action": Choice(instructions="What enforcement action should be taken?", criteria={v: v for v in ACTIONS}),
    }


def distributions(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    answers = result["answers"]
    return {
        name: normalize({v: float(answers[name]["probabilities"][v]) for v in values})
        for name, values in (("risk", RISKS), ("violation", VIOLATIONS), ("action", ACTIONS))
    }


@dataclass
class PolicyCircuit:
    groups: dict[str, dict[str, int]]
    manager: SddManager
    root: Any
    compile_ms: float

    @classmethod
    def build(cls) -> "PolicyCircuit":
        groups: dict[str, dict[str, int]] = {}
        next_var = 1
        for name, values in (("risk", RISKS), ("violation", VIOLATIONS), ("action", ACTIONS)):
            groups[name] = {value: next_var + i for i, value in enumerate(values)}
            next_var += len(values)
        started = time.perf_counter()
        manager = SddManager.from_vtree(Vtree(var_count=next_var - 1, var_order=list(range(1, next_var)), vtree_type="balanced"))
        root = manager.true()
        for group in groups.values():
            literals = [manager.literal(var) for var in group.values()]
            at_least = manager.false()
            for literal in literals:
                at_least |= literal
            root &= at_least
            for left, right in itertools.combinations(literals, 2):
                root &= (~left) | (~right)
        relation = manager.false()
        for risk, violation, action in valid_triples():
            relation |= manager.literal(groups["risk"][risk]) & manager.literal(groups["violation"][violation]) & manager.literal(groups["action"][action])
        root &= relation
        root.ref()
        return cls(groups, manager, root, 1000 * (time.perf_counter() - started))

    def infer(self, local: dict[str, dict[str, float]]) -> dict[str, Any]:
        count = sum(map(len, self.groups.values()))
        positive, negative = [1.0] * count, [1.0] * count
        for name, group in self.groups.items():
            for value, var in group.items():
                positive[var - 1] = local[name][value]
        started = time.perf_counter()
        wmc = self.root.wmc(log_mode=False)
        for var in range(1, count + 1):
            wmc.set_literal_weight(var, positive[var - 1])
            wmc.set_literal_weight(-var, negative[var - 1])
        z = float(wmc.propagate())
        marginals = {name: {value: float(wmc.literal_pr(var)) for value, var in group.items()} for name, group in self.groups.items()}
        mpe = max_product(self.root, self.manager.vtree(), positive, negative, self.manager.true())
        joint = {name: next(value for value, var in group.items() if mpe.assignment[var] == 1) for name, group in self.groups.items()}
        return {"z": z, "marginals": marginals, "joint_map": joint, "inference_ms": 1000 * (time.perf_counter() - started)}


def argmax(values: dict[str, float]) -> str:
    return max(values, key=values.__getitem__)


def tuple_of(distributions_: dict[str, dict[str, float]]) -> dict[str, str]:
    return {name: argmax(values) for name, values in distributions_.items()}


def is_valid(decision: dict[str, str]) -> bool:
    return (decision["risk"], decision["violation"], decision["action"]) in valid_triples()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"samples": len(records)}
    for mode in ("local", "marginal", "map"):
        predictions = [record[mode] for record in records]
        out[mode] = {
            "joint_exact_match": float(np.mean([all(pred[k] == rec["gold"][k] for k in ("risk", "violation", "action")) for pred, rec in zip(predictions, records)])),
            "action_accuracy": float(np.mean([pred["action"] == rec["gold"]["action"] for pred, rec in zip(predictions, records)])),
            "policy_conflicts": sum(not is_valid(pred) for pred in predictions),
        }
    out["mean_z"] = float(np.mean([record["z"] for record in records]))
    out["mean_model_ms"] = float(np.mean([record["model_ms"] for record in records]))
    out["mean_circuit_ms"] = float(np.mean([record["circuit_ms"] for record in records]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "sop_traffic_spoofing_probjev.json")
    args = parser.parse_args()
    with DATA_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[args.offset : args.offset + args.limit]
    engine = LocalJev(args.model, device=args.device, dtype=args.dtype)
    circuit = PolicyCircuit.build()
    records = []
    for index, row in enumerate(rows, start=args.offset):
        started = time.perf_counter()
        local_dist = distributions(engine.system_one(state(row), questions()))
        model_ms = 1000 * (time.perf_counter() - started)
        inferred = circuit.infer(local_dist)
        record = {
            "sample_index": index, "partner_id": row["partner_id"],
            "gold": {"risk": row["risk_level"], "violation": row["violation_type"], "action": row["enforcement_action"]},
            "local": tuple_of(local_dist), "marginal": tuple_of(inferred["marginals"]),
            "map": inferred["joint_map"], "z": inferred["z"],
            "model_ms": model_ms, "circuit_ms": circuit.compile_ms + inferred["inference_ms"],
        }
        records.append(record)
        print(f"[{len(records):>3}/{len(rows)}] {row['partner_id']} local={record['local']['action']} map={record['map']['action']} gold={record['gold']['action']}", flush=True)
    report = {"dataset": "SOP-Bench traffic_spoofing_detection", "hard_policy_worlds": len(valid_triples()), "summary": summarize(records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
