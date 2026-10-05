#!/usr/bin/env python3
"""Multi-step OpenJev/ProbJev pilot for SOP-Bench dangerous goods."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openjev import Choice, LocalJev, Noul
from pysdd.sdd import SddManager, Vtree

from sop_dangerous_goods_probjev import (
    ARTIFACT_DIR,
    CLASS_VALUES,
    DATA_PATH,
    MODEL_ID,
    RESULT_DIR,
    class_for,
    max_product,
    normalize,
    valid_product_id,
)


TOOLS = (
    "calculate_sds_label_score",
    "calculate_handling_score",
    "calculate_transportation_score",
    "calculate_disposal_score",
)
ACTIONS = TOOLS + ("finalize",)
CLASS_OPTION_ORDER = (
    "Hazard Class A",
    "Hazard Class B",
    "Hazard Class C",
    "Hazard Class D",
    "Unable to Decide",
)
TOOL_COLUMNS = {
    "calculate_sds_label_score": "sds_label_score",
    "calculate_handling_score": "handling_score",
    "calculate_transportation_score": "transportation_score",
    "calculate_disposal_score": "disposal_score",
}
TOOL_SHORT = {
    "calculate_sds_label_score": "SDS",
    "calculate_handling_score": "handling",
    "calculate_transportation_score": "transportation",
    "calculate_disposal_score": "disposal",
}


def value_from_row(row: dict[str, str], tool: str) -> int:
    text = row[TOOL_COLUMNS[tool]].strip()
    return 0 if not text else int(float(text))


def final_class(row: dict[str, str], observations: dict[str, int]) -> str:
    values = tuple(observations[tool] for tool in TOOLS)
    return class_for(values, valid_product_id(row["product_id"]))


def ready_for_final(row: dict[str, str], observations: dict[str, int]) -> bool:
    return not valid_product_id(row["product_id"]) or len(observations) == len(TOOLS)


def allowed_actions(row: dict[str, str], observations: dict[str, int]) -> set[str]:
    if ready_for_final(row, observations):
        return {"finalize"}
    return set(TOOLS) - observations.keys()


def possible_classes(row: dict[str, str], observations: dict[str, int]) -> set[str]:
    if not valid_product_id(row["product_id"]):
        return {"Unable to Decide"}
    if len(observations) == len(TOOLS):
        return {final_class(row, observations)}

    remaining = [tool for tool in TOOLS if tool not in observations]
    possible: set[str] = set()
    # Future tool outputs have finite support 0..5. Enumerating at most 6^4
    # completions is a transparent grounding step, not probabilistic inference.
    import itertools

    for completion in itertools.product(range(6), repeat=len(remaining)):
        candidate = dict(observations)
        candidate.update(zip(remaining, completion, strict=True))
        possible.add(final_class(row, candidate))
    return possible


def make_state(row: dict[str, str], observations: dict[str, int]) -> str:
    observed = (
        "none"
        if not observations
        else ", ".join(
            f"{TOOL_SHORT[tool]} score={score}" for tool, score in observations.items()
        )
    )
    completed = ", ".join(TOOL_SHORT[tool] for tool in observations) or "none"
    return (
        "Dangerous-goods SOP execution state:\n"
        f"Product ID: {row['product_id']}\n"
        f"SDS Section 2: {row['sds_label_text']}\n"
        f"Handling and storage: {row['handling_and_storage_guidelines']}\n"
        f"Transportation requirements: {row['transportation_requirements']}\n"
        f"Disposal guidelines: {row['disposal_guidelines']}\n"
        f"Completed scoring tools: {completed}\n"
        f"Observed tool results: {observed}\n\n"
        "Policy: a valid ID is P_ followed by five digits. For a valid ID, call "
        "each of the four scoring tools exactly once before finalizing. For an "
        "invalid ID, call no scoring tool and finalize Unable to Decide. One zero "
        "score is replaced by the maximum other score; two or more zero scores "
        "give Unable to Decide. Class bands are A=4--7, B=8--12, C=13--16, "
        "and D=17--20."
    )


def questions() -> dict[str, Any]:
    return {
        "next_action": Choice(
            instructions="Which single action should the agent take next?",
            criteria={
                "calculate_sds_label_score": "Calculate the SDS severity score.",
                "calculate_handling_score": "Calculate the handling and storage score.",
                "calculate_transportation_score": "Calculate the transportation score.",
                "calculate_disposal_score": "Calculate the disposal score.",
                "finalize": "Return the final hazard classification and stop.",
            },
        ),
        "evidence_sufficient": Noul(
            instructions="Is the available evidence sufficient to finalize now?"
        ),
        "hazard_class": Choice(
            instructions="What final hazard class is supported by the current state?",
            criteria={label: label for label in CLASS_OPTION_ORDER},
        ),
    }


def distributions(result: dict[str, Any]) -> dict[str, dict[Any, float]]:
    answers = result["answers"]
    ready = float(answers["evidence_sufficient"]["noul"])
    return {
        "action": normalize(
            {
                action: float(answers["next_action"]["probabilities"][action])
                for action in ACTIONS
            }
        ),
        "ready": normalize({False: 1.0 - ready, True: ready}),
        "class": normalize(
            {
                label: float(answers["hazard_class"]["probabilities"][label])
                for label in CLASS_VALUES
            }
        ),
    }


@dataclass
class StepCircuit:
    groups: dict[str, dict[Any, int]]
    manager: SddManager
    root: Any
    compile_ms: float

    @classmethod
    def build(
        cls,
        legal_actions: set[str],
        ready_evidence: bool,
        legal_classes: set[str],
    ) -> "StepCircuit":
        groups: dict[str, dict[Any, int]] = {}
        next_var = 1
        groups["action"] = {value: next_var + i for i, value in enumerate(ACTIONS)}
        next_var += len(ACTIONS)
        groups["ready"] = {False: next_var, True: next_var + 1}
        next_var += 2
        groups["class"] = {
            value: next_var + i for i, value in enumerate(CLASS_VALUES)
        }
        next_var += len(CLASS_VALUES)

        start = time.perf_counter()
        vtree = Vtree(
            var_count=next_var - 1,
            var_order=list(range(1, next_var)),
            vtree_type="balanced",
        )
        manager = SddManager.from_vtree(vtree)
        root = manager.true()
        for group in groups.values():
            literals = [manager.literal(var) for var in group.values()]
            at_least_one = manager.false()
            for literal in literals:
                at_least_one |= literal
            root &= at_least_one
            for index, left in enumerate(literals):
                for right in literals[index + 1 :]:
                    root &= (~left) | (~right)

        for action, var in groups["action"].items():
            if action not in legal_actions:
                root &= ~manager.literal(var)
        root &= manager.literal(groups["ready"][ready_evidence])
        for label, var in groups["class"].items():
            if label not in legal_classes:
                root &= ~manager.literal(var)

        # Cross-head policy relation, retained even though readiness is also
        # grounded as evidence for the current state.
        finalize = manager.literal(groups["action"]["finalize"])
        ready = manager.literal(groups["ready"][True])
        root &= ((~finalize) | ready) & ((~ready) | finalize)
        if root.is_false():
            raise RuntimeError("Grounded step constraints are unsatisfiable")
        root.ref()
        return cls(groups, manager, root, 1000.0 * (time.perf_counter() - start))

    def infer(self, local: dict[str, dict[Any, float]]) -> dict[str, Any]:
        var_count = sum(len(group) for group in self.groups.values())
        positive = [1.0] * var_count
        negative = [1.0] * var_count
        for name, group in self.groups.items():
            for value, var in group.items():
                positive[var - 1] = local[name][value]

        start = time.perf_counter()
        wmc = self.root.wmc(log_mode=False)
        for var in range(1, var_count + 1):
            wmc.set_literal_weight(var, positive[var - 1])
            wmc.set_literal_weight(-var, negative[var - 1])
        z = float(wmc.propagate())
        marginals = {
            name: {
                value: float(wmc.literal_pr(var)) for value, var in group.items()
            }
            for name, group in self.groups.items()
        }
        mpe = max_product(
            self.root,
            self.manager.vtree(),
            positive,
            negative,
            self.manager.true(),
        )
        joint_map = {}
        for name, group in self.groups.items():
            selected = [value for value, var in group.items() if mpe.assignment[var] == 1]
            if len(selected) != 1:
                raise RuntimeError(f"Invalid MAP assignment for {name}: {selected}")
            joint_map[name] = selected[0]
        return {
            "z": z,
            "marginals": marginals,
            "joint_map": joint_map,
            "joint_map_probability": mpe.weight / z,
            "inference_ms": 1000.0 * (time.perf_counter() - start),
        }


def argmax(values: dict[Any, float]) -> Any:
    return max(values, key=values.__getitem__)


def local_tuple(local: dict[str, dict[Any, float]]) -> dict[str, Any]:
    return {name: argmax(values) for name, values in local.items()}


def tuple_is_legal(
    decision: dict[str, Any],
    legal_actions: set[str],
    ready: bool,
    legal_classes: set[str],
) -> bool:
    return (
        decision["action"] in legal_actions
        and decision["ready"] == ready
        and decision["class"] in legal_classes
        and (decision["action"] == "finalize") == ready
    )


def action_is_operationally_legal(
    action: str,
    legal_actions: set[str],
    ready: bool,
) -> bool:
    return action in legal_actions and (action == "finalize") == ready


def rollout(
    engine: LocalJev,
    row: dict[str, str],
    mode: str,
    max_steps: int = 8,
) -> dict[str, Any]:
    observations: dict[str, int] = {}
    trace = []
    for step in range(max_steps):
        state = make_state(row, observations)
        start = time.perf_counter()
        result = engine.system_one(state, questions())
        model_ms = 1000.0 * (time.perf_counter() - start)
        local = distributions(result)
        local_decision = local_tuple(local)
        legal_actions = allowed_actions(row, observations)
        ready = ready_for_final(row, observations)
        legal_classes = possible_classes(row, observations)

        circuit = StepCircuit.build(legal_actions, ready, legal_classes)
        inferred = circuit.infer(local)
        if mode == "local":
            decision = local_decision
        elif mode == "masked":
            decision = {
                "action": argmax(
                    {a: p for a, p in local["action"].items() if a in legal_actions}
                ),
                "ready": ready,
                "class": argmax(local["class"]),
            }
        elif mode == "probjev":
            decision = inferred["joint_map"]
        else:
            raise ValueError(mode)

        joint_consistent = tuple_is_legal(
            decision, legal_actions, ready, legal_classes
        )
        operationally_legal = action_is_operationally_legal(
            decision["action"], legal_actions, ready
        )
        trace.append(
            {
                "step": step + 1,
                "observations_before": dict(observations),
                "legal_actions": sorted(legal_actions),
                "possible_classes": sorted(legal_classes),
                "local": local_decision,
                "selected": decision,
                "selected_is_operationally_legal": operationally_legal,
                "selected_joint_is_consistent": joint_consistent,
                "z": inferred["z"],
                "model_ms": model_ms,
                "compile_ms": circuit.compile_ms,
                "circuit_inference_ms": inferred["inference_ms"],
            }
        )
        if not operationally_legal:
            return {
                "success": False,
                "failure": "policy_violation",
                "trace": trace,
                "final_class": None,
            }
        action = decision["action"]
        if action == "finalize":
            predicted_class = decision["class"]
            return {
                "success": predicted_class == row["hazard_class"],
                "failure": None if predicted_class == row["hazard_class"] else "wrong_class",
                "trace": trace,
                "final_class": predicted_class,
            }
        observations[action] = value_from_row(row, action)

    return {
        "success": False,
        "failure": "step_limit",
        "trace": trace,
        "final_class": None,
    }


def procedural_from_masked(
    row: dict[str, str], masked_run: dict[str, Any]
) -> dict[str, Any]:
    """Add a deterministic final rule evaluator to the action-mask baseline."""
    run = copy.deepcopy(masked_run)
    if not run["trace"] or run["failure"] == "policy_violation":
        return run
    last = run["trace"][-1]
    if last["selected"]["action"] != "finalize":
        return run
    observations = dict(last["observations_before"])
    if valid_product_id(row["product_id"]):
        for step in run["trace"]:
            action = step["selected"]["action"]
            if action in TOOLS:
                observations[action] = value_from_row(row, action)
        if not ready_for_final(row, observations):
            return run
        label = final_class(row, observations)
    else:
        label = "Unable to Decide"
    last["selected"]["class"] = label
    last["selected_joint_is_consistent"] = True
    run["final_class"] = label
    run["success"] = label == row["hazard_class"]
    run["failure"] = None if run["success"] else "wrong_class"
    return run


def summarize(records: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(records), "model": model_id}
    for mode in ("local", "masked", "procedural", "probjev"):
        runs = [record[mode] for record in records]
        all_steps = [step for run in runs for step in run["trace"]]
        summary[mode] = {
            "task_success_rate": sum(run["success"] for run in runs) / len(runs),
            "policy_violation_rate": sum(
                run["failure"] == "policy_violation" for run in runs
            ) / len(runs),
            "wrong_class_rate": sum(run["failure"] == "wrong_class" for run in runs)
            / len(runs),
            "mean_steps": float(np.mean([len(run["trace"]) for run in runs])),
            "joint_inconsistency_rate_per_step": sum(
                not step["selected_joint_is_consistent"] for step in all_steps
            )
            / len(all_steps),
            "mean_model_ms_per_step": float(
                np.mean([step["model_ms"] for step in all_steps])
            ),
            "mean_valid_mass_z": float(np.mean([step["z"] for step in all_steps])),
            "mean_circuit_ms_per_step": float(
                np.mean(
                    [step["compile_ms"] + step["circuit_inference_ms"] for step in all_steps]
                )
            ),
        }
    return summary


def run(
    limit: int,
    offset: int,
    model_id: str,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    with DATA_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))[offset : offset + limit]
    engine = LocalJev(model_id, device=device, dtype=dtype)
    records = []
    for index, row in enumerate(rows, start=offset):
        item = {"sample_index": index, "product_id": row["product_id"]}
        for mode in ("local", "masked", "probjev"):
            item[mode] = rollout(engine, row, mode)
        item["procedural"] = procedural_from_masked(row, item["masked"])
        records.append(item)
        print(
            f"[{len(records):>3}/{len(rows)}] {row['product_id']} "
            + " ".join(
                f"{mode}={'OK' if item[mode]['success'] else item[mode]['failure']}"
                for mode in ("local", "masked", "procedural", "probjev")
            ),
            flush=True,
        )
    return {
        "dataset": "Amazon SOP-Bench / dangerous_goods / multi-step",
        "summary": summarize(records, model_id),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "sop_dangerous_goods_multistep.json",
    )
    args = parser.parse_args()
    report = run(args.limit, args.offset, args.model, args.device, args.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
