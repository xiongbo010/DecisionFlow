"""Full pilot reproduction with inference delegated to DecisionFlow.

The tracked pilot modules remain the reference implementation of dataset
loading, label policies, metrics, bootstrap intervals, and audits. This module
replaces their enumeration/SDD inference functions at runtime with calls to the
public DecisionFlow API, then invokes their unchanged result builders.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from decisionflow import DecisionFlow, LocalPotentials

from .registry import EXPERIMENTS


PILOTS = {
    "openai_moderation": "openai_moderation_pilot",
    "toxigen": "toxigen_sdd_pilot",
    "goemotions": "goemotions_sdd_pilot",
    "toxicchat": "toxicchat_sdd_pilot",
    "helpsteer2": "helpsteer2_sdd_pilot",
}


def _load_pilot(repo_root: Path, name: str):
    pilot_dir = repo_root / "pilot"
    if not pilot_dir.is_dir():
        raise FileNotFoundError("tracked pilot source is missing: %s" % pilot_dir)
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    module = importlib.import_module(PILOTS[name])
    parquet = None
    if hasattr(module, "pd"):
        # The local pilot bundle contains PyArrow for parquet decoding. Convert
        # through Python records to avoid coupling pandas' optional Arrow
        # extension registration to that historical binary build.
        parquet = importlib.import_module("pyarrow.parquet")
    # Historical pilots added a local dependency directory to sys.path for
    # one-off execution. The reproducibility environment installs dependencies
    # normally; leaving that directory first can mix incompatible PyArrow
    # builds when pandas imports its parquet engine lazily.
    local_deps = str(pilot_dir / ".deps")
    while local_deps in sys.path:
        sys.path.remove(local_deps)
    if parquet is not None:
        module.pd.read_parquet = lambda path: module.pd.DataFrame(
            parquet.read_table(path).to_pylist()
        )
    return module


def _circuit_info(result, samples: int, elapsed: float) -> Dict[str, Any]:
    info = result.inference
    return {
        "library": "DecisionFlow/PySDD" if info.backend == "sdd" else "DecisionFlow",
        "version": "0.0.0",
        "vtree": "balanced" if info.backend == "sdd" else None,
        "node_count": info.circuit_nodes,
        "size_elements": info.circuit_elements,
        "model_count": info.valid_world_count,
        "reloaded_model_count": info.valid_world_count,
        "compile_seconds": info.compile_ms / 1000.0,
        "inference_seconds": elapsed,
        "milliseconds_per_sample": 1000.0 * elapsed / samples,
        "backend": info.backend,
        "public_api": "DecisionFlow.compile + DecisionFlow.infer_many",
    }


def _infer(
    questions: Sequence[Mapping[str, Any]],
    constraints: Mapping[str, Any],
    probability_rows: Sequence[Mapping[str, Mapping[Any, float]]],
):
    engine = DecisionFlow(backend="sdd")
    request = {"state": {}, "questions": list(questions)}
    program = engine.compile(request, constraints)
    started = time.perf_counter()
    results = list(
        engine.infer_many(
            (program, LocalPotentials(probabilities)) for probabilities in probability_rows
        )
    )
    elapsed = time.perf_counter() - started
    return results, elapsed


def _bool_rows(names: Sequence[str], probabilities: np.ndarray):
    return [
        {
            name: {False: 1.0 - float(row[index]), True: float(row[index])}
            for index, name in enumerate(names)
        }
        for row in probabilities
    ]


def _patch_openai(module):
    names = tuple(module.HEADS)

    def condition(probabilities):
        results, _ = _infer(
            EXPERIMENTS["openai_moderation"].questions,
            EXPERIMENTS["openai_moderation"].constraints,
            _bool_rows(names, probabilities),
        )
        marginals = np.asarray(
            [[result.marginals[name][True] for name in names] for result in results]
        )
        z = np.asarray([result.valid_mass for result in results])
        maps = np.asarray(
            [[int(result.joint_map[name]) for name in names] for result in results],
            dtype=np.int8,
        )
        return marginals, z, maps

    module.exact_condition = condition


def _toxigen_constraints(policy: str):
    if policy == "anchors":
        return EXPERIMENTS["toxigen"].constraints
    if policy == "full_semantics":
        return {
            "name": "toxigen-full-semantics",
            "hard": [{
                "name": "toxic-iff-nonbenign",
                "expr": {"iff": [
                    {"eq": [{"var": "toxic"}, True]},
                    {"gt": [{"var": "toxicity"}, 0]},
                ]},
            }],
        }
    raise ValueError(policy)


def _patch_toxigen(module):
    def condition(noul, score_probs, policy, include_info=False):
        rows = [
            {
                "toxic": {False: 1.0 - float(toxic), True: float(toxic)},
                "toxicity": {level: float(scores[level]) for level in range(5)},
            }
            for toxic, scores in zip(noul, score_probs, strict=True)
        ]
        results, elapsed = _infer(
            EXPERIMENTS["toxigen"].questions, _toxigen_constraints(policy), rows
        )
        toxic_m = np.asarray([result.marginals["toxic"][True] for result in results])
        score_m = np.asarray(
            [[result.marginals["toxicity"][level] for level in range(5)] for result in results]
        )
        z = np.asarray([result.valid_mass for result in results])
        maps = np.asarray(
            [[int(result.joint_map["toxic"]), int(result.joint_map["toxicity"])] for result in results],
            dtype=np.int8,
        )
        weights = np.asarray(
            [result.map_probability * result.diagnostics["normalizer"] for result in results]
        )
        payload = (toxic_m, score_m, z, maps, weights)
        if include_info:
            payload += (_circuit_info(results[0], len(results), elapsed),)
        return payload

    module.enumerate_condition = lambda noul, scores, policy: condition(
        noul, scores, policy, False
    )
    module.circuit_condition = lambda noul, scores, policy, artifact_dir: condition(
        noul, scores, policy, True
    )


def _patch_toxicchat(module):
    names = tuple(module.HEADS)

    def condition(probabilities, include_info=False):
        results, elapsed = _infer(
            EXPERIMENTS["toxicchat"].questions,
            EXPERIMENTS["toxicchat"].constraints,
            _bool_rows(names, probabilities),
        )
        marginals = np.asarray(
            [[result.marginals[name][True] for name in names] for result in results]
        )
        z = np.asarray([result.valid_mass for result in results])
        maps = np.asarray(
            [[int(result.joint_map[name]) for name in names] for result in results],
            dtype=np.int8,
        )
        weights = np.asarray(
            [result.map_probability * result.diagnostics["normalizer"] for result in results]
        )
        payload = (marginals, z, maps, weights)
        if include_info:
            payload += (_circuit_info(results[0], len(results), elapsed),)
        return payload

    module.enumerate_condition = lambda probabilities: condition(probabilities, False)
    module.circuit_condition = lambda probabilities, artifact_dir: condition(
        probabilities, True
    )


def _patch_goemotions(module):
    names = tuple(module.LABELS)
    original_analytic = module.analytic_condition

    def condition(probabilities, constrained, include_info=False):
        constraints = {
            "name": "goemotions-neutral-exclusion",
            "hard": [
                {
                    "name": "neutral-excludes-%s" % names[index],
                    "expr": {"implies": [
                        {"eq": [{"var": "neutral"}, True]},
                        {"eq": [{"var": names[index]}, False]},
                    ]},
                }
                for index in constrained
            ],
        }
        results, elapsed = _infer(
            EXPERIMENTS["goemotions"].questions,
            constraints,
            _bool_rows(names, probabilities),
        )
        marginals = np.asarray(
            [[result.marginals[name][True] for name in names] for result in results]
        )
        z = np.asarray([result.valid_mass for result in results])
        maps = np.asarray(
            [[int(result.joint_map[name]) for name in names] for result in results],
            dtype=np.int8,
        )
        weights = np.asarray(
            [result.map_probability * result.diagnostics["normalizer"] for result in results]
        )
        # These three derived OR-event queries are retained from the audited
        # closed form. All decision marginals, Z, and MAP come from DecisionFlow.
        parents = original_analytic(probabilities, constrained)[4]
        payload = (marginals, z, maps, weights, parents)
        if include_info:
            payload += (_circuit_info(results[0], len(results), elapsed),)
        return payload

    module.analytic_condition = lambda probabilities, constrained: condition(
        probabilities, constrained, False
    )
    module.circuit_condition = lambda probabilities, name, constrained, artifact_dir: condition(
        probabilities, constrained, True
    )


def _helpsteer_constraints(module, penalties):
    document = dict(EXPERIMENTS["helpsteer2"].constraints)
    if penalties is None:
        return document
    expressions = (
        {"implies": [{"ge": [{"var": "helpfulness"}, 3]}, {"ge": [{"var": "correctness"}, 2]}]},
        {"implies": [{"ge": [{"var": "helpfulness"}, 3]}, {"ge": [{"var": "coherence"}, 2]}]},
        {"implies": [{"ge": [{"var": "correctness"}, 4]}, {"ge": [{"var": "helpfulness"}, 3]}]},
    )
    document["soft"] = [
        {
            "name": rule.name,
            "penalty": -math.log(float(weight)),
            "expr": expression,
        }
        for rule, weight, expression in zip(module.SOFT_RULES, penalties, expressions, strict=True)
    ]
    return document


def _patch_helpsteer(module):
    names = tuple(module.HEADS)

    def condition(probabilities, penalties, include_info=False):
        rows = [
            {
                name: {level: float(sample[head, level]) for level in range(5)}
                for head, name in enumerate(names)
            }
            for sample in probabilities
        ]
        results, elapsed = _infer(
            EXPERIMENTS["helpsteer2"].questions,
            _helpsteer_constraints(module, penalties),
            rows,
        )
        marginals = np.asarray(
            [
                [[result.marginals[name][level] for level in range(5)] for name in names]
                for result in results
            ]
        )
        partition = np.asarray([result.diagnostics["normalizer"] for result in results])
        maps = np.asarray(
            [[int(result.joint_map[name]) for name in names] for result in results],
            dtype=np.int8,
        )
        weights = np.asarray(
            [result.map_probability * result.diagnostics["normalizer"] for result in results]
        )
        payload = (marginals, partition, maps, weights)
        if include_info:
            payload += (_circuit_info(results[0], len(results), elapsed),)
        return payload

    module.enumerate_inference = lambda probabilities, penalties: condition(
        probabilities, penalties, False
    )
    module.circuit_inference = lambda probabilities, penalties, artifact_dir: condition(
        probabilities, penalties, True
    )


PATCHERS = {
    "openai_moderation": _patch_openai,
    "toxigen": _patch_toxigen,
    "goemotions": _patch_goemotions,
    "toxicchat": _patch_toxicchat,
    "helpsteer2": _patch_helpsteer,
}


def run_full(repo_root: Path, name: str) -> Dict[str, Any]:
    module = _load_pilot(repo_root, name)
    PATCHERS[name](module)
    tmp = repo_root / "tmp"
    responses = tmp / "jev-benchmarking" / "cache-responses.db"
    artifacts = repo_root / "experiments" / "artifacts" / name
    if name == "openai_moderation":
        return module.run(tmp / "jev-benchmarking" / "samples-1680.jsonl.gz", responses)
    if name == "toxigen":
        return module.run(tmp / "toxigen-test.parquet", responses, artifacts)
    if name == "goemotions":
        return module.run(
            tmp / "go-emotions-test.parquet",
            tmp / "go-emotions-validation.parquet",
            responses,
            artifacts,
        )
    if name == "toxicchat":
        return module.run(
            tmp / "toxicchat-test.parquet",
            tmp / "toxicchat-train.parquet",
            responses,
            artifacts,
        )
    if name == "helpsteer2":
        return module.run(
            tmp / "helpsteer2-validation.parquet",
            tmp / "helpsteer2-train.parquet",
            responses,
            artifacts,
        )
    raise KeyError(name)


def write_full(repo_root: Path, name: str, output: Path) -> Dict[str, Any]:
    result = run_full(repo_root, name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
