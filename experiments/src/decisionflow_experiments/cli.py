"""CLI for research reproduction; intentionally separate from decisionflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decisionflow import DecisionFlow

from .registry import EXPERIMENTS, get_experiment
from .runner import ExperimentRunner, TrajectoryExperimentRunner
from .metrics import summarize


def _open(path: str, mode: str):
    if path == "-":
        return sys.stdin if "r" in mode else sys.stdout
    target = Path(path)
    if "w" in mode:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target.open(mode, encoding="utf-8")


def command_verify(args) -> None:
    rows = []
    for spec in EXPERIMENTS.values():
        if spec.record_mode == "fixed":
            probabilities = {}
            for question in spec.questions:
                if question["type"] == "noul":
                    probabilities[question["id"]] = 0.5
                else:
                    probabilities[question["id"]] = [1.0] * len(question["options"])
            record = {"id": "schema-check", "state": {}, "probabilities": probabilities}
            result = ExperimentRunner.create(spec, args.backend).run_record(record)
            rows.append({
                "experiment": spec.name,
                "status": "ok",
                "worlds": result["inference"]["world_count"],
                "valid_worlds": result["inference"]["valid_world_count"],
            })
        else:
            rows.append({
                "experiment": spec.name,
                "status": "record-grounded",
                "note": "validated when a grounded cached record is supplied",
            })
    json.dump({"experiments": rows}, sys.stdout, indent=2)
    sys.stdout.write("\n")


def command_run(args) -> None:
    runner = ExperimentRunner.create(get_experiment(args.experiment), args.backend)
    source = _open(args.input, "r")
    target = _open(args.output, "w")
    try:
        runner.run_stream(source, target)
    finally:
        if source is not sys.stdin:
            source.close()
        if target is not sys.stdout:
            target.close()


def command_summarize(args) -> None:
    source = _open(args.input, "r")
    try:
        rows = [json.loads(line) for line in source if line.strip()]
    finally:
        if source is not sys.stdin:
            source.close()
    report = summarize(rows)
    target = _open(args.output, "w")
    try:
        json.dump(report, target, indent=2, sort_keys=True)
        target.write("\n")
    finally:
        if target is not sys.stdout:
            target.close()


def command_trajectory(args) -> None:
    source = _open(args.input, "r")
    target = _open(args.output, "w")
    try:
        TrajectoryExperimentRunner.create().run_stream(source, target)
    finally:
        if source is not sys.stdin:
            source.close()
        if target is not sys.stdout:
            target.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decisionflow-experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="validate registered experiment programs")
    verify.add_argument("--backend", choices=("auto", "enumeration", "sdd"), default="auto")
    verify.set_defaults(handler=command_verify)
    run = commands.add_parser("run", help="condition a standardized cached-score JSONL file")
    run.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    run.add_argument("--input", required=True)
    run.add_argument("--output", default="-")
    run.add_argument("--backend", choices=("auto", "enumeration", "sdd"), default="auto")
    run.set_defaults(handler=command_run)
    summary = commands.add_parser("summarize", help="aggregate a generated result JSONL file")
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", default="-")
    summary.set_defaults(handler=command_summarize)
    trajectory = commands.add_parser(
        "run-trajectories", help="evaluate grounded finite-horizon trajectory records"
    )
    trajectory.add_argument("--input", required=True)
    trajectory.add_argument("--output", default="-")
    trajectory.set_defaults(handler=command_trajectory)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
