from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .engine import WorldJev
from .scorers.callable import CallableScorer
from .trajectory import TrajectoryEngine, parse_trajectory


def _load_json(path: str):
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _scorer(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("scorer must use module:attribute syntax")
    target = getattr(importlib.import_module(module_name), attribute)
    if isinstance(target, type):
        target = target()
    if hasattr(target, "score"):
        return target
    return CallableScorer(target)


def _engine(args):
    return WorldJev(
        scorer=_scorer(args.scorer) if getattr(args, "scorer", None) else None,
        backend=args.backend,
        enumeration_limit=args.enumeration_limit,
    )


def _constraints(args):
    return _load_json(args.constraints) if args.constraints else None


def command_infer(args):
    result = _engine(args).infer(_load_json(args.request), _constraints(args))
    json.dump(result.to_dict(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def command_evaluate(args):
    engine = _engine(args)
    constraints = _constraints(args)
    source = sys.stdin if args.input == "-" else Path(args.input).open(encoding="utf-8")
    target = sys.stdout if args.output == "-" else Path(args.output).open("w", encoding="utf-8")
    try:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                result = engine.infer(record, constraints)
                output = {"id": record.get("id", line_number), "status": "ok", **result.to_dict()}
            except Exception as error:
                output = {
                    "id": record.get("id", line_number),
                    "status": "error",
                    "error": type(error).__name__,
                    "message": str(error),
                }
            target.write(json.dumps(output, default=str) + "\n")
            target.flush()
    finally:
        if source is not sys.stdin:
            source.close()
        if target is not sys.stdout:
            target.close()


def command_trajectory(args):
    result = TrajectoryEngine().infer(parse_trajectory(_load_json(args.spec)))
    json.dump(result.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")


def _common(parser):
    parser.add_argument("--constraints")
    parser.add_argument("--scorer", help="Python plugin as module:attribute")
    parser.add_argument("--backend", choices=("auto", "enumeration", "sdd"), default="auto")
    parser.add_argument("--enumeration-limit", type=int, default=100_000)


def build_parser():
    parser = argparse.ArgumentParser(prog="worldjev")
    subcommands = parser.add_subparsers(dest="command", required=True)
    infer = subcommands.add_parser("infer", help="infer one typed decision request")
    infer.add_argument("request")
    _common(infer)
    infer.set_defaults(handler=command_infer)
    evaluate = subcommands.add_parser("evaluate", help="evaluate JSONL requests")
    evaluate.add_argument("input")
    evaluate.add_argument("--output", default="-")
    _common(evaluate)
    evaluate.set_defaults(handler=command_evaluate)
    trajectory = subcommands.add_parser("trajectory", help="infer a finite trajectory model")
    trajectory.add_argument("spec")
    trajectory.set_defaults(handler=command_trajectory)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

