from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .api import DecisionFlow
from .scorers.callable import CallableScorer
from .workflows.inference import create_flow_backend_registry


def _load_json(path: str):
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _model(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("model must use module:attribute syntax")
    target = getattr(importlib.import_module(module_name), attribute)
    if isinstance(target, type):
        target = target()
    if hasattr(target, "score"):
        return target
    return CallableScorer(target)


def _backend_options(value):
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return _load_json(value)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("backend options must be a JSON object")
    return parsed


def command_run(args):
    flow = DecisionFlow.from_file(
        workflow=args.workflow,
        model=_model(args.model),
        policies=args.policies,
        backend=args.backend,
        backend_options=_backend_options(args.backend_options),
        max_worlds=args.max_worlds,
    )
    result = flow.infer(_load_json(args.state))
    json.dump(result.to_dict(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def command_backends(args):
    payload = {
        name: descriptor.to_dict()
        for name, descriptor in create_flow_backend_registry().describe().items()
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(prog="decisionflow")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser(
        "run", help="compile and infer a declarative decision workflow"
    )
    run.add_argument("workflow", help="JSON or YAML workflow file")
    run.add_argument("state", help="JSON state file, or - for stdin")
    run.add_argument("--model", required=True, help="model plugin as module:attribute")
    run.add_argument("--policies", help="optional JSON or YAML policy file")
    run.add_argument("--backend", default="auto")
    run.add_argument("--backend-options")
    run.add_argument("--max-worlds", type=int, default=100_000)
    run.set_defaults(handler=command_run)
    backends = subcommands.add_parser(
        "backends", help="list structured-flow inference methods"
    )
    backends.set_defaults(handler=command_backends)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
