"""CLI for research reproduction; intentionally separate from decisionflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import summarize
from .registry import EXPERIMENTS, get_experiment
from .runner import ExperimentRunner, TrajectoryExperimentRunner

FULL_PILOTS = (
    "openai_moderation",
    "toxigen",
    "goemotions",
    "toxicchat",
    "helpsteer2",
)


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
            rows.append(
                {
                    "experiment": spec.name,
                    "status": "ok",
                    "worlds": result["inference"]["world_count"],
                    "valid_worlds": result["inference"]["valid_world_count"],
                }
            )
        else:
            rows.append(
                {
                    "experiment": spec.name,
                    "status": "record-grounded",
                    "note": "validated when a grounded cached record is supplied",
                }
            )
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


def command_full(args) -> None:
    from .full import write_full

    root = Path(args.repo_root).resolve()
    names = FULL_PILOTS if args.experiment == "all" else (args.experiment,)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    completed = []
    for name in names:
        output = output_dir / (name + ".json")
        result = write_full(root, name, output)
        completed.append(
            {"experiment": name, "output": str(output), "keys": list(result)}
        )
    json.dump({"completed": completed}, sys.stdout, indent=2)
    sys.stdout.write("\n")


def command_gliner25(args) -> None:
    from .gliner25_fast_decisions import RULE_DATASETS, run, write_report

    datasets = RULE_DATASETS if args.dataset == "all" else (args.dataset,)
    report = run(
        datasets,
        model_id=args.model,
        batch_size=args.batch_size,
        backend=args.backend,
    )
    output = Path(args.output)
    write_report(report, output)
    json.dump(
        {
            "output": str(output),
            "datasets": [row["dataset"] for row in report["datasets"]],
        },
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


def command_logistics(args) -> None:
    from .logistics import run_pilot

    root = Path(args.dataset_root) / "logistics_planbench"
    result = run_pilot(
        root / "domain.pddl",
        root / "adapted_instances" / f"instance-{args.instance}.pddl",
        root / "adapted_gold_plans" / f"instance-{args.instance}_gold_plan.txt",
        horizon=args.horizon,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        cache_path=Path(args.cache) if args.cache else None,
        scorer=args.scorer,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    json.dump({"output": str(output), "summary": result}, sys.stdout, indent=2)
    sys.stdout.write("\n")


def command_depot(args) -> None:
    from .depot import run_pilot

    root = Path(args.dataset_root) / "depot_planbench"
    result = run_pilot(
        root / "domain.pddl",
        root / "orig_problems" / f"instance-{args.instance}.pddl",
        root / "orig_gold_plans" / f"instance-{args.instance}_gold_plan.txt",
        horizon=args.horizon,
        model_id=args.model,
        device=args.device,
        cache_path=Path(args.cache) if args.cache else None,
        batch_size=args.batch_size,
        scorer_name=args.scorer,
        beam_widths=args.beam_widths,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    json.dump({"output": str(output), "summary": result}, sys.stdout, indent=2)
    sys.stdout.write("\n")


def command_trex(args) -> None:
    from .trex import run_pilot

    result = run_pilot(
        clm_root=Path(args.clm_root),
        seeds=args.seeds,
        scenarios_per_seed=args.scenarios_per_seed,
        horizon=args.horizon,
        decision_frames=args.decision_frames,
        capture_distance=args.capture_distance,
        scorer_name=args.scorer,
        encoder=args.encoder,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        head_repo=args.head_repo,
        head_filename=args.head_filename,
        cache_path=Path(args.cache) if args.cache else None,
        beam_widths=args.beam_widths,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    json.dump(
        {"output": str(output), "aggregate": result["aggregate"]}, sys.stdout, indent=2
    )
    sys.stdout.write("\n")


def command_trex_full(args) -> None:
    from .trex_full import run_full_courses

    result = run_full_courses(
        clm_root=Path(args.clm_root),
        seeds=args.seeds,
        duration_seconds=args.duration,
        horizon=args.horizon,
        decision_frames=args.decision_frames,
        capture_distance=args.capture_distance,
        encoder=args.encoder,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        head_repo=args.head_repo,
        head_filename=args.head_filename,
        cache_path=Path(args.cache) if args.cache else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    json.dump(
        {"output": str(output), "summary": result["summary"]}, sys.stdout, indent=2
    )
    sys.stdout.write("\n")


def command_fleetflow(args) -> None:
    from .fleetflow import run_suite

    result = run_suite(
        seeds=args.seeds,
        horizon=args.horizon,
        scorer_name=args.scorer,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        clm_cache_path=Path(args.clm_cache) if args.clm_cache else None,
        clm_encoder=args.clm_encoder,
        clm_checkpoint=Path(args.clm_checkpoint) if args.clm_checkpoint else None,
        clm_head_repo=args.clm_head_repo,
        clm_head_filename=args.clm_head_filename,
        clm_include_budget_context=not args.clm_budget_blind,
        beam_widths=args.beam_widths,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    json.dump(
        {"output": str(output), "aggregate": result["aggregate"]}, sys.stdout, indent=2
    )
    sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decisionflow-experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser(
        "verify", help="validate registered experiment programs"
    )
    verify.add_argument(
        "--backend", choices=("auto", "enumeration", "sdd"), default="auto"
    )
    verify.set_defaults(handler=command_verify)
    run = commands.add_parser(
        "run", help="condition a standardized cached-score JSONL file"
    )
    run.add_argument("--experiment", required=True, choices=tuple(EXPERIMENTS))
    run.add_argument("--input", required=True)
    run.add_argument("--output", default="-")
    run.add_argument(
        "--backend", choices=("auto", "enumeration", "sdd"), default="auto"
    )
    run.set_defaults(handler=command_run)
    summary = commands.add_parser(
        "summarize", help="aggregate a generated result JSONL file"
    )
    summary.add_argument("--input", required=True)
    summary.add_argument("--output", default="-")
    summary.set_defaults(handler=command_summarize)
    trajectory = commands.add_parser(
        "run-trajectories", help="evaluate grounded finite-horizon trajectory records"
    )
    trajectory.add_argument("--input", required=True)
    trajectory.add_argument("--output", default="-")
    trajectory.set_defaults(handler=command_trajectory)
    full = commands.add_parser(
        "full", help="reproduce complete pilot metrics through DecisionFlow"
    )
    full.add_argument("--experiment", choices=("all", *FULL_PILOTS), default="all")
    full.add_argument("--repo-root", default=".")
    full.add_argument("--output-dir", default="experiments/results/generated/full")
    full.set_defaults(handler=command_full)
    gliner = commands.add_parser(
        "gliner25-fast-decisions",
        help="audit candidate rules and compare independent, constrained MAP, and DecisionFlow",
    )
    from .gliner25_fast_decisions import MODEL_ID, RULE_DATASETS

    gliner.add_argument("--dataset", choices=("all", *RULE_DATASETS), default="all")
    gliner.add_argument("--model", default=MODEL_ID)
    gliner.add_argument("--batch-size", type=int, default=8)
    gliner.add_argument("--backend", choices=("enumeration", "sdd"), default="sdd")
    gliner.add_argument(
        "--output",
        default="experiments/results/generated/gliner25-fast-decisions.json",
    )
    gliner.set_defaults(handler=command_gliner25)
    logistics = commands.add_parser(
        "logistics",
        help="run the AutoPlanBench Logistics OpenJev/DecisionFlow pilot",
    )
    logistics.add_argument("--dataset-root", required=True)
    logistics.add_argument("--instance", default="0")
    logistics.add_argument("--horizon", type=int, default=5)
    logistics.add_argument("--model", default="Qwen/Qwen3-0.6B")
    logistics.add_argument("--scorer", choices=("openjev", "laya"), default="openjev")
    logistics.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    logistics.add_argument("--dtype", default="float32")
    logistics.add_argument("--batch-size", type=int, default=4)
    logistics.add_argument("--cache")
    logistics.add_argument(
        "--output",
        default="experiments/results/generated/logistics-instance-0.json",
    )
    logistics.set_defaults(handler=command_logistics)
    depot = commands.add_parser(
        "depot",
        help="run the AutoPlanBench Depot multi-step DecisionFlow experiment",
    )
    depot.add_argument("--dataset-root", required=True)
    depot.add_argument("--instance", default="2")
    depot.add_argument("--horizon", type=int, default=12)
    depot.add_argument("--model", default="convaiinnovations/laya-typed-decisions")
    depot.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    depot.add_argument("--batch-size", type=int, default=8)
    depot.add_argument("--scorer", choices=("laya", "uniform"), default="laya")
    depot.add_argument("--beam-widths", type=int, nargs="+", default=(4, 16, 64))
    depot.add_argument("--cache")
    depot.add_argument(
        "--output",
        default="experiments/results/generated/depot-instance-2.json",
    )
    depot.set_defaults(handler=command_depot)
    trex = commands.add_parser(
        "trex",
        help="run the neutral-prompt CLM-8B/DecisionFlow T-Rex pilot",
    )
    trex.add_argument("--clm-root", default="third_party/clm")
    trex.add_argument("--seeds", type=int, nargs="+", default=(0,))
    trex.add_argument("--scenarios-per-seed", type=int, default=3)
    trex.add_argument("--horizon", type=int, default=4)
    trex.add_argument("--decision-frames", type=int, default=6)
    trex.add_argument("--capture-distance", type=int, default=120)
    trex.add_argument("--scorer", choices=("clm", "uniform"), default="clm")
    trex.add_argument("--encoder", default="mlx-community/Qwen3-8B-8bit")
    trex.add_argument("--checkpoint")
    trex.add_argument("--head-repo", default="Contrastive-LM/CLM-v0.1-8B")
    trex.add_argument("--head-filename", default="CLM_v0.1-8B.pt")
    trex.add_argument("--beam-widths", type=int, nargs="+", default=(1, 4, 16))
    trex.add_argument("--cache")
    trex.add_argument(
        "--output",
        default="experiments/results/generated/trex-clm8b.json",
    )
    trex.set_defaults(handler=command_trex)
    trex_full = commands.add_parser(
        "trex-full",
        help="run full seeded T-Rex courses with receding-horizon DecisionFlow",
    )
    trex_full.add_argument("--clm-root", default="third_party/clm")
    trex_full.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    trex_full.add_argument("--duration", type=float, default=60.0)
    trex_full.add_argument("--horizon", type=int, default=4)
    trex_full.add_argument("--decision-frames", type=int, default=6)
    trex_full.add_argument("--capture-distance", type=int, default=140)
    trex_full.add_argument("--encoder", default="mlx-community/Qwen3-8B-8bit")
    trex_full.add_argument("--checkpoint")
    trex_full.add_argument("--head-repo", default="Contrastive-LM/CLM-v0.1-8B")
    trex_full.add_argument("--head-filename", default="CLM_v0.1-8B.pt")
    trex_full.add_argument("--cache")
    trex_full.add_argument(
        "--output",
        default="experiments/results/generated/trex-decisionflow-full.json",
    )
    trex_full.set_defaults(handler=command_trex_full)
    fleet = commands.add_parser(
        "fleetflow",
        help="run the parameterized JevAny-inspired fleet-dispatch experiment",
    )
    fleet.add_argument("--seeds", type=int, nargs="+", default=tuple(range(10)))
    fleet.add_argument("--horizon", type=int, default=7)
    fleet.add_argument(
        "--scorer", choices=("myopic", "jevany", "clm"), default="myopic"
    )
    fleet.add_argument("--base-url", default="http://127.0.0.1:8008")
    fleet.add_argument("--model", default="jevany-latest")
    fleet.add_argument("--api-key", default="local")
    fleet.add_argument("--cache-dir")
    fleet.add_argument("--clm-cache")
    fleet.add_argument("--clm-encoder", default="mlx-community/Qwen3-8B-8bit")
    fleet.add_argument("--clm-checkpoint")
    fleet.add_argument("--clm-head-repo", default="Contrastive-LM/CLM-v0.1-8B")
    fleet.add_argument("--clm-head-filename", default="CLM_v0.1-8B.pt")
    fleet.add_argument("--clm-budget-blind", action="store_true")
    fleet.add_argument("--beam-widths", type=int, nargs="+", default=(1, 2, 4, 16))
    fleet.add_argument(
        "--output",
        default="experiments/results/generated/fleetflow.json",
    )
    fleet.set_defaults(handler=command_fleetflow)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
