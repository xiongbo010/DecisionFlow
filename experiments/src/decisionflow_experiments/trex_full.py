"""Full seeded T-Rex courses controlled by receding-horizon DecisionFlow.

This is an experiment adapter. It imports the official deterministic game and
uses DecisionFlow-Core unchanged for each finite-horizon inference call.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from decisionflow.errors import UnsatisfiableError

from .trex import (
    CLMScorer,
    Scenario,
    build_scenario_graph,
    infer_graph,
    load_trex,
)


def apply_control(game: Any, held: str, action: str) -> str:
    if action == "jump":
        if not game.trex.jumping:
            if game.trex.ducking:
                game.release_duck()
            game.press_jump()
        return "run"
    return action


def enforce_control(game: Any, held: str) -> None:
    if not game.playing or game.crashed:
        return
    if held == "duck":
        game.press_duck()
    else:
        game.release_duck()


def should_plan(snap: Any, held: str, capture_distance: int) -> bool:
    if not snap.obstacles:
        return False
    first = snap.obstacles[0]
    distance = first.x - (snap.trex_x + 44)
    return distance <= capture_distance or bool(snap.trex[2]) or held == "duck"


def run_course(
    *,
    seed: int,
    duration_seconds: float,
    horizon: int,
    decision_frames: int,
    capture_distance: int,
    scorer: CLMScorer,
    engine_module: Any,
    planner_module: Any,
) -> dict[str, Any]:
    game = engine_module.Game(seed)
    planner = planner_module.Planner()
    held = "run"
    game.press_jump()
    duration_frames = int(duration_seconds * engine_module.FPS)
    best_score = 0
    decisions = 0
    trivial_run_decisions = 0
    infeasible = 0
    valid_masses: list[float] = []
    inference_ms: list[float] = []
    scored_graph_states = 0
    action_counts = dict.fromkeys(("jump", "duck", "run"), 0)
    restart_wait = 0

    for frame in range(duration_frames):
        best_score = max(best_score, game.score)
        if game.crashed:
            restart_wait += 1
            game.step()
            if restart_wait >= int(1.5 * engine_module.FPS):
                game.restart()
                held = "run"
                restart_wait = 0
            continue

        if frame % decision_frames == 0 and game.playing:
            snap = planner_module.snapshot(game, held)
            if should_plan(snap, held, capture_distance):
                plan = planner.plan(
                    snap,
                    first=(0, 0),
                    gap=(decision_frames, decision_frames),
                )
                scenario = Scenario(seed, frame, game.obstacle_index, snap, plan)
                graph = build_scenario_graph(
                    scenario,
                    planner_module,
                    horizon=horizon,
                    decision_frames=decision_frames,
                )
                scored_graph_states += sum(
                    node.depth < graph.horizon for node in graph.nodes.values()
                )
                try:
                    policy = scorer.score((graph,))
                    result = infer_graph(graph, policy)
                    action = result["marginal_action"]
                    valid_masses.append(result["valid_mass"])
                    inference_ms.append(result["inference_ms"])
                except UnsatisfiableError:
                    infeasible += 1
                    action = "run"
                decisions += 1
            else:
                action = "run"
                trivial_run_decisions += 1
            held = apply_control(game, held, action)
            action_counts[action] += 1

        enforce_control(game, held)
        game.step()

    best_score = max(best_score, game.score)
    return {
        "seed": seed,
        "survived": game.deaths == 0,
        "deaths": game.deaths,
        "best_score": best_score,
        "structured_decisions": decisions,
        "trivial_run_decisions": trivial_run_decisions,
        "infeasible_decisions": infeasible,
        "scored_graph_states": scored_graph_states,
        "action_counts": action_counts,
        "mean_valid_mass": statistics.fmean(valid_masses) if valid_masses else None,
        "mean_inference_ms": statistics.fmean(inference_ms) if inference_ms else None,
    }


def run_full_courses(
    *,
    clm_root: Path,
    seeds: Sequence[int],
    duration_seconds: float,
    horizon: int,
    decision_frames: int,
    capture_distance: int,
    encoder: str,
    checkpoint: Path | None,
    head_repo: str,
    head_filename: str,
    cache_path: Path | None,
) -> dict[str, Any]:
    engine_module, planner_module = load_trex(clm_root)
    scorer = CLMScorer(
        encoder=encoder,
        cache_path=cache_path,
        checkpoint=checkpoint,
        head_repo=head_repo,
        head_filename=head_filename,
    )
    results = []
    for seed in seeds:
        row = run_course(
            seed=seed,
            duration_seconds=duration_seconds,
            horizon=horizon,
            decision_frames=decision_frames,
            capture_distance=capture_distance,
            scorer=scorer,
            engine_module=engine_module,
            planner_module=planner_module,
        )
        results.append(row)
        print(
            f"[decisionflow seed={seed}] survived={row['survived']} "
            f"deaths={row['deaths']} best_score={row['best_score']} "
            f"structured_decisions={row['structured_decisions']} "
            f"infeasible={row['infeasible_decisions']}",
            flush=True,
        )
    masses = [row["mean_valid_mass"] for row in results if row["mean_valid_mass"] is not None]
    return {
        "benchmark": "Contrastive-LM T-Rex",
        "controller": "DecisionFlow receding horizon",
        "course_style": "original",
        "duration_s": duration_seconds,
        "seeds": list(seeds),
        "horizon_decisions": horizon,
        "decision_frames": decision_frames,
        "capture_distance": capture_distance,
        "summary": {
            "survived": sum(row["survived"] for row in results),
            "deaths": sum(row["deaths"] for row in results),
            "mean_best_score": statistics.fmean(row["best_score"] for row in results),
            "structured_decisions": sum(row["structured_decisions"] for row in results),
            "infeasible_decisions": sum(row["infeasible_decisions"] for row in results),
            "mean_valid_mass": statistics.fmean(masses) if masses else None,
        },
        "scoring": scorer.metadata(),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clm-root", type=Path, default=Path("third_party/clm"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--decision-frames", type=int, default=6)
    parser.add_argument("--capture-distance", type=int, default=140)
    parser.add_argument("--encoder", default="mlx-community/Qwen3-8B-8bit")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--head-repo", default="Contrastive-LM/CLM-v0.1-8B")
    parser.add_argument("--head-filename", default="CLM_v0.1-8B.pt")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_full_courses(
        clm_root=args.clm_root,
        seeds=args.seeds,
        duration_seconds=args.duration,
        horizon=args.horizon,
        decision_frames=args.decision_frames,
        capture_distance=args.capture_distance,
        encoder=args.encoder,
        checkpoint=args.checkpoint,
        head_repo=args.head_repo,
        head_filename=args.head_filename,
        cache_path=args.cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
