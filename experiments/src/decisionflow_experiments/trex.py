"""CLM-8B + DecisionFlow on the deterministic T-Rex benchmark.

The benchmark adapter, prompts, physics graph, baselines, and caches live in
the experiments package.  DecisionFlow-Core is used unchanged through its
public finite-horizon trajectory API.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from decisionflow.trajectory import TrajectoryEngine, TrajectorySpec, Transition


ACTIONS = ("jump", "duck", "run")
ACTION_DESCRIPTIONS = {
    "jump": "Tap jump, then continue running after the tap.",
    "duck": "Hold the down control so the dinosaur ducks or descends faster.",
    "run": "Release the down control and continue running.",
}


def load_trex(clm_root: Path) -> tuple[Any, Any]:
    """Load the benchmark from an external CLM checkout."""
    source = (clm_root / "examples" / "t_rex").resolve()
    if not source.exists():
        raise FileNotFoundError(f"T-Rex benchmark not found under {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return importlib.import_module("trex.engine"), importlib.import_module("trex.planner")


@dataclass(frozen=True)
class Scenario:
    seed: int
    frame: int
    obstacle_index: int
    snapshot: Any
    oracle_plan: Any

    @property
    def key(self) -> str:
        return f"seed-{self.seed}-obstacle-{self.obstacle_index}-frame-{self.frame}"


@dataclass(frozen=True)
class SearchNode:
    trex: tuple
    held: str
    frame: int
    depth: int


@dataclass
class ScenarioGraph:
    initial_node: str
    nodes: dict[str, SearchNode]
    transitions: dict[str, dict[str, str]]
    terminals: set[str]
    horizon: int
    world: Any
    snapshot: Any
    obstacle_step: Any


def apply_action(search: Any, node: SearchNode, action: str) -> tuple[Any, str]:
    state = search.press(node.trex, action, node.frame)
    return state, "run" if action == "jump" else action


def _node_key(node: SearchNode, prefix: str) -> str:
    values = ",".join(
        f"{value:.5f}" if isinstance(value, float) else str(value) for value in node.trex
    )
    return f"{prefix}|{node.depth}:{node.frame}:{node.held}:{values}"


def build_scenario_graph(
    scenario: Scenario,
    planner_module: Any,
    *,
    horizon: int,
    decision_frames: int,
) -> ScenarioGraph:
    """Expand every collision-free action sequence for one benchmark snapshot."""
    world = planner_module.World(scenario.snapshot, horizon * decision_frames)
    search = planner_module.Search(world, (decision_frames, decision_frames), world.horizon)
    initial = SearchNode(scenario.snapshot.trex, scenario.snapshot.held, 0, 0)
    initial_key = _node_key(initial, scenario.key)
    nodes = {initial_key: initial}
    layer = {initial_key}
    transitions: dict[str, dict[str, str]] = {}
    terminals: set[str] = set()
    for depth in range(horizon):
        next_layer: set[str] = set()
        for key in sorted(layer):
            node = nodes[key]
            row: dict[str, str] = {}
            for action in ACTIONS:
                pressed, held = apply_action(search, node, action)
                successor = search.run_frames(
                    pressed, held, node.frame, decision_frames
                )
                if successor is None:
                    continue
                child = SearchNode(
                    successor,
                    held,
                    node.frame + decision_frames,
                    depth + 1,
                )
                child_key = _node_key(child, scenario.key)
                nodes.setdefault(child_key, child)
                row[action] = child_key
                next_layer.add(child_key)
            transitions[key] = row
        layer = next_layer
    terminals.update(layer)
    return ScenarioGraph(
        initial_key,
        nodes,
        transitions,
        terminals,
        horizon,
        world,
        scenario.snapshot,
        planner_module.obstacle_step,
    )


def collect_scenarios(
    clm_root: Path,
    *,
    seeds: Sequence[int],
    scenarios_per_seed: int,
    decision_frames: int,
    capture_distance: int,
    max_frames: int = 6_000,
) -> tuple[list[Scenario], Any]:
    """Use the official physics planner only to produce reproducible snapshots."""
    engine, planner_module = load_trex(clm_root)
    scenarios: list[Scenario] = []
    for seed in seeds:
        game = engine.Game(seed)
        planner = planner_module.Planner()
        held = "run"
        game.press_jump()
        seen: set[int] = set()
        for frame in range(max_frames):
            if frame % decision_frames == 0 and game.playing and not game.crashed:
                snap = planner_module.snapshot(game, held)
                plan = planner.plan(
                    snap,
                    first=(0, 0),
                    gap=(decision_frames, decision_frames),
                )
                if snap.obstacles and plan.distance is not None:
                    obstacle_id = snap.obstacles[0].id
                    if obstacle_id not in seen and plan.distance <= capture_distance:
                        seen.add(obstacle_id)
                        scenarios.append(
                            Scenario(seed, frame, game.obstacle_index, snap, plan)
                        )
                        if sum(item.seed == seed for item in scenarios) >= scenarios_per_seed:
                            break
                if plan.best == "jump":
                    if not game.trex.jumping:
                        if game.trex.ducking:
                            game.release_duck()
                        game.press_jump()
                    held = "run"
                else:
                    held = plan.best
            if held == "duck":
                game.press_duck()
            else:
                game.release_duck()
            game.step()
        if sum(item.seed == seed for item in scenarios) < scenarios_per_seed:
            raise RuntimeError(f"seed {seed} yielded too few T-Rex scenarios")
    return scenarios, planner_module


def state_text(graph: ScenarioGraph, node: SearchNode) -> str:
    y, velocity, jumping, ducking, speed_drop, _, status = node.trex
    obstacle_lines = []
    for obstacle in graph.snapshot.obstacles[:3]:
        travelled = sum(
            graph.obstacle_step(graph.world.speed[k], obstacle.offset)
            for k in range(1, node.frame + 1)
        )
        distance = obstacle.x - travelled - graph.snapshot.trex_x
        if distance + obstacle.width > 0:
            obstacle_lines.append(
                f"{obstacle.label}: {distance:.0f} px ahead, vertical position {obstacle.y:.0f}"
            )
    posture = "jumping" if jumping else "ducking" if ducking else "running"
    return (
        "Chrome dinosaur runner. The world advances at 60 frames per second.\n"
        f"Dinosaur posture: {posture}; height y={y:.1f}; vertical velocity={velocity:.1f}; "
        f"held control={node.held}; speed-drop={bool(speed_drop)}.\n"
        f"Runner speed: {graph.snapshot.speed:.3f}.\n"
        "Visible obstacles:\n- "
        + ("\n- ".join(obstacle_lines) if obstacle_lines else "none")
        + "\nChoose the control for the next short interval."
    )


class UniformScorer:
    def score(self, graphs: Iterable[ScenarioGraph]) -> dict[str, dict[str, float]]:
        return {
            key: {action: 1.0 / len(ACTIONS) for action in ACTIONS}
            for graph in graphs
            for key, node in graph.nodes.items()
            if node.depth < graph.horizon
        }

    def metadata(self) -> dict[str, Any]:
        return {"scorer": "uniform"}


class CLMScorer:
    """Batched CLM-8B scorer with a persistent experiment-only probability cache."""

    def __init__(
        self,
        *,
        encoder: str,
        cache_path: Path | None,
        checkpoint: Path | None,
        head_repo: str,
        head_filename: str,
    ) -> None:
        self.encoder = encoder
        self.cache_path = cache_path
        self.checkpoint = checkpoint
        self.head_repo = head_repo
        self.head_filename = head_filename
        self.cache: dict[str, dict[str, float]] = {}
        self.engine = None
        self.score_ms = 0.0
        self.scored_states = 0
        if cache_path and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("encoder") == encoder and payload.get("prompt_version") == 2:
                self.cache = payload.get("scores", {})

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {"encoder": self.encoder, "prompt_version": 2, "scores": self.cache},
                indent=2,
            ),
            encoding="utf-8",
        )

    def _ensure_engine(self) -> None:
        if self.engine is not None:
            return
        from clm_tune_mlx import load_engine
        from huggingface_hub import hf_hub_download

        checkpoint = self.checkpoint or Path(
            hf_hub_download(repo_id=self.head_repo, filename=self.head_filename)
        )
        self.engine = load_engine(encoder=self.encoder, checkpoint=str(checkpoint))

    def _score_requests(
        self, requests: Iterable[tuple[str, Any, dict[str, Any]]]
    ) -> dict[str, dict[str, float]]:
        from clm_tune_mlx.schema import build_pairs

        rows: list[tuple[str, str, tuple[str, list[str], list[str]]]] = []
        missing: dict[str, tuple[str, list[str], list[str]]] = {}
        for output_key, state, question in requests:
            pair = build_pairs(state, question)["action"]
            fingerprint = hashlib.sha256(
                json.dumps(pair, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            rows.append((output_key, fingerprint, pair))
            if fingerprint not in self.cache:
                missing.setdefault(fingerprint, pair)

        started = time.perf_counter()
        if missing:
            self._ensure_engine()
            import numpy as np

            fingerprints = list(missing)
            pairs = [missing[key] for key in fingerprints]
            state_embeddings = np.asarray(
                self.engine._project([pair[0] for pair in pairs], "state"),
                dtype=np.float64,
            )
            flat_options = [text for pair in pairs for text in pair[2]]
            action_embeddings = np.asarray(
                self.engine._project(flat_options, "action"), dtype=np.float64
            )
            offset = 0
            for index, (fingerprint, pair) in enumerate(zip(fingerprints, pairs)):
                keys, texts = pair[1], pair[2]
                count = len(texts)
                logits = self.engine.heads.scale * (
                    action_embeddings[offset : offset + count] @ state_embeddings[index]
                )
                offset += count
                logits -= logits.max()
                probabilities = np.exp(logits)
                probabilities /= probabilities.sum()
                self.cache[fingerprint] = {
                    key: float(value) for key, value in zip(keys, probabilities)
                }
            self.scored_states += len(missing)
            self._save()
        self.score_ms += 1000.0 * (time.perf_counter() - started)
        return {output_key: self.cache[fingerprint] for output_key, fingerprint, _ in rows}

    def score(self, graphs: Iterable[ScenarioGraph]) -> dict[str, dict[str, float]]:
        question = {
            "action": {
                "type": "choice",
                "instructions": "Which control should the dinosaur use next?",
                "criteria": ACTION_DESCRIPTIONS,
            }
        }
        requests = []
        for graph in graphs:
            for key, node in graph.nodes.items():
                if node.depth >= graph.horizon:
                    continue
                requests.append((key, state_text(graph, node), question))
        return self._score_requests(requests)

    def score_official_roots(
        self, scenarios: Iterable[Scenario]
    ) -> dict[str, dict[str, float]]:
        from trex.backends import build_question

        requests = []
        for scenario in scenarios:
            state, question = build_question(scenario.oracle_plan, "labeled")
            requests.append((scenario.key, state, question))
        return self._score_requests(requests)

    def metadata(self) -> dict[str, Any]:
        return {
            "scorer": "clm-8b-mlx",
            "encoder": self.encoder,
            "head": str(self.checkpoint) if self.checkpoint else f"{self.head_repo}/{self.head_filename}",
            "cached_states": len(self.cache),
            "new_states_this_run": self.scored_states,
            "score_ms_this_run": self.score_ms,
        }


def greedy_rollout(graph: ScenarioGraph, policy: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    node = graph.initial_node
    actions = []
    probability = 1.0
    for _ in range(graph.horizon):
        row = policy[node]
        action = max(ACTIONS, key=lambda item: (row[item], item))
        actions.append(action)
        probability *= row[action]
        target = graph.transitions.get(node, {}).get(action)
        if target is None:
            return {"success": False, "actions": actions, "path_probability": probability}
        node = target
    return {
        "success": node in graph.terminals,
        "actions": actions,
        "path_probability": probability,
    }


def beam_search(
    graph: ScenarioGraph,
    policy: Mapping[str, Mapping[str, float]],
    width: int,
) -> dict[str, Any]:
    beam = [(0.0, graph.initial_node, ())]
    expanded = 0
    for _ in range(graph.horizon):
        candidates = []
        for logp, node, path in beam:
            expanded += 1
            for action, target in graph.transitions.get(node, {}).items():
                candidates.append(
                    (logp + math.log(max(policy[node][action], 1e-300)), target, path + (action,))
                )
        candidates.sort(key=lambda item: (-item[0], item[2]))
        beam = candidates[:width]
        if not beam:
            break
    completed = [item for item in beam if item[1] in graph.terminals]
    if not completed:
        return {"success": False, "actions": [], "expanded_states": expanded, "width": width}
    best = max(completed, key=lambda item: (item[0], item[2]))
    return {
        "success": True,
        "actions": list(best[2]),
        "path_probability": math.exp(best[0]),
        "expanded_states": expanded,
        "width": width,
    }


def infer_graph(graph: ScenarioGraph, policy: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    result = TrajectoryEngine().infer(
        TrajectorySpec(
            initial_state=graph.initial_node,
            actions=ACTIONS,
            horizon=graph.horizon,
            terminal_states=tuple(graph.terminals),
            transitions={
                node: {action: (Transition(target),) for action, target in row.items()}
                for node, row in graph.transitions.items()
            },
            policy=policy,
            metadata={"benchmark": "CLM T-Rex"},
        )
    )
    report = result.to_dict()
    report["actions"] = [action for action, _ in result.trajectory_map]
    report["marginal_action"] = max(
        ACTIONS, key=lambda action: (result.first_action_marginals[action], action)
    )
    report["success"] = True
    return report


def run_pilot(
    *,
    clm_root: Path,
    seeds: Sequence[int],
    scenarios_per_seed: int,
    horizon: int,
    decision_frames: int,
    capture_distance: int,
    scorer_name: str,
    encoder: str,
    checkpoint: Path | None,
    head_repo: str,
    head_filename: str,
    cache_path: Path | None,
    beam_widths: Sequence[int],
) -> dict[str, Any]:
    scenarios, planner_module = collect_scenarios(
        clm_root,
        seeds=seeds,
        scenarios_per_seed=scenarios_per_seed,
        decision_frames=decision_frames,
        capture_distance=capture_distance,
    )
    graphs = [
        build_scenario_graph(
            scenario,
            planner_module,
            horizon=horizon,
            decision_frames=decision_frames,
        )
        for scenario in scenarios
    ]
    scorer: Any = (
        UniformScorer()
        if scorer_name == "uniform"
        else CLMScorer(
            encoder=encoder,
            cache_path=cache_path,
            checkpoint=checkpoint,
            head_repo=head_repo,
            head_filename=head_filename,
        )
    )
    all_policy = scorer.score(graphs)
    official_root_policy = (
        scorer.score_official_roots(scenarios) if scorer_name == "clm" else {}
    )
    reports = []
    for scenario, graph in zip(scenarios, graphs):
        policy = {key: all_policy[key] for key in graph.nodes if key in all_policy}
        local = greedy_rollout(graph, policy)
        exact = infer_graph(graph, policy)
        raw_action = local["actions"][0]
        official = None
        if official_root_policy:
            official_probabilities = official_root_policy[scenario.key]
            proposed = max(
                ACTIONS,
                key=lambda action: (
                    official_probabilities[action],
                    -ACTIONS.index(action),
                ),
            )
            allowed = scenario.oracle_plan.safe_actions
            executed = (
                proposed
                if proposed in allowed or not allowed
                else max(allowed, key=official_probabilities.__getitem__)
            )
            equivalent = proposed == scenario.oracle_plan.best or (
                scenario.oracle_plan.airborne
                and {proposed, scenario.oracle_plan.best} <= {"jump", "run"}
            )
            official = {
                "probabilities": official_probabilities,
                "proposed_action": proposed,
                "executed_with_shield": executed,
                "shield_intervened": executed != proposed,
                "agreed_with_planner": equivalent,
            }
        reports.append(
            {
                "scenario": scenario.key,
                "seed": scenario.seed,
                "frame": scenario.frame,
                "obstacle": scenario.oracle_plan.ahead,
                "distance_px": scenario.oracle_plan.distance,
                "oracle_safe_actions": scenario.oracle_plan.safe_actions,
                "oracle_best_action": scenario.oracle_plan.best,
                "reachable_states": len(graph.nodes),
                "safe_edges": sum(len(row) for row in graph.transitions.values()),
                "safe_terminal_states": len(graph.terminals),
                "local_probabilities": policy[graph.initial_node],
                "official_labeled_clm": official,
                "local_greedy": local,
                "beam_search": {
                    str(width): beam_search(graph, policy, width) for width in beam_widths
                },
                "decisionflow": exact,
                "raw_first_action_oracle_safe": raw_action in scenario.oracle_plan.safe_actions,
                "conditioned_first_action_oracle_safe": exact["marginal_action"]
                in scenario.oracle_plan.safe_actions,
            }
        )
    count = len(reports)
    beam_rates = {
        str(width): sum(row["beam_search"][str(width)]["success"] for row in reports) / count
        for width in beam_widths
    }
    official_rows = [row["official_labeled_clm"] for row in reports if row["official_labeled_clm"]]
    return {
        "benchmark": "Contrastive-LM T-Rex",
        "mode": "offline deterministic finite-horizon safety",
        "seeds": list(seeds),
        "scenario_count": count,
        "horizon_decisions": horizon,
        "decision_frames": decision_frames,
        "horizon_frames": horizon * decision_frames,
        "action_prompt": {
            "instructions": "Which control should the dinosaur use next?",
            "criteria": ACTION_DESCRIPTIONS,
            "contains_planner_labels": False,
        },
        "aggregate": {
            "local_greedy_survival_rate": sum(row["local_greedy"]["success"] for row in reports) / count,
            "decisionflow_survival_rate": sum(row["decisionflow"]["success"] for row in reports) / count,
            "constrained_beam_survival_rate": beam_rates,
            "raw_first_action_oracle_safe_rate": sum(row["raw_first_action_oracle_safe"] for row in reports) / count,
            "conditioned_first_action_oracle_safe_rate": sum(row["conditioned_first_action_oracle_safe"] for row in reports) / count,
            "mean_valid_mass": sum(row["decisionflow"]["valid_mass"] for row in reports) / count,
            "mean_inference_ms": sum(row["decisionflow"]["inference_ms"] for row in reports) / count,
            "marginal_map_first_action_disagreements": sum(
                row["decisionflow"]["marginal_action"]
                != row["decisionflow"]["actions"][0]
                for row in reports
            ),
            "conflicts_removed": sum(
                (not row["local_greedy"]["success"]) and row["decisionflow"]["success"]
                for row in reports
            ),
            "official_labeled_planner_agreement": (
                sum(row["agreed_with_planner"] for row in official_rows) / len(official_rows)
                if official_rows
                else None
            ),
            "official_labeled_shield_interventions": sum(
                row["shield_intervened"] for row in official_rows
            ),
        },
        "scoring": scorer.metadata(),
        "scenarios": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clm-root", type=Path, default=Path("third_party/clm"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--scenarios-per-seed", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--decision-frames", type=int, default=6)
    parser.add_argument("--capture-distance", type=int, default=120)
    parser.add_argument("--scorer", choices=("clm", "uniform"), default="clm")
    parser.add_argument("--encoder", default="mlx-community/Qwen3-8B-8bit")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--head-repo", default="Contrastive-LM/CLM-v0.1-8B")
    parser.add_argument("--head-filename", default="CLM_v0.1-8B.pt")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=(1, 4, 16))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run_pilot(
        clm_root=args.clm_root,
        seeds=args.seeds,
        scenarios_per_seed=args.scenarios_per_seed,
        horizon=args.horizon,
        decision_frames=args.decision_frames,
        capture_distance=args.capture_distance,
        scorer_name=args.scorer,
        encoder=args.encoder,
        checkpoint=args.checkpoint,
        head_repo=args.head_repo,
        head_filename=args.head_filename,
        cache_path=args.cache,
        beam_widths=args.beam_widths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
