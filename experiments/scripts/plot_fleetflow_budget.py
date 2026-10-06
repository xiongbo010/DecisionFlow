"""Render the FleetFlow CLM-8B budget sweep as a dependency-free SVG."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def polyline(points: list[tuple[float, float]], color: str, width: float = 3.5) -> str:
    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def render(inputs: list[Path], output: Path) -> None:
    rows = []
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        aggregate = payload["aggregate"]
        rows.append(
            {
                "budget": payload["instances"][0]["horizon"],
                "local": aggregate["local_argmax_success_rate"],
                "shield": aggregate["one_step_shield_success_rate"],
                "beam": aggregate["beam_success_rate"]["256"],
                "decisionflow": aggregate["decisionflow_success_rate"],
                "best_first": aggregate["best_first_success_rate"],
                "dp": aggregate["dynamic_programming_success_rate"],
                "z": aggregate["mean_valid_mass"],
                "z_min": min(
                    x["decisionflow"]["valid_mass"] for x in payload["instances"]
                ),
                "z_max": max(
                    x["decisionflow"]["valid_mass"] for x in payload["instances"]
                ),
                "ms": aggregate["mean_decisionflow_inference_ms"],
                "best_first_ms": aggregate["mean_best_first_inference_ms"],
                "dp_ms": aggregate["mean_dynamic_programming_inference_ms"],
                "states": aggregate["mean_goal_path_states"],
            }
        )
    rows.sort(key=lambda row: row["budget"])

    width, height = 1530, 570
    panels = [(70, 155, 410, 315), (560, 155, 410, 315), (1050, 155, 410, 315)]
    budgets = [row["budget"] for row in rows]
    xmin, xmax = min(budgets), max(budgets)

    def sx(value: float, panel: tuple[int, int, int, int]) -> float:
        x, _, w, _ = panel
        return x + (value - xmin) / (xmax - xmin) * w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfa"/>',
        "<style>text{font-family:Inter,Arial,sans-serif;fill:#202326}.title{font-size:28px;font-weight:700}.sub{font-size:15px;fill:#62676d}.pt{font-size:17px;font-weight:650}.axis{font-size:13px;fill:#5c6268}.note{font-size:12px;fill:#62676d}.grid{stroke:#dfe2e3;stroke-width:1}.frame{stroke:#aeb4b7;stroke-width:1.2}</style>",
        '<text x="70" y="52" class="title">FleetFlow budget sensitivity under a fair inference protocol</text>',
        '<text x="70" y="82" class="sub">One shared CLM-8B probability table and transition graph; three parameterized instances per budget</text>',
    ]

    titles = ["Task success", "Consistent probability mass", "Exact inference cost"]
    subtitles = [
        "Fraction of instances completed",
        "Mean Z with min–max range (log scale)",
        "Shared scores; inference only (log scale)",
    ]
    for panel, title, subtitle in zip(panels, titles, subtitles):
        x, y, w, h = panel
        parts += [
            f'<text x="{x}" y="118" class="pt">{esc(title)}</text>',
            f'<text x="{x}" y="139" class="note">{esc(subtitle)}</text>',
            f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" class="frame"/>',
            f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" class="frame"/>',
        ]
        for budget in budgets:
            px = sx(budget, panel)
            parts += [
                f'<line x1="{px}" y1="{y}" x2="{px}" y2="{y + h}" class="grid"/>',
                f'<text x="{px}" y="{y + h + 23}" text-anchor="middle" class="axis">{budget}</text>',
            ]
        parts.append(
            f'<text x="{x + w / 2}" y="{y + h + 50}" text-anchor="middle" class="axis">Operation budget</text>'
        )

    # Panel 1: success.
    x, y, w, h = panels[0]
    for value in (0, 0.5, 1):
        py = y + h - value * h
        parts += [
            f'<line x1="{x}" y1="{py}" x2="{x + w}" y2="{py}" class="grid"/>',
            f'<text x="{x - 12}" y="{py + 5}" text-anchor="end" class="axis">{int(value * 100)}%</text>',
        ]
    exact_points = [
        (sx(row["budget"], panels[0]), y + h - row["decisionflow"] * h) for row in rows
    ]
    base_points = [(sx(row["budget"], panels[0]), y + h) for row in rows]
    parts += [polyline(exact_points, "#1f7a68"), polyline(base_points, "#c45b4d")]
    for px, py in exact_points:
        parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#1f7a68"/>')
    for px, py in base_points:
        parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#c45b4d"/>')
    parts += [
        f'<rect x="{x + 16}" y="{y + 16}" width="13" height="13" rx="6" fill="#1f7a68"/><text x="{x + 38}" y="{y + 28}" class="axis">Exact: A*, DP, DecisionFlow</text>',
        f'<rect x="{x + 210}" y="{y + 16}" width="13" height="13" rx="6" fill="#c45b4d"/><text x="{x + 232}" y="{y + 28}" class="axis">Local, shield, beam ≤256</text>',
    ]

    # Panel 2: valid mass.
    x, y, w, h = panels[1]
    log_min, log_max = -13.0, -8.0

    def zy(value: float) -> float:
        return y + h - (math.log10(value) - log_min) / (log_max - log_min) * h

    for exponent in range(-13, -7):
        py = zy(10**exponent)
        parts += [
            f'<line x1="{x}" y1="{py}" x2="{x + w}" y2="{py}" class="grid"/>',
            f'<text x="{x - 12}" y="{py + 5}" text-anchor="end" class="axis">10<tspan baseline-shift="super" font-size="9">{exponent}</tspan></text>',
        ]
    z_points = []
    for row in rows:
        px, py = sx(row["budget"], panels[1]), zy(row["z"])
        z_points.append((px, py))
        parts += [
            f'<line x1="{px}" y1="{zy(row["z_max"])}" x2="{px}" y2="{zy(row["z_min"])}" stroke="#486b9b" stroke-width="2"/>',
            f'<line x1="{px - 5}" y1="{zy(row["z_max"])}" x2="{px + 5}" y2="{zy(row["z_max"])}" stroke="#486b9b" stroke-width="2"/>',
            f'<line x1="{px - 5}" y1="{zy(row["z_min"])}" x2="{px + 5}" y2="{zy(row["z_min"])}" stroke="#486b9b" stroke-width="2"/>',
        ]
    parts.append(polyline(z_points, "#486b9b"))
    for px, py in z_points:
        parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#486b9b"/>')
    parts.append(
        f'<text x="{x + w - 4}" y="{y + 20}" text-anchor="end" class="note">24,707× increase from B=7 to B=32</text>'
    )

    # Panel 3: inference time.
    x, y, w, h = panels[2]
    ms_min, ms_max = math.log10(0.1), math.log10(100)

    def my(value: float) -> float:
        return y + h - (math.log10(value) - ms_min) / (ms_max - ms_min) * h

    for value in (0.1, 1, 10, 100):
        py = my(value)
        parts += [
            f'<line x1="{x}" y1="{py}" x2="{x + w}" y2="{py}" class="grid"/>',
            f'<text x="{x - 12}" y="{py + 5}" text-anchor="end" class="axis">{value:g} ms</text>',
        ]
    series = [
        ("A* / best-first (MAP)", "#c2873f", "best_first_ms"),
        ("Backward DP (task-specific)", "#486b9b", "dp_ms"),
        ("DecisionFlow (all queries)", "#8b63a7", "ms"),
    ]
    for _, color, key in series:
        points = [(sx(row["budget"], panels[2]), my(row[key])) for row in rows]
        parts.append(polyline(points, color, 3.0))
        for px, py in points:
            parts.append(f'<circle cx="{px}" cy="{py}" r="4.5" fill="{color}"/>')
    for index, (label, color, _) in enumerate(series):
        ly = y + 20 + index * 22
        parts += [
            f'<line x1="{x + 14}" y1="{ly}" x2="{x + 34}" y2="{ly}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{x + 42}" y="{ly + 5}" class="axis">{label}</text>',
        ]
    parts += [
        '<text x="70" y="552" class="note">A* matches the MAP result; task-specific DP is fastest here; DecisionFlow returns MAP together with Z, marginals, and conditionals through one reusable interface.</text>',
        "</svg>",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.inputs, args.output)


if __name__ == "__main__":
    main()
