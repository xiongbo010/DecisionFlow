#!/usr/bin/env python3
"""Generate resumable JevAny predictions for the Typed Decisions test split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / ".deps"))

import pyarrow.parquet as pq
from jevany.runtime import JevModel


def rows(path: Path) -> list[dict]:
    table = pq.read_table(path)
    return [
        {name: table[name][row].as_py() for name in table.column_names}
        for row in range(table.num_rows)
    ]


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if item.get("status") == "ok":
                result.add(item["id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("tmp/typed-decisions-review/test.parquet")
    )
    parser.add_argument("--checkpoint", default="SimpleJev/JevAny-Qwen3.5-4B-LoRA")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--workflow",
        choices=(
            "agent_trace_observability",
            "customer_service",
            "invoice_processing",
            "security_incidents",
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pilot/results/typed_decisions_jevany_4b_predictions.jsonl"),
    )
    args = parser.parse_args()

    records = rows(args.data)
    if args.workflow:
        records = [row for row in records if row["workflow"] == args.workflow]
    if args.limit is not None:
        records = records[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = completed(args.out)
    print(
        json.dumps(
            {"loading": args.checkpoint, "device": args.device, "dtype": args.dtype}
        ),
        flush=True,
    )
    model = JevModel.from_pretrained(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        model_name="jevany-qwen3.5-4b",
    )
    description = model.describe()
    print(json.dumps(description), flush=True)
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest = {
        "dataset": str(args.data),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "dtype": args.dtype,
        "workflow": args.workflow,
        "limit": args.limit,
        "model": description,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {
            key: previous.get(key)
            for key in manifest
            if key not in ("workflow", "limit")
        }
        expected = {
            key: value
            for key, value in manifest.items()
            if key not in ("workflow", "limit")
        }
        if comparable != expected:
            raise ValueError(f"resume configuration differs from {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    with args.out.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(records):
            if row["id"] in done:
                continue
            request = {
                "model": model.model_id,
                "state": json.loads(row["state"]),
                "questions": json.loads(row["questions"]),
            }
            started = time.perf_counter()
            try:
                response = model(request)
                item = {
                    "id": row["id"],
                    "workflow": row["workflow"],
                    "status": "ok",
                    "wall_latency_ms": 1000.0 * (time.perf_counter() - started),
                    "response": response,
                }
            except Exception as error:
                item = {
                    "id": row["id"],
                    "workflow": row["workflow"],
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                raise
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {"completed": index + 1, "total": len(records), "id": row["id"]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
