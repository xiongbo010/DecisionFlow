"""Deterministic cached-score inference through DecisionFlow public APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, TextIO

from decisionflow import DecisionFlow, TrajectoryEngine, parse_trajectory
from decisionflow.frontends.json import parse_probabilities

from .registry import ExperimentSpec


def read_jsonl(stream: TextIO) -> Iterator[Mapping[str, Any]]:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, Mapping):
            raise ValueError("JSONL line %d is not an object" % line_number)
        yield record


@dataclass
class ExperimentRunner:
    """Run one experiment without owning data acquisition or model inference."""

    spec: ExperimentSpec
    engine: DecisionFlow

    @classmethod
    def create(cls, spec: ExperimentSpec, backend: str = "auto") -> "ExperimentRunner":
        return cls(spec=spec, engine=DecisionFlow(backend=backend))

    def run_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        request_payload = self.spec.request(record)
        constraints = self.spec.constraint_document(record)
        program = self.engine.compile(request_payload, constraints)
        potentials = parse_probabilities(request_payload, program.request)
        result = self.engine.infer_program(program, potentials)
        return {
            "id": record.get("id"),
            "experiment": self.spec.name,
            "questions": [
                {
                    "id": question.id,
                    "type": question.type,
                    "options": list(question.options),
                }
                for question in program.request.questions
            ],
            "gold": record.get("gold"),
            "provenance": record.get("provenance", {}),
            **result.to_dict(),
        }

    def run(self, records: Iterable[Mapping[str, Any]]) -> Iterator[Dict[str, Any]]:
        for record in records:
            yield self.run_record(record)

    def run_stream(self, source: TextIO, target: TextIO) -> int:
        count = 0
        for output in self.run(read_jsonl(source)):
            target.write(json.dumps(output, default=str, sort_keys=True) + "\n")
            target.flush()
            count += 1
        return count


@dataclass
class TrajectoryExperimentRunner:
    """Run grounded finite-horizon records through the public trajectory API."""

    engine: TrajectoryEngine

    @classmethod
    def create(cls) -> "TrajectoryExperimentRunner":
        return cls(engine=TrajectoryEngine())

    def run_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        result = self.engine.infer(parse_trajectory(record["trajectory"]))
        return {
            "id": record.get("id"),
            "experiment": record.get("experiment", "multistep"),
            "gold": record.get("gold"),
            "provenance": record.get("provenance", {}),
            **result.to_dict(),
        }

    def run_stream(self, source: TextIO, target: TextIO) -> int:
        count = 0
        for record in read_jsonl(source):
            target.write(json.dumps(self.run_record(record), sort_keys=True) + "\n")
            target.flush()
            count += 1
        return count
