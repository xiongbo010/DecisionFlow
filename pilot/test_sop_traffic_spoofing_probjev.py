"""Exactness and policy tests for traffic-spoofing ProbJev."""

from __future__ import annotations

import itertools

from sop_dangerous_goods_probjev import normalize
from sop_traffic_spoofing_probjev import ACTIONS, RISKS, VIOLATIONS, PolicyCircuit, valid_triples


def test_policy_has_expected_non_deterministic_worlds() -> None:
    triples = valid_triples()
    assert len(triples) == 25
    assert ("Medium", "Spoofing Traffic", "Temporary Suspension") in triples
    assert ("Medium", "Spoofing Traffic", "Warning Issued") in triples
    assert ("Low", "None", "No Action") in triples


def test_sdd_matches_enumeration() -> None:
    local = {
        "risk": normalize({value: i + 1 for i, value in enumerate(RISKS)}),
        "violation": normalize({value: i + 1 for i, value in enumerate(VIOLATIONS)}),
        "action": normalize({value: i + 1 for i, value in enumerate(ACTIONS)}),
    }
    result = PolicyCircuit.build().infer(local)
    worlds = []
    for risk, violation, action in itertools.product(RISKS, VIOLATIONS, ACTIONS):
        if (risk, violation, action) in valid_triples():
            worlds.append(((risk, violation, action), local["risk"][risk] * local["violation"][violation] * local["action"][action]))
    z = sum(weight for _, weight in worlds)
    assert abs(result["z"] - z) < 1e-12
    best = max(worlds, key=lambda item: item[1])[0]
    assert result["joint_map"] == {"risk": best[0], "violation": best[1], "action": best[2]}
    for name, position in (("risk", 0), ("violation", 1), ("action", 2)):
        for value in local[name]:
            expected = sum(weight for world, weight in worlds if world[position] == value) / z
            assert abs(result["marginals"][name][value] - expected) < 1e-12
