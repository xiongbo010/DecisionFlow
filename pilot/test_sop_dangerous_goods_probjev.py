"""Exactness checks for the SOP-Bench dangerous-goods circuit."""

from __future__ import annotations

import itertools

import numpy as np

from sop_dangerous_goods_probjev import (
    CLASS_VALUES,
    DATA_PATH,
    SCORE_NAMES,
    SCORE_VALUES,
    DangerousGoodsCircuit,
    audit_dataset,
    class_for,
    normalize,
)


def test_released_dataset_matches_declared_rules() -> None:
    import csv

    with DATA_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    audit = audit_dataset(rows)
    assert audit["rows"] == 274
    assert audit["rule_mismatches"] == []
    assert audit["invalid_product_ids"] == 5


def test_sdd_matches_exhaustive_enumeration() -> None:
    rng = np.random.default_rng(7)
    circuit = DangerousGoodsCircuit()
    for product_valid_evidence in (False, True):
        for _ in range(3):
            local = {
                "product_valid": normalize(
                    {False: float(rng.random()), True: float(rng.random())}
                ),
                "hazard_class": normalize(
                    {label: float(rng.random()) for label in CLASS_VALUES}
                ),
            }
            for name in SCORE_NAMES:
                local[name] = normalize(
                    {value: float(rng.random()) for value in SCORE_VALUES}
                )

            actual = circuit.infer(
                local, product_valid_evidence=product_valid_evidence
            )
            z = 0.0
            class_mass = {label: 0.0 for label in CLASS_VALUES}
            score_mass = {
                name: {value: 0.0 for value in SCORE_VALUES}
                for name in SCORE_NAMES
            }
            best_weight = -1.0
            best_world = None
            for values in itertools.product(SCORE_VALUES, repeat=4):
                label = class_for(values, product_valid_evidence)
                weight = local["product_valid"][product_valid_evidence]
                weight *= local["hazard_class"][label]
                for name, value in zip(SCORE_NAMES, values, strict=True):
                    weight *= local[name][value]
                z += weight
                class_mass[label] += weight
                for name, value in zip(SCORE_NAMES, values, strict=True):
                    score_mass[name][value] += weight
                if weight > best_weight:
                    best_weight = weight
                    best_world = values, label

            assert np.isclose(actual["z"], z, atol=1e-12)
            for label, mass in class_mass.items():
                assert np.isclose(
                    actual["marginals"]["hazard_class"][label], mass / z, atol=1e-12
                )
            for name in SCORE_NAMES:
                for value, mass in score_mass[name].items():
                    assert np.isclose(
                        actual["marginals"][name][value], mass / z, atol=1e-12
                    )
            assert actual["joint_map"]["product_valid"] == product_valid_evidence
            assert tuple(actual["joint_map"][name] for name in SCORE_NAMES) == best_world[0]
            assert actual["joint_map"]["hazard_class"] == best_world[1]
            assert np.isclose(
                actual["joint_map_unnormalized_probability"], best_weight, atol=1e-12
            )
