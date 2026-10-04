"""Regression tests for every structural family exercised by the pilots.

These tests use synthetic probabilities. Dataset loading and metrics remain in
the experiment code; WorldJev core is tested only on the resulting finite
decision schemas and rules.
"""

import importlib.util
import unittest

from worldjev import WorldJev


HAS_SDD = importlib.util.find_spec("pysdd") is not None


def noul_questions(names):
    return [{"id": name, "type": "noul"} for name in names]


def implication(left, right, right_value=True):
    return {"implies": [
        {"eq": [{"var": left}, True]},
        {"eq": [{"var": right}, right_value]},
    ]}


@unittest.skipUnless(HAS_SDD, "PySDD is optional")
class PilotShapeTests(unittest.TestCase):
    def test_openai_moderation_shape(self):
        names = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]
        request = {
            "state": {},
            "questions": noul_questions(names),
            "probabilities": {name: 0.2 + index * 0.05 for index, name in enumerate(names)},
        }
        constraints = {"hard": [
            {"name": "S3-S", "expr": implication("S3", "S")},
            {"name": "H2-H", "expr": implication("H2", "H")},
            {"name": "V2-V", "expr": implication("V2", "V")},
        ]}
        result = WorldJev(backend="sdd").infer(request, constraints)
        self.assertEqual(result.inference.valid_world_count, 108)

    def test_toxigen_shape(self):
        request = {
            "state": {},
            "questions": [
                {"id": "toxic", "type": "noul"},
                {"id": "toxicity", "type": "score", "options": [0, 1, 2, 3, 4]},
            ],
            "probabilities": {"toxic": 0.4, "toxicity": [0.1, 0.2, 0.3, 0.2, 0.2]},
        }
        constraints = {"hard": [
            {"expr": {"implies": [
                {"eq": [{"var": "toxicity"}, 0]},
                {"eq": [{"var": "toxic"}, False]},
            ]}},
            {"expr": {"implies": [
                {"eq": [{"var": "toxicity"}, 4]},
                {"eq": [{"var": "toxic"}, True]},
            ]}},
        ]}
        result = WorldJev(backend="sdd").infer(request, constraints)
        self.assertEqual(result.inference.valid_world_count, 8)

    def test_goemotions_shape(self):
        emotions = ["emotion_%d" % index for index in range(27)]
        request = {
            "state": {},
            "questions": noul_questions(emotions + ["neutral"]),
            "probabilities": {name: 0.1 for name in emotions + ["neutral"]},
        }
        constraints = {"hard": [
            {"expr": implication("neutral", name, False)} for name in emotions
        ]}
        result = WorldJev(backend="sdd").infer(request, constraints)
        self.assertEqual(result.inference.valid_world_count, 2 ** 27 + 1)

    def test_toxicchat_shape(self):
        request = {
            "state": {},
            "questions": noul_questions(["toxic", "jailbreak"]),
            "probabilities": {"toxic": 0.3, "jailbreak": 0.6},
        }
        result = WorldJev(backend="sdd").infer(
            request, {"hard": [{"expr": implication("jailbreak", "toxic")}]}
        )
        self.assertEqual(result.inference.valid_world_count, 3)

    def test_helpsteer2_shape(self):
        names = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
        request = {
            "state": {},
            "questions": [{"id": name, "type": "score", "options": [0, 1, 2, 3, 4]} for name in names],
            "probabilities": {name: [0.2] * 5 for name in names},
        }
        constraints = {"hard": [
            {"expr": {"implies": [
                {"ge": [{"var": "helpfulness"}, 3]},
                {"ge": [{"var": "correctness"}, 1]},
            ]}},
            {"expr": {"implies": [
                {"ge": [{"var": "helpfulness"}, 3]},
                {"ge": [{"var": "coherence"}, 1]},
            ]}},
            {"expr": {"implies": [
                {"ge": [{"var": "helpfulness"}, 4]},
                {"ge": [{"var": "coherence"}, 2]},
            ]}},
        ]}
        result = WorldJev(backend="sdd").infer(request, constraints)
        self.assertEqual(result.inference.valid_world_count, 2575)


if __name__ == "__main__":
    unittest.main()
