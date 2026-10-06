import importlib.util
import unittest

from decisionflow.engine import DecisionEngine


@unittest.skipUnless(importlib.util.find_spec("pysdd"), "PySDD is optional")
class SDDTests(unittest.TestCase):
    def test_sdd_matches_enumeration_with_soft_constraint(self):
        request = {
            "state": {},
            "questions": [
                {"id": "toxic", "type": "noul"},
                {"id": "score", "type": "score", "options": [0, 1, 2, 3, 4]},
            ],
            "probabilities": {
                "toxic": 0.63,
                "score": [0.12, 0.18, 0.21, 0.27, 0.22],
            },
        }
        constraints = {
            "hard": [
                {
                    "name": "benign",
                    "expr": {
                        "implies": [
                            {"eq": [{"var": "score"}, 0]},
                            {"eq": [{"var": "toxic"}, False]},
                        ]
                    },
                },
                {
                    "name": "extreme",
                    "expr": {
                        "implies": [
                            {"eq": [{"var": "score"}, 4]},
                            {"eq": [{"var": "toxic"}, True]},
                        ]
                    },
                },
            ],
            "soft": [
                {
                    "name": "high",
                    "penalty": 0.7,
                    "expr": {
                        "implies": [
                            {"ge": [{"var": "score"}, 3]},
                            {"eq": [{"var": "toxic"}, True]},
                        ]
                    },
                }
            ],
        }
        enum = DecisionEngine(backend="enumeration").infer(request, constraints)
        sdd = DecisionEngine(backend="sdd").infer(request, constraints)
        self.assertAlmostEqual(enum.valid_mass, sdd.valid_mass, places=12)
        self.assertAlmostEqual(
            enum.diagnostics["normalizer"], sdd.diagnostics["normalizer"], places=12
        )
        self.assertEqual(enum.joint_map, sdd.joint_map)
        for name in enum.marginals:
            for value in enum.marginals[name]:
                self.assertAlmostEqual(
                    enum.marginals[name][value], sdd.marginals[name][value], places=12
                )

    def test_large_boolean_star(self):
        questions = [{"id": "neutral", "type": "noul"}]
        probabilities = {"neutral": 0.2}
        hard = []
        for index in range(27):
            name = "emotion_%d" % index
            questions.append({"id": name, "type": "noul"})
            probabilities[name] = 0.1 + (index % 5) * 0.03
            hard.append(
                {
                    "name": "neutral-excludes-%s" % name,
                    "expr": {
                        "implies": [
                            {"eq": [{"var": "neutral"}, True]},
                            {"eq": [{"var": name}, False]},
                        ]
                    },
                }
            )
        result = DecisionEngine(backend="sdd").infer(
            {"state": {}, "questions": questions, "probabilities": probabilities},
            {"hard": hard},
        )
        self.assertTrue(result.inference.exact)
        self.assertEqual(result.inference.world_count, 2**28)
        self.assertGreater(result.valid_mass, 0)

    def test_compilation_cache_ignores_unreferenced_state(self):
        engine = DecisionEngine(backend="sdd")
        constraints = {
            "hard": [
                {
                    "expr": {
                        "implies": [
                            {"eq": [{"var": "a"}, True]},
                            {"eq": [{"var": "b"}, True]},
                        ]
                    }
                }
            ]
        }
        base = {
            "questions": [{"id": "a", "type": "noul"}, {"id": "b", "type": "noul"}],
            "probabilities": {"a": 0.7, "b": 0.2},
        }
        first = engine.infer({**base, "state": {"text": "first"}}, constraints)
        second = engine.infer({**base, "state": {"text": "second"}}, constraints)
        self.assertFalse(first.diagnostics["compile_cache_hit"])
        self.assertTrue(second.diagnostics["compile_cache_hit"])
        self.assertEqual(second.inference.compile_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
