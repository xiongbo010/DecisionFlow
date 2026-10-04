import unittest

from worldjev.frontends.json import parse_request
from worldjev.scorers import TypedResponseScorer


class ScorerTests(unittest.TestCase):
    def test_typed_response_adapter(self):
        request = parse_request({
            "state": "x",
            "questions": [
                {"id": "route", "type": "choice", "options": ["a", "b"]},
                {"id": "flag", "type": "noul"},
                {"id": "score", "type": "score", "options": [0, 1, 2]},
            ],
        })

        def call(payload):
            self.assertEqual(len(payload["questions"]), 3)
            return {"answers": {
                "route": {"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}},
                "flag": {"type": "noul", "noul": 0.7},
                "score": {"type": "score", "probabilities": {0: 0.2, 1: 0.3, 2: 0.5}},
            }}

        result = TypedResponseScorer(call).score(request)
        self.assertAlmostEqual(result.values["flag"][True], 0.7)
        self.assertAlmostEqual(result.values["route"]["b"], 0.6)


if __name__ == "__main__":
    unittest.main()
