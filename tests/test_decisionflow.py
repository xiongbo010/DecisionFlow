import math
import unittest

from decisionflow import DecisionRequest, LocalPotentials, Question, DecisionFlow
from decisionflow.frontends.json import parse_program, parse_request
from decisionflow.trajectory import TrajectoryEngine, parse_trajectory


class DecisionFlowTests(unittest.TestCase):
    def test_dataclass_request_accepts_json_constraints(self):
        request = DecisionRequest(
            state={},
            questions=(Question("decision", "choice", ("allow", "deny")),),
        )
        result = DecisionFlow(backend="enumeration").infer(
            request,
            {"hard": [{"expr": {"eq": [{"var": "decision"}, "deny"]}}]},
            LocalPotentials({"decision": {"allow": 0.9, "deny": 0.1}}),
        )
        self.assertEqual(result.joint_map["decision"], "deny")

    def test_compile_and_infer_program_public_boundary(self):
        engine = DecisionFlow(backend="enumeration")
        request = {
            "state": {},
            "questions": [{"id": "decision", "type": "choice", "options": ["a", "b"]}],
        }
        program = engine.compile(
            request,
            {"hard": [{"expr": {"eq": [{"var": "decision"}, "b"]}}]},
        )
        result = engine.infer_program(
            program,
            LocalPotentials({"decision": {"a": 0.9, "b": 0.1}}),
        )
        self.assertEqual(result.joint_map["decision"], "b")

    def test_infer_many_streams_prepared_programs(self):
        engine = DecisionFlow(backend="enumeration")
        request = DecisionRequest(
            state={}, questions=(Question("decision", "choice", ("a", "b")),)
        )
        program = engine.compile(request)
        rows = list(
            engine.infer_many(
                [
                    (program, LocalPotentials({"decision": {"a": 0.8, "b": 0.2}})),
                    (program, LocalPotentials({"decision": {"a": 0.1, "b": 0.9}})),
                ]
            )
        )
        self.assertEqual([row.joint_map["decision"] for row in rows], ["a", "b"])

    def test_mixed_typed_hard_constraints(self):
        request = {
            "state": {},
            "questions": [
                {"id": "route", "type": "choice", "options": ["billing", "security"]},
                {"id": "fraud", "type": "noul"},
                {"id": "risk", "type": "score", "options": [0, 1, 2]},
            ],
            "probabilities": {
                "route": {"billing": 0.6, "security": 0.4},
                "fraud": 0.7,
                "risk": [0.2, 0.3, 0.5],
            },
        }
        constraints = {
            "hard": [
                {
                    "name": "fraud-security",
                    "expr": {"implies": [
                        {"eq": [{"var": "fraud"}, True]},
                        {"eq": [{"var": "route"}, "security"]},
                    ]},
                },
                {
                    "name": "high-risk-fraud",
                    "expr": {"implies": [
                        {"eq": [{"var": "risk"}, 2]},
                        {"eq": [{"var": "fraud"}, True]},
                    ]},
                },
            ]
        }
        result = DecisionFlow(backend="enumeration").infer(request, constraints)
        self.assertGreater(result.valid_mass, 0)
        self.assertEqual(result.joint_map, {"route": "security", "fraud": True, "risk": 2})
        for marginal in result.marginals.values():
            self.assertAlmostEqual(sum(marginal.values()), 1.0)
        self.assertTrue(result.inference.exact)

    def test_soft_constraints_report_hard_z_separately(self):
        request = {
            "state": {},
            "questions": [
                {"id": "help", "type": "score", "options": [0, 1, 2]},
                {"id": "correct", "type": "score", "options": [0, 1, 2]},
            ],
            "probabilities": {"help": [0.1, 0.2, 0.7], "correct": [0.6, 0.3, 0.1]},
        }
        constraints = {
            "soft": [{
                "name": "high-help-prefers-correct",
                "penalty": 2.0,
                "expr": {"implies": [
                    {"eq": [{"var": "help"}, 2]},
                    {"eq": [{"var": "correct"}, 2]},
                ]},
            }]
        }
        result = DecisionFlow(backend="enumeration").infer(request, constraints)
        self.assertAlmostEqual(result.valid_mass, 1.0)
        self.assertLess(result.diagnostics["normalizer"], 1.0)
        self.assertGreater(result.marginal("correct")[2], 0.1)

    def test_table_constraint(self):
        request = {
            "state": {},
            "questions": [
                {"id": "a", "type": "choice", "options": ["x", "y"]},
                {"id": "b", "type": "choice", "options": [0, 1]},
            ],
            "probabilities": {"a": [0.8, 0.2], "b": [0.3, 0.7]},
        }
        constraints = {"hard": [{"expr": {"allowed_table": {
            "variables": ["a", "b"], "rows": [["x", 0], ["y", 1]]
        }}}]}
        result = DecisionFlow(backend="enumeration").infer(request, constraints)
        self.assertEqual(result.joint_map, {"a": "x", "b": 0})
        self.assertAlmostEqual(result.valid_mass, 0.8 * 0.3 + 0.2 * 0.7)

    def test_state_grounding(self):
        request = {
            "state": {"account": {"locked": True}},
            "questions": [{"id": "action", "type": "choice", "options": ["unlock", "ignore"]}],
            "probabilities": {"action": [0.2, 0.8]},
        }
        constraints = {"hard": [{"expr": {"implies": [
            {"eq": [{"state": "account.locked"}, True]},
            {"eq": [{"var": "action"}, "unlock"]},
        ]}}]}
        result = DecisionFlow().infer(request, constraints)
        self.assertEqual(result.joint_map["action"], "unlock")
        self.assertAlmostEqual(result.valid_mass, 0.2)

    def test_trajectory(self):
        payload = {
            "initial_state": "s0",
            "actions": ["inspect", "finish"],
            "horizon": 2,
            "terminal_states": ["done"],
            "policy": {
                "s0": {"inspect": 0.4, "finish": 0.6},
                "s1": {"inspect": 0.1, "finish": 0.9},
            },
            "transitions": {"s0": {"inspect": "s1"}, "s1": {"finish": "done"}},
        }
        result = TrajectoryEngine().infer(parse_trajectory(payload))
        self.assertAlmostEqual(result.valid_mass, 0.4 * 0.9)
        self.assertAlmostEqual(result.first_action_marginals["inspect"], 1.0)
        self.assertEqual(result.trajectory_map[0], ("inspect", "s1"))


if __name__ == "__main__":
    unittest.main()
