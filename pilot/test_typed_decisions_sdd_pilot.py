from pathlib import Path

from typed_decisions_sdd_pilot import run

ROOT = Path(__file__).resolve().parents[1]


def test_soft_gold_diagnostic_exactness(tmp_path):
    result = run(
        ROOT / "tmp/typed-decisions-review/train.parquet",
        ROOT / "tmp/typed-decisions-review/test.parquet",
        None,
        tmp_path,
        False,
    )
    assert set(result["workflows"]) == {
        "agent_trace_observability",
        "customer_service",
        "invoice_processing",
        "security_incidents",
    }
    for workflow in result["workflows"].values():
        assert workflow["exactness"]["status"] == "pass"
        assert workflow["exactness"]["z"] < 1e-11
        assert workflow["exactness"]["marginal"] < 1e-11
        assert workflow["exactness"]["map_weight"] < 1e-11
        assert workflow["metrics"]["map_conflicting_examples"] == 0
