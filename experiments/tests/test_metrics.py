from decisionflow_experiments import EXPERIMENTS, ExperimentRunner
from decisionflow_experiments.metrics import summarize


def test_summary_compares_local_marginal_and_joint_predictions():
    record = {
        "id": "conflict",
        "state": {},
        "probabilities": {"toxic": 0.45, "jailbreak": 0.9},
        "gold": {"toxic": True, "jailbreak": True},
    }
    output = ExperimentRunner.create(
        EXPERIMENTS["toxicchat"], backend="enumeration"
    ).run_record(record)
    report = summarize([output])
    assert report["consistency"]["raw_conflicting_records"] == 1
    assert report["consistency"]["joint_map_conflicting_records"] == 0
    assert report["prediction"]["joint_map_exact_match"] == 1.0
