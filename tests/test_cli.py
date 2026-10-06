import json
import sys
import types

from decisionflow.cli import main

from workflow_fixtures import decision_model, workflow_document


def test_cli_run(tmp_path, capsys, monkeypatch):
    workflow_path = tmp_path / "workflow.json"
    state_path = tmp_path / "state.json"
    workflow_path.write_text(json.dumps(workflow_document()), encoding="utf-8")
    state_path.write_text(json.dumps({"status": "open"}), encoding="utf-8")
    plugin = types.ModuleType("decisionflow_test_model")
    plugin.model = decision_model
    monkeypatch.setitem(sys.modules, "decisionflow_test_model", plugin)

    main(
        [
            "run",
            str(workflow_path),
            str(state_path),
            "--model",
            "decisionflow_test_model:model",
            "--backend",
            "enumeration",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert output["prediction_kind"] == "joint_map"
    assert output["inference"]["exact"] is True


def test_cli_lists_workflow_backends(capsys):
    main(["backends"])
    output = json.loads(capsys.readouterr().out)

    assert "enumeration" in output
    assert "greedy" in output
