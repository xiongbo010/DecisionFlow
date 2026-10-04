import json

from worldjev.cli import main


def _request(identifier="case-1"):
    return {
        "id": identifier,
        "state": {"channel": "card"},
        "questions": [
            {"id": "route", "type": "choice", "options": ["billing", "security"]},
            {"id": "fraud", "type": "noul"},
        ],
        "probabilities": {
            "route": {"billing": 0.7, "security": 0.3},
            "fraud": 0.8,
        },
    }


def _constraints():
    return {
        "hard": [
            {
                "name": "fraud-routes-security",
                "expr": {
                    "implies": [
                        {"eq": [{"var": "fraud"}, True]},
                        {"eq": [{"var": "route"}, "security"]},
                    ]
                },
            }
        ]
    }


def test_cli_infer(tmp_path, capsys):
    request_path = tmp_path / "request.json"
    constraints_path = tmp_path / "constraints.json"
    request_path.write_text(json.dumps(_request()), encoding="utf-8")
    constraints_path.write_text(json.dumps(_constraints()), encoding="utf-8")

    main(["infer", str(request_path), "--constraints", str(constraints_path)])
    output = json.loads(capsys.readouterr().out)

    assert output["inference"]["exact"] is True
    assert output["joint_map"] == {"route": "security", "fraud": True}
    assert output["valid_mass"] > 0


def test_cli_jsonl_evaluation(tmp_path):
    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "results.jsonl"
    constraints_path = tmp_path / "constraints.json"
    input_path.write_text(
        "\n".join(json.dumps(_request("case-%d" % index)) for index in range(2)) + "\n",
        encoding="utf-8",
    )
    constraints_path.write_text(json.dumps(_constraints()), encoding="utf-8")

    main(
        [
            "evaluate",
            str(input_path),
            "--constraints",
            str(constraints_path),
            "--output",
            str(output_path),
        ]
    )
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [row["id"] for row in rows] == ["case-0", "case-1"]
    assert all(row["status"] == "ok" for row in rows)
    assert all("marginals" in row and "joint_map" in row for row in rows)
