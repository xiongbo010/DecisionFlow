def workflow_document():
    return {
        "name": "service-ticket",
        "version": 1,
        "start": "classify",
        "max_steps": 3,
        "steps": {
            "classify": {
                "questions": [
                    {
                        "id": "route",
                        "type": "choice",
                        "options": ["billing", "security"],
                    },
                    {"id": "fraud", "type": "noul"},
                ],
                "transitions": [
                    {
                        "name": "possible-fraud",
                        "when": {"eq": [{"var": "fraud"}, True]},
                        "goto": "review",
                        "set": {"queue": {"const": "security"}},
                    },
                    {
                        "name": "ordinary-ticket",
                        "otherwise": True,
                        "goto": "resolved",
                        "set": {"queue": {"var": "route"}},
                    },
                ],
            },
            "review": {
                "questions": [
                    {
                        "id": "action",
                        "type": "choice",
                        "options": ["freeze", "dismiss"],
                    }
                ],
                "transitions": [
                    {
                        "otherwise": True,
                        "goto": "resolved",
                        "set": {"status": {"const": "resolved"}},
                    }
                ],
            },
            "resolved": {"terminal": True},
        },
        "constraints": {
            "hard": [
                {
                    "name": "fraud-routes-security",
                    "expr": {
                        "implies": [
                            {"eq": [{"var": "classify.fraud"}, True]},
                            {"eq": [{"var": "classify.route"}, "security"]},
                        ]
                    },
                },
                {
                    "name": "fraud-cannot-be-dismissed",
                    "expr": {
                        "implies": [
                            {"eq": [{"var": "classify.fraud"}, True]},
                            {"eq": [{"var": "review.action"}, "freeze"]},
                        ]
                    },
                },
            ]
        },
    }


def decision_model(request):
    if request.metadata["step"] == "classify":
        return {
            "route": {"billing": 0.7, "security": 0.3},
            "fraud": {False: 0.2, True: 0.8},
        }
    if request.metadata["step"] == "review":
        return {"action": {"freeze": 0.4, "dismiss": 0.6}}
    raise AssertionError("terminal steps must not call the decision model")
