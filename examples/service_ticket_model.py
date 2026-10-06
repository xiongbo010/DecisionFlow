"""Small deterministic scorer used by the README workflow example."""


def model(request):
    if request.metadata["step"] == "classify":
        return {
            "route": {"billing": 0.7, "security": 0.3},
            "fraud": {False: 0.2, True: 0.8},
        }
    if request.metadata["step"] == "review":
        return {"action": {"freeze": 0.4, "dismiss": 0.6}}
    raise ValueError("unexpected workflow step")
