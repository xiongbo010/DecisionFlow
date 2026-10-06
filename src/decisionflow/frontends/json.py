"""JSON frontend for typed questions, probabilities, and constraints."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ..core import (
    Constraint,
    DecisionProgram,
    DecisionRequest,
    LocalPotentials,
    Question,
)
from ..errors import ConstraintSyntaxError
from ..expressions import (
    All,
    AnyOf,
    Bool,
    Cardinality,
    Compare,
    Contains,
    Expr,
    Iff,
    Implies,
    Literal,
    Not,
    Operand,
    StateValue,
    Table,
    Variable,
)


class JsonFrontend:
    name = "json"

    def compile(
        self, request: Mapping[str, Any], constraints: Any = None
    ) -> DecisionProgram:
        return parse_program(request, constraints)


def _question_options(payload: Mapping[str, Any]) -> Tuple[Any, ...]:
    kind = str(payload.get("type", "choice")).lower()
    if kind in {"noul", "boolean"}:
        return (False, True)
    raw = payload.get("options", payload.get("criteria"))
    if isinstance(raw, Mapping):
        return tuple(raw.keys())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ConstraintSyntaxError("question options must be a list or object")
    values = []
    for item in raw:
        values.append(
            item.get("value") if isinstance(item, Mapping) and "value" in item else item
        )
    return tuple(values)


def parse_request(payload: Mapping[str, Any]) -> DecisionRequest:
    raw_questions = payload.get("questions")
    if isinstance(raw_questions, Mapping):
        entries = [dict(value, id=key) for key, value in raw_questions.items()]
    elif isinstance(raw_questions, Sequence):
        entries = list(raw_questions)
    else:
        raise ConstraintSyntaxError("request.questions must be a list or object")
    questions = []
    for raw in entries:
        if not isinstance(raw, Mapping) or "id" not in raw:
            raise ConstraintSyntaxError("each question needs an id")
        criteria = raw.get("criteria", {})
        questions.append(
            Question(
                id=str(raw["id"]),
                type=str(raw.get("type", "choice")),
                options=_question_options(raw),
                instruction=str(raw.get("instruction", raw.get("instructions", ""))),
                criteria=criteria if isinstance(criteria, Mapping) else {},
                metadata=raw.get("metadata", {}),
            )
        )
    return DecisionRequest(
        state=payload.get("state"),
        questions=tuple(questions),
        evidence=payload.get("evidence", {}),
        metadata=payload.get("metadata", {}),
    )


def parse_operand(payload: Any) -> Operand:
    if isinstance(payload, Mapping):
        if set(payload) == {"var"}:
            return Variable(str(payload["var"]))
        if set(payload) == {"state"}:
            return StateValue(str(payload["state"]))
        if set(payload) == {"const"}:
            return Literal(payload["const"])
    return Literal(payload)


def _pair(name: str, payload: Any) -> Tuple[Any, Any]:
    if (
        not isinstance(payload, Sequence)
        or isinstance(payload, (str, bytes))
        or len(payload) != 2
    ):
        raise ConstraintSyntaxError("%s expects a two-item list" % name)
    return payload[0], payload[1]


def parse_expr(payload: Any) -> Expr:
    if isinstance(payload, bool):
        return Bool(payload)
    if not isinstance(payload, Mapping) or len(payload) != 1:
        raise ConstraintSyntaxError("expression must contain exactly one operator")
    op, body = next(iter(payload.items()))
    if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
        left, right = _pair(op, body)
        return Compare(op, parse_operand(left), parse_operand(right))
    if op in {"in", "not_in"}:
        value, choices = _pair(op, body)
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ConstraintSyntaxError("%s choices must be a list" % op)
        return Contains(parse_operand(value), tuple(choices), negate=op == "not_in")
    if op == "not":
        return Not(parse_expr(body))
    if op in {"all", "any"}:
        if not isinstance(body, Sequence) or isinstance(body, (str, bytes)):
            raise ConstraintSyntaxError("%s expects a list" % op)
        children = tuple(parse_expr(item) for item in body)
        return All(children) if op == "all" else AnyOf(children)
    if op in {"implies", "iff"}:
        left, right = _pair(op, body)
        return (
            Implies(parse_expr(left), parse_expr(right))
            if op == "implies"
            else Iff(parse_expr(left), parse_expr(right))
        )
    if op in {"at_most", "at_least", "exactly"}:
        if not isinstance(body, Mapping) or "k" not in body or "items" not in body:
            raise ConstraintSyntaxError("%s expects {k, items}" % op)
        return Cardinality(
            op, int(body["k"]), tuple(parse_expr(item) for item in body["items"])
        )
    if op in {"allowed_table", "forbidden_table"}:
        if (
            not isinstance(body, Mapping)
            or "variables" not in body
            or "rows" not in body
        ):
            raise ConstraintSyntaxError("%s expects {variables, rows}" % op)
        variables = tuple(map(str, body["variables"]))
        rows = tuple(tuple(row) for row in body["rows"])
        if any(len(row) != len(variables) for row in rows):
            raise ConstraintSyntaxError("table row width does not match variables")
        return Table(variables, rows, allowed=op == "allowed_table")
    raise ConstraintSyntaxError("unknown expression operator: %s" % op)


def parse_constraints(payload: Any) -> Tuple[Constraint, ...]:
    if payload is None:
        return ()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        hard_rows, soft_rows = payload, ()
    elif isinstance(payload, Mapping):
        hard_rows = payload.get("hard", ())
        soft_rows = payload.get("soft", ())
    else:
        raise ConstraintSyntaxError("constraints must be a list or object")
    result = []
    for hard, rows in ((True, hard_rows), (False, soft_rows)):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                row = {"expr": row}
            expression = row.get("expr", row.get("constraint"))
            if expression is None:
                raise ConstraintSyntaxError("constraint entry needs expr")
            result.append(
                Constraint(
                    name=str(
                        row.get("name", ("hard" if hard else "soft") + "-%d" % index)
                    ),
                    expr=parse_expr(expression),
                    hard=hard,
                    penalty=0.0
                    if hard
                    else float(row.get("penalty", row.get("weight", 1.0))),
                    source=row.get("source"),
                    metadata=row.get("metadata", {}),
                )
            )
    return tuple(result)


def parse_program(
    request_payload: Mapping[str, Any], constraints_payload: Any = None
) -> DecisionProgram:
    request = parse_request(request_payload)
    constraints_doc = (
        constraints_payload
        if constraints_payload is not None
        else request_payload.get("constraints", {})
    )
    constraints = list(parse_constraints(constraints_doc))
    for variable, value in request.evidence.items():
        constraints.append(
            Constraint(
                name="evidence:%s" % variable,
                expr=Compare("eq", Variable(variable), Literal(value)),
                hard=True,
                source="request.evidence",
            )
        )
    meta = constraints_doc if isinstance(constraints_doc, Mapping) else {}
    return DecisionProgram(
        request=request,
        constraints=tuple(constraints),
        name=str(meta.get("name", "anonymous")),
        version=str(meta.get("version", "0")),
        metadata=meta.get("metadata", {}),
    )


def parse_probabilities(
    payload: Mapping[str, Any], request: DecisionRequest
) -> LocalPotentials:
    raw = payload.get("probabilities")
    if raw is None:
        raise ConstraintSyntaxError(
            "request needs probabilities when no scorer is configured"
        )
    values: Dict[str, Dict[Any, float]] = {}
    for question in request.questions:
        row = raw.get(question.id)
        if row is None:
            raise ConstraintSyntaxError("missing probabilities for %s" % question.id)
        if isinstance(row, Mapping):
            if (
                question.type in {"noul", "boolean"}
                and "true" not in row
                and "false" not in row
            ):
                positive = float(row.get("probability_true", row.get("noul")))
                values[question.id] = {False: 1.0 - positive, True: positive}
            else:
                values[question.id] = dict(row)
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            if len(row) != len(question.options):
                raise ConstraintSyntaxError(
                    "probability vector length mismatch for %s" % question.id
                )
            values[question.id] = dict(zip(question.options, map(float, row)))
        elif question.type in {"noul", "boolean"}:
            positive = float(row)
            values[question.id] = {False: 1.0 - positive, True: positive}
        else:
            raise ConstraintSyntaxError(
                "unsupported probability representation for %s" % question.id
            )
    return LocalPotentials(
        values, payload.get("probability_metadata", {})
    ).normalized_for(request)
