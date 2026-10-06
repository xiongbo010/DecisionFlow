"""Small typed-STRIPS adapter used by planning experiments.

The module intentionally lives in the experiment package.  DecisionFlow only
consumes finite decision variables, policies, and transitions; PDDL parsing and
grounding are benchmark-specific front-end concerns.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


Atom = tuple[str, ...]
State = frozenset[Atom]


@dataclass(frozen=True)
class Parameter:
    name: str
    type_name: str = "object"


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: tuple[Parameter, ...]
    positive_preconditions: frozenset[Atom]
    negative_preconditions: frozenset[Atom]
    add_effects: frozenset[Atom]
    delete_effects: frozenset[Atom]


@dataclass(frozen=True, order=True)
class GroundAction:
    name: str
    arguments: tuple[str, ...]
    positive_preconditions: frozenset[Atom]
    negative_preconditions: frozenset[Atom]
    add_effects: frozenset[Atom]
    delete_effects: frozenset[Atom]

    @property
    def key(self) -> str:
        suffix = " " + " ".join(self.arguments) if self.arguments else ""
        return f"({self.name}{suffix})"


@dataclass(frozen=True)
class StripsTask:
    object_types: Mapping[str, str]
    type_parents: Mapping[str, str]
    initial_state: State
    positive_goals: frozenset[Atom]
    negative_goals: frozenset[Atom]
    actions: tuple[GroundAction, ...]
    dynamic_predicates: frozenset[str]

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(sorted(self.object_types))

    def is_goal(self, state: State) -> bool:
        return self.positive_goals <= state and not (self.negative_goals & state)

    @staticmethod
    def applicable(state: State, action: GroundAction) -> bool:
        return action.positive_preconditions <= state and not (
            action.negative_preconditions & state
        )

    @staticmethod
    def apply(state: State, action: GroundAction) -> State:
        return frozenset((state - action.delete_effects) | action.add_effects)


def parse_sexpr(text: str) -> list[Any]:
    text = re.sub(r";[^\n]*", "", text.lower())
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    root: list[Any] = []
    stack = [root]
    for token in tokens:
        if token == "(":
            child: list[Any] = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced closing parenthesis")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1 or len(root) != 1:
        raise ValueError("invalid PDDL S-expression")
    return root[0]


def section(tree: list[Any], name: str) -> list[Any]:
    for row in tree:
        if isinstance(row, list) and row and row[0] == name:
            return row
    raise KeyError(name)


def typed_symbols(tokens: Iterable[Any], default: str = "object") -> tuple[tuple[str, str], ...]:
    """Parse a PDDL typed list, including a final untyped group."""
    raw = [str(token) for token in tokens]
    result: list[tuple[str, str]] = []
    pending: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "-":
            if not pending or index + 1 >= len(raw):
                raise ValueError("malformed PDDL typed list")
            type_name = raw[index + 1]
            result.extend((name, type_name) for name in pending)
            pending = []
            index += 2
        else:
            pending.append(token)
            index += 1
    result.extend((name, default) for name in pending)
    return tuple(result)


def conjunction(expr: Any) -> tuple[frozenset[Atom], frozenset[Atom]]:
    if not isinstance(expr, list) or not expr:
        raise ValueError("expected a PDDL formula")
    rows = expr[1:] if expr[0] == "and" else [expr]
    positive: set[Atom] = set()
    negative: set[Atom] = set()
    for row in rows:
        if not isinstance(row, list) or not row:
            raise ValueError("expected a PDDL atom")
        if row[0] == "not":
            negative.add(tuple(row[1]))
        else:
            positive.add(tuple(row))
    return frozenset(positive), frozenset(negative)


def _substitute(atom: Atom, binding: Mapping[str, str]) -> Atom:
    return tuple(binding.get(token, token) for token in atom)


def _is_subtype(type_name: str, expected: str, parents: Mapping[str, str]) -> bool:
    current = type_name
    seen: set[str] = set()
    while current not in seen:
        if current == expected:
            return True
        seen.add(current)
        if current == "object":
            break
        current = parents.get(current, "object")
    return expected == "object"


def load_typed_strips_task(domain_path: Path, problem_path: Path) -> StripsTask:
    domain = parse_sexpr(domain_path.read_text(encoding="utf-8"))
    try:
        type_rows = typed_symbols(section(domain, ":types")[1:])
    except KeyError:
        type_rows = ()
    type_parents = {name: parent for name, parent in type_rows}
    type_parents.setdefault("object", "object")

    schemas: list[ActionSchema] = []
    dynamic: set[str] = set()
    for row in domain:
        if not (isinstance(row, list) and row and row[0] == ":action"):
            continue
        fields = {row[index]: row[index + 1] for index in range(2, len(row), 2)}
        pre_pos, pre_neg = conjunction(fields[":precondition"])
        add, delete = conjunction(fields[":effect"])
        dynamic.update(atom[0] for atom in add | delete)
        schemas.append(
            ActionSchema(
                name=str(row[1]),
                parameters=tuple(
                    Parameter(name, type_name)
                    for name, type_name in typed_symbols(fields[":parameters"])
                ),
                positive_preconditions=pre_pos,
                negative_preconditions=pre_neg,
                add_effects=add,
                delete_effects=delete,
            )
        )

    problem = parse_sexpr(problem_path.read_text(encoding="utf-8"))
    object_types = dict(typed_symbols(section(problem, ":objects")[1:]))
    initial = frozenset(tuple(atom) for atom in section(problem, ":init")[1:])
    goal_pos, goal_neg = conjunction(section(problem, ":goal")[1])

    objects_by_type: dict[str, tuple[str, ...]] = {}
    needed_types = {parameter.type_name for schema in schemas for parameter in schema.parameters}
    for expected in needed_types:
        objects_by_type[expected] = tuple(
            sorted(
                name
                for name, type_name in object_types.items()
                if _is_subtype(type_name, expected, type_parents)
            )
        )

    actions: list[GroundAction] = []
    for schema in schemas:
        domains = [objects_by_type[parameter.type_name] for parameter in schema.parameters]
        for values in itertools.product(*domains):
            binding = {
                parameter.name: value
                for parameter, value in zip(schema.parameters, values)
            }
            grounded_add = frozenset(_substitute(atom, binding) for atom in schema.add_effects)
            grounded_delete = frozenset(
                _substitute(atom, binding) for atom in schema.delete_effects
            )
            # Delete/add overlap produces a PDDL semantic no-op for that atom.
            effective_add = grounded_add - grounded_delete
            effective_delete = grounded_delete - grounded_add
            if not effective_add and not effective_delete:
                continue
            actions.append(
                GroundAction(
                    name=schema.name,
                    arguments=tuple(values),
                    positive_preconditions=frozenset(
                        _substitute(atom, binding) for atom in schema.positive_preconditions
                    ),
                    negative_preconditions=frozenset(
                        _substitute(atom, binding) for atom in schema.negative_preconditions
                    ),
                    add_effects=effective_add,
                    delete_effects=effective_delete,
                )
            )

    return StripsTask(
        object_types=object_types,
        type_parents=type_parents,
        initial_state=initial,
        positive_goals=goal_pos,
        negative_goals=goal_neg,
        actions=tuple(sorted(actions)),
        dynamic_predicates=frozenset(dynamic),
    )


def state_key(state: State) -> str:
    serialized = "|".join("(" + " ".join(atom) + ")" for atom in sorted(state))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def action_index(task: StripsTask) -> dict[str, GroundAction]:
    return {action.key: action for action in task.actions}
