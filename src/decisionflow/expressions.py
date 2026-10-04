"""Safe, backend-neutral expression tree for grounded decision constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .errors import ConstraintSyntaxError, StatePathError


class Operand:
    def resolve(self, assignment: Mapping[str, Any], state: Any) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class Literal(Operand):
    value: Any

    def resolve(self, assignment: Mapping[str, Any], state: Any) -> Any:
        return self.value


@dataclass(frozen=True)
class Variable(Operand):
    name: str

    def resolve(self, assignment: Mapping[str, Any], state: Any) -> Any:
        return assignment[self.name]


@dataclass(frozen=True)
class StateValue(Operand):
    path: str

    def resolve(self, assignment: Mapping[str, Any], state: Any) -> Any:
        current = state
        if not self.path:
            return current
        for part in self.path.split("."):
            try:
                if isinstance(current, Mapping):
                    current = current[part]
                elif isinstance(current, (list, tuple)):
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            except (KeyError, IndexError, AttributeError, ValueError) as error:
                raise StatePathError("cannot resolve state path %s" % self.path) from error
        return current


class Expr:
    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Bool(Expr):
    value: bool

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        return self.value


@dataclass(frozen=True)
class Compare(Expr):
    op: str
    left: Operand
    right: Operand

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        left = self.left.resolve(assignment, state)
        right = self.right.resolve(assignment, state)
        if self.op == "eq":
            return left == right
        if self.op == "ne":
            return left != right
        if self.op == "lt":
            return left < right
        if self.op == "le":
            return left <= right
        if self.op == "gt":
            return left > right
        if self.op == "ge":
            return left >= right
        raise ConstraintSyntaxError("unknown comparison %s" % self.op)


@dataclass(frozen=True)
class Contains(Expr):
    value: Operand
    choices: Tuple[Any, ...]
    negate: bool = False

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        result = self.value.resolve(assignment, state) in self.choices
        return not result if self.negate else result


@dataclass(frozen=True)
class Not(Expr):
    child: Expr

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        return not self.child.evaluate(assignment, state)


@dataclass(frozen=True)
class All(Expr):
    children: Tuple[Expr, ...]

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        return all(child.evaluate(assignment, state) for child in self.children)


@dataclass(frozen=True)
class AnyOf(Expr):
    children: Tuple[Expr, ...]

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        return any(child.evaluate(assignment, state) for child in self.children)


@dataclass(frozen=True)
class Implies(Expr):
    antecedent: Expr
    consequent: Expr

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        return (not self.antecedent.evaluate(assignment, state)) or self.consequent.evaluate(
            assignment, state
        )


@dataclass(frozen=True)
class Iff(Expr):
    left: Expr
    right: Expr

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        return self.left.evaluate(assignment, state) == self.right.evaluate(assignment, state)


@dataclass(frozen=True)
class Cardinality(Expr):
    kind: str
    k: int
    children: Tuple[Expr, ...]

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        count = sum(child.evaluate(assignment, state) for child in self.children)
        if self.kind == "at_most":
            return count <= self.k
        if self.kind == "at_least":
            return count >= self.k
        if self.kind == "exactly":
            return count == self.k
        raise ConstraintSyntaxError("unknown cardinality operator %s" % self.kind)


@dataclass(frozen=True)
class Table(Expr):
    """An arbitrary finite relation over decision variables.

    Table constraints are the universal escape hatch for frontends such as
    ontology grounders, SOP compilers, and legacy rule engines. They keep those
    systems out of the inference backend while preserving exact semantics.
    """

    variables: Tuple[str, ...]
    rows: Tuple[Tuple[Any, ...], ...]
    allowed: bool = True

    def evaluate(self, assignment: Mapping[str, Any], state: Any) -> bool:
        values = tuple(assignment[name] for name in self.variables)
        contained = values in self.rows
        return contained if self.allowed else not contained


def var(name: str) -> Variable:
    return Variable(name)


def lit(value: Any) -> Literal:
    return Literal(value)


def eq(left: Operand, right: Operand) -> Compare:
    return Compare("eq", left, right)


def implies(left: Expr, right: Expr) -> Implies:
    return Implies(left, right)
