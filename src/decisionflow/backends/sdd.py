"""Exact SDD backend for independent typed potentials and declarative rules."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import fields, is_dataclass
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from pysdd.sdd import SddManager, SddNode, Vtree

from ..core import DecisionProgram, DecisionResult, InferenceInfo, LocalPotentials, Question
from ..errors import ConstraintSyntaxError, UnsatisfiableError
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
from ..joints import IndependentJoint, JointBuilder


@dataclass(frozen=True)
class _MapResult:
    weight: float
    assignment: Mapping[int, int]


def _assignment_mask(assignment: Mapping[int, int]) -> int:
    return sum(int(value) << (var - 1) for var, value in assignment.items())


def _better(left: Optional[_MapResult], right: _MapResult, atol: float = 1e-15) -> _MapResult:
    if left is None or right.weight > left.weight + atol:
        return right
    if math.isclose(right.weight, left.weight, rel_tol=0.0, abs_tol=atol):
        return right if _assignment_mask(right.assignment) < _assignment_mask(left.assignment) else left
    return left


def _merge(left: _MapResult, right: _MapResult) -> _MapResult:
    if left.assignment.keys() & right.assignment.keys():
        raise RuntimeError("non-decomposable SDD element")
    assignment = dict(left.assignment)
    assignment.update(right.assignment)
    return _MapResult(left.weight * right.weight, assignment)


def _max_product(
    node: SddNode,
    vtree: Vtree,
    positive: Sequence[float],
    negative: Sequence[float],
    true_node: SddNode,
) -> _MapResult:
    if node.is_false():
        return _MapResult(-math.inf, {})
    if vtree.is_leaf():
        var = int(vtree.var())
        if node.is_true():
            if positive[var - 1] > negative[var - 1]:
                return _MapResult(float(positive[var - 1]), {var: 1})
            return _MapResult(float(negative[var - 1]), {var: 0})
        if not node.is_literal() or abs(int(node.literal)) != var:
            raise RuntimeError("unexpected SDD terminal/vtree pairing")
        value = int(node.literal > 0)
        weight = positive[var - 1] if value else negative[var - 1]
        return _MapResult(float(weight), {var: value})
    if node.is_true():
        return _merge(
            _max_product(true_node, vtree.left(), positive, negative, true_node),
            _max_product(true_node, vtree.right(), positive, negative, true_node),
        )
    node_vtree = node.vtree()
    if node_vtree == vtree:
        best = None
        for prime, sub in node.elements():
            if sub.is_false():
                continue
            candidate = _merge(
                _max_product(prime, vtree.left(), positive, negative, true_node),
                _max_product(sub, vtree.right(), positive, negative, true_node),
            )
            best = _better(best, candidate)
        return best if best is not None else _MapResult(-math.inf, {})
    if Vtree.is_sub(node_vtree, vtree.left()):
        return _merge(
            _max_product(node, vtree.left(), positive, negative, true_node),
            _max_product(true_node, vtree.right(), positive, negative, true_node),
        )
    return _merge(
        _max_product(true_node, vtree.left(), positive, negative, true_node),
        _max_product(node, vtree.right(), positive, negative, true_node),
    )


def _compare(op: str, left: Any, right: Any) -> bool:
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "lt":
        return left < right
    if op == "le":
        return left <= right
    if op == "gt":
        return left > right
    if op == "ge":
        return left >= right
    raise ConstraintSyntaxError("unknown comparison %s" % op)


@dataclass
class _Compiled:
    manager: SddManager
    root: SddNode
    variables: Mapping[str, Mapping[Any, int]]
    soft_variables: Mapping[str, int]
    compile_ms: float


class _ExpressionCompiler:
    def __init__(self, manager, variables, questions, state):
        self.manager = manager
        self.variables = variables
        self.questions = questions
        self.state = state

    def _operand(self, operand: Operand):
        if isinstance(operand, Variable):
            if operand.name not in self.variables:
                raise ConstraintSyntaxError("unknown decision variable %s" % operand.name)
            return [
                (self.manager.literal(var), value)
                for value, var in self.variables[operand.name].items()
            ]
        value = operand.resolve({}, self.state)
        return [(self.manager.true(), value)]

    def compile(self, expr: Expr) -> SddNode:
        if isinstance(expr, Bool):
            return self.manager.true() if expr.value else self.manager.false()
        if isinstance(expr, Compare):
            result = self.manager.false()
            for left_node, left_value in self._operand(expr.left):
                for right_node, right_value in self._operand(expr.right):
                    if _compare(expr.op, left_value, right_value):
                        result |= left_node & right_node
            return result
        if isinstance(expr, Contains):
            result = self.manager.false()
            for node, value in self._operand(expr.value):
                accepted = value in expr.choices
                if expr.negate:
                    accepted = not accepted
                if accepted:
                    result |= node
            return result
        if isinstance(expr, Not):
            return ~self.compile(expr.child)
        if isinstance(expr, All):
            result = self.manager.true()
            for child in expr.children:
                result &= self.compile(child)
            return result
        if isinstance(expr, AnyOf):
            result = self.manager.false()
            for child in expr.children:
                result |= self.compile(child)
            return result
        if isinstance(expr, Implies):
            left = self.compile(expr.antecedent)
            return (~left) | self.compile(expr.consequent)
        if isinstance(expr, Iff):
            left, right = self.compile(expr.left), self.compile(expr.right)
            return (left & right) | ((~left) & (~right))
        if isinstance(expr, Cardinality):
            children = [self.compile(child) for child in expr.children]
            exact = [self.manager.false() for _ in range(len(children) + 1)]
            exact[0] = self.manager.true()
            used = 0
            for child in children:
                updated = [self.manager.false() for _ in range(len(children) + 1)]
                for count in range(used + 1):
                    updated[count] |= exact[count] & (~child)
                    updated[count + 1] |= exact[count] & child
                exact = updated
                used += 1
            result = self.manager.false()
            for count, node in enumerate(exact):
                if (
                    (expr.kind == "at_most" and count <= expr.k)
                    or (expr.kind == "at_least" and count >= expr.k)
                    or (expr.kind == "exactly" and count == expr.k)
                ):
                    result |= node
            return result
        if isinstance(expr, Table):
            result = self.manager.false()
            for row in expr.rows:
                term = self.manager.true()
                for name, value in zip(expr.variables, row):
                    try:
                        term &= self.manager.literal(self.variables[name][value])
                    except KeyError as error:
                        raise ConstraintSyntaxError(
                            "table references unknown value %s=%r" % (name, value)
                        ) from error
                result |= term
            return result if expr.allowed else ~result
        raise ConstraintSyntaxError("unsupported expression type %s" % type(expr).__name__)


class SDDBackend:
    name = "sdd"
    exact = True

    def __init__(self, cache_size: int = 32):
        self.cache_size = cache_size
        self._cache = OrderedDict()

    @staticmethod
    def _state_paths(value):
        result = set()
        if isinstance(value, StateValue):
            result.add(value.path)
        elif is_dataclass(value):
            for item in fields(value):
                result.update(SDDBackend._state_paths(getattr(value, item.name)))
        elif isinstance(value, (tuple, list)):
            for item in value:
                result.update(SDDBackend._state_paths(item))
        elif isinstance(value, Mapping):
            for item in value.values():
                result.update(SDDBackend._state_paths(item))
        return result

    def _cache_key(self, program: DecisionProgram):
        paths = set()
        for constraint in program.constraints:
            paths.update(self._state_paths(constraint.expr))
        state_signature = tuple(
            (path, repr(StateValue(path).resolve({}, program.request.state)))
            for path in sorted(paths)
        )
        schema = tuple(
            (question.id, question.type, tuple(map(repr, question.options)))
            for question in program.request.questions
        )
        constraints = tuple(
            (item.name, item.hard, item.penalty, repr(item.expr))
            for item in program.constraints
        )
        return schema, constraints, state_signature

    def _cached_compile(self, program: DecisionProgram):
        key = self._cache_key(program)
        if key in self._cache:
            compiled = self._cache.pop(key)
            self._cache[key] = compiled
            return compiled, True
        compiled = self._compile(program)
        self._cache[key] = compiled
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return compiled, False

    def _compile(self, program: DecisionProgram) -> _Compiled:
        started = time.perf_counter()
        variables: Dict[str, Dict[Any, int]] = {}
        next_var = 1
        for question in program.request.questions:
            variables[question.id] = {}
            for option in question.options:
                variables[question.id][option] = next_var
                next_var += 1
        soft_variables = {}
        for constraint in program.soft_constraints:
            soft_variables[constraint.name] = next_var
            next_var += 1
        vtree = Vtree(
            var_count=next_var - 1,
            var_order=list(range(1, next_var)),
            vtree_type="balanced",
        )
        manager = SddManager.from_vtree(vtree)
        root = manager.true()
        for group in variables.values():
            literals = [manager.literal(var) for var in group.values()]
            at_least_one = manager.false()
            for literal in literals:
                at_least_one |= literal
            root &= at_least_one
            for index, left in enumerate(literals):
                for right in literals[index + 1 :]:
                    root &= (~left) | (~right)
        compiler = _ExpressionCompiler(
            manager,
            variables,
            program.request.by_id,
            program.request.state,
        )
        for constraint in program.hard_constraints:
            root &= compiler.compile(constraint.expr)
        for constraint in program.soft_constraints:
            violation = manager.literal(soft_variables[constraint.name])
            desired = compiler.compile(constraint.expr)
            root &= (violation & (~desired)) | ((~violation) & desired)
        if root.is_false():
            raise UnsatisfiableError("grounded constraints are unsatisfiable")
        root.ref()
        return _Compiled(
            manager,
            root,
            variables,
            soft_variables,
            1000.0 * (time.perf_counter() - started),
        )

    @staticmethod
    def _wmc(compiled: _Compiled, positive, negative):
        wmc = compiled.root.wmc(log_mode=False)
        for var in range(1, len(positive) + 1):
            wmc.set_literal_weight(var, float(positive[var - 1]))
            wmc.set_literal_weight(-var, float(negative[var - 1]))
        return wmc, float(wmc.propagate())

    def infer(
        self,
        program: DecisionProgram,
        potentials: LocalPotentials,
        joint: JointBuilder,
    ) -> DecisionResult:
        if not isinstance(joint, IndependentJoint):
            raise NotImplementedError("SDD v0.0 supports IndependentJoint leaf weights")
        compiled, cache_hit = self._cached_compile(program)
        var_count = sum(len(group) for group in compiled.variables.values()) + len(
            compiled.soft_variables
        )
        positive = [1.0] * var_count
        negative = [1.0] * var_count
        for name, group in compiled.variables.items():
            for value, var in group.items():
                positive[var - 1] = potentials.values[name][value]
        for constraint in program.soft_constraints:
            var = compiled.soft_variables[constraint.name]
            positive[var - 1] = math.exp(-constraint.penalty)
        started = time.perf_counter()
        wmc, normalizer = self._wmc(compiled, positive, negative)
        if normalizer <= 0:
            raise UnsatisfiableError("no positive-mass assignment satisfies the grounded constraints")
        marginals = {
            name: {value: float(wmc.literal_pr(var)) for value, var in group.items()}
            for name, group in compiled.variables.items()
        }
        mpe = _max_product(
            compiled.root,
            compiled.manager.vtree(),
            positive,
            negative,
            compiled.manager.true(),
        )
        joint_map = {}
        for name, group in compiled.variables.items():
            selected = [value for value, var in group.items() if mpe.assignment[var] == 1]
            if len(selected) != 1:
                raise RuntimeError("invalid SDD MAP assignment for %s: %r" % (name, selected))
            joint_map[name] = selected[0]
        hard_positive = list(positive)
        for var in compiled.soft_variables.values():
            hard_positive[var - 1] = 1.0
        _, hard_z = self._wmc(compiled, hard_positive, negative)
        elapsed = 1000.0 * (time.perf_counter() - started)
        raw_map = {
            question.id: max(
                question.options,
                key=lambda value: potentials.values[question.id][value],
            )
            for question in program.request.questions
        }
        violations = [
            item.name
            for item in program.hard_constraints
            if not item.expr.evaluate(raw_map, program.request.state)
        ]
        marginal_map = {
            name: max(row, key=row.get) for name, row in marginals.items()
        }
        marginal_violations = [
            item.name
            for item in program.hard_constraints
            if not item.expr.evaluate(marginal_map, program.request.state)
        ]
        nodes = None
        elements = None
        try:
            nodes = int(compiled.root.count())
            elements = int(compiled.root.size())
        except (AttributeError, TypeError):
            pass
        return DecisionResult(
            marginals=marginals,
            joint_map=joint_map,
            valid_mass=hard_z,
            map_probability=mpe.weight / normalizer,
            local_potentials=potentials.values,
            inference=InferenceInfo(
                backend=self.name,
                exact=True,
                scope="finite candidate space specified by the request",
                compile_ms=0.0 if cache_hit else compiled.compile_ms,
                inference_ms=elapsed,
                world_count=math.prod(
                    len(question.options) for question in program.request.questions
                ),
                valid_world_count=int(compiled.root.global_model_count()),
                circuit_nodes=nodes,
                circuit_elements=elements,
            ),
            diagnostics={
                "program": {"name": program.name, "version": program.version},
                "joint_builder": joint.name,
                "normalizer": normalizer,
                "hard_valid_mass": hard_z,
                "raw_local_map": raw_map,
                "raw_local_map_violations": violations,
                "marginal_map": marginal_map,
                "marginal_map_violations": marginal_violations,
                "hard_constraints": [item.name for item in program.hard_constraints],
                "soft_constraints": [item.name for item in program.soft_constraints],
                "compile_cache_hit": cache_hit,
            },
        )
