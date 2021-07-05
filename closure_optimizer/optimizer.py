import ast
import contextlib
import copy
import functools
import sys
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    cast,
)

from closure_optimizer import constants, settings
from closure_optimizer.utils import (
    CONSTANT_NODES,
    CONSTANT_NODE_BY_TYPE,
    IterableOrPredicate,
    NodeTransformation,
    Predicate,
    as_predicate,
    get_captured,
    get_function_ast,
    is_pure_for_loop,
    map_parameters,
)


class IsCached:
    def __init__(self):
        self._is_cached: Optional[bool] = None

    def __bool__(self):
        return bool(self._is_cached)

    def __iand__(self, other):
        if self._is_cached is None:
            self._is_cached = True
        self._is_cached &= bool(other)
        return self


Undefined = object()


class Optimizer(ast.NodeTransformer):
    def __init__(
        self,
        globalns: Mapping[str, Any],
        captured: Dict[str, Any],
        skip: Collection[str],
        inline: Predicate[Callable],
    ):
        self.inline = inline
        self._captured = captured
        self._skipped = set(skip)
        self._counter = 0
        self._cached_sub_expr = IsCached()
        self._functions: Dict[str, ast.FunctionDef] = {}
        self._main_function = False
        self._namespace: Dict[str, Any] = {**globalns, **captured}
        self._scopes: List[Set[str]] = []
        for exc in skip:
            self._namespace.pop(exc, ...)

    def _cache(self, value: Any) -> ast.expr:
        try:
            return CONSTANT_NODE_BY_TYPE[value.__class__](value)
        except KeyError:
            self._counter += 1
            name = constants.PREFIX + str(self._counter)
            self._captured[name] = value
            self._namespace[name] = value
            return ast.Name(id=name, ctx=ast.Load())

    def _get_value(self, expr: ast.expr) -> Any:
        if isinstance(expr, ast.Name) and expr.id in self._namespace:
            assert isinstance(expr.ctx, ast.Load)
            return self._namespace[expr.id]
        elif isinstance(expr, ast.Ellipsis):
            return ...
        elif isinstance(expr, CONSTANT_NODES):
            return getattr(expr, expr._fields[0])
        else:
            return Undefined

    def _visit_value(self, expr: ast.expr) -> Tuple[ast.expr, Any]:
        result = self.visit(expr)
        assert isinstance(result, ast.expr)
        return result, self._get_value(result)

    def _eval(self, expr: ast.expr) -> Any:
        expr = ast.fix_missing_locations(expr)
        return eval(
            compile(ast.Expression(expr), "<ast>", mode="eval"), self._namespace, {}
        )

    def visit(self, node: ast.AST) -> NodeTransformation:
        if isinstance(node, ast.expr):
            cached_expr = self._cached_sub_expr
            self._cached_sub_expr = IsCached()
            try:
                result = super().visit(node)
                cached_expr &= self._cached_sub_expr
                return result
            finally:
                self._cached_sub_expr = cached_expr
        else:
            return super().visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.expr):
            result = super().generic_visit(node)
            assert isinstance(result, ast.expr)
            if self._cached_sub_expr:
                with contextlib.suppress(Exception):
                    value = self._eval(result)
                    if not as_predicate(settings.skipped_functions)(value):
                        return self._cache(value)
            return result
        if isinstance(node, ast.stmt):
            self._scopes.append(set())
            try:
                return super().generic_visit(node)
            finally:
                for name in self._scopes.pop():
                    self._namespace.pop(name, ...)
        else:
            return super().generic_visit(node)

    def _visit_and_flatten(self, nodes: Iterable[ast.AST]) -> Iterable[ast.AST]:
        for node in nodes:
            result = self.visit(node)
            if isinstance(result, ast.AST):
                yield result
            elif result is not None:
                yield from result

    def _discard(self, target: ast.expr):
        if isinstance(target, ast.Name):
            self._namespace.pop(target.id, ...)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._discard(elt)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            self._discard(target.value)
        else:
            raise NotImplementedError

    def _assign(
        self, target: ast.expr, value: Any, fallback: Callable[[ast.expr], Any] = None
    ):
        if fallback is None:
            fallback = self._discard
        if isinstance(target, ast.Name) and value is not Undefined:
            self._scopes[-1].add(target.id)
            self._namespace[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value, Collection
        ):
            if not isinstance(value, Sequence):
                value = list(value)
            offset = 0
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Starred):
                    assert offset == 0
                    offset = len(value) - len(target.elts)
                    self._assign(elt, list(value[i : i + offset + 1]))
                else:
                    self._assign(elt, value[i + offset])
        else:
            fallback(target)

    def visit_Assign(self, node: ast.Assign) -> NodeTransformation:
        node.value, value = self._visit_value(node.value)
        for target in node.targets:
            self._assign(target, value)
        return node  # no need to call generic_visit because node.value is visited

    def visit_AnnAssign(self, node: ast.AnnAssign) -> NodeTransformation:
        if not hasattr(node, "value") or node.value is None:
            return node
        node.value, value = self._visit_value(node.value)
        self._assign(node.target, value)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> NodeTransformation:
        self._discard(node.target)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> NodeTransformation:
        self._cached_sub_expr &= True
        return node

    visit_Num = visit_Str = visit_Bytes = visit_NameConstant = visit_Ellipsis = visit_Constant  # type: ignore

    def visit_For(self, node: ast.For) -> NodeTransformation:
        node.iter, value = self._visit_value(node.iter)
        if (
            not is_pure_for_loop(node)
            or not isinstance(value, Collection)
            or len(value) > settings.loop_enrolling_limit
        ):
            return self.generic_visit(node)
        for elt in value:
            yield from self._visit_and_flatten(
                [ast.Assign(targets=[node.target], value=self._cache(elt))]
            )
            yield from self._visit_and_flatten(copy.deepcopy(node.body))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> NodeTransformation:
        if not self._main_function:
            self._main_function = True
        else:
            self._functions[node.name] = node
            self._namespace.pop(node.name, ...)
        return self.generic_visit(node)

    def visit_If(self, node: ast.If) -> NodeTransformation:
        node.test, value = self._visit_value(node.test)
        if value is Undefined:
            return self.generic_visit(node)
        elif value:
            return self._visit_and_flatten(node.body)
        else:
            return self._visit_and_flatten(node.orelse)

    def visit_IfExp(self, node: ast.IfExp) -> NodeTransformation:
        node.test, value = self._visit_value(node.test)
        if value is Undefined:
            return self.generic_visit(node)
        elif value:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Lambda(self, node: ast.Lambda) -> NodeTransformation:
        return node

    def visit_Name(self, node: ast.Name) -> NodeTransformation:
        if isinstance(node.ctx, ast.Load):
            self._cached_sub_expr &= (
                node.id in self._namespace or node.id in constants.BUILTINS
            )
        return node


Func = TypeVar("Func", bound=Callable)


def optimize(
    func: Func,
    *,
    skip: Collection[str] = (),
    inline: IterableOrPredicate[Callable] = (),
) -> Func:
    if isinstance(func, functools.partial):
        partial_args = map_parameters(func.func, func.args, func.keywords, partial=True)
        orig_func, func = func, func.func  # type: ignore
    else:
        orig_func, partial_args = None, {}  # type: ignore

    captured = get_captured(func)
    captured.update(partial_args)
    optimizer = Optimizer(
        func.__globals__,  # type: ignore
        captured,
        skip,
        lambda f, pred=as_predicate(inline): f != func and pred(f),  # type: ignore
    )
    optimized_ast = optimizer.visit(get_function_ast(func))
    for key in partial_args:
        del captured[key]
    # Generating a function which instantiate a closure is the only way I've found to
    # to add values to the closure captured namespace.
    factory = ast.Module(
        body=[
            ast.FunctionDef(
                name="factory",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="captured")],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=[
                    *(
                        ast.Assign(
                            targets=[ast.Name(id=var, ctx=ast.Store())],
                            value=ast.Subscript(
                                value=ast.Name(id="captured", ctx=ast.Load()),
                                slice=(
                                    ast.Index(ast.Constant(value=var))
                                    if sys.version_info < (3, 9)
                                    else ast.Constant(value=var)
                                ),
                                ctx=ast.Load(),
                            ),
                        )
                        for var in captured
                    ),
                    optimized_ast,
                    ast.Return(value=ast.Name(id=func.__name__, ctx=ast.Load())),
                ],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(factory)
    compiled = compile(factory, "<ast>", "exec")
    localns: dict = {}
    exec(compiled, func.__globals__, localns)  # type: ignore
    optimized_func = localns["factory"](captured)
    setattr(optimized_func, constants.OPTIMIZED_AST_ATTR, optimized_ast)
    result = functools.wraps(func)(optimized_func)
    if partial_args:
        result = functools.partial(result, *orig_func.args, **orig_func.keywords)
    return cast(Func, result)
