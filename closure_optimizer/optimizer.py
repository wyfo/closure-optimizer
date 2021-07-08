import ast
import copy
import functools
import itertools
import sys
import typing
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    cast,
)

from closure_optimizer import constants
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


is_builtin = as_predicate(constants.BUILTINS)
Undefined = object()

Node = TypeVar("Node", bound=ast.AST)
T = TypeVar("T")


class Optimizer(ast.NodeTransformer):
    def __init__(
        self,
        globalns: Mapping[str, Any],
        captured: Dict[str, Any],
        execute: Predicate[Callable],
        inline: Predicate[Callable],
        skip: Collection[str],
    ):
        self.execute = execute
        self.inline = inline
        self.skip = set(skip)
        self._captured = captured
        self._counter = 0
        self._cached_sub_expr = IsCached()
        self._functions: Dict[str, ast.FunctionDef] = {}
        self._main_function = False
        self._namespace: Dict[str, Any] = {**globalns, **captured}
        self._replacement: Dict[str, Any] = {}
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
        if isinstance(expr, ast.Name):
            if expr.id in self._namespace:
                assert isinstance(expr.ctx, ast.Load)
                return self._namespace[expr.id]
            if expr.id in constants.BUILTIN_NAMES:
                return self._eval(expr)
        if isinstance(expr, ast.Ellipsis):
            return ...
        if isinstance(expr, CONSTANT_NODES):
            return getattr(expr, expr._fields[0])
        return Undefined

    def _visit_value(self, expr: ast.expr) -> Tuple[ast.expr, Any]:
        result = self.visit(expr)
        return result, self._get_value(result)

    def _eval(self, expr: ast.expr) -> Any:
        expr = ast.fix_missing_locations(expr)
        return eval(
            compile(ast.Expression(expr), "<ast>", mode="eval"), self._namespace, {}
        )

    @typing.overload
    def visit(self, node: ast.expr) -> ast.expr:
        ...

    @typing.overload
    def visit(self, node: ast.AST) -> NodeTransformation:
        ...

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
                return self._cache(self._eval(result))
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

    def _to_assign(
        self, target: ast.expr, value: Any, fallback: Callable[[ast.expr], Any] = None
    ) -> Iterator[Tuple[str, Any]]:
        if isinstance(target, ast.Name) and value is not Undefined:
            yield target.id, value
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
                    yield from self._to_assign(
                        elt, list(value[i : i + offset + 1]), fallback
                    )
                else:
                    yield from self._to_assign(elt, value[i + offset], fallback)
        elif fallback is not None:
            fallback(target)
        else:
            raise NotImplementedError

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

    def _assign(self, target: ast.expr, value: Any):
        for name, val in self._to_assign(target, value, self._discard):
            if name in self.skip:
                self._namespace.pop(name, ...)
            else:
                self._scopes[-1].add(name)
                self._namespace[name] = value

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

    def visit_Call(self, node: ast.Call) -> NodeTransformation:
        node.func, func = self._visit_value(node.func)
        self._cached_sub_expr &= func is not Undefined and self.execute(
            getattr(type(func.__self__), func.__name__)
            if hasattr(func, "__self__")
            else func
        )
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> NodeTransformation:
        self._cached_sub_expr &= True
        return node

    visit_Num = visit_Str = visit_Bytes = visit_NameConstant = visit_Ellipsis = visit_Constant  # type: ignore

    def visit_DictComp(self, node: ast.DictComp) -> NodeTransformation:
        key_values = self._visit_comp(
            node.generators,
            lambda: (
                self.visit(copy.deepcopy(node.key)),
                self.visit(copy.deepcopy(node.value)),
            ),
        )
        if key_values is not None:
            keys, values = zip(*key_values)
            return ast.Dict(keys=list(keys), values=list(values))
        else:
            return self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> NodeTransformation:
        return node

    def visit_For(self, node: ast.For) -> NodeTransformation:
        node.iter, value = self._visit_value(node.iter)
        if not is_pure_for_loop(node) or not isinstance(value, Collection):
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

    def _visit_comp(
        self, generators: Sequence[ast.comprehension], visit: Callable[[], T]
    ) -> Optional[Sequence[T]]:
        iters = []
        for generator in generators:
            generator.iter, value = self._visit_value(generator.iter)
            if value is Undefined:
                return None
            iters.append(value)
        elts = []
        for values in itertools.product(*iters):
            replacement_save = self._replacement.copy()
            skip_elt = False
            try:
                for generator, value in zip(generators, values):
                    self._replacement.update(
                        {
                            name: self._cache(value)
                            for name, value in self._to_assign(generator.target, value)
                        }
                    )
                    for if_expr in generator.ifs:
                        _, if_value = self._visit_value(copy.deepcopy(if_expr))
                        if if_value is Undefined:
                            return None
                        elif not if_value:
                            skip_elt = True
                            break
                    if skip_elt:
                        break
                if not skip_elt:
                    elts.append(visit())
            finally:
                self._replacement = replacement_save
        return elts

    def visit_ListComp(self, node: ast.ListComp) -> NodeTransformation:
        elts = self._visit_comp(
            node.generators, lambda: self.visit(copy.deepcopy(node.elt))
        )
        if elts is not None:
            return ast.List(elts=elts, ctx=ast.Load())
        else:
            return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> NodeTransformation:
        if isinstance(node.ctx, ast.Load):
            if node.id in self._replacement:
                self._cached_sub_expr &= True
                return self._replacement[node.id]
            self._cached_sub_expr &= (
                node.id in self._namespace or node.id in constants.BUILTIN_NAMES
            )
        return node

    def visit_SetComp(self, node: ast.SetComp) -> NodeTransformation:
        elts = self._visit_comp(
            node.generators, lambda: self.visit(copy.deepcopy(node.elt))
        )
        if elts is not None:
            return ast.Set(elts=elts)
        else:
            return self.generic_visit(node)


Func = TypeVar("Func", bound=Callable)


def optimize(
    func: Func,
    *,
    execute: IterableOrPredicate[Callable] = (),
    inline: IterableOrPredicate[Callable] = (),
    skip: Collection[str] = (),
) -> Func:
    if isinstance(func, functools.partial):
        partial_args = map_parameters(func.func, func.args, func.keywords, partial=True)
        orig_func, func = func, func.func  # type: ignore
    else:
        orig_func, partial_args = None, {}  # type: ignore
    captured = get_captured(func)
    captured.update(partial_args)
    inline_pred = as_predicate(inline)
    execute_pred = as_predicate(execute)

    def inline_guard(f: Callable) -> bool:
        if isinstance(f, functools.partial):
            f = f.func
        return f != func and inline_pred(f)

    def builtin_or_exec(f: Callable):
        return is_builtin(f) or execute_pred(f)

    optimizer = Optimizer(
        func.__globals__, captured, builtin_or_exec, inline_guard, skip  # type: ignore
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


def is_optimized(func: Callable) -> bool:
    return hasattr(func, constants.OPTIMIZED_AST_ATTR)
