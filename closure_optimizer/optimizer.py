import ast
import copy
import functools
import inspect
import itertools
import sys
import warnings
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
    Union,
    cast,
    overload,
)

from closure_optimizer.constants import (
    BUILTINS,
    BUILTIN_NAMES,
    CONSTANT_NODES,
    CONSTANT_NODE_BY_TYPE,
    PREFIX,
)
from closure_optimizer.utils import (
    IterableOrPredicate,
    METADATA_ATTR,
    NameGenerator,
    NodeTransformation,
    Predicate,
    as_predicate,
    ast_parameters,
    flatten,
    get_captured,
    get_function_ast,
    is_inlinable,
    is_unrollable,
    map_parameters,
    rename,
    replace_return,
)


class NotLoad:
    def __bool__(self):
        return False

    def __and__(self, other) -> bool:
        return bool(other)


is_builtin = as_predicate(BUILTINS)
Undefined = object()

Node = TypeVar("Node", bound=ast.AST)
T = TypeVar("T")
Method = TypeVar("Method", bound=Callable[["Optimizer", ast.AST], NodeTransformation])


def assignments(
    target: ast.expr, value: Any, fallback: Callable[[ast.expr], Any] = None
) -> Iterator[Tuple[str, Any]]:
    if isinstance(target, ast.Name) and value is not Undefined:
        yield target.id, value
    elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, Collection):
        if not isinstance(value, Sequence):
            value = list(value)
        offset = 0
        for i, elt in enumerate(target.elts):
            if isinstance(elt, ast.Starred):
                assert offset == 0
                offset = len(value) - len(target.elts)
                yield from assignments(
                    elt.value, list(value[i : i + offset + 1]), fallback
                )
            else:
                yield from assignments(elt, value[i + offset], fallback)
    elif fallback is not None:
        fallback(target)
    else:
        raise NotImplementedError


Comp = TypeVar("Comp", bound=Union[ast.DictComp, ast.ListComp, ast.SetComp])


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
        self.name_generator = NameGenerator(PREFIX)
        self.skip = set(skip)
        self._captured = captured
        self._counter = 0
        self._cached_sub_expr: Union[bool, NotLoad] = NotLoad()
        self._functions: Dict[str, ast.FunctionDef] = {}
        self._inlining: List[ast.stmt] = []
        self._inline_guard: Set[str] = set()
        self._main_function = False
        self._namespace: Dict[str, Any] = {**globalns, **captured}
        self._replacement: Dict[str, Any] = {}
        self._scopes: List[Set[str]] = []
        for exc in skip:
            self._namespace.pop(exc, ...)

    def _cache(self, value: Any) -> ast.expr:
        if value.__class__ in CONSTANT_NODE_BY_TYPE:
            return CONSTANT_NODE_BY_TYPE[value.__class__](value)
        else:
            name = self.name_generator()
            self._captured[name] = value
            self._namespace[name] = value
            return ast.Name(id=name, ctx=ast.Load())

    def _get_value(self, expr: ast.expr, as_boolean=False) -> Any:
        if isinstance(expr, ast.Name):
            if expr.id in self._namespace:
                assert isinstance(expr.ctx, ast.Load)
                return self._namespace[expr.id]
            if expr.id in BUILTIN_NAMES:
                return self._eval(expr)
        if isinstance(expr, ast.Ellipsis):
            return ...
        if isinstance(expr, CONSTANT_NODES):
            return getattr(expr, expr._fields[0])
        if as_boolean:
            if isinstance(expr, ast.BoolOp):
                for operand in expr.values:
                    value = self._get_value(operand, as_boolean=True)
                    if value is not Undefined:
                        if (isinstance(expr.op, ast.And) and not value) or (
                            isinstance(expr.op, ast.Or) and value
                        ):
                            return bool(value)
                    elif not isinstance(operand, ast.Name):
                        break
            if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
                value = self._get_value(expr.operand, as_boolean=True)
                if value is not Undefined:
                    return not value
        return Undefined

    def _visit_value(self, expr: ast.expr, as_boolean=False) -> Tuple[ast.expr, Any]:
        result = self.visit(expr)
        return result, self._get_value(result, as_boolean)

    def _eval(self, expr: ast.expr) -> Any:
        expr = ast.fix_missing_locations(expr)
        return eval(
            compile(ast.Expression(expr), "<ast>", mode="eval"), self._namespace, {}
        )

    @overload
    def visit(self, node: ast.expr) -> ast.expr:
        ...

    @overload
    def visit(self, node: ast.keyword) -> ast.keyword:
        ...

    @overload
    def visit(self, node: ast.AST) -> NodeTransformation:
        ...

    def visit(self, node: ast.AST) -> NodeTransformation:
        if isinstance(node, ast.expr):
            cached_expr = self._cached_sub_expr
            self._cached_sub_expr = NotLoad()
            try:
                result = super().visit(node)
                cached_expr &= bool(self._cached_sub_expr)
                return result
            finally:
                self._cached_sub_expr = cached_expr
        elif isinstance(node, ast.stmt):
            inlining = self._inlining
            self._inlining = []
            try:
                result = super().visit(node)
                return (*self._inlining, *flatten(result)) if self._inlining else result
            finally:
                self._inlining = inlining
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
                    self._functions.pop(name, ...)
        return super().generic_visit(node)

    def _visit_and_flatten(self, nodes: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
        for node in nodes:
            yield from cast(Iterable[ast.stmt], flatten(self.visit(node)))

    def _discard(self, target: ast.expr):
        if isinstance(target, ast.Name):
            self._namespace.pop(target.id, ...)
            self._functions.pop(target.id, ...)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._discard(elt)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            self._discard(target.value)
        else:
            raise NotImplementedError

    def _assign(self, target: ast.expr, value: Any):
        for name, val in assignments(target, value, self._discard):
            if name in self.skip or val.__class__ not in CONSTANT_NODE_BY_TYPE:
                self._namespace.pop(name, ...)
            else:
                self._scopes[-1].add(name)
                self._namespace[name] = value
            self._functions.pop(name, ...)

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

    def visit_BoolOp(self, node: ast.BoolOp) -> NodeTransformation:
        values = enumerate(node.values.copy())
        for i, operand in values:
            node.values[i], value = self._visit_value(operand)
            if value is not Undefined:
                if (isinstance(node.op, ast.And) and not value) or (
                    isinstance(node.op, ast.Or) and value
                ):
                    return node.values[i]
            else:
                break
        for i, operand in values:
            node.values[i] = self.visit(operand)
        return node

    def _inline(
        self,
        node: ast.Call,
        func_ast: ast.FunctionDef,
        parameters: Iterable[inspect.Parameter],
        args: List[ast.expr] = None,
        kwargs: Dict[str, ast.expr] = None,
    ) -> Optional[ast.Name]:
        args, kwargs = args or [], kwargs or {}
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                arg_value = self._get_value(arg.value)
                if arg_value is Undefined or not isinstance(arg_value, Collection):
                    return None
                args.extend(map(self._cache, arg_value))
            else:
                arg_value = self._get_value(arg)
                args.append(arg if arg_value is Undefined else self._cache(arg_value))
        for kw in node.keywords:
            kw_value = self._get_value(kw.value)
            if kw.arg is None:
                if kw_value is Undefined or not isinstance(kw_value, Mapping):
                    return None
                kwargs.update((k, self._cache(v)) for k, v in kw_value.items())
            elif kw_value is not Undefined:
                kwargs[kw.arg] = self._cache([kw_value])
            else:
                kwargs[kw.arg] = kw.value
        args_mapping = map_parameters(parameters, args, kwargs)
        return_name = self.name_generator()
        self._inlining.extend(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=arg_node)
            for name, arg_node in args_mapping.items()
        )
        self._inlining.extend(
            replace_return(stmt, return_name) for stmt in func_ast.body
        )
        return ast.Name(id=return_name, ctx=ast.Load())

    def visit_Call(self, node: ast.Call) -> NodeTransformation:
        node.func, func = self._visit_value(node.func)
        # Use list(map(...)) instead of only map because of pypy issue
        # https://foss.heptapod.net/pypy/pypy/-/issues/3440
        node.args[:] = list(map(self.visit, node.args))
        node.keywords[:] = list(map(self.visit, node.keywords))
        inlined = None
        if func is not Undefined:
            args: List[ast.expr] = []
            kwargs: Dict[str, ast.expr] = {}
            if isinstance(func, functools.partial):
                args.extend(map(self._cache, func.args))
                kwargs.update((k, self._cache(v)) for k, v in func.keywords.items())
                func = func.func
            if hasattr(func, "__self__"):
                func_or_method = getattr(type(func.__self__), func.__name__)
            else:
                func_or_method = func
            if self._cached_sub_expr and self.execute(func_or_method):
                return self._cache(self._eval(node))
            else:
                self._cached_sub_expr &= False
            if self.inline(func_or_method):
                try:
                    func_ast = get_function_ast(func)
                    if is_inlinable(func_ast):
                        func_namespace = {**func.__globals__, **get_captured(func)}
                        rename_mapping = rename(func_ast, self.name_generator)
                        renamed_namespace = {
                            renamed: func_namespace[name]
                            for name, renamed in rename_mapping.items()
                            if name in func_namespace
                        }
                        self._namespace.update(renamed_namespace)
                        self._captured.update(renamed_namespace)
                        parameters = [
                            param.replace(
                                name=rename_mapping[param.name],
                                default=self._cache(param.default)
                                if param.default is not inspect.Parameter.empty
                                else inspect.Parameter.empty,
                            )
                            for param in inspect.signature(func).parameters.values()
                        ]
                        inlined = self._inline(node, func_ast, parameters, args, kwargs)
                    else:
                        raise ValueError
                except Exception:
                    warnings.warn(f"{func} cannot be inlined", UserWarning)
        elif isinstance(node.func, ast.Name) and node.func.id in self._functions:
            func_ast = copy.deepcopy(self._functions[node.func.id])
            rename(func_ast, self.name_generator, only_declared=True)
            inlined = self._inline(node, func_ast, ast_parameters(func_ast))
        return inlined if inlined is not None else node

    def visit_Constant(self, node: ast.Constant) -> NodeTransformation:
        self._cached_sub_expr &= True
        return node

    visit_Num = visit_Str = visit_Bytes = visit_NameConstant = visit_Ellipsis = visit_Constant  # type: ignore

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
                            for name, value in assignments(generator.target, value)
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
            except Exception:
                return None
            finally:
                self._replacement = replacement_save
        return elts

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
        if not is_unrollable(node) or not isinstance(value, Collection):
            return self.generic_visit(node)
        for elt in value:
            yield from self._visit_and_flatten(
                [ast.Assign(targets=[node.target], value=self._cache(elt))]
            )
            yield from self._visit_and_flatten(copy.deepcopy(node.body))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> NodeTransformation:
        if not self._main_function:
            self._main_function = True
            return self.generic_visit(node)
        else:
            self._functions[node.name] = node
            self._namespace.pop(node.name, ...)
            return node

    def visit_If(self, node: ast.If) -> NodeTransformation:
        node.test, value = self._visit_value(node.test, as_boolean=True)
        if value is Undefined:
            return self.generic_visit(node)
        elif value:
            return self._visit_and_flatten(node.body)
        else:
            return self._visit_and_flatten(node.orelse)

    def visit_IfExp(self, node: ast.IfExp) -> NodeTransformation:
        node.test, value = self._visit_value(node.test, as_boolean=True)
        if value is Undefined:
            return self.generic_visit(node)
        elif value:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Lambda(self, node: ast.Lambda) -> NodeTransformation:
        return node

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
            if node.id in self._namespace:
                self._cached_sub_expr &= True
                value = self._namespace[node.id]
                if value.__class__ in CONSTANT_NODE_BY_TYPE:
                    return CONSTANT_NODE_BY_TYPE[value.__class__](value)
            else:
                self._cached_sub_expr &= node.id in BUILTIN_NAMES
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
        params = inspect.signature(func.func).parameters.values()
        partial_args = map_parameters(params, func.args, func.keywords, partial=True)
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
    setattr(optimized_func, METADATA_ATTR, optimized_ast)
    result = functools.wraps(func)(optimized_func)
    if partial_args:
        result = functools.partial(result, *orig_func.args, **orig_func.keywords)
    return cast(Func, result)


def is_optimized(func: Callable) -> bool:
    return hasattr(func, METADATA_ATTR)
