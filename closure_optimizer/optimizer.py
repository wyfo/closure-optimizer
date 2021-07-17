import ast
import contextlib
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
    Undefined,
    as_predicate,
    assigned_names,
    assignments,
    ast_parameters,
    flatten,
    get_captured,
    get_function_ast,
    get_skipped,
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

Node = TypeVar("Node", bound=ast.AST)
T = TypeVar("T")
Method = TypeVar("Method", bound=Callable[["Optimizer", ast.AST], NodeTransformation])

Comprehension = Union[ast.DictComp, ast.ListComp, ast.SetComp]
Comp = TypeVar("Comp", bound=Comprehension)


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
        self._enable_inlining = True
        self._functions: Dict[str, ast.FunctionDef] = {}
        self._inlined: List[ast.stmt] = []
        self._inline_guard: Set[str] = set()
        self._main_function = False
        self._namespace: Dict[str, Any] = {**globalns, **captured}
        self._substitution: Dict[str, Any] = {}
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
            elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
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

    def _add_inlined(self, stmts: Iterable[ast.stmt]):
        self._inlined.extend(self._visit_and_flatten(stmts))

    @overload
    def visit(self, node: ast.expr) -> ast.expr:
        ...

    @overload
    def visit(self, node: ast.keyword) -> ast.keyword:
        ...

    @overload
    def visit(self, node: ast.arguments) -> ast.arguments:
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
            inlined = self._inlined
            self._inlined = []
            try:
                result = super().visit(node)
                return (*self._inlined, *flatten(result)) if self._inlined else result
            finally:
                self._inlined = inlined
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

    def _visit_and_flatten(self, stmts: Iterable[ast.stmt]) -> Iterable[ast.stmt]:
        for node in stmts:
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
            if name in self.skip:
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

    @contextlib.contextmanager
    def _disable_inlining(self):
        enable = self._enable_inlining
        self._enable_inlining = False
        try:
            yield
        finally:
            self._enable_inlining = enable

    def _inline(
        self,
        func: Any,
        node: ast.Call,
        func_ast: ast.FunctionDef,
        parameters: Iterable[inspect.Parameter],
        args: List[ast.expr] = None,
        kwargs: Dict[str, ast.expr] = None,
    ) -> Optional[ast.Name]:
        if not self._enable_inlining or func in self._inline_guard:
            return None
        self._inline_guard.add(func)
        try:
            args, kwargs = args or [], kwargs or {}
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    arg_value = self._get_value(arg.value)
                    if arg_value is Undefined or not isinstance(arg_value, Collection):
                        return None
                    args.extend(map(self._cache, arg_value))
                else:
                    arg_value = self._get_value(arg)
                    args.append(
                        arg if arg_value is Undefined else self._cache(arg_value)
                    )
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
            self._add_inlined(
                ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=arg_node)
                for name, arg_node in args_mapping.items()
            )
            self._add_inlined(
                replace_return(stmt, return_name) for stmt in func_ast.body
            )
            return ast.Name(id=return_name, ctx=ast.Load())
        finally:
            self._inline_guard.remove(func)

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
                self._cached_sub_expr = False
            if self.inline(func_or_method):
                func_ast = copy.deepcopy(get_function_ast(func))
                if is_inlinable(func_ast):
                    func_namespace = {
                        **func.__globals__,
                        **get_captured(func_or_method),
                    }
                    rename_mapping = rename(func_ast, self.name_generator.replace)
                    skipped = get_skipped(func_or_method)
                    renamed_namespace = {
                        renamed: func_namespace[name]
                        for name, renamed in rename_mapping.items()
                        if name in func_namespace and name not in skipped
                    }
                    self.skip.update(rename_mapping[name] for name in skipped)
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
                    inlined = self._inline(
                        func_or_method, node, func_ast, parameters, args, kwargs
                    )
                else:
                    warnings.warn(f"{func} cannot be inlined", UserWarning)
        elif isinstance(node.func, ast.Name) and node.func.id in self._functions:
            local_func = self._functions[node.func.id]
            func_ast = copy.deepcopy(local_func)
            rename(func_ast, self.name_generator.replace, only_declared=True)
            inlined = self._inline(local_func, node, func_ast, ast_parameters(func_ast))
        return inlined if inlined is not None else node

    def visit_Constant(self, node: ast.Constant) -> NodeTransformation:
        self._cached_sub_expr &= True
        return node

    visit_Num = visit_Str = visit_Bytes = visit_NameConstant = visit_Ellipsis = visit_Constant  # type: ignore

    def visit_Dict(self, node: ast.Dict) -> NodeTransformation:
        self._cached_sub_expr = False
        return self.generic_visit(node)

    def _visit_comprehension(
        self,
        node: Comp,
        visit: Callable[[Comp], T],
        flatten: Callable[[Sequence[T]], ast.expr],
        insert: Callable[[Comp, ast.Name], ast.stmt],
    ) -> Any:
        self._cached_sub_expr = False
        # Rename targets in order to not mess the scopes up
        for i, generator in enumerate(node.generators):
            names = assigned_names(generator.target)
            if names is None:
                return node
            renaming = {name: self.name_generator() for name in names}
            rename(generator.target, renaming)
            for if_expr in generator.ifs:
                rename(if_expr, renaming)
            for gen in node.generators[i + 1 :]:
                rename(gen, renaming)
            for attr in ["elt", "key", "value"]:
                if hasattr(node, attr):
                    rename(getattr(node, attr), renaming)
        iter_values = []
        for generator in node.generators:
            generator.iter, value = self._visit_value(generator.iter)
            iter_values.append(value)
        if Undefined not in iter_values:
            flattened, renaming = [], {}
            for values in itertools.product(*iter_values):
                keep_elt: Optional[bool] = True
                substitution = self._substitution.copy()
                try:
                    for generator, value in zip(node.generators, values):
                        self._substitution.update(
                            (name, self._cache(value))
                            # assignments can't raise because assigned_names above
                            for name, value in assignments(generator.target, value)
                        )
                        for if_expr in generator.ifs:
                            with self._disable_inlining():
                                _, if_value = self._visit_value(copy.deepcopy(if_expr))
                            if if_value is Undefined:
                                keep_elt = None
                                break
                            elif not if_value:
                                keep_elt = False
                                break
                        if not keep_elt:
                            break
                    if keep_elt:
                        flattened.append(visit(node))
                    elif keep_elt is None:
                        break
                finally:
                    self._substitution = substitution
            else:
                return flatten(flattened)
        name = self.name_generator()
        variable = ast.Name(id=name, ctx=ast.Load())
        stmts: List[ast.stmt] = [
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=flatten([]))
        ]
        body = stmts
        for generator in node.generators:
            ast_for = ast.For(
                target=generator.target,
                iter=generator.iter,
                body=[],
                orelse=[],
                type_comment=None,
            )
            body.append(ast_for)
            body = ast_for.body
            for if_expr in generator.ifs:
                ast_if = ast.If(test=if_expr, body=[], orelse=[])
                body.append(ast_if)
                body = ast_if.body
        body.append(insert(node, variable))
        self._add_inlined(stmts)
        return variable

    def visit_DictComp(self, node: ast.DictComp) -> NodeTransformation:
        def flatten(elts: Sequence[Tuple[ast.expr, ast.expr]]) -> ast.Dict:
            keys, values = zip(*elts) if elts else ((), ())
            return ast.Dict(keys=list(keys), values=list(values))

        return self._visit_comprehension(
            node,
            lambda node: (
                self.visit(copy.deepcopy(node.key)),
                self.visit(copy.deepcopy(node.value)),
            ),
            flatten,
            lambda node, name: ast.Assign(
                targets=[ast.Subscript(value=name, slice=node.key, ctx=ast.Store())],
                value=node.value,
            ),
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> NodeTransformation:
        return node

    def visit_For(self, node: ast.For) -> NodeTransformation:
        node.iter, value = self._visit_value(node.iter)
        if not is_unrollable(node) or not isinstance(value, Collection):
            return self.generic_visit(node)
        stmts: List[ast.stmt] = []
        for elt in value:
            stmts.append(ast.Assign(targets=[node.target], value=self._cache(elt)))
            stmts.extend(copy.deepcopy(node.body))
        return self._visit_and_flatten(stmts)

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
        node.args = self.visit(node.args)
        return node

    def visit_List(self, node: ast.List) -> NodeTransformation:
        self._cached_sub_expr = False
        return self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> NodeTransformation:
        return self._visit_comprehension(
            node,
            lambda node: self.visit(copy.deepcopy(node.elt)),
            lambda elts: ast.List(elts=elts, ctx=ast.Load()),
            lambda node, name: ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=name, attr="append", ctx=ast.Load()),
                    args=[node.elt],
                    keywords=[],
                )
            ),
        )

    def visit_Name(self, node: ast.Name) -> NodeTransformation:
        if isinstance(node.ctx, ast.Load):
            if node.id in self._substitution:
                self._cached_sub_expr &= True
                return self._substitution[node.id]
            elif node.id in self._namespace:
                self._cached_sub_expr &= True
                value = self._namespace[node.id]
                if value.__class__ in CONSTANT_NODE_BY_TYPE:
                    return CONSTANT_NODE_BY_TYPE[value.__class__](value)
            else:
                self._cached_sub_expr &= node.id in BUILTIN_NAMES
        return node

    def visit_Set(self, node: ast.Set) -> NodeTransformation:
        self._cached_sub_expr = False
        if not node.elts:
            return ast.Call(
                func=ast.Name(id="set", ctx=ast.Load()), args=[], keywords=[]
            )
        return self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> NodeTransformation:
        return self._visit_comprehension(
            node,
            lambda node: self.visit(copy.deepcopy(node.elt)),
            lambda elts: ast.Set(elts=elts, ctx=ast.Load()),
            lambda node, name: ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(value=name, attr="add", ctx=ast.Load()),
                    args=[node.elt],
                    keywords=[],
                )
            ),
        )

    def visit_Tuple(self, node: ast.Tuple) -> NodeTransformation:
        self._cached_sub_expr &= True
        return self.generic_visit(node)


Func = TypeVar("Func", bound=Callable)


def is_optimized(func: Callable) -> bool:
    return hasattr(func, METADATA_ATTR)


def optimize(
    func: Func,
    *,
    execute: IterableOrPredicate[Callable] = (),
    inline: IterableOrPredicate[Callable] = (),
    inline_optimized: bool = True,
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
        return f != func and (inline_pred(f) or (inline_optimized and is_optimized(f)))

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
    setattr(optimized_func, METADATA_ATTR, (optimized_ast, optimizer.skip))
    result = functools.wraps(func)(optimized_func)
    if partial_args:
        result = functools.partial(result, *orig_func.args, **orig_func.keywords)
    return cast(Func, result)
