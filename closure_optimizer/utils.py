import ast
import inspect
import sys
import textwrap
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from closure_optimizer.constants import BUILTIN_NAMES, PREFIX

METADATA_ATTR = f"{PREFIX}ast"

T = TypeVar("T")


def get_function_ast(obj: Callable) -> ast.FunctionDef:
    if obj.__name__ == "<lambda>":
        raise ValueError("Lambda are not supported")
    if hasattr(obj, METADATA_ATTR):
        return getattr(obj, METADATA_ATTR)[0]
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__  # type: ignore
    node = ast.parse(textwrap.dedent(inspect.getsource(obj))).body[0]
    if not isinstance(node, ast.FunctionDef):
        raise ValueError(f"Unsupported object {obj}")
    node.decorator_list = []
    return node


def get_captured(func: Callable) -> Dict[str, Any]:
    cells = [cell.cell_contents for cell in func.__closure__ or ()]  # type: ignore
    return dict(zip(func.__code__.co_freevars, cells))


def get_skipped(func: Callable) -> Collection[str]:
    return getattr(func, METADATA_ATTR, (..., ()))[1]


def map_parameters(
    parameters: Iterable[inspect.Parameter],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    partial: bool = False,
) -> Mapping[str, Any]:
    offset = 0
    kwargs = dict(kwargs)
    result: Dict[str, Any] = {}
    for param in parameters:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            if not partial:
                result[param.name] = tuple(args[offset:])
            offset = len(args)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            if not partial:
                result[param.name] = kwargs
            assert offset == len(args)
            return result
        elif offset < len(args) and param.kind != inspect.Parameter.KEYWORD_ONLY:
            result[param.name] = args[offset]
            offset += 1
        elif param.name in kwargs and param.kind != inspect.Parameter.POSITIONAL_ONLY:
            result[param.name] = kwargs.pop(param.name)
        elif partial:
            pass
        else:
            assert param.default is not inspect.Parameter.empty
            result[param.name] = param.default
    assert offset == len(args) and not kwargs
    return result


def ast_parameters(node: ast.FunctionDef) -> Iterable[inspect.Parameter]:
    posonly_args = node.args.posonlyargs if sys.version_info >= (3, 8) else []
    for arg, kind, default in zip(
        posonly_args + node.args.args,
        [inspect.Parameter.POSITIONAL_ONLY] * len(posonly_args)
        + [inspect.Parameter.POSITIONAL_OR_KEYWORD] * len(node.args.args),  # type: ignore
        [inspect.Parameter.empty]
        * (len(posonly_args) + len(node.args.args) - len(node.args.defaults))
        + node.args.defaults,
    ):
        yield inspect.Parameter(arg.arg, kind, default=default)
    if node.args.vararg is not None:
        yield inspect.Parameter(node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        yield inspect.Parameter(
            arg.arg,
            inspect.Parameter.KEYWORD_ONLY,
            default=default or inspect.Parameter.empty,
        )
    if node.args.kwarg is not None:
        yield inspect.Parameter(node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD)


Predicate = Callable[[T], bool]
IterableOrPredicate = Union[Collection[T], Predicate[T]]


def as_predicate(iter_or_pred: IterableOrPredicate[T]) -> Predicate[T]:
    predicate = iter_or_pred if callable(iter_or_pred) else iter_or_pred.__contains__

    def wrap(arg: T) -> bool:
        try:
            return predicate(arg)  # type: ignore
        except Exception:
            return False

    return wrap


class NotUnrollable(Exception):
    pass


class UnrollableChecker(ast.NodeVisitor):
    def visit_Break(self, node: ast.Break):
        raise NotUnrollable

    def visit_Continue(self, node: ast.Continue):
        raise NotUnrollable

    def visit_For(self, node: ast.For):
        if node.orelse:
            raise NotUnrollable
        self.generic_visit(node)


def is_unrollable(node: ast.For) -> bool:
    try:
        UnrollableChecker().visit(node)
    except NotUnrollable:
        return False
    else:
        return True


NodeTransformation = Union[ast.AST, Iterable[ast.AST], None]


def flatten(node: NodeTransformation) -> Iterable[ast.AST]:
    if isinstance(node, ast.AST):
        yield node
    elif node is not None:
        yield from node


class NameGenerator:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self._counter = 0

    def __call__(self):
        self._counter += 1
        return f"{self.prefix}{self._counter}"

    def replace(self, _: str) -> str:
        return self()


class Renamer(ast.NodeVisitor):
    def __init__(self, renamer: Callable[[str], str], only_declared: bool):
        self.generate_name = renamer
        self.only_declared = only_declared
        self._declared: Set[str] = set()
        self._mapping: Dict[str, str] = {}

    def _rename(self, name: str) -> str:
        if name not in self._mapping:
            self._mapping[name] = self.generate_name(name)
        return self._mapping[name]

    def visit_arg(self, node: ast.arg):
        node.arg = self._rename(node.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node.name = self._rename(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = self._rename(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.name = self._rename(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        raise NotImplementedError

    def visit_Nonlocal(self, node: ast.Nonlocal):
        raise NotImplementedError

    def visit_Name(self, node: ast.Name):
        if node.id in BUILTIN_NAMES:
            return
        if not self.only_declared:
            node.id = self._rename(node.id)
        elif isinstance(node.ctx, ast.Store):
            node.id = self._rename(node.id)
        elif node.id in self._mapping:
            node.id = self._mapping[node.id]
        self.generic_visit(node)


def rename(
    node: ast.AST,
    renaming: Union[Callable[[str], str], Mapping[str, str]],
    *,
    only_declared: bool = False,
) -> Mapping[str, str]:
    if not callable(renaming):
        mapping = renaming

        def renaming(name: str) -> str:
            try:
                return mapping[name]
            except KeyError:
                return name

    renamer = Renamer(cast(Callable[[str], str], renaming), only_declared)
    renamer.visit(node)
    return renamer._mapping


class NotInlinable(Exception):
    pass


class Return(Exception):
    pass


class InlinableChecker(ast.NodeVisitor):
    def visit_Global(self, node: ast.Global):
        raise NotInlinable

    def visit_Import(self, node: ast.Import):
        raise NotInlinable

    def visit_ImportFrom(self, node: ast.ImportFrom):
        raise NotInlinable

    def visit_Nonlocal(self, node: ast.Nonlocal):
        raise NotInlinable

    def visit_Return(self, node: ast.Return):
        raise Return

    def _cannot_return(self, node: ast.stmt):
        try:
            self.generic_visit(node)
        except Return:
            raise NotInlinable

    def visit_For(self, node: ast.For):
        return self._cannot_return(node)

    def visit_While(self, node: ast.While):
        return self._cannot_return(node)

    def visit_With(self, node: ast.With):
        return self._cannot_return(node)

    def _does_return(self, stmts: Sequence[ast.stmt]) -> bool:
        try:
            for stmt in stmts:
                self.visit(stmt)
        except Return:
            return False
        else:
            return True

    def visit_If(self, node: ast.If):
        if self._does_return(node.body) != self._does_return(node.orelse):
            raise NotInlinable


def is_inlinable(node: ast.AST) -> bool:
    try:
        InlinableChecker().visit(node)
    except Return:
        return True
    except NotUnrollable:
        return False
    else:
        return True


class ReturnReplacer(ast.NodeTransformer):
    def __init__(self, name: str):
        self.name = name

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> NodeTransformation:
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> NodeTransformation:
        return node

    def visit_Return(self, node: ast.Return) -> NodeTransformation:
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=node.value
        )

    def visit(self, node: ast.AST) -> NodeTransformation:
        return node if isinstance(node, ast.expr) else super().visit(node)


def replace_return(node: ast.stmt, name: str) -> ast.stmt:
    return cast(ast.stmt, ReturnReplacer(name).visit(node))


Undefined = object()


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


def assigned_names(target: ast.expr) -> Optional[Collection[str]]:
    result = set()
    if isinstance(target, ast.Name):
        result.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            if isinstance(elt, ast.Starred):
                elt = elt.value
            names = assigned_names(elt)
            if names is None:
                return None
            result.update(names)
    else:
        return None
    return result


def ast_slice(value: ast.expr) -> ast.expr:
    return ast.Index(value=value) if sys.version_info < (3, 9) else value  # type: ignore
