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
    Mapping,
    Sequence,
    TypeVar,
    Union,
)

from closure_optimizer.constants import OPTIMIZED_AST_ATTR

T = TypeVar("T")


def get_function_ast(obj: Callable) -> ast.FunctionDef:
    if obj.__name__ == "<lambda>":
        raise ValueError("Lambda are not supported")
    if hasattr(obj, OPTIMIZED_AST_ATTR):
        return getattr(obj, OPTIMIZED_AST_ATTR)
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


def map_parameters(
    func: Callable,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    partial: bool = False,
) -> Mapping[str, Any]:
    if hasattr(func, "__wrapped__") or hasattr(func, "__signature__"):
        return {}
    offset = 0
    kwargs = dict(kwargs)
    result: Dict[str, Any] = {}
    for param in inspect.signature(func).parameters.values():
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


class ImpurForLoop(Exception):
    pass


class PureForLoopChecker(ast.NodeVisitor):
    def visit_Break(self, node: ast.Break):
        raise ImpurForLoop

    def visit_Continue(self, node: ast.Continue):
        raise ImpurForLoop

    def visit_For(self, node: ast.For):
        if node.orelse:
            raise ImpurForLoop
        self.generic_visit(node)


def is_pure_for_loop(node: ast.For) -> bool:
    try:
        PureForLoopChecker().visit(node)
    except ImpurForLoop:
        return False
    else:
        return True


CONSTANT_NODES = (
    ast.Constant,
    ast.Num,
    ast.Str,
    ast.Bytes,
    ast.NameConstant,
    ast.Ellipsis,
)
CONSTANT_NODE_BY_TYPE: Mapping[type, Callable[[Any], ast.expr]] = {}
if sys.version_info < (3, 8):
    CONSTANT_NODE_BY_TYPE = {
        int: ast.Num,
        float: ast.Num,
        str: ast.Str,
        bytes: ast.Bytes,
        bool: ast.NameConstant,
        type(None): ast.NameConstant,
        type(...): lambda _: ast.Ellipsis(),
    }
else:
    CONSTANT_NODE_BY_TYPE = dict.fromkeys(
        (int, float, str, bytes, bool, type(None), type(...)), ast.Constant
    )

NodeTransformation = Union[ast.AST, Iterable[ast.AST], None]


class Renamer(ast.NodeVisitor):
    def __init__(self, generate_name: Callable[[], str]):
        self.generate_name = generate_name
        self._mapping: Dict[str, str] = {}

    def visit_Name(self, node: ast.Name):
        if node.id not in self._mapping:
            self._mapping[node.id] = self.generate_name()
        node.id = self._mapping[node.id]


def rename(node: ast.AST, generate_name: Callable[[], str]) -> Mapping[str, str]:
    renamer = Renamer(generate_name)
    renamer.visit(node)
    return renamer._mapping
