import ast
import sys
from typing import Any, Callable, Mapping

PREFIX = "_closure_optimizer_"
BUILTINS = (
    abs,
    all,
    any,
    ascii,
    bin,
    bool,
    bytearray,
    bytes,
    callable,
    chr,
    classmethod,
    complex,
    dict,
    dir,
    divmod,
    enumerate,
    float,
    format,
    frozenset,
    getattr,
    hasattr,
    hash,
    hex,
    id,
    int,
    isinstance,
    issubclass,
    len,
    max,
    memoryview,
    min,
    object,
    oct,
    ord,
    pow,
    property,
    range,
    repr,
    reversed,
    round,
    slice,
    sorted,
    staticmethod,
    str,
    sum,
    tuple,
    type,
)
BUILTIN_NAMES = set(f.__name__ for f in BUILTINS)  # type: ignore
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
