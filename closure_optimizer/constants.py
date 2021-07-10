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
    filter,
    float,
    format,
    frozenset,
    getattr,
    globals,
    hasattr,
    hash,
    hex,
    id,
    int,
    isinstance,
    issubclass,
    iter,
    len,
    list,
    map,
    max,
    memoryview,
    min,
    next,
    object,
    oct,
    ord,
    pow,
    print,
    property,
    range,
    repr,
    reversed,
    round,
    set,
    slice,
    sorted,
    staticmethod,
    str,
    sum,
    tuple,
    tuple,
    type,
    zip,
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
