import ast
import functools
import sys
from functools import partial
from typing import Any, Callable, Dict, Optional

from closure_optimizer import optimize
from closure_optimizer.constants import PREFIX
from closure_optimizer.utils import METADATA_ATTR, get_function_ast


def compare_ast(
    node1: Any, node2: Any, name_mapping: Optional[Dict[str, str]] = None
) -> bool:
    if name_mapping is None:
        name_mapping = {}
    if (
        isinstance(node1, ast.Name)
        and isinstance(node2, ast.Name)
        and type(node1.ctx) == type(node2.ctx)  # noqa: E721
    ):
        if node1.id in name_mapping:
            return node2.id == name_mapping.get(node1.id)
        elif node1.id.startswith(PREFIX) or node2.id.startswith(PREFIX):
            name_mapping[node1.id] = node2.id
            return True
        else:
            return node1.id == node2.id
    elif isinstance(node1, ast.AST) and isinstance(node2, ast.AST):
        return type(node1) == type(node2) and all(
            compare_ast(field1, field2, name_mapping)
            for (name, field1), (_, field2) in zip(
                ast.iter_fields(node1), ast.iter_fields(node2)
            )
            if name not in {"lineno", "col_offset", "end_lineno", "end_col_offset"}
        )
    elif isinstance(node1, list) and isinstance(node2, list):
        return len(node1) == len(node2) and all(
            compare_ast(elt1, elt2, name_mapping) for elt1, elt2 in zip(node1, node2)
        )
    else:
        return node1 == node2


def compare_func_body(func1: Callable, func2: Callable) -> bool:
    if isinstance(func1, partial):
        func1 = func1.func
    if isinstance(func2, partial):
        func2 = func2.func
    ast1, ast2 = get_function_ast(func1), get_function_ast(func2)
    result = compare_ast(ast1.body, ast2.body)
    if not result and sys.version_info >= (3, 9):
        print("==========")
        print(ast.unparse(ast1))
        print("==========")
        print(ast.unparse(ast2))
        print("==========")
    return result


def assert_optimized(__func: Callable, __ref: Callable, *args, **kwargs):
    if isinstance(__func, functools.partial):
        __ref = functools.partial(__ref, *__func.args, **__func.keywords)
    if not hasattr(__func, METADATA_ATTR):
        __func = optimize(__func)
    assert compare_func_body(__func, __ref)
    assert __func(*args, **kwargs) == __ref(*args, **kwargs)
