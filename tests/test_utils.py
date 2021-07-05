import pytest

from closure_optimizer import constants
from closure_optimizer.utils import get_function_ast, is_pure_for_loop, map_parameters
from tests.utils import compare_func_body


@pytest.mark.parametrize(
    "args, kwargs, partial, expected",
    [
        ((0, 1), {"d": 2}, False, {"__a": 0, "b": 1, "c": (), "d": 2, "e": {}}),
        ((0, 1), {"d": 2}, True, {"__a": 0, "b": 1, "d": 2}),
        (
            (0, 1, 2, 3),
            {"d": 4, "e": 5, "f": 6},
            False,
            {"__a": 0, "b": 1, "c": (2, 3), "d": 4, "e": {"e": 5, "f": 6}},
        ),
        ((0, 1, 2, 3), {"d": 4, "e": 5, "f": 6}, True, {"__a": 0, "b": 1, "d": 4}),
        ((), {}, True, {}),
    ],
)
def test_map_parameters(args, kwargs, partial, expected):
    def func_param(__a, b, *c, d, **e):
        ...

    assert map_parameters(func_param, args, kwargs, partial=partial) == expected


def func1(a, b):
    c = 0
    return a + c


def func2(a, b):
    _1 = 0
    return a + _1


def func3(a, b):
    d = 0
    return a + d


def func4(a, b):
    ...


@pytest.mark.parametrize(
    "other_func, expected", [(func2, True), (func3, False), (func4, False)]
)
def test_compare_func(other_func, expected, monkeypatch):
    monkeypatch.setattr(constants, "PREFIX", "_")
    assert compare_func_body(func1, other_func) == expected


def for_loops():
    for _ in ():
        ...
    for _ in ():
        break
    for _ in ():
        continue
    for _ in ():
        ...
    else:
        ...


def test_is_pure_for_loop():
    pure, with_break, with_continue, with_else = get_function_ast(for_loops).body
    assert is_pure_for_loop(pure)
    assert not is_pure_for_loop(with_break)
    assert not is_pure_for_loop(with_continue)
    assert not is_pure_for_loop(with_else)
