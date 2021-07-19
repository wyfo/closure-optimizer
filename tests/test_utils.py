import inspect
from typing import Callable, Iterable

import pytest

from closure_optimizer.utils import (
    NameGenerator,
    ast_parameters,
    get_function_ast,
    is_inlinable,
    is_unrollable,
    map_parameters,
    rename,
)
from tests.utils import compare_ast, compare_func_body


def func_params(func: Callable) -> Iterable[inspect.Parameter]:
    return inspect.signature(func).parameters.values()


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
    def func(__a, b, *c, d, **e):
        ...

    assert map_parameters(func_params(func), args, kwargs, partial=partial) == expected


@pytest.mark.parametrize(
    "args, kwargs, a, b", [((), {}, 0, 1), ((2,), {}, 2, 1), ((), {"b": 2}, 0, 2)]
)
def test_map_parameters_default(args, kwargs, a, b):
    def func(a=0, b=1):
        ...

    assert map_parameters(func_params(func), args, kwargs) == dict(a=a, b=b)


def test_ast_parameters():
    def f(a, b=0, *c, d, e=0, **f):
        ...

    def g(a, b=0, *, d, e=0):
        ...

    for func in (f, g):
        assert [
            p.replace(default=0) if p.default is not inspect.Parameter.empty else p
            for p in ast_parameters(get_function_ast(func))
        ] == list(inspect.signature(func).parameters.values())


def func1(a, b):
    c = 0
    return a + c


def func2(a, b):
    _closure_optimizer_1 = 0
    return a + _closure_optimizer_1


def func3(a, b):
    d = 0
    return a + d


def func4(a, b):
    ...


@pytest.mark.parametrize(
    "other_func, expected", [(func2, True), (func3, False), (func4, False)]
)
def test_compare_func(other_func, expected, monkeypatch):
    assert compare_func_body(func1, other_func) == expected


def test_is_pure_for_loop():
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

    pure, with_break, with_continue, with_else = get_function_ast(for_loops).body
    assert is_unrollable(pure)
    assert not is_unrollable(with_break)
    assert not is_unrollable(with_continue)
    assert not is_unrollable(with_else)


def test_inlinable():
    def f():
        pass

    assert is_inlinable(get_function_ast(f))

    def f():
        global test_inlinable

    assert not is_inlinable(get_function_ast(f))

    def f():
        nonlocal f

    assert not is_inlinable(get_function_ast(f))

    def f():
        for _ in ...:
            return

    assert not is_inlinable(get_function_ast(f))

    def f():
        while ...:
            return

    assert not is_inlinable(get_function_ast(f))

    def f():
        with ...:
            return

    assert not is_inlinable(get_function_ast(f))

    def f():
        if ...:
            return

    assert not is_inlinable(get_function_ast(f))

    def f():
        if ...:
            ...
        else:
            return

    assert not is_inlinable(get_function_ast(f))

    def f():
        if ...:
            ...
        else:
            ...
        if ...:
            return
        else:
            return

    assert is_inlinable(get_function_ast(f))


def test_rename():
    captured = ...

    def f(a):
        b = abs(a)

        def closure(*arg):
            return arg

        return closure(a, b, captured)

    _6 = ...

    def _1(_2):
        _3 = abs(_2)

        def _4(*_5):
            return _5

        return _4(_2, _3, _6)

    renamed = get_function_ast(f)
    assert rename(renamed, NameGenerator("_").replace) == {
        "f": "_1",
        "a": "_2",
        "b": "_3",
        "closure": "_4",
        "arg": "_5",
        "captured": "_6",
    }
    assert compare_ast(renamed, get_function_ast(_1))

    def _1(_2):
        _3 = abs(_2)

        def _4(*_5):
            return _5

        return _4(_2, _3, captured)

    renamed = get_function_ast(f)
    assert rename(renamed, NameGenerator("_").replace, only_declared=True) == {
        "f": "_1",
        "a": "_2",
        "b": "_3",
        "closure": "_4",
        "arg": "_5",
    }
    assert compare_ast(renamed, get_function_ast(_1))


def test_normalization():
    def f(a, b, c, d, e):
        if a:
            return
        else:
            a = ...
        if b:
            ...
        elif c:
            return
        for _ in d:
            if e:
                continue
            e = ...

    def g(a, b, c, d, e):
        if a:
            return
        else:
            a = ...
            if b:
                ...
            elif c:
                return
        for _ in d:
            if e:
                pass
            else:
                e = ...

    assert compare_func_body(f, g)
