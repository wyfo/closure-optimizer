# flake8: noqa
import functools
import types

import pytest

from closure_optimizer import optimize
from closure_optimizer.utils import get_skipped
from tests.utils import assert_optimized


def test_evaluate_constants():
    a = 0

    def f():
        return a + 1

    def g():
        return 1

    assert_optimized(f, g)


def test_evaluate_builtins():
    def f():
        return int(0)

    def g():
        return 0

    assert_optimized(f, g)


def test_does_not_evaluate_skipped():
    a = 0

    def f():
        return eval("42") + (a + 0)

    assert_optimized(optimize(f, skip={"a"}), f)


def test_partial():
    def f(a):
        return a + 1

    def g(a):
        return 1

    assert_optimized(functools.partial(f, 0), g)


def test_evaluate_nested_expressions():
    a = 1

    def f():
        return (a + a) * 2

    def g():
        return 4

    assert_optimized(f, g)


def test_assign():
    a = 0

    def f():
        b = a + 1
        return b

    def g():
        b = 1
        return 1

    assert_optimized(f, g)


def test_scopes():
    def f(a):
        if a:
            b = 0
        else:
            b = 1
        return b + 1

    assert_optimized(f, f, True)

    def g(a):
        b = 0
        return 1

    assert_optimized(functools.partial(f, True), g)


def test_reassign_captured_with_not_evaluable_value():
    a = 0

    def f():
        nonlocal a
        a = eval("42")
        return a + 1

    assert_optimized(f, f)


def test_if():
    def f(a):
        if a:
            return 0
        else:
            return 1

    def g(a):
        return 0

    def h(a):
        return 1

    assert_optimized(f, f, True)
    assert_optimized(functools.partial(f, True), g)
    assert_optimized(functools.partial(f, False), h)


def test_if_expr():
    def f(a):
        return 0 if a else 1

    def g(a):
        return 0

    def h(a):
        return 1

    assert_optimized(f, f, True)
    assert_optimized(functools.partial(f, True), g)
    assert_optimized(functools.partial(f, False), h)


def test_for_loop():
    l = [0, 1]

    def f():
        for i in l:
            eval(str(i))

    def g():
        i = 0
        eval("0")
        i = 1
        eval("1")

    assert_optimized(f, g)


def test_comprehension():
    def f():
        return (
            [i for i in (0, 1)],
            {(i, j) for i, j in ((0, 1), (0, 2))},
            {i: i + 1 for i in (0, 1) if i != 0},
        )

    _1, _2 = (0, 1), (0, 2)

    def g():
        return ([0, 1], {_1, _2}, {1: 2})

    assert_optimized(f, g)


def test_comprehension_inlined():
    def f(l):
        return (
            [i for i in l],
            {(i, j) for i, j in zip(l, l)},
            {i: j + 1 for i in l if i != 0 for j in l},
        )

    def g(l):
        _1 = []
        for _2 in l:
            _1.append(_2)
        _3 = set()
        for _4, _5 in zip(l, l):
            _3.add((_4, _5))
        _6 = {}
        for _7 in l:
            if _7 != 0:
                for _8 in l:
                    _6[_7] = _8 + 1
        return _1, _3, _6

    assert_optimized(f, g, [0, 1])


def test_exec_function():
    def identity(a):
        return a

    def f():
        return identity(0)

    def g():
        return 0

    assert_optimized(f, f)
    assert_optimized(optimize(f, execute={identity}), g)


def test_exec_method():
    a = {0: 1}

    def f():
        return list(a.items())

    _1 = {0: 1}.items

    def g():
        return list(_1())

    _2 = {0: 1}.items()

    def h():
        return list(_2)

    assert_optimized(f, g)
    assert_optimized(optimize(f, execute={dict.items}), h)


def test_inline():
    def inlined(a, b=1):
        c = 2
        return a + b + c

    def f(a):
        return inlined(a)

    def g(a):
        _1 = a
        _2 = 1
        _3 = 2
        _4 = _1 + 1 + 2
        return _4

    assert_optimized(f, f, 0)
    assert_optimized(optimize(f, inline={inlined}), g, 0)


def test_inline_local():
    def f(a):
        def nested(a, b=1):
            c = 2
            return a + b + c

        return nested(a)

    def g(a):
        def nested(a, b=1):
            c = 2
            return a + b + c

        _1 = a
        _2 = 1
        _3 = 2
        _4 = _1 + 1 + 2
        return _4

    assert_optimized(f, g, 0)


def test_bool_op():
    def inlined(x):
        return x + []

    def f(a):
        print(0 or 0)
        print(a and inlined(a) and a)
        if a and 0:
            raise
        if not (a or 1):
            raise
        if (a + 1) and 0:
            raise
        if not (a + 1 or 1):
            raise
        return (0 and a) + (1 and a) + (a and 0) + (0 or a) + (1 or a) + (a or 1)

    def g(a):
        print(0)
        _1 = a
        if _1:
            _2 = a
            _3 = _2 + []
            _1 = _3 and a
        print(_1)
        if (a + 1) and 0:
            raise
        if not (a + 1 or 1):
            raise
        return 0 + a + (a and 0) + a + 1 + (a or 1)

    assert_optimized(optimize(f, inline={inlined}), g, 0)


def test_recursive_inlining():
    def rec(i):
        return rec(i - 1) if i > 0 else 0

    def f(a):
        def nested(i):
            return nested(i - 1) if i > 0 else 0

        return rec(0) + rec(a) + nested(0) + nested(a)

    _rec = rec

    def _nested(i):
        return _nested(i - 1) if i > 0 else 0

    def g(a):
        def nested(i):
            return nested(i - 1) if i > 0 else 0

        # rec(0)
        _1 = 0
        _2 = 0
        # rec(a)
        _3 = a
        _4 = _rec(_3 - 1) if _3 > 0 else 0
        # nested(0)
        _5 = 0
        _6 = 0
        # nested(a)
        _7 = a
        _8 = _nested(_7 - 1) if _7 > 0 else 0
        return 0 + _4 + 0 + _8

    assert_optimized(optimize(f, inline={rec}), g, 0)


def test_skip_is_preserved_in_inlined_optimized():
    def to_optimize():
        a = 0
        return a

    optimized = optimize(to_optimize, skip={"a"})
    assert_optimized(optimized, to_optimize)

    def f():
        return optimized()

    def g():
        _1 = 0
        _2 = _1
        return _2

    assert_optimized(f, g)
    # optimized f has skipped variables
    optimized_f = optimize(f)
    assert get_skipped(optimized_f)

    def h():
        return optimized_f()

    def i():
        _1 = 0
        _2 = _1
        _3 = _2
        return _3

    assert_optimized(h, i)


def test_list_map():
    def f(a):
        l = list(())
        return list(map(int, a))

    _1 = ()

    def g(a):
        l = list(_1)
        _2 = []
        for _3 in a:
            _2.append(int(_3))
        return _2

    def h(a):
        l = list(_1)
        return [0, 1]

    assert_optimized(f, g, (0, 1))
    assert_optimized(functools.partial(f, (0, 1)), h)


def test_use_builtin_when_captured():
    a = abs
    b = types.SimpleNamespace(a=abs)

    def f(arg):
        return a(arg) + (b.a or arg)(arg)

    def g(arg):
        return abs(arg) + abs(arg)

    assert_optimized(f, g, 0)


def test_properties():
    class A:
        @property
        def zero(self):
            return 0

    a = A()

    def f():
        return a.zero

    def g():
        return 0

    def h():
        _1 = 0
        return _1

    assert_optimized(optimize(f, execute={A.zero}), g)
    assert_optimized(optimize(f, execute={A.zero.fget}), g)
    assert_optimized(optimize(f, inline={A.zero}), h)
    assert_optimized(optimize(f, inline={A.zero.fget}), h)


def test_cache_function_default():
    a = (0, 1)

    def f(a=a):
        return a

    _1 = (0, 1)

    def g(a=_1):
        return a

    assert_optimized(f, g)


def test_optimize_gettatr():
    attr = "attr"

    def f(obj):
        return getattr(obj, attr)

    def g(obj):
        return obj.attr

    assert_optimized(f, g, types.SimpleNamespace(attr=0))


@pytest.mark.parametrize("wraps_optimized", [False, True])
def test_functools_wraps(wraps_optimized):
    a = 0

    def tmp():
        return a

    optimized = optimize(tmp)

    @functools.wraps(optimized if wraps_optimized else tmp)
    def f():
        return optimized()

    def g():
        _1 = 0
        return _1

    assert_optimized(f, g)
