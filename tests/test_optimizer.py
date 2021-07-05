import functools

from closure_optimizer import optimize
from tests.utils import assert_optimized, compare_func_body


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
        return b

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
