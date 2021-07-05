from typing import Any

from closure_optimizer.utils import IterableOrPredicate

skipped_functions: IterableOrPredicate[Any] = {
    compile,
    eval,
    exec,
    globals,
    input,
    locals,
    vars,
    open,
    print,
    __import__,
}
loop_enrolling_limit: int = 100
