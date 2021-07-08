PREFIX = "_closure_optimizer_"
OPTIMIZED_AST_ATTR = f"{PREFIX}ast"
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
