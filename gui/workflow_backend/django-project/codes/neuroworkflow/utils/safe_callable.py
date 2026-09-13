"""Turn text into a callable, for node parameters that hold a function.

A workflow graph is saved as JSON, and JSON cannot carry a function. So a parameter whose
value is a callable — a connectivity rule, a stimulus profile, a custom weight function —
arrives from the GUI as **text**, and the node has to compile it. That is ``eval``, which
runs whatever the text says, so it needs a guard.

This lives here rather than in one node on purpose: every node with a callable parameter
faces the same problem, and the optimizer is node-agnostic — it will happily tune a custom
connectivity node written by someone else. Without a shared helper each of those nodes
writes its own ``eval`` and reopens the same hole.

What is allowed: any Python **expression** that produces a callable — a lambda, a named
function, a conditional expression, ``functools.partial(...)``. What is refused: names that
reach outside the process, and dunder attribute access, which is how a restricted ``eval``
is normally escaped::

    safe_callable("lambda src, tgt: 1 if np.random.rand() < 0.1 else 0")   # fine
    safe_callable("np.random.default_rng(src['node_id']).random")          # fine
    safe_callable("lambda s, t: ().__class__.__bases__[0].__subclasses__()")  # ValueError

Names are checked when the text is compiled, not when the function is first called. A rule
using something unavailable therefore fails where the user set it, with a message naming
what *is* available — rather than raising ``NameError`` thousands of calls later, in the
middle of building a network.

This is a guard, not a sandbox: real isolation means a separate process. It stops accidents
and casual misuse by someone who can already edit a workflow.
"""

import ast
import builtins
import functools
import math
import random
import statistics
from typing import Any, Callable, Dict, Optional

#: Names that reach outside the expression: imports, code execution, the filesystem,
#: attribute machinery, and the namespace itself. Everything else in builtins is allowed,
#: because a rule is arithmetic and an allowlist of arithmetic has to grow forever.
DENIED_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "object",
        "open",
        "quit",
        "setattr",
        "super",
        "type",
        "vars",
    }
)

#: Modules a scientific rule reasonably reaches for.
DEFAULT_GLOBALS: Dict[str, Any] = {
    "np": None,  # filled in below, so numpy stays an optional dependency
    "numpy": None,
    "math": math,
    "random": random,
    "statistics": statistics,
    "functools": functools,
}

try:  # numpy is not required to import this module
    import numpy as _np

    DEFAULT_GLOBALS["np"] = _np
    DEFAULT_GLOBALS["numpy"] = _np
except ImportError:  # pragma: no cover - depends on environment
    DEFAULT_GLOBALS.pop("np")
    DEFAULT_GLOBALS.pop("numpy")


def safe_builtins() -> Dict[str, Any]:
    """Every public builtin except the denied ones."""
    return {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if name not in DENIED_NAMES and not name.startswith("_")
    }


def bound_names(tree: ast.AST) -> set:
    """Names the expression binds itself, so they are not flagged as unavailable.

    Without this, ``sum(w for w in [1, 0, 1])`` is rejected because ``w`` looks like a
    missing global — it is bound by the comprehension.
    """
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                bound.add(arg.arg)
            for extra in (args.vararg, args.kwarg):
                if extra is not None:
                    bound.add(extra.arg)
            if not isinstance(node, ast.Lambda):
                bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)  # assignments, loop targets, walrus, comprehensions
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def check(tree: ast.AST, available: set, what: str = "value") -> None:
    """Raise ValueError if the expression reaches outside what it is given."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(f"{what} may not import")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # ().__class__.__bases__[0].__subclasses__() reaches every loaded class,
            # and needs no builtin at all, so restricting builtins does not stop it.
            raise ValueError(f"{what} may not use the attribute {node.attr!r}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in DENIED_NAMES:
                raise ValueError(f"{what} may not use {node.id!r}")
            if isinstance(node.ctx, ast.Load) and node.id not in available:
                raise ValueError(
                    f"{what} uses {node.id!r}, which is not available. Available: "
                    f"{', '.join(sorted(available - set(safe_builtins())))}, its own "
                    f"arguments, and the Python built-ins."
                )


def safe_callable(
    text: str,
    extra_globals: Optional[Dict[str, Any]] = None,
    what: str = "value",
) -> Callable:
    """Compile ``text`` into a callable, refusing anything that reaches outside it.

    Args:
        text: a Python expression producing a callable, e.g. ``"lambda src, tgt: 1"``.
        extra_globals: names the expression may use in addition to the defaults
            (``np``, ``math``, ``random``, ``statistics``, ``functools``).
        what: how to name the value in error messages, e.g. ``"connection_rule"``.

    Raises:
        ValueError: the text is not valid Python, reaches outside what it is given, or
            does not produce a callable.
    """
    namespace = dict(DEFAULT_GLOBALS)
    namespace.update(extra_globals or {})

    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        # Only expressions for now. A multi-line `def` block would need mode="exec",
        # taking the last function defined; the checks above already cover that shape.
        raise ValueError(f"{what} is not a valid Python expression: {exc}") from exc

    allowed = safe_builtins()
    check(tree, set(allowed) | set(namespace) | bound_names(tree), what)

    value = eval(  # noqa: S307 - guarded above; see the module docstring
        compile(tree, f"<{what}>", "eval"), {"__builtins__": allowed, **namespace}
    )
    if not callable(value):
        raise ValueError(f"{what} did not produce a callable: {value!r}")
    return value
