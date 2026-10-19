"""Live-kernel Python execution for the Claude agent's ``run_code`` tool.

``run_code`` executes against the notebook's live namespace dict so variables
persist across calls. The agent sends it to the kernel's main thread (see
``loop.py``), so it never runs alongside a user's cell; we still ``exec`` the
code rather than calling ``ipython.run_cell``, which would record it as a cell
of its own. Captured output, rich displays included, is shown to the user
(``show``) *and* summarised for the model.
"""

import ast
import contextlib
import io
import signal
import threading

_MAX_RESULT_CHARS = 8000


def _truncate(text: str) -> str:
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + "\n...[truncated]"
    return text


@contextlib.contextmanager
def _interruptible():
    """Let a kernel interrupt raise KeyboardInterrupt in the code.

    ipykernel only installs the handler while it handles a request; code the
    ChatPanel runs from the IO loop would otherwise ignore the interrupt.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _capture():
    """Capture stdout/stderr and, inside IPython, rich displays too."""
    try:
        from IPython import get_ipython
        from IPython.utils.capture import capture_output
    except ImportError:
        get_ipython = None
    if get_ipython is not None and get_ipython() is not None:
        return capture_output()
    return _PlainCapture()


class _PlainCapture:
    """``capture_output`` stand-in without IPython (stdout/stderr only)."""

    outputs = ()

    def __enter__(self):
        self._out, self._err = io.StringIO(), io.StringIO()
        self._redirects = contextlib.ExitStack()
        self._redirects.enter_context(contextlib.redirect_stdout(self._out))
        self._redirects.enter_context(contextlib.redirect_stderr(self._err))
        return self

    def __exit__(self, *exc):
        self._redirects.close()
        self.stdout, self.stderr = self._out.getvalue(), self._err.getvalue()
        return False

    def show(self):
        import sys

        sys.stdout.write(self.stdout)
        sys.stdout.write(self.stderr)


def run_code(code: str, namespace: dict, show=None) -> tuple[str, bool]:
    """Execute ``code`` against ``namespace``.

    Returns the text for the model and whether a kernel interrupt stopped the
    code. ``show(captured)`` displays the output (default: in the current cell).
    """
    result_repr = None
    error_text = None
    interrupted = False
    with _capture() as captured:
        try:
            parsed = ast.parse(code)
            last_expr = None
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                last_expr = parsed.body.pop()
            module = ast.Module(body=parsed.body, type_ignores=[])
            ast.fix_missing_locations(module)
            with _interruptible():
                exec(compile(module, "<chat>", "exec"), namespace)
                if last_expr is not None:
                    expr = ast.Expression(last_expr.value)
                    ast.fix_missing_locations(expr)
                    value = eval(compile(expr, "<chat>", "eval"), namespace)
                    if value is not None:
                        result_repr = repr(value)
        except KeyboardInterrupt:
            interrupted = True
            error_text = "interrupted by the user"
        except Exception as e:  # surface execution errors back to the model
            error_text = f"{type(e).__name__}: {e}"

    if show is None:
        captured.show()
    else:
        show(captured)

    parts = []
    if captured.stdout:
        parts.append(captured.stdout.rstrip())
    if captured.stderr:
        parts.append("[stderr]\n" + captured.stderr.rstrip())
    rich = captured.outputs
    if rich:
        texts = [o.data.get("text/plain", "") for o in rich if o.data.get("text/plain")]
        parts.append(f"[display] {len(rich)} rich output(s) shown to the user")
        parts.extend(texts)
    if error_text:
        parts.append("[error] " + error_text)
    elif result_repr is not None:
        parts.append("[result] " + result_repr)
    text = _truncate("\n".join(parts)) if parts else "(no output)"
    return text, interrupted
