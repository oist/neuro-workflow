"""A persistent ipywidget chat panel for the notebook agent.

The agent runs in the background (``Agent.start``): the Send handler returns
at once, so the kernel keeps handling the panel's buttons (Approve / Deny /
Stop) while the agent works. What the agent shows comes back to the kernel's
main thread through its IO loop and is written into the panel explicitly; an
``Output`` context manager would stop capturing once the click handler has
returned.
"""

import concurrent.futures
import html

from .permissions import PermissionDecision

HINT = (
    "Tool calls that need your approval appear here. Change what is asked "
    "with <code>%chat_permissions</code>."
)
RUNNING_HINT = (
    "Agent running — cells you run now share its namespace. "
    "<b>Stop</b> ends the turn."
)

# Run of the most recently displayed panel. Displaying a panel again stops it,
# so no approval waits in a panel nobody can see.
_active = None


class _Transcript:
    """The panel's log: text segments and rich outputs (main thread only).

    Streamed text is buffered and synced at most every ``FLUSH_S`` seconds,
    and only the current text segment is resent; rich outputs get their own
    ``Output`` widgets.
    """

    FLUSH_S = 0.15

    def __init__(self, widgets, io_loop):
        self._widgets = widgets
        self._io_loop = io_loop
        self.box = widgets.VBox(layout=widgets.Layout(flex="0 0 auto", padding="6px"))
        self._out = None
        self._text = ""
        self._pending = ""
        self._scheduled = False

    def write(self, text: str):
        if not text:
            return
        self._pending += text
        if not self._scheduled:
            self._scheduled = True
            self._io_loop.call_later(self.FLUSH_S, self.flush)

    def flush(self):
        self._scheduled = False
        if not self._pending:
            return
        if self._out is None:
            self._out = self._widgets.Output()
            self._text = ""
            self.box.children += (self._out,)
        self._text += self._pending
        self._pending = ""
        self._out.outputs = (
            {"output_type": "stream", "name": "stdout", "text": self._text},
        )

    def add(self, widget):
        self.flush()
        self._out = None
        self.box.children += (widget,)

    def captured(self, captured):
        """Show what ``run_code`` printed and displayed (IPython ``CapturedIO``)."""
        self.write(captured.stdout)
        self.write(captured.stderr)
        for output in captured.outputs:
            out = self._widgets.Output()
            out.outputs = (
                {
                    "output_type": "display_data",
                    "data": output.data,
                    "metadata": output.metadata or {},
                },
            )
            self.add(out)
        self.flush()


def _approval_card(widgets, request, transcript, io_loop):
    """Show an approval prompt; return a future the buttons resolve.

    The agent cancels the future when nobody answers in time or the turn is
    stopped; the card then disables itself.
    """
    future = concurrent.futures.Future()
    title = widgets.HTML(
        f"<b>Allow {html.escape(request.tool)}?</b> "
        f"<span style='color:#666'>({html.escape(request.category)})</span>"
    )
    body = widgets.HTML(
        "<pre style='max-height:240px;overflow:auto;margin:4px 0;"
        f"white-space:pre-wrap'>{html.escape(request.summary)}</pre>"
    )
    reason = widgets.Text(
        placeholder="Reason (optional, told to the agent when you deny)",
        layout=widgets.Layout(width="100%"),
    )
    approve = widgets.Button(description="Approve", button_style="success")
    deny = widgets.Button(description="Deny", button_style="danger")
    buttons = [approve, deny]
    if request.can_always_allow:
        always = widgets.Button(
            description="Always allow",
            tooltip=f"Allow {request.tool} for the rest of this conversation",
        )
        buttons.append(always)
        always.on_click(lambda _: answer(PermissionDecision(allow=True, always=True)))
    status = widgets.HTML()
    card = widgets.VBox(
        [title, body, widgets.HBox(buttons), reason, status],
        layout=widgets.Layout(
            border="1px solid #e0a800", padding="6px", margin="4px 0"
        ),
    )

    def answer(decision):
        try:
            future.set_result(decision)
        except concurrent.futures.InvalidStateError:
            pass  # already timed out or stopped

    approve.on_click(lambda _: answer(PermissionDecision(allow=True)))
    deny.on_click(
        lambda _: answer(PermissionDecision(allow=False, reason=reason.value.strip()))
    )

    def close(done):
        for control in [*buttons, reason]:
            control.disabled = True
        if done.cancelled():
            status.value = "<i>Not answered in time or stopped: not run.</i>"
        elif done.result().allow:
            status.value = "<i>Approved.</i>"
        else:
            status.value = "<i>Denied.</i>"

    # Resolved from a click (main thread) or cancelled by the agent's worker.
    future.add_done_callback(lambda done: io_loop.add_callback(close, done))
    transcript.add(card)
    return future


def ChatPanel(*, user_token=None, project_id=None):
    """Display a chat panel for the notebook agent in the current cell.

    MCP workflow tools are enabled automatically when the notebook lives in a
    project folder (``codes/projects/<uuid>/``) whose Jupyter tab is open in the
    app; ``user_token`` (a Keycloak access token) overrides that relay.
    """
    import ipywidgets as widgets
    from IPython import get_ipython
    from IPython.display import display

    from . import get_agent, list_models

    ip = get_ipython()
    if ip is None or getattr(ip, "kernel", None) is None:
        raise RuntimeError("ChatPanel needs a Jupyter kernel; use chat() instead.")
    io_loop = ip.kernel.io_loop

    if _active is not None and not _active.done.done():
        _active.cancel()
    agent = get_agent(user_token=user_token, project_id=project_id)

    # Model picker, shown only when the backend offers non-Claude models.
    try:
        models = list_models()
    except Exception:
        models = [""]
    picker = None
    if len(models) > 1:
        current = agent._config.provider_model or ""
        picker = widgets.Dropdown(
            options=[("Claude (default)", ""), *[(m, m) for m in models[1:]]],
            value=current if current in models else "",
            layout=widgets.Layout(width="148px"),
        )

    transcript = _Transcript(widgets, io_loop)
    # The log must not shrink (flex 0 0 auto) so it overflows the box; the
    # column-reverse box then anchors its scroll position to the newest output
    # and the log follows the streamed reply without any JavaScript.
    log_box = widgets.Box(
        [transcript.box],
        layout=widgets.Layout(
            border="1px solid #ccc",
            height="360px",
            overflow="auto",
            display="flex",
            flex_flow="column-reverse",
        ),
    )
    text = widgets.Textarea(
        placeholder="Ask the NeuroWorkflow agent…",
        layout=widgets.Layout(width="100%", height="70px"),
    )
    send = widgets.Button(description="Send", button_style="primary")
    stop = widgets.Button(description="Stop", button_style="warning", disabled=True)
    hint = widgets.HTML(f"<small>{HINT}</small>")
    pending = set()
    state = {"handle": None}

    def on_tool(name, args):
        preview = ", ".join(f"{k}={v!r}"[:60] for k, v in args.items())
        transcript.write(f"\n  ⚙ {name}({preview})\n")

    def approve(request):
        future = _approval_card(widgets, request, transcript, io_loop)
        pending.add(future)
        future.add_done_callback(pending.discard)
        return future

    def finished(done):
        for future in list(pending):
            future.cancel()
        if done.cancelled():
            transcript.write("\n(stopped)\n")
        elif done.exception() is not None:
            transcript.write(f"\n[error] {done.exception()}\n")
        else:
            transcript.write("\n")
        transcript.flush()
        state["handle"] = None
        send.disabled = False
        stop.disabled = True
        hint.value = f"<small>{HINT}</small>"

    def _submit(_=None):
        global _active
        nonlocal agent
        message = text.value.strip()
        if not message or state["handle"] is not None:
            return
        text.value = ""
        transcript.write(f"\n🧑 {message}\n🤖 ")
        try:
            if picker is not None:
                # Read the picker at send time; switching rebuilds the agent.
                chosen = get_agent(
                    user_token=user_token, project_id=project_id, model=picker.value
                )
                if chosen is not agent:
                    agent = chosen
                    transcript.write(f"({picker.label}: new conversation) ")
            handle = agent.start(
                message,
                dispatch=io_loop.add_callback,
                on_text=transcript.write,
                on_tool=on_tool,
                on_output=transcript.captured,
                approve=approve,
            )
        except Exception as e:
            transcript.write(f"\n[error] {e}\n")
            transcript.flush()
            return
        state["handle"] = _active = handle
        send.disabled = True
        stop.disabled = False
        hint.value = f"<small>{RUNNING_HINT}</small>"
        handle.done.add_done_callback(lambda done: io_loop.add_callback(finished, done))

    def _stop(_=None):
        if state["handle"] is not None:
            state["handle"].cancel()

    send.on_click(_submit)
    stop.on_click(_stop)
    buttons = widgets.VBox([send, stop] if picker is None else [send, stop, picker])
    panel = widgets.VBox([log_box, widgets.HBox([text, buttons]), hint])
    # display() renders it once; returning it too would make Jupyter
    # auto-display the cell result and show a second copy.
    display(panel)
