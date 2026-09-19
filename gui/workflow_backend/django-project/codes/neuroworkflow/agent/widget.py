"""A persistent ipywidget chat panel for the notebook agent."""


def ChatPanel(*, user_token=None, project_id=None):
    """Display a chat panel for the notebook agent in the current cell.

    MCP workflow tools are enabled automatically when the notebook lives in a
    project folder (``codes/projects/<uuid>/``) whose Jupyter tab is open in the
    app; ``user_token`` (a Keycloak access token) overrides that relay.
    """
    import ipywidgets as widgets
    from IPython.display import display

    from . import get_agent

    agent = get_agent(user_token=user_token, project_id=project_id)

    # The Output must not shrink (flex 0 0 auto) so it overflows the box; the
    # column-reverse box then anchors its scroll position to the newest output
    # and the log follows the streamed reply without any JavaScript.
    log = widgets.Output(layout=widgets.Layout(flex="0 0 auto", padding="6px"))
    log_box = widgets.Box(
        [log],
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

    def _submit(_=None):
        message = text.value.strip()
        if not message:
            return
        text.value = ""
        send.disabled = True
        with log:
            print(f"\n🧑 {message}\n🤖 ", end="")
        try:
            def on_text(delta):
                with log:
                    print(delta, end="")

            def on_tool(name, args):
                preview = ", ".join(f"{k}={v!r}"[:60] for k, v in args.items())
                with log:
                    print(f"\n  ⚙ {name}({preview})")

            agent.run(message, on_text=on_text, on_tool=on_tool)
            with log:
                print()
        except Exception as e:
            with log:
                print(f"\n[error] {e}")
        finally:
            send.disabled = False

    send.on_click(_submit)
    panel = widgets.VBox([log_box, widgets.HBox([text, send])])
    # display() renders it once; returning it too would make Jupyter
    # auto-display the cell result and show a second copy.
    display(panel)
