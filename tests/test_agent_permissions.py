"""Kernel side of the notebook agent's tool permissions (Issue #123).

Each tool call is allowed, denied or sent to the user for approval, by
category or tool, within the administrator's ceilings. The agent runs on a
worker thread; callbacks, approvals and ``run_code`` go through ``dispatch``
to the calling thread. Requires ``claude_agent_sdk`` (run inside the nest
kernel image).
"""

import asyncio
import concurrent.futures
import threading
import time

import pytest

from neuroworkflow.agent.client import BackendClient
from neuroworkflow.agent.loop import (
    RunHandle,
    _make_can_use_tool,
    _Run,
)
from neuroworkflow.agent.permissions import (
    DEFAULTS,
    PermissionDecision,
    Permissions,
    Policy,
    describe,
    parse_answer,
)
from neuroworkflow.agent.tools import run_code
from test_agent_conversation_record import (
    _FakeResponse,
    _agent,
    _config,
)

ROOT = "/home/jovyan/codes"


def _permissions(**policy) -> Permissions:
    data = {"workflow_read_tools": ["get_flow", "list_nodes"], **policy}
    return Permissions(Policy.from_json(data))


# ---------------------------------------------------------------------------
# permissions.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool, sdk_name, category",
    [
        ("Read", "Read", "read"),
        ("ToolSearch", "ToolSearch", "read"),
        ("Edit", "Edit", "file-write"),
        ("Bash", "Bash", "code-exec"),
        ("run_code", "mcp__notebook__run_code", "code-exec"),
        ("WebFetch", "WebFetch", "web"),
        ("get_flow", "mcp__workflow__get_flow", "workflow-read"),
        ("add_node", "mcp__workflow__add_node", "workflow-edit"),
        ("brand_new_tool", "mcp__workflow__brand_new_tool", "workflow-edit"),
        ("Task", "Task", "other"),
        ("mcp__other__x", "mcp__other__x", "other"),
    ],
)
def test_category(tool, sdk_name, category):
    assert _permissions().category(tool, sdk_name) == category


def test_defaults_ask_for_everything_but_reads():
    p = _permissions()
    assert p.mode("Read", "Read") == "allow"
    assert p.mode("get_flow", "mcp__workflow__get_flow") == "allow"
    for tool, sdk in [
        ("add_node", "mcp__workflow__add_node"),
        ("Write", "Write"),
        ("Bash", "Bash"),
        ("run_code", "mcp__notebook__run_code"),
        ("WebSearch", "WebSearch"),
        ("Task", "Task"),
    ]:
        assert p.mode(tool, sdk) == "ask", tool


def test_user_settings_tool_beats_category():
    p = _permissions()
    p.set("code-exec", "allow")
    p.set("Bash", "deny")
    assert p.mode("run_code", "mcp__notebook__run_code") == "allow"
    assert p.mode("Bash", "Bash") == "deny"
    p.reset()
    assert p.mode("Bash", "Bash") == "ask"


def test_set_cannot_loosen_past_the_admin_ceiling():
    p = _permissions(category_ceilings={"code-exec": "ask"})
    with pytest.raises(ValueError, match="limit is 'ask'"):
        p.set("code-exec", "allow")
    with pytest.raises(ValueError):
        p.set("Bash", "allow")
    p.set("code-exec", "deny")  # tightening is fine
    assert p.mode("Bash", "Bash") == "deny"


def test_ceiling_caps_defaults_and_tool_ceiling_wins():
    p = _permissions(
        category_ceilings={"read": "deny", "workflow-edit": "deny"},
        tool_ceilings={"add_node": "ask", "Glob": "allow"},
    )
    assert p.mode("Read", "Read") == "deny"
    assert p.mode("Glob", "Glob") == "allow"
    assert p.mode("delete_node", "mcp__workflow__delete_node") == "deny"
    assert p.mode("add_node", "mcp__workflow__add_node") == "ask"


def test_always_allow_only_below_an_allow_ceiling():
    p = _permissions(tool_ceilings={"Bash": "ask"})
    p.allow_for_conversation("Bash", "Bash")
    assert p.mode("Bash", "Bash") == "ask"
    p.allow_for_conversation("Write", "Write")
    assert p.mode("Write", "Write") == "allow"
    p.new_conversation()
    assert p.mode("Write", "Write") == "ask"


def test_policy_from_json_drops_bad_values():
    policy = Policy.from_json(
        {
            "category_ceilings": {"code-exec": "ask", "nope": "deny", "web": "x"},
            "tool_ceilings": {"Bash": "deny", "": "deny", "Edit": 3},
        }
    )
    assert policy.category_ceilings == {"code-exec": "ask"}
    assert policy.tool_ceilings == {"Bash": "deny"}


def test_fallback_policy_blocks_loosening():
    p = Permissions()  # backend policy not known yet
    assert p.mode("Read", "Read") == "allow"
    with pytest.raises(ValueError):
        p.set("code-exec", "allow")


def test_denied_builtins():
    p = _permissions()
    p.set("web", "deny")
    p.set("run_code", "deny")
    assert set(p.denied_builtins()) == {
        "WebFetch",
        "WebSearch",
        "mcp__notebook__run_code",
    }


def test_describe():
    assert describe("run_code", {"code": "print(1)"}) == "print(1)"
    assert describe("Bash", {"command": "ls", "description": "list"}) == "# list\nls"
    edit = describe(
        "Edit", {"file_path": "/a.py", "old_string": "x", "new_string": "y"}
    )
    assert edit == "/a.py\n--- old\nx\n+++ new\ny"
    long = describe("run_code", {"code": "x" * 5000})
    assert long.startswith("x" * 4000) and long.endswith("(1000 chars truncated)")
    assert '"workflow_id": "w"' in describe("add_node", {"workflow_id": "w"})


@pytest.mark.parametrize(
    "answer, can_always, expected",
    [
        ("y", True, PermissionDecision(allow=True)),
        ("YES", False, PermissionDecision(allow=True)),
        ("a", True, PermissionDecision(allow=True, always=True)),
        ("a", False, PermissionDecision(allow=True, always=False)),
        ("", True, PermissionDecision(allow=False)),
        ("n", True, PermissionDecision(allow=False)),
        ("use pandas", True, PermissionDecision(allow=False, reason="use pandas")),
    ],
)
def test_parse_answer(answer, can_always, expected):
    assert parse_answer(answer, can_always) == expected


def test_get_notebook_policy_uses_the_service_token(monkeypatch):
    calls = []

    class _Httpx:
        class Client:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, url, headers=None):
                calls.append((url, headers))
                return _FakeResponse(200, {"category_ceilings": {}})

    monkeypatch.setattr(BackendClient, "_httpx", lambda self: _Httpx)

    assert BackendClient(_config()).get_notebook_policy() == {"category_ceilings": {}}
    assert calls == [
        (
            "http://backend:3000/api/chat/notebook-policy/",
            {"x-api-key": "svc-token-123"},
        )
    ]


# ---------------------------------------------------------------------------
# The permission gate (can_use_tool)
# ---------------------------------------------------------------------------


def _gate(permissions, approve=None, timeout=None, decisions=None):
    run = _Run(lambda job: job(), RunHandle(), None, None, None, approve, timeout)

    def on_decision(request, decision):
        if decisions is not None:
            decisions.append((request.tool, decision.source, decision.allow))

    return _make_can_use_tool(ROOT, permissions, run, on_decision)


def _call(gate, tool_name, tool_input=None):
    return asyncio.run(gate(tool_name, tool_input or {}, None))


class _Approver:
    def __init__(self, decision):
        self.decision = decision
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.decision


def test_allow_and_deny_never_ask():
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    p = _permissions()
    p.set("web", "deny")
    approver = _Approver(PermissionDecision(allow=True))
    gate = _gate(p, approver)

    assert isinstance(_call(gate, "Read", {"file_path": "/x"}), PermissionResultAllow)
    denied = _call(gate, "WebFetch", {"url": "https://example.com"})
    assert isinstance(denied, PermissionResultDeny)
    assert "set to deny" in denied.message
    assert approver.requests == []


def test_ask_approve_deny_and_always():
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    p = _permissions()
    decisions = []
    approver = _Approver(PermissionDecision(allow=True))
    gate = _gate(p, approver, decisions=decisions)

    assert isinstance(_call(gate, "Bash", {"command": "ls"}), PermissionResultAllow)
    (request,) = approver.requests
    assert (request.tool, request.category, request.summary) == (
        "Bash",
        "code-exec",
        "ls",
    )
    assert request.can_always_allow

    approver.decision = PermissionDecision(allow=False, reason="not now")
    denied = _call(gate, "mcp__notebook__run_code", {"code": "1"})
    assert isinstance(denied, PermissionResultDeny)
    assert "declined this run_code call: not now" in denied.message

    approver.decision = PermissionDecision(allow=True, always=True)
    _call(gate, "Write", {"file_path": f"{ROOT}/a.py", "content": ""})
    asked = len(approver.requests)
    _call(gate, "Write", {"file_path": f"{ROOT}/b.py", "content": ""})
    assert len(approver.requests) == asked  # not asked again
    assert decisions[-1] == ("Write", "policy", True)
    assert decisions[0] == ("Bash", "user", True)


def test_ask_times_out_and_cancels_the_prompt():
    from claude_agent_sdk import PermissionResultDeny

    prompts = []

    def approve(request):
        future = concurrent.futures.Future()
        prompts.append(future)
        return future

    result = _call(
        _gate(_permissions(), approve, timeout=0.05), "Bash", {"command": "ls"}
    )

    assert isinstance(result, PermissionResultDeny)
    assert "No approval" in result.message
    assert prompts[0].cancelled()


def test_ask_without_approver_denies():
    from claude_agent_sdk import PermissionResultDeny

    result = _call(_gate(_permissions()), "Bash", {"command": "ls"})
    assert isinstance(result, PermissionResultDeny)
    assert "nobody can approve" in result.message


def test_hard_denies_win_over_allow():
    from claude_agent_sdk import PermissionResultDeny

    p = _permissions()
    p.set("code-exec", "allow")
    p.set("file-write", "allow")
    gate = _gate(p)

    bash = _call(gate, "Bash", {"command": "rm -rf /"})
    assert isinstance(bash, PermissionResultDeny) and "destructive" in bash.message
    write = _call(gate, "Write", {"file_path": "/etc/passwd"})
    assert (
        isinstance(write, PermissionResultDeny) and "outside workspace" in write.message
    )


# ---------------------------------------------------------------------------
# Agent.start / run: the worker thread and the main-thread bridge
# ---------------------------------------------------------------------------


def _agent_with(monkeypatch, arun):
    agent = _agent()
    agent._client.get_notebook_policy = lambda: {"workflow_read_tools": []}
    monkeypatch.setattr(type(agent), "_arun", arun)
    return agent


def test_run_services_callbacks_on_the_calling_thread(monkeypatch):
    caller = threading.get_ident()
    seen = []

    async def arun(self, message, run):
        run.notify(run.on_text, "hello ")
        decision = await asyncio.wrap_future(run.call_main(run.approve, "req"))
        run.notify(run.on_text, "approved" if decision.allow else "denied")
        return "final"

    agent = _agent_with(monkeypatch, arun)

    def on_text(delta):
        seen.append((delta, threading.get_ident() == caller))

    def approve(request):
        seen.append((request, threading.get_ident() == caller))
        return PermissionDecision(allow=True)

    assert agent.run("hi", on_text=on_text, approve=approve) == "final"
    assert seen == [("hello ", True), ("req", True), ("approved", True)]
    assert not agent.busy


def test_start_is_exclusive_and_cancel_stops_the_turn(monkeypatch):
    started = threading.Event()

    async def arun(self, message, run):
        started.set()
        await asyncio.sleep(30)

    agent = _agent_with(monkeypatch, arun)
    handle = agent.start("hi", dispatch=lambda job: job())
    assert started.wait(5)
    with pytest.raises(RuntimeError, match="still answering"):
        agent.start("again", dispatch=lambda job: job())

    handle.cancel()
    deadline = time.monotonic() + 5
    while not handle.done.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert handle.done.cancelled()
    assert not agent.busy


def test_interrupt_at_an_approval_prompt_stops_the_turn(monkeypatch, capsys):
    async def arun(self, message, run):
        await asyncio.wrap_future(run.call_main(run.approve, "req"))
        await asyncio.sleep(30)

    def approve(request):
        raise KeyboardInterrupt

    agent = _agent_with(monkeypatch, arun)
    assert agent.run("hi", approve=approve) == ""
    assert "[interrupted]" in capsys.readouterr().out
    assert not agent.busy


def test_run_reports_errors(monkeypatch):
    async def arun(self, message, run):
        raise RuntimeError("cli failed")

    agent = _agent_with(monkeypatch, arun)
    with pytest.raises(RuntimeError, match="cli failed"):
        agent.run("hi")
    assert not agent.busy


def test_policy_is_refreshed_at_start(monkeypatch):
    async def arun(self, message, run):
        return ""

    agent = _agent_with(monkeypatch, arun)
    agent._client.get_notebook_policy = lambda: {
        "category_ceilings": {"code-exec": "deny"}
    }
    agent.run("hi")
    assert agent.permissions.mode("Bash", "Bash") == "deny"

    def unreachable():
        raise OSError("down")

    agent._client.get_notebook_policy = unreachable
    agent.run("hi")  # keeps the last known policy
    assert agent.permissions.mode("Bash", "Bash") == "deny"


# ---------------------------------------------------------------------------
# run_code
# ---------------------------------------------------------------------------


def test_run_code_output_result_and_error():
    shown = []
    ns = {}

    text, interrupted = run_code("x = 2\nprint('hi')\nx * 3", ns, shown.append)
    assert text == "hi\n[result] 6"
    assert not interrupted
    assert ns["x"] == 2
    assert shown[0].stdout == "hi\n"

    text, _ = run_code("1/0", ns, shown.append)
    assert text == "[error] ZeroDivisionError: division by zero"


def test_run_code_reports_an_interrupt():
    text, interrupted = run_code("raise KeyboardInterrupt", {}, lambda c: None)
    assert interrupted
    assert text == "[error] interrupted by the user"


def test_defaults_cover_every_category():
    from neuroworkflow.agent.permissions import CATEGORIES

    assert set(DEFAULTS) == set(CATEGORIES)
