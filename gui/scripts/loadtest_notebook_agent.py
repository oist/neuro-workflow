#!/usr/bin/env python3
"""Concurrent load test for the in-notebook Claude agent.

Opens N kernels in one shared JupyterLab (as N users with a notebook open
would), sends one agent message in all of them at the same moment and reports:

* per-kernel success, latency, tool calls, failed workflow tool calls,
  turns / tokens / cost from the SDK result
* the Lab container's memory (idle with N kernels, and peak during the run),
  peak CPU and peak number of ``claude`` CLI processes, sampled from a separate
  monitor kernel via cgroup v2 files

Runs inside the backend container (it needs httpx + websockets and reaches the
hub at JUPYTERHUB_INTERNAL_HOST with JUPYTERHUB_API_TOKEN):

    docker compose exec -T -e NW_TOKEN=eyJ... backend \\
        python - -n 10 < gui/scripts/loadtest_notebook_agent.py

NW_TOKEN is a Keycloak access token (see loadtest_chat.py); it is handed to the
agent as ``user_token`` so workflow tools work without the browser token relay.
Each run makes real Anthropic calls and is billed accordingly.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid

import httpx
from websockets.asyncio.client import connect

_base = os.environ.get("JUPYTERHUB_BASE_URL", "/").strip("/")
HUB = os.environ.get("JUPYTERHUB_INTERNAL_HOST", "http://jupyterhub:8000").rstrip("/")
HUB = f"{HUB}/{_base}" if _base else HUB
HEADERS = {"Authorization": f"token {os.environ['JUPYTERHUB_API_TOKEN']}"}

DEFAULT_MESSAGE = (
    "Use the list_projects workflow tool once, then reply with only the number "
    "of projects. Do not use any other tool."
)

# Monitor kernel: background thread sampling the Lab container's cgroup.
MONITOR_START = r"""
import os, threading, time
_mon = {"stop": False, "samples": []}
def _claude_procs():
    n = rss = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
            if b"claude" not in cmd or b"ipykernel" in cmd:
                continue
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1]) * 1024
            n += 1
        except OSError:
            pass
    return n, rss
def _cpu_usec():
    for line in open("/sys/fs/cgroup/cpu.stat"):
        if line.startswith("usage_usec"):
            return int(line.split()[1])
def _sample():
    last_t, last_cpu = time.monotonic(), _cpu_usec()
    while not _mon["stop"]:
        time.sleep(0.5)
        t, cpu = time.monotonic(), _cpu_usec()
        n, rss = _claude_procs()
        _mon["samples"].append({
            "t": t,
            "mem": int(open("/sys/fs/cgroup/memory.current").read()),
            "cores": (cpu - last_cpu) / 1e6 / (t - last_t),
            "claude": n,
            "claude_rss": rss,
        })
        last_t, last_cpu = t, cpu
threading.Thread(target=_sample, daemon=True).start()
"""

MONITOR_READ = r"""
import json
_s = [x for x in _mon["samples"] if x["t"] >= {since}]
print("@@RESULT@@" + json.dumps({{
    "mem_now": int(open("/sys/fs/cgroup/memory.current").read()),
    "mem_peak": max((x["mem"] for x in _s), default=None),
    "cores_peak": max((x["cores"] for x in _s), default=None),
    "claude_peak": max((x["claude"] for x in _s), default=None),
    "claude_rss_peak": max((x["claude_rss"] for x in _s), default=None),
    "now": time.monotonic(),
}}))
"""

AGENT_SETUP = r"""
import json, time
import claude_agent_sdk as _sdk
from neuroworkflow.agent import get_agent
{extra_import}
_res = {{}}
_orig_query = _sdk.query
def _recording_query(*a, **k):
    async def gen():
        async for m in _orig_query(*a, **k):
            if isinstance(m, _sdk.ResultMessage):
                _res.update(is_error=m.is_error, subtype=m.subtype,
                            turns=m.num_turns, cost=m.total_cost_usd,
                            usage=m.usage)
            yield m
    return gen()
_sdk.query = _recording_query  # loop.py imports query at call time
_agent = get_agent(user_token={token!r})
_tool_results = []
_orig_call = _agent._client.call_mcp_tool
def _recording_call(name, args):
    out = _orig_call(name, args)
    _tool_results.append(out)
    return out
_agent._client.call_mcp_tool = _recording_call
print("@@RESULT@@" + json.dumps({{"workflow_tools": "workflow" in _agent._servers}}))
"""

AGENT_RUN = r"""
_tools, _err, _text = [], None, ""
_t0 = time.monotonic()
try:
    _text = _agent.run({message!r}, on_tool=lambda n, a: _tools.append(n))
except Exception as e:
    _err = repr(e)
print("@@RESULT@@" + json.dumps({{
    "seconds": time.monotonic() - _t0, "tools": _tools, "error": _err,
    "text": _text[:200], "tool_results": [str(r)[:200] for r in _tool_results],
    "result": _res,
}}))
"""


async def ensure_server(client, user):
    r = await client.get(f"{HUB}/hub/api/users/{user}", headers=HEADERS)
    r.raise_for_status()
    if r.json().get("servers", {}).get("", {}).get("ready"):
        return
    await client.post(f"{HUB}/hub/api/users/{user}/server", headers=HEADERS)
    for _ in range(150):
        await asyncio.sleep(2)
        r = await client.get(f"{HUB}/hub/api/users/{user}", headers=HEADERS)
        if r.json().get("servers", {}).get("", {}).get("ready"):
            return
    raise TimeoutError(f"server for {user} did not start")


async def execute(user, kernel_id, code, timeout):
    """Run code in a kernel; return the JSON after the @@RESULT@@ marker."""
    ws_url = (
        HUB.replace("http", "ws", 1) + f"/user/{user}/api/kernels/{kernel_id}/channels"
    )
    msg_id = uuid.uuid4().hex
    request = {
        "header": {
            "msg_id": msg_id,
            "msg_type": "execute_request",
            "username": user,
            "session": uuid.uuid4().hex,
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "buffers": [],
        "channel": "shell",
    }
    out, error = [], None
    async with connect(ws_url, additional_headers=HEADERS, open_timeout=60) as ws:
        await ws.send(json.dumps(request))
        deadline = time.monotonic() + timeout
        while True:
            raw = await asyncio.wait_for(ws.recv(), max(1, deadline - time.monotonic()))
            msg = json.loads(raw)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind = msg["header"]["msg_type"]
            if kind == "stream":
                out.append(msg["content"]["text"])
            elif kind == "error":
                error = f"{msg['content']['ename']}: {msg['content']['evalue']}"
            elif kind == "status" and msg["content"]["execution_state"] == "idle":
                break
    text = "".join(out)
    if "@@RESULT@@" in text:
        return json.loads(text.split("@@RESULT@@", 1)[1].splitlines()[0])
    raise RuntimeError(error or f"no result; output: {text[-500:]}")


def is_tool_error(text):
    if text.startswith(("[error]", "Error")):
        return True
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("status") == "error"


def gb(n):
    return f"{n / 2**30:.2f} GB" if n is not None else "-"


async def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-n", "--concurrency", type=int, default=5)
    p.add_argument(
        "--hub-user", default=os.environ.get("JUPYTERHUB_PROJECT_USER", "internal")
    )
    p.add_argument("--message", default=DEFAULT_MESSAGE)
    p.add_argument(
        "--import-nest",
        action="store_true",
        help="also `import nest` in every kernel, like a typical notebook",
    )
    p.add_argument("--timeout", type=float, default=900, help="per-message seconds")
    args = p.parse_args()
    token = os.environ.get("NW_TOKEN")
    if not token:
        p.error("set NW_TOKEN")
    user, n = args.hub_user, args.concurrency
    base = f"{HUB}/user/{user}/api/kernels"

    async with httpx.AsyncClient(timeout=120) as client:
        await ensure_server(client, user)

        async def new_kernel():
            r = await client.post(base, headers=HEADERS, json={"name": "python3"})
            r.raise_for_status()
            return r.json()["id"]

        monitor = await new_kernel()
        kernels = []
        try:
            await execute(user, monitor, MONITOR_START + "print('@@RESULT@@{}')", 60)
            before = await execute(user, monitor, MONITOR_READ.format(since=0), 60)

            print(f"Starting {n} kernels in the '{user}' Lab ...")
            kernels = list(await asyncio.gather(*(new_kernel() for _ in range(n))))
            setup = AGENT_SETUP.format(
                token=token, extra_import="import nest" if args.import_nest else ""
            )
            setups = await asyncio.gather(
                *(execute(user, k, setup, 300) for k in kernels), return_exceptions=True
            )
            bad = [
                s for s in setups if isinstance(s, Exception) or not s["workflow_tools"]
            ]
            if bad:
                print(
                    f"  {len(bad)} kernels failed setup / have no workflow tools: {bad[:3]}"
                )
            idle = await execute(user, monitor, MONITOR_READ.format(since=0), 60)

            print(f"Sending {n} agent messages at once ...")
            run_code = AGENT_RUN.format(message=args.message)
            wall = time.monotonic()
            results = await asyncio.gather(
                *(execute(user, k, run_code, args.timeout) for k in kernels),
                return_exceptions=True,
            )
            wall = time.monotonic() - wall
            during = await execute(
                user, monitor, MONITOR_READ.format(since=idle["now"]), 60
            )
        finally:
            for k in kernels + [monitor]:
                await client.delete(f"{base}/{k}", headers=HEADERS)

    ok, costs, inputs = [], [], []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"#{i:>3} FAIL {type(r).__name__}: {r}")
            continue
        tool_errors = [t for t in r["tool_results"] if is_tool_error(t)]
        res = r["result"]
        good = (
            not r["error"]
            and not tool_errors
            and not res.get("is_error")
            # Only the default prompt must call a tool; custom ones may not.
            and (r["tools"] or args.message != DEFAULT_MESSAGE)
        )
        if good:
            ok.append(r["seconds"])
        if res.get("cost") is not None:
            costs.append(res["cost"])
        usage = res.get("usage") or {}
        inputs.append(
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
        print(
            f"#{i:>3} {'OK  ' if good else 'FAIL'} {r['seconds']:.1f}s "
            f"turns={res.get('turns')} tools={r['tools']} "
            f"in_tokens={inputs[-1]} out_tokens={usage.get('output_tokens')}"
        )
        for msg in [r["error"]] + tool_errors:
            if msg:
                print(f"       {msg}")
        if not good and not r["error"] and not tool_errors:
            print(f"       is_error={res.get('is_error')} text={r['text']!r}")

    print("\nSummary")
    print(f"  succeeded          : {len(ok)}/{n}")
    print(f"  wall time          : {wall:.1f}s")
    if ok:
        print(
            f"  latency (ok)       : p50={statistics.median(ok):.1f}s max={max(ok):.1f}s"
        )
    print(f"  Lab memory before  : {gb(before['mem_now'])}")
    print(f"  Lab memory idle    : {gb(idle['mem_now'])}  (with {n} agent kernels)")
    print(f"  Lab memory peak    : {gb(during['mem_peak'])}  (during the run)")
    print(
        f"  claude CLI peak    : {during['claude_peak']} procs, {gb(during['claude_rss_peak'])} RSS"
    )
    print(f"  CPU peak           : {during['cores_peak']:.1f} cores")
    if inputs:
        print(
            f"  input tokens/msg   : p50={statistics.median(inputs):.0f} (incl. cache)"
        )
    if costs:
        print(
            f"  cost               : total ${sum(costs):.2f}, ${statistics.median(costs):.3f}/msg"
        )


if __name__ == "__main__":
    asyncio.run(main())
