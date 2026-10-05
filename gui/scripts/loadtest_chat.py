#!/usr/bin/env python3
"""Concurrent load test for the browser AI Assistant (``/api/chat/stream``).

Fires N chat messages at the same moment, reads each SSE stream to the end and
reports latency plus the failure modes that cap concurrency:

* HTTP errors / dropped streams (backend worker exhausted or killed)
* failed tool results (MCP -> backend call starved or timed out); the model
  still answers in that case, so these are not visible as HTTP errors
* ``error`` events, incl. OpenAI 429 rate limits

Every request uses the same Keycloak access token, each in its own new
conversation; the conversations are soft-deleted afterwards unless --keep.
Each run makes real OpenAI calls and is billed accordingly.

Getting a token: log in to the GUI, open DevTools > Network, pick any /api/
request and copy the value after "Authorization: Bearer ". Tokens expire
(realm default 1 h).

Usage (stdlib only, Python 3.8+):
    NW_TOKEN=eyJ... python3 gui/scripts/loadtest_chat.py -n 10
    python3 gui/scripts/loadtest_chat.py --base-url https://example.org -n 40 \\
        --token eyJ... --message "Say hello."
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Default prompt forces one MCP tool call so the MCP -> backend round trip is
# exercised, not just the OpenAI stream.
DEFAULT_MESSAGE = (
    "Call the list_projects tool once, then reply with only the number of "
    "projects. Do not call any other tool."
)


def is_tool_error(text):
    # Orchestrator-side failures are plain "Error ..." strings; MCP tools that
    # could not reach the backend return {"status": "error", ...} instead.
    if text.startswith("Error"):
        return True
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("status") == "error"


def run_one(idx, args, barrier):
    result = {
        "idx": idx,
        "status": None,
        "first_event_s": None,
        "total_s": None,
        "tool_calls": 0,
        "tool_errors": [],
        "errors": [],
        "done": False,
        "conversation_id": None,
    }
    body = {"message": args.message}
    if args.project_id:
        body["project_id"] = args.project_id
    req = urllib.request.Request(
        f"{args.base_url}/api/chat/stream/",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {args.token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    barrier.wait()
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            result["status"] = resp.status
            event = None
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("event: "):
                    event = line[7:]
                    continue
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if event == "conversation_id":
                    result["conversation_id"] = data.get("id")
                    continue
                if result["first_event_s"] is None:
                    result["first_event_s"] = time.monotonic() - start
                if event == "tool_call_start":
                    result["tool_calls"] += 1
                elif event == "tool_result":
                    text = str(data.get("result", ""))
                    if is_tool_error(text):
                        result["tool_errors"].append(text[:200])
                elif event == "error":
                    result["errors"].append(str(data.get("message", ""))[:300])
                elif event == "done":
                    result["done"] = True
    except urllib.error.HTTPError as e:
        result["status"] = e.code
        result["errors"].append(f"HTTP {e.code}: {e.read()[:200]!r}")
    except Exception as e:  # timeouts, resets, truncated streams
        result["errors"].append(f"{type(e).__name__}: {e}")
    # A text-only answer to the default prompt (tools disabled by a chat
    # profile, or MCP tool discovery failed) skips the MCP -> backend round
    # trip this test exists to exercise.
    if args.message == DEFAULT_MESSAGE and result["done"] and not result["tool_calls"]:
        result["errors"].append("no tool call: the default prompt expects one")
    result["total_s"] = time.monotonic() - start
    return result


def delete_conversation(args, conversation_id):
    req = urllib.request.Request(
        f"{args.base_url}/api/chat/conversations/{conversation_id}/",
        headers={"Authorization": f"Bearer {args.token}"},
        method="DELETE",
    )
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except Exception as e:
        print(f"  cleanup failed for {conversation_id}: {e}", file=sys.stderr)


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://localhost:3000")
    p.add_argument("--token", default=os.environ.get("NW_TOKEN"))
    p.add_argument("-n", "--concurrency", type=int, default=5)
    p.add_argument("--message", default=DEFAULT_MESSAGE)
    p.add_argument("--project-id", help="optional project UUID for context")
    p.add_argument("--timeout", type=float, default=600, help="per-request seconds")
    p.add_argument("--keep", action="store_true", help="keep test conversations")
    args = p.parse_args()
    if not args.token:
        p.error("pass --token or set NW_TOKEN")
    args.base_url = args.base_url.rstrip("/")

    n = args.concurrency
    barrier = threading.Barrier(n)
    print(f"Sending {n} concurrent chat messages to {args.base_url} ...")
    wall = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda i: run_one(i, args, barrier), range(n)))
    wall = time.monotonic() - wall

    for r in results:
        ok = r["done"] and not r["errors"] and not r["tool_errors"]
        first = f"{r['first_event_s']:.1f}s" if r["first_event_s"] else "-"
        print(
            f"#{r['idx']:>3} {'OK  ' if ok else 'FAIL'} http={r['status']} "
            f"first={first} total={r['total_s']:.1f}s tools={r['tool_calls']}"
        )
        for msg in r["errors"] + r["tool_errors"]:
            print(f"       {msg}")

    ok = [r for r in results if r["done"] and not r["errors"] and not r["tool_errors"]]
    totals = [r["total_s"] for r in ok]
    rate_limited = sum(any("429" in e for e in r["errors"]) for r in results)
    print("\nSummary")
    print(f"  succeeded      : {len(ok)}/{n}")
    print(f"  tool errors    : {sum(bool(r['tool_errors']) for r in results)}")
    print(f"  OpenAI 429     : {rate_limited}")
    print(f"  wall time      : {wall:.1f}s")
    if totals:
        print(
            f"  latency (ok)   : p50={statistics.median(totals):.1f}s "
            f"p95={pct(totals, 0.95):.1f}s max={max(totals):.1f}s"
        )

    if not args.keep:
        for r in results:
            if r["conversation_id"]:
                delete_conversation(args, r["conversation_id"])
    return 0 if len(ok) == n else 1


if __name__ == "__main__":
    sys.exit(main())
