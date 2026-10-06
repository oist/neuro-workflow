import httpx
import json
import logging
import os

from .llm_providers import minimax_api_key, minimax_models, minimax_openai_base_url

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
# Reasoning effort for reasoning models (e.g. "low", or "none" for gpt-5.6);
# non-reasoning models such as gpt-4o reject it, so it is only sent when set.
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "")
# The Responses API is used because newer models (gpt-6.1+) reject function
# tools combined with reasoning on /v1/chat/completions.
OPENAI_API_URL = "https://api.openai.com/v1/responses"


def _to_responses_input(messages: list[dict]) -> list[dict]:
    """Convert Chat Completions-style messages to Responses API input items."""
    items = []
    for msg in messages:
        role = msg["role"]
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": msg.get("content") or "",
            })
            continue
        if msg.get("content"):
            items.append({"role": role, "content": msg["content"]})
        for tc in msg.get("tool_calls") or []:
            items.append({
                "type": "function_call",
                "call_id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            })
    return items


def _to_responses_tools(tools: list[dict]) -> list[dict]:
    """Flatten Chat Completions function tools into the Responses API shape."""
    return [
        {
            "type": "function",
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"]["parameters"],
            # The Responses API defaults to strict schemas, which MCP tool
            # inputSchemas do not satisfy.
            "strict": False,
        }
        for t in tools
    ]


def _merge_leading_system_messages(messages: list[dict]) -> list[dict]:
    """Collapse the leading system messages (prompt + viewer state) into one."""
    count = 0
    while count < len(messages) and messages[count]["role"] == "system":
        count += 1
    if count < 2:
        return messages
    merged = "\n\n".join(m["content"] for m in messages[:count])
    return [{"role": "system", "content": merged}, *messages[count:]]


async def _stream_minimax(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
):
    """Stream a model response from MiniMax's OpenAI-compatible Chat Completions.

    Yields the same chunks as ``stream_chat_completion`` plus "reasoning_delta"
    (the model's thinking, kept apart from the answer by ``reasoning_split``).
    """
    headers = {
        "Authorization": f"Bearer {minimax_api_key()}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": _merge_leading_system_messages(messages),
        "stream": True,
        "reasoning_split": True,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{minimax_openai_base_url()}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield {"type": "error", "message": f"MiniMax API error {response.status_code}: {body.decode()}"}
                    return

                finish_reason = None
                has_tool_calls = False
                named_tool_calls = set()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # The trailing usage chunk has no choices.
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    if delta.get("reasoning_content"):
                        yield {"type": "reasoning_delta", "content": delta["reasoning_content"]}
                    if delta.get("content"):
                        yield {"type": "content_delta", "content": delta["content"]}

                    for tc in delta.get("tool_calls") or []:
                        has_tool_calls = True
                        index = tc.get("index", 0)
                        function = tc.get("function") or {}
                        # Report the name once per call: the orchestrator
                        # announces a tool call each time it sees one.
                        name = None
                        if function.get("name") and index not in named_tool_calls:
                            named_tool_calls.add(index)
                            name = function["name"]
                        yield {
                            "type": "tool_call_delta",
                            "index": index,
                            "id": tc.get("id"),
                            "function_name": name,
                            "arguments_delta": function.get("arguments") or "",
                        }

                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]

                # A truncated response may carry truncated tool arguments, so it
                # must not be executed or saved as a finished reply.
                if finish_reason == "length":
                    yield {"type": "error", "message": "MiniMax response incomplete: max output tokens reached"}
                elif finish_reason is None:
                    yield {"type": "error", "message": "MiniMax stream ended without a result"}
                else:
                    yield {"type": "tool_calls_complete" if has_tool_calls else "done"}

    except httpx.HTTPError as e:
        logger.error("MiniMax HTTP error: %s", e)
        yield {"type": "error", "message": f"MiniMax connection error: {str(e)}"}
    except Exception as e:
        logger.error("MiniMax unexpected error: %s", e, exc_info=True)
        yield {"type": "error", "message": f"Unexpected error: {str(e)}"}


async def stream_chat_completion(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
):
    """Stream a model response from the OpenAI Responses API.

    ``model`` selects a MiniMax model when it is one of ``minimax_models()``;
    anything else (including None) uses ``OPENAI_MODEL``.

    ``messages`` and ``tools`` use the Chat Completions format and are
    converted here. Yields parsed chunks as dicts. Each chunk has a "type" field:
      - "content_delta": partial text content
      - "tool_call_delta": partial tool call data
      - "tool_calls_complete": stream finished with tool calls
      - "done": stream finished
      - "error": an error occurred
    """
    if model in minimax_models():
        async for chunk in _stream_minimax(model, messages, tools):
            yield chunk
        return

    if not OPENAI_API_KEY:
        yield {"type": "error", "message": "OpenAI API key is not configured"}
        return

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": _to_responses_input(messages),
        "stream": True,
        "store": False,
    }

    if OPENAI_REASONING_EFFORT:
        payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}

    if tools:
        payload["tools"] = _to_responses_tools(tools)
        payload["tool_choice"] = "auto"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", OPENAI_API_URL, json=payload, headers=headers,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield {"type": "error", "message": f"OpenAI API error {response.status_code}: {body.decode()}"}
                    return

                has_tool_calls = False
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    # Text content delta (refusals are shown as text too)
                    if event_type in ("response.output_text.delta", "response.refusal.delta"):
                        yield {"type": "content_delta", "content": event.get("delta", "")}

                    # Tool call start (id + name) and argument deltas
                    elif event_type == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            has_tool_calls = True
                            yield {
                                "type": "tool_call_delta",
                                "index": event.get("output_index", 0),
                                "id": item.get("call_id"),
                                "function_name": item.get("name"),
                                "arguments_delta": item.get("arguments", ""),
                            }
                    elif event_type == "response.function_call_arguments.delta":
                        yield {
                            "type": "tool_call_delta",
                            "index": event.get("output_index", 0),
                            "id": None,
                            "function_name": None,
                            "arguments_delta": event.get("delta", ""),
                        }

                    elif event_type == "response.completed":
                        yield {"type": "tool_calls_complete" if has_tool_calls else "done"}
                        return
                    # An incomplete response may carry truncated tool arguments,
                    # so it must not be executed or saved as a finished reply.
                    elif event_type == "response.incomplete":
                        details = (event.get("response") or {}).get("incomplete_details") or {}
                        yield {"type": "error", "message": f"OpenAI response incomplete: {details.get('reason', details)}"}
                        return
                    elif event_type == "response.failed":
                        error = (event.get("response") or {}).get("error") or {}
                        yield {"type": "error", "message": f"OpenAI response failed: {error.get('message', error)}"}
                        return
                    elif event_type == "error":
                        yield {"type": "error", "message": f"OpenAI stream error: {event.get('message', event)}"}
                        return

    except httpx.HTTPError as e:
        logger.error("OpenAI HTTP error: %s", e)
        yield {"type": "error", "message": f"OpenAI connection error: {str(e)}"}
    except Exception as e:
        logger.error("OpenAI unexpected error: %s", e, exc_info=True)
        yield {"type": "error", "message": f"Unexpected error: {str(e)}"}
