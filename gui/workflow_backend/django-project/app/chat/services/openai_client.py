import httpx
import json
import logging
import os

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


async def stream_chat_completion(
    messages: list[dict],
    tools: list[dict] | None = None,
):
    """Stream a model response from the OpenAI Responses API.

    ``messages`` and ``tools`` use the Chat Completions format and are
    converted here. Yields parsed chunks as dicts. Each chunk has a "type" field:
      - "content_delta": partial text content
      - "tool_call_delta": partial tool call data
      - "tool_calls_complete": stream finished with tool calls
      - "done": stream finished
      - "error": an error occurred
    """
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
