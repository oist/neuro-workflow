import json
import logging

from asgiref.sync import sync_to_async

from ..models import Conversation, Message
from .mcp_client import MCPClient, mcp_tools_to_openai_functions
from .openai_client import stream_chat_completion

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for NeuroWorkflow, a visual workflow editor for neuroscience simulations. "
    "You can help users manage workflow projects, nodes, edges, and parameters using the available tools.\n\n"
    "Key rules:\n"
    "- When reading parameter values, always use 'default_value' as the current value. Ignore any 'value' field.\n"
    "- When a user asks to change a parameter, call update_node_parameter with parameter_field='default_value'.\n"
    "- When a user asks to generate a report, methods section, or paper section about a workflow:\n"
    "  1. Call get_workflow_facts (NOT get_flow) to collect all node parameters and results.\n"
    "  2. Write the report as a proper scientific Methods section following these rules:\n"
    "     - Write in continuous, flowing scientific prose — no bullet lists, no UI jargon.\n"
    "     - NEVER mention node names, class names, or workflow software internals (no 'Simulator node',\n"
    "       'ConnectivitySetUp node', 'node', 'edge', 'workflow', 'NeuroWorkflow', etc.).\n"
    "     - Describe the neuroscience experiment: what model was used, what parameters, what stimulation,\n"
    "       what was recorded, and how results were stored — as a scientist would write it.\n"
    "     - Embed ALL parameter values directly in the narrative (e.g. 'a simulation duration of\n"
    "       30,000 ms', 'a coupling scaling factor of 0.0075', 'a time step of 0.5 ms').\n"
    "     - Use generated_code to understand the exact computational steps and library calls made.\n"
    "     - Use notebook_outputs (stdout, printed values, errors) to describe what actually ran and\n"
    "       what results were produced — mention key output values if present.\n"
    "     - For result files: interpret array shapes as (time_steps, output_channels) and describe\n"
    "       them scientifically based on the variable names and node parameters.\n"
    "     - Use standard scientific terminology appropriate to the domain (neuroscience, physics, etc.).\n"
    "     - Mention specific software/libraries by name as found in the generated code.\n"
    "     - Use sections: Abstract (1 paragraph), Computational Model, Simulation Setup,\n"
    "       Input Data / Connectivity (if any), Stimulation Protocol (if any), Data Recording and Output.\n"
    "  3. Call save_report (NOT upload_python_file) to save it to the project folder.\n"
    "  4. Reply with only a short confirmation — do NOT print the full report text in chat.\n"
    "- Never use get_flow for report generation. Never use upload_python_file to save reports.\n"
    "- Always respond in the same language the user uses."
)

MAX_AGENT_LOOPS = 10


@sync_to_async
def _create_message(**kwargs):
    return Message.objects.create(**kwargs)


@sync_to_async
def _build_openai_messages(conversation: Conversation) -> list[dict]:
    """Build the OpenAI messages array from conversation history."""
    messages = []

    # System prompt — inject active project context if available
    system_prompt = conversation.system_prompt or DEFAULT_SYSTEM_PROMPT
    if conversation.project_id:
        try:
            project = conversation.project
            project_context = (
                f"\n\nACTIVE PROJECT: \"{project.name}\" (workflow_id: {project.id}). "
                "Use this workflow_id in all tool calls that require it unless the user specifies otherwise."
            )
            system_prompt = system_prompt + project_context
        except Exception:
            pass
    messages.append({"role": "system", "content": system_prompt})

    # History
    for msg in conversation.messages.all():
        messages.append(msg.to_openai_format())

    return messages


async def orchestrate_chat(
    conversation: Conversation,
    user_message: str,
    auth_token: str | None = None,
):
    """Run the agent loop: LLM -> tool calls -> LLM -> ... -> final response.

    ``auth_token`` is the end-user's bearer JWT, forwarded through MCPClient
    so workflow_mcp tools can present it when calling the Django API.

    This is an async generator that yields SSE event dicts.
    """
    # 1. Save user message
    await _create_message(
        conversation=conversation,
        role="user",
        content=user_message,
    )

    # 2. Initialize MCP client and get tools
    mcp = MCPClient(auth_token=auth_token)
    try:
        await mcp.initialize()
    except Exception as e:
        logger.warning("MCP initialize failed (may already be initialized): %s", e)

    try:
        mcp_tools = await mcp.list_tools()
        openai_tools = mcp_tools_to_openai_functions(mcp_tools)
    except Exception as e:
        logger.error("Failed to get MCP tools: %s", e)
        openai_tools = []

    # 3. Agent loop
    for loop_idx in range(MAX_AGENT_LOOPS):
        # Build message history for OpenAI
        messages = await _build_openai_messages(conversation)

        # Stream OpenAI response
        full_content = ""
        tool_calls_map = {}  # index -> {id, name, arguments}

        async for chunk in stream_chat_completion(messages, openai_tools or None):
            if chunk["type"] == "content_delta":
                full_content += chunk["content"]
                yield {
                    "type": "text_delta",
                    "data": {"content": chunk["content"]},
                }

            elif chunk["type"] == "tool_call_delta":
                idx = chunk["index"]
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        "id": chunk.get("id", ""),
                        "name": chunk.get("function_name", ""),
                        "arguments": "",
                    }
                tc = tool_calls_map[idx]
                if chunk.get("id"):
                    tc["id"] = chunk["id"]
                if chunk.get("function_name"):
                    tc["name"] = chunk["function_name"]
                    yield {
                        "type": "tool_call_start",
                        "data": {
                            "tool_name": tc["name"],
                            "tool_call_id": tc["id"],
                        },
                    }
                if chunk.get("arguments_delta"):
                    tc["arguments"] += chunk["arguments_delta"]
                    yield {
                        "type": "tool_call_args_delta",
                        "data": {"content": chunk["arguments_delta"]},
                    }

            elif chunk["type"] == "error":
                yield {"type": "error", "data": {"message": chunk["message"]}}
                return

            elif chunk["type"] in ("done", "tool_calls_complete"):
                break

        # Save assistant message
        if tool_calls_map:
            openai_tool_calls = []
            for idx in sorted(tool_calls_map.keys()):
                tc = tool_calls_map[idx]
                openai_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                })

            await _create_message(
                conversation=conversation,
                role="assistant",
                content=full_content,
                tool_calls=openai_tool_calls,
            )

            # Execute each tool call via MCP
            for tc_data in openai_tool_calls:
                tool_name = tc_data["function"]["name"]
                tool_call_id = tc_data["id"]
                try:
                    arguments = json.loads(tc_data["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                try:
                    result = await mcp.call_tool(tool_name, arguments)
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"
                    logger.error("MCP tool call error for %s: %s", tool_name, e)

                # Save tool result message
                await _create_message(
                    conversation=conversation,
                    role="tool",
                    content=result,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                )

                yield {
                    "type": "tool_result",
                    "data": {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "result": result,
                    },
                }

            # Continue the loop — LLM will process tool results
            continue

        else:
            # No tool calls — final text response
            if full_content:
                await _create_message(
                    conversation=conversation,
                    role="assistant",
                    content=full_content,
                )
            yield {"type": "done", "data": {}}
            return

    # Max loops reached
    yield {"type": "done", "data": {}}
