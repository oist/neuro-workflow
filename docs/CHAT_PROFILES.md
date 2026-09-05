# Chat Profiles (browser chat)

Chat Profiles let each user control which MCP tools the browser **AI Assistant**
may use, and optionally override its system prompt. Profiles are stored per user
on the backend and switched from the chat header.

## Concepts

| Term | Meaning |
|---|---|
| **Default** (no profile) | Unchanged behaviour: every MCP tool is offered and `DEFAULT_SYSTEM_PROMPT` is used. |
| `allowed_tools` | Explicit allowlist of MCP tool names. Only these tools are offered to OpenAI **and** allowed to execute. |
| `allowed_tools = []` | Tools disabled: the backend skips MCP discovery entirely and appends `TOOLS_DISABLED_NOTE` to the system prompt. |
| `system_prompt` | Optional override. Precedence: profile prompt > `Conversation.system_prompt` > `DEFAULT_SYSTEM_PROMPT`. |

Because the allowlist is explicit, **new MCP tools start unchecked in existing
profiles**. They appear under "Other" in the editor until they are categorised
in `chatToolCategories.ts`.

When a profile restricts (but does not disable) tools, the backend appends
`TOOLS_RESTRICTED_NOTE` listing the enabled tools, because the default prompt
refers to tools by name.

## Using it

1. **Settings → Chat Profiles** (`/settings/chat-profiles`): create a profile —
   name, optional system prompt, and the tool picker (grouped by category, with
   per-category and per-tool checkboxes plus *Select all* / *Select none*).
2. In the chat header, pick the profile from the dropdown next to the
   conversation selector. The selection is remembered per user in this browser
   (`localStorage` key `chatProfileId:<user key>`, where the key is the Keycloak
   `sub`, falling back to `preferred_username` / email when the access token
   carries no `sub` — the same order the backend maps users by) and sent as
   `profile_id` with every message, so it can be switched mid-conversation.
3. The **Generate report** button is disabled when the selected profile lacks
   `get_workflow_facts` or `save_report`.

## API

| Method | Endpoint | Notes |
|---|---|---|
| GET / POST | `/api/chat/profiles/` | List / create the caller's profiles. Body: `{name, allowed_tools: string[], system_prompt}` |
| GET / PUT / DELETE | `/api/chat/profiles/<uuid>/` | Owner-scoped (404 otherwise). PUT is partial. DELETE returns 204 |
| POST | `/api/chat/stream/` | Accepts optional `profile_id`; an unknown or foreign id returns 404 before any conversation is created |
| GET | `/api/chat/mcp-tools/` | Tool catalog used by the editor (shared with the notebook agent; shape unchanged) |

## Code map

Backend (`gui/workflow_backend/django-project/app/chat/`):
`models.py` (`ChatProfile`, migration `0002_chatprofile`),
`serializers.py` (`ChatProfileSerializer`, `SendMessageSerializer.profile_id`),
`views.py` (`ChatProfileListCreateView`, `ChatProfileDetailView`, `ChatStreamView`),
`services/mcp_client.py` (`mcp_tools_to_openai_functions(..., allowed=)`),
`services/chat_orchestrator.py` (`orchestrate_chat(..., profile=)`).

Frontend (`gui/workflow_frontend/src/`):
`api/chatProfileApi.ts`, `stores/chatProfileStore.ts`,
`views/home/components/ChatProfileSelector.tsx`, `ChatProfileManager.tsx`,
`ChatProfileModal.tsx`, `chatToolCategories.ts`; wired in `chatbotView.tsx`,
`components/tabs/TabManager.tsx` and `shared/header/header.tsx`.

Tests: `gui/workflow_backend/django-project/tests/test_chat_profiles.py`.
