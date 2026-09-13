# Chat Profiles (browser chat)

Chat Profiles control which MCP tools the browser **AI Assistant** may use and
optionally override its system prompt. Profiles are **shared presets managed by
administrators**: staff create and edit them under Settings, and every user picks
one from the chat header.

## Concepts

| Term | Meaning |
|---|---|
| **Default** (no profile) | Unchanged behaviour: every MCP tool the server advertises is offered and `DEFAULT_SYSTEM_PROMPT` is used. Available to staff, and to everyone while no default profile is set. |
| **Default profile** | At most one profile flagged `is_default`. While it is set, non-staff users cannot pick "Default (all tools)"; the backend applies the default profile whenever a non-staff request carries no `profile_id`. |
| **Admin / staff** | A Django user with `is_staff = True` (the same flag that gates custom-database management). Only staff may create, edit, delete or flag profiles. |
| `allowed_tools` | Explicit allowlist of MCP tool names. Only these tools are offered to OpenAI **and** allowed to execute. |
| `allowed_tools = []` | Tools disabled: the backend skips MCP discovery entirely and appends `TOOLS_DISABLED_NOTE` to the system prompt. |
| `system_prompt` | Optional override. Precedence: profile prompt > `Conversation.system_prompt` > `DEFAULT_SYSTEM_PROMPT`. |

Because the allowlist is explicit, **new MCP tools start unchecked in existing
profiles**. They appear under "Other" in the editor until they are categorised
in `chatToolCategories.ts`. A **new** profile starts with every tool currently
listed in the picker checked; the user can still Select none (`allowed_tools =
[]` remains valid).

When a profile restricts (but does not disable) tools, the backend appends
`TOOLS_RESTRICTED_NOTE` listing the enabled tools, because the default prompt
refers to tools by name.

"Default (all tools)" exposes **every** tool the MCP server advertises,
including any that manage user credentials or other sensitive state. Set a
default profile with an explicit allowlist to keep such tools away from
non-staff chats.

Chat Profiles apply only to the browser AI Assistant (`POST /api/chat/stream/`).
The Jupyter notebook agent and its proxies (`GET /api/chat/mcp-tools/` and
`POST /api/chat/mcp-call/`) do not use profiles.

## Granting admin rights

Users are provisioned in Django automatically on their first Keycloak login, so
flag the existing row rather than creating a new account. From `gui/`:

```bash
docker-compose exec backend python django-project/manage.py shell -c \
  "from django.contrib.auth import get_user_model as U; u = U().objects.get(email='alice@example.com'); u.is_staff = True; u.save()"
```

Alternatively tick **Staff status** on the user in the Django admin (`/admin/`;
creating the superuser needed to log in there is described in
`docs/RECOVER_LEGACY_PROJECTS.md`, Step 1).

## Using it

1. **Settings → Chat Profiles** (`/settings/chat-profiles`), as staff: create a
   profile — name, optional system prompt, and the tool picker (grouped by
   category, with per-category and per-tool checkboxes plus *Select all* /
   *Select none*). The star button flags a profile as the default (or clears
   it). Non-staff users see the same page read-only.
2. In the chat header, pick the profile from the dropdown next to the
   conversation selector. The selection is remembered per user in this browser
   (`localStorage` key `chatProfileId:<user key>`, where the key is the Keycloak
   `sub`, falling back to `preferred_username` / email when the access token
   carries no `sub` — the same order the backend maps users by) and sent as
   `profile_id` with every message, so it can be switched mid-conversation.
   Non-staff users start on the default profile when one is set, and the
   "Default (all tools)" entry is hidden from them.
3. The **Generate report** button is disabled when the selected profile lacks
   `get_workflow_facts` or `save_report`.

## API

| Method | Endpoint | Who | Notes |
|---|---|---|---|
| GET | `/api/chat/profiles/` | any user | List the shared profiles |
| POST | `/api/chat/profiles/` | staff | Create. Body: `{name, allowed_tools: string[], system_prompt, is_default?}`. Names are unique; `system_prompt` ≤ 16000 chars |
| GET | `/api/chat/profiles/<uuid>/` | any user | Retrieve one profile |
| PUT / DELETE | `/api/chat/profiles/<uuid>/` | staff | PUT is partial; `is_default: true` clears the flag on every other profile. DELETE returns 204. Non-staff get 403 |
| POST | `/api/chat/stream/` | any user | Optional `profile_id` (unknown id → 404 before any conversation is created). Without it, non-staff users get the default profile if one is set |
| GET | `/api/profile/` | any user | Returns `user.is_staff`, which the frontend uses to decide what to show |
| GET | `/api/chat/mcp-tools/` | any user | Tool catalog used by the editor (shared with the notebook agent; shape unchanged) |

## Code map

Backend (`gui/workflow_backend/django-project/app/chat/`):
`models.py` (`ChatProfile`, migration `0002_chatprofile`),
`serializers.py` (`ChatProfileSerializer`, `SendMessageSerializer.profile_id`),
`views.py` (`ChatProfileListCreateView`, `ChatProfileDetailView` — staff-only
writes via `IsAdminUser`; `ChatStreamView` applies the default profile),
`services/mcp_client.py` (`mcp_tools_to_openai_functions(..., allowed=)`),
`services/chat_orchestrator.py` (`orchestrate_chat(..., profile=)`).

Frontend (`gui/workflow_frontend/src/`):
`api/chatProfileApi.ts`, `stores/chatProfileStore.ts`,
`views/home/components/ChatProfileSelector.tsx`, `ChatProfileManager.tsx`,
`ChatProfileModal.tsx`, `chatToolCategories.ts`; wired in `chatbotView.tsx`,
`components/tabs/TabManager.tsx` and `shared/header/header.tsx`.

Tests: `gui/workflow_backend/django-project/tests/test_chat_profiles.py`.
