# Notebook Chat Agent

An AI chat agent that runs **inside a Jupyter notebook** (Issue #52). It helps you
write, run, and debug code and build node-based workflows without leaving the
notebook. It also reads the skills tracked in the repository's `.claude/skills/`
and uses them as guidance (for example, the node-creation skill).

## How it works

```
Jupyter kernel                              Django backend              Anthropic / MCP
──────────────                              ──────────────              ───────────────
neuroworkflow.agent (Claude Agent SDK)
  ├─ agent loop + run_code/Read/Write/Edit  (run locally in the kernel)
  ├─ model calls    ── ANTHROPIC_BASE_URL ──▶ /api/chat/anthropic ──▶ Anthropic API
  │                 (other model picked)    ▶ /api/chat/llm       ──▶ provider's Anthropic-compatible API
  └─ workflow tools ── POST /api/chat/mcp-call/ ──▶ MCPClient ──▶ MCP server ──▶ workflow API
     (hub token       ── GET  /api/chat/mcp-tools/  ▲ relayed Keycloak token
      + project_id)                                 │ (bound to the kernel's
                                                    │  Jupyter space)
Browser (app, project's Jupyter tab open) ── POST /api/chat/notebook-token/ ─┘
```

- The **agent loop runs in the kernel** (the Claude Agent SDK drives the `claude` CLI),
  so notebook-native tools (run_code, file edits) act directly on your live workspace.
- The **Anthropic key and the MCP workflow tools stay on the backend**; the kernel
  reaches them over HTTP. The kernel cannot reach the MCP server or Anthropic directly,
  so both are proxied through the backend.
- **Model calls** go through the backend Anthropic proxy: the kernel sets
  `ANTHROPIC_BASE_URL` to the backend and presents its own per-server **JupyterHub
  token** as its API key; the backend checks it with the hub and swaps in the real
  Anthropic key. **Workflow
  tools** act with your own Keycloak token so per-user data is scoped correctly — but
  the kernel never holds that token: while a project's Jupyter tab is open, the app
  relays it to the backend, and the backend uses it on the kernel's behalf for that
  project (the kernel only sends its own JupyterHub token and the project id; the hub
  token tells the backend which Jupyter space the kernel is in).
- **Each turn is saved** on the backend (`POST /api/chat/notebook-conversations/`, same
  hub token) so it can be read later in the app; see
  [Saved conversations](#saved-conversations).

## Prerequisites

Run the Docker stack as usual:

```bash
cd gui
docker-compose build && docker-compose up
```

The single-user **kernel** image (`nest-jupyterlab:latest`) is built **separately** — it is
*not* built by `docker-compose` (that only builds the JupyterHub *hub* image). After
changing `Dockerfile.nest` (e.g. this migration adds Node.js + the Claude Code CLI +
`claude-agent-sdk`), rebuild it and re-spawn your container:

```bash
cd gui/workflow_backend/django-project/neuroworkflow
./build-nest-image.sh        # docker build -t nest-jupyterlab -f Dockerfile.nest .
docker rm -f jupyter-user1   # drop the old single-user container, then reopen Jupyter
```

Set `ANTHROPIC_API_KEY` in `gui/.env` (alongside `OPENAI_API_KEY`). This is the **real**
key; it stays on the backend and powers the `/api/chat/anthropic` proxy. The kernel never
receives it.

To let users switch the agent to MiniMax models, also set `MINIMAX_API_KEY` (and
optionally `MINIMAX_MODELS`, a comma-separated list, default `MiniMax-M3`) in `gui/.env`;
other providers come from `gui/workflow_backend/llm_providers.json` — see
[Choosing the model](#choosing-the-model). Those keys stay on the backend as well.

The JupyterHub spawner wires everything into each single-user container automatically:

| Variable | Purpose | Default |
| --- | --- | --- |
| `NEUROWORKFLOW_BACKEND_URL` | Backend base URL the kernel calls | `http://backend:3000` |
| `JUPYTERHUB_API_TOKEN` | Set by JupyterHub itself: the server's own token, which the agent presents to the backend proxies (incl. the Anthropic proxy). The hub's admin service token is never passed into containers | (per server) |
| `ANTHROPIC_BASE_URL` | Backend Anthropic proxy the `claude` CLI calls | `<backend>/api/chat/anthropic` |
| `ANTHROPIC_MODEL` | Optional model override (empty = CLI default) | unset |
| `NEUROWORKFLOW_SKILLS_DIR` | Where `.claude` skills are read from | `/home/jovyan/.claude/skills` |
| `PYTHONPATH` | Makes `import neuroworkflow` resolve | `/home/jovyan/codes` |
| `NEUROWORKFLOW_USER_TOKEN` | Manual override: a Keycloak token forwarded directly (normally not needed) | unset |
| `NEUROWORKFLOW_PROJECT_ID` | Workflow id; derived from the notebook folder `codes/projects/<uuid>/` when unset | unset |
| `NEUROWORKFLOW_WORKSPACE_ROOT` | Root that file edits (Write/Edit) are confined to | `/home/jovyan/codes` |

The repository's `.claude/skills/` directory is mounted read-only at
`/home/jovyan/.claude/skills`, so the agent reads the git-tracked skills. The rest of
`.claude/` (local settings, worktrees) is not mounted.

> A new single-user container picks up this wiring only when it is **(re)spawned**.
> After changing spawner config, remove the old container (`docker rm -f jupyter-user1`)
> and reopen Jupyter.

## Quick start

### 1. Load the extension

```python
%load_ext neuroworkflow.agent
```

### 2a. Inline chat (magics)

One line:

```python
%chat Create a numpy array of spike times and print the mean inter-spike interval.
```

Multi-line cell:

```python
%%chat
How do I build a SONATA network with the neuroworkflow library?
Show me the minimal code.
```

### 2b. Persistent chat panel (ipywidget)

```python
from neuroworkflow.agent import ChatPanel
ChatPanel()
```

This shows a docked panel with an output area and an input box. Workflow tools are
available when the notebook lives in a project folder opened from the app (see below);
otherwise only notebook-native tools are available.

## Choosing the model

By default the agent uses Claude (`ANTHROPIC_MODEL`, or the CLI default). The models of
OpenAI-compatible providers that also have an Anthropic-compatible API can be picked as
well: MiniMax (when the backend has `MINIMAX_API_KEY` set, the models in
`MINIMAX_MODELS`) and every provider in `llm_providers.json` with an `anthropic_base_url`
(see "Model selection" in `docs/CHAT_PROFILES.md`). Providers without one are offered in
the browser chat only.

- **`ChatPanel()`** shows a model dropdown under the Send button (only when such models
  are offered). The choice applies from the next message.
- **Magics:** `%chat --model MiniMax-M3 <message>` (or `%%chat --model MiniMax-M3` as the
  first line of a cell) switches the model for this and later messages;
  `%chat --model default <message>` returns to Claude.
- **Python:** `list_models()` returns the ids (`""` is the default Claude model) and
  `get_agent(model="MiniMax-M3")` switches.

**Switching the model starts a new conversation**: the agent is rebuilt, because a
session cannot be carried from one provider to the other. The kernel's variables and
files are untouched.

For such a model the kernel points the `claude` CLI at the backend's second proxy route,
`/api/chat/llm`, which reads the `model` of each request, finds the provider offering it
and forwards to that provider's Anthropic-compatible API (for MiniMax
`MINIMAX_ANTHROPIC_BASE_URL`, default `https://api.minimax.io/anthropic`) with the
backend's key as a bearer token. Models that are not on offer are rejected (400), and the
route only serves `v1/messages` and `v1/messages/count_tokens`. As with Anthropic, the
kernel only ever holds its own hub token.

Choosing such a model sends your prompts, notebook code and tool results to that provider
instead of Anthropic. Models differ in what they accept (for example MiniMax's M2.x
models take no image input).

## Enabling workflow tools

To let the agent operate on your saved workflow projects (add nodes, read the flow,
generate code, update parameters, …) it needs to act with your Keycloak identity. This is
wired up automatically:

1. In the app (`http://localhost:5173`), select the project and open its **Jupyter tab**.
   While that tab is open the app relays your Keycloak access token to the backend for
   that project (checked every minute, re-sent whenever the token is refreshed).
2. In JupyterLab, work in a notebook inside the project folder
   `codes/projects/<project uuid>/` (that is where the app opens `workflow.py`). The agent
   derives the project id from the folder.
3. `%load_ext neuroworkflow.agent`, then `ChatPanel()` or `%chat …` — the workflow tools
   are listed and the agent can use them.

The kernel never receives your token: it calls the backend with its own JupyterHub token
and the project id. The backend asks JupyterHub which user (Jupyter
space) that token belongs to and forwards the token you relayed for **that project in that
space** to the MCP server. The kernel only gets the tools that take a `workflow_id`, and
the backend always sets it to that project. The relay needs write access to the project,
so opening someone else's project read-only (e.g. a public community project from the
project space) does not give its notebook agent workflow tools.

Closing the project's Jupyter tab stops the relay; the stored token expires shortly after
(Keycloak access-token lifetime) and workflow tools stop working until you reopen the tab.
The agent picks the new token up on its next tool call — no kernel-side action needed.

You can check the wiring from the kernel:

```python
from neuroworkflow.agent import get_agent
a = get_agent()
print(a._config.project_id, "workflow" in a._servers)   # -> '<uuid>', True
```

### Manual fallback

If the notebook is not in a project folder (or you want to act on another project), pass
the token and project id explicitly:

```python
from neuroworkflow.agent import ChatPanel
ChatPanel(
    user_token="eyJ...",                                  # your Keycloak access token
    project_id="4b5023b0-8f1e-4dfc-87f0-1579c1a9bf00",    # target workflow id
)
```

The token is a short-lived JWT: in the app, open the browser console while logged in and
run `window.__NEURO_WORKFLOW_KEYCLOAK__.token`, then copy the whole `eyJ...` string. The
project id is the folder name under `codes/projects/` (also in the app URL). When the
token expires, recreate the panel with a fresh one — the agent is rebuilt when the token
changes. Avoid leaving tokens in saved notebooks.

## Tools the agent can use

**In the kernel (always available):**

- `run_code` — execute Python in the live kernel (shared namespace); returns stdout/result.
  Runs in-process so variables persist and output displays inline.
- `Read` / `Write` / `Edit` — read and edit files (Claude Agent SDK built-ins). Writes are
  confined to `NEUROWORKFLOW_WORKSPACE_ROOT` (default `/home/jovyan/codes`); paths outside
  it are rejected.
- `Bash` — run shell commands (obviously destructive commands are blocked); use `run_code`
  for Python, not Bash.

**Workflow tools via MCP (require the relayed or a manually supplied token):** `add_node`, `get_flow`, `list_nodes`,
`add_edge`, `update_node_parameter`, `generate_code_batch`, `get_workflow_facts`,
`save_report`, and the rest of the workflow MCP tools that take a `workflow_id`. With the
relayed token, tools that are not tied to one project (`list_projects`, `create_project`,
`upload_python_file`, `catalog_*`, …) are not offered.

## Skills (`.claude/skills`)

On startup the agent loads every `*.md` file in `NEUROWORKFLOW_SKILLS_DIR`
(`/home/jovyan/.claude/skills`) and appends it to the system prompt under
`# Available skills`. The skill content is used as guidance — the agent does not run
Claude's skill machinery, it simply follows the instructions in the markdown.

Currently this is `create-node.md` (the node-creation guide). You can verify it is loaded:

```python
from neuroworkflow.agent import reset_agent, get_agent
reset_agent()
print("Skill: create-node.md" in get_agent()._append_prompt)   # -> True
```

To add a skill, commit a new `*.md` under `.claude/skills/`; it is picked up on the next
agent start.

## Python API

```python
from neuroworkflow.agent import chat, get_agent, list_models, reset_agent, ChatPanel

chat("explain the BuildSonataNetworkNode ports")   # one-shot, streams to stdout
agent = get_agent()                                 # shared singleton; project id from the notebook folder
agent = get_agent(user_token="eyJ...")              # manual token override (rebuilt if token changes)
agent = get_agent(model="MiniMax-M3")               # switch model (ids from list_models(); new conversation)
reset_agent()                                       # clear history / re-read config
```

## Saved conversations

Every turn with the agent (`%chat` or `ChatPanel`) is saved automatically on the backend,
so a conversation can be read again after the kernel restarts or the page reloads.
Open the project in the app and click **💬 Notebook chats** in the Flow Designer panel:
the list shows the project's conversations (newest first); select one to read it, or
download it as Markdown (e.g. for lab notes) or JSON.

- **What is saved:** your message, the agent's replies, and each tool call with its
  arguments and result (each clipped to 4,000 characters, marked `…(N chars truncated)`),
  the model, and the SDK session id (kept for a future *resume*). A turn that fails or is
  interrupted is saved up to that point, marked `…(interrupted)` / `[error] …`.
- **What is not saved:** thinking blocks, the internals of subagents (their call and
  result are saved), and conversations in notebooks outside `codes/projects/<uuid>/`
  (there is no project to file them under).
- **Secrets:** the kernel replaces its hub token and a manual `user_token` with
  `[REDACTED]` anywhere in the turn before sending it. The relayed
  Keycloak token never reaches the kernel, so it cannot end up in a record.
- **Who can read them:** anyone who can open the project — only the owner for a private
  project, everyone in the same tenant for a public one. This matches the shared Jupyter
  space, where the notebooks themselves are visible the same way. The user shown on a
  conversation is a best guess: whoever last opened the project's Jupyter tab in the app
  (see *Token relay trust boundary* below; Issue #28).
- **Deleting:** the project owner or the user shown on the conversation can delete it (it
  is hidden, not erased from the database).
- **New conversation:** `reset_agent()`, switching the model, or a kernel restart starts
  a new saved conversation; earlier ones stay in the list.
- If saving fails (e.g. the backend is unreachable) the chat carries on and prints
  `[neuroworkflow.agent] conversation not saved: …` once.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'neuroworkflow.agent'` | The container mounts the synced copy at `codes/neuroworkflow/`, not `src/`. Make sure `codes/neuroworkflow/agent/` exists (see *Maintenance* below) and the container was respawned. |
| Jupyter shows **500 / "Client secret mismatch"** on spawn | Do **not** set `JUPYTERHUB_API_TOKEN` in the spawner env — it is reserved for the single-user server's own hub OAuth (the agent reads that per-server value; it must not be replaced by the hub's service token). |
| `ChatPanel()` shows **two panels** | Fixed: the panel is displayed once. If you still see two, restart the kernel so the updated module is reloaded. |
| Workflow tools return `401 … No relayed user token` | The project's Jupyter tab in the app is closed or the relayed token expired. Reopen the project's Jupyter tab in the app; the next tool call picks the new token up. (Manual `user_token`: get a fresh one and recreate the panel.) |
| No workflow tools are listed (`"workflow" in get_agent()._servers` is `False`) | Check `os.getcwd()` is `/home/jovyan/codes/projects/<uuid>` and the project's Jupyter tab is open in the app, then `reset_agent()` — the tool list is fetched when the agent is created. |
| Skills don't seem to apply | Check the mount: `import os; os.listdir("/home/jovyan/.claude/skills")` should list `create-node.md`. If empty, respawn the single-user container so the `.claude` volume is mounted. |

## Known limitations

- **No resume or history in the notebook yet.** Saved conversations are read in the app;
  the agent cannot continue an old one after a kernel restart, and `ChatPanel` does not
  list them. If you interrupt the first turn of a conversation and send the next message
  right away, the two turns may be saved as two conversations.

- **Skill path mapping.** Skills authored against repository paths (e.g.
  `src/neuroworkflow/nodes/`, `NODE_CREATION_GUIDE.md`) do not match the container layout,
  where only `/home/jovyan/codes/` is mounted. For node creation, the writable directory is
  `/home/jovyan/codes/nodes/`, and repo-root docs are not present in the kernel. The agent
  still follows the skill's guidance, but file paths from the skill text may need adjusting.
- **Auth scope.** The Anthropic proxy accepts any kernel's per-server hub token (it
  identifies the Jupyter space, not a person) and uses a single shared Anthropic key
  (acceptable for a trusted lab/hackathon). It only serves `POST` to `v1/messages` and
  `v1/messages/count_tokens`, so the key cannot reach the rest of the Anthropic API, and
  it logs each call with the kernel's space and model. It does not restrict which Claude
  model is asked for. Per-user identity applies only to the workflow (MCP) tools via your
  Keycloak token. The `/api/chat/llm` route works the same way with each provider's
  shared key, and rejects models that are not on offer.
- **Token relay trust boundary.** A kernel authenticates to the MCP proxies with its own
  JupyterHub token, which the backend verifies with the hub to learn the kernel's Jupyter
  space (hub user); the hub's service token is refused. Relayed tokens are
  bound to the relaying user's space, so a kernel in the community space cannot use a
  project-space token and vice versa. Within one space, however, all users still share the
  hub user: any kernel there (any user, or code the agent runs) can drive the workflow tools
  as *whoever last opened that project's Jupyter tab in the app*, limited to that project:
  only tools that take a `workflow_id` are served, with the `workflow_id` set to the
  project. Issue #28 (per-person hub users) closes this gap. Mitigations in place: only
  short-lived access tokens are stored (never refresh tokens), expired rows are purged,
  the kernel never sees the JWT, and the relay only accepts projects the caller may
  modify. Keep the realm's access-token lifespan short (≤ 5 min recommended in
  production). If two users open the same project's Jupyter tab, the most recent relay
  wins.

## Maintenance

The agent package lives in **two places**, matching the existing `core/` and `utils/`
convention:

- `src/neuroworkflow/agent/` — the library source.
- `gui/workflow_backend/django-project/codes/neuroworkflow/agent/` — the synced copy the
  single-user container mounts and imports.

When you edit the agent, update **both** copies (and commit the synced copy too).

Kernel-side unit tests (`tests/test_agent_token_relay.py`,
`tests/test_agent_conversation_record.py`) need `claude_agent_sdk`, so run them inside the
kernel image:

```bash
docker run --rm -v "$PWD":/repo -w /repo -e PYTHONPATH=/repo/src nest-jupyterlab:latest \
  python -m pytest tests/test_agent_token_relay.py tests/test_agent_conversation_record.py \
  -q -p no:cacheprovider
```

Backend tests for the relay live in `gui/workflow_backend/django-project/tests/test_notebook_token.py`,
and for saved conversations in `tests/test_notebook_conversations.py` next to it.
