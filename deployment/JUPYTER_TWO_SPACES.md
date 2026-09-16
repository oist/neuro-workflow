# Two Jupyter spaces (project vs community)

This is the operator runbook for the two shared Labs. It is **not** per-user
container isolation.

Canonical names in the app/DB/API/UI are **project** and **community**.
Live snnbuilder still uses the first #88 cutover names. Those stay accepted
aliases until an operator explicitly renames Keycloak groups, Hub users, and
the disk tree.

## Honesty bound

| Boundary | Mechanism | Strength |
|----------|-----------|----------|
| Community vs project | Separate bind-mounts (`codes/` vs `codes-hackathon/` today; `codes-community/` if that dir exists) | **Real** |
| Alice vs Bob inside one Lab | ContentsManager hides dirs using a viewer token | **Visual only** |
| App project/node lists | Postgres `tenant` + existing visibility | **Real** for GUI/API |

The kernel and terminal in a Lab can still `ls` every path **mounted in that
Lab**. Do not treat JupyterLab as a security boundary inside a group.

## Hub users

FirstUse (or Dummy in local compose). Operator-managed passwords.

**Live-safe defaults** (do not change without recreating Labs):

| Env | Default Hub user | Container name | Host tree |
|-----|------------------|----------------|-----------|
| `JUPYTERHUB_PROJECT_USER` | `internal` | `jupyter-internal` | `codes/projects`, `codes/nodes` |
| `JUPYTERHUB_COMMUNITY_USER` | `hackathon` | `jupyter-hackathon` | `codes-hackathon/` (or `codes-community/` if present) |

Aliases the Hub still accepts:

- `user1` authenticates as `internal` (pre-cutover cookie).
- Usernames `project` / `community` map to the same volume trees as
  `internal` / `hackathon` **once those Hub users exist**. Creating them is
  an operator cutover (below). It is **not** done by this branch alone.

The app stays on Keycloak. Frontend/backend pick `/user/<hub_user>/` from
`hub_username_for_tenant` (defaults stay `internal` / `hackathon` so a
migrate cannot 403 `/user/project/` before those Hub users exist).

`JUPYTERHUB_ALLOWED_USERS` default:
`project,community,internal,hackathon,user1`.

## Keycloak groups

Canonical Django/Keycloak groups (created in migration; copy members from the
legacy groups; **leave the old groups in place**):

- `nw-project` — project members (default for existing users)
- `nw-community` — community / outside users
- `node-reviewers` — can set review labels **in their own tenant** (live group
  is empty; Django staff still count). There is no Keycloak admin API sync.

**Live aliases** still recognized in tokens and Django membership:

- `nw-internal` → project
- `nw-hackathon` → community

On login Django syncs the canonical group. If the token has no tenant claim,
existing membership is left as-is; users with neither group are assigned
`nw-project`. Project wins if a user is in both.

Exact name match only: `nw-internal-mentees` / `nw-internal-readonly` are
**not** project.

Renaming the Keycloak realm groups is optional operator work. Do not do it as
part of a code deploy.

## Disk

Django never uses `HOST_*` paths (those are host paths; the backend runs in
a container). `path_utils.community_codes_root()` prefers `codes-community/`
if that directory exists, otherwise the live **`codes-hackathon/`** tree.

Hub uses `HOST_COMMUNITY_PATH`, falling back to `HOST_HACKATHON_PATH`, then
the same directory rule.

Do **not** `mv` the live tree in a code-only pass. An operator may later:

```
# explicit OK only — not executed by this branch
mv codes-hackathon codes-community
```

and/or set `HOST_COMMUNITY_PATH` to the new path.

The `hackathon/` participant kit (repo docs/examples) is **not** this tree
and must not be renamed.

## Env (compose)

```
JUPYTERHUB_ALLOWED_USERS=project,community,internal,hackathon,user1
JUPYTERHUB_PROJECT_USER=internal
JUPYTERHUB_COMMUNITY_USER=hackathon
JUPYTER_GRANT_SUDO=no
JUPYTER_MEM_LIMIT=8G          # tune: ~half remaining RAM per Lab
JUPYTER_CPU_LIMIT=4
HOST_PROJECT_PATH=.../django-project
# HOST_COMMUNITY_PATH defaults to codes-community/ if present, else codes-hackathon/
# HOST_HACKATHON_PATH is an alias for HOST_COMMUNITY_PATH
NODE_PUBLISH_REQUIRES_REVIEW=0
```

Do not publish Jupyter/Docker ports on `0.0.0.0`. Hub stays behind nginx
`/jupyter`.

## Opening nodes vs review labels

Opening (palette-visible to others in the same tenant) is **owner publish**:
`status=public`. Owners also have Close (unpublish).

Review is metadata only:

| `review_status` | meaning |
|-----------------|--------|
| `unreviewed` | default for new uploads |
| `in_review` | owner submitted |
| `reviewed` | a reviewer approved |

The review pipeline (submit / approve / reject / queue / audit /
`node-reviewers`) stays. Approve does **not** open the node. Reject does not
change public/private.

`NODE_PUBLISH_REQUIRES_REVIEW` defaults **off**. When an operator later sets
it true, owner publish requires `review_status=reviewed`. That forced policy
is disabled now.

Catalog files (`uploaded_by` null) stay `public` + `reviewed` + tenant
`project`.

## Optional Hub rename (needs explicit OK — this recreates Labs)

Not executed by this branch. Listed so a later operator can flip names after
Keycloak/DB are already on project/community:

1. Create FirstUse users `project` and `community` (or copy the live
   `internal` / `hackathon` passwords into those accounts).
2. Set `JUPYTERHUB_PROJECT_USER=project` and
   `JUPYTERHUB_COMMUNITY_USER=community`.
3. Recreate JupyterHub so spawners pick up the new usernames. **Warn:** this
   drops running kernels (`jupyter-internal` → `jupyter-project`, etc.).
4. Smoke `/user/project/` and `/user/community/` from the app. Keep
   `internal` / `hackathon` / `user1` in `JUPYTERHUB_ALLOWED_USERS` until
   cookies rotate.

## First two-space cutover (already done on live, 2026-08-14)

Live already has Hub users `internal` / `hackathon` and `codes-hackathon/`.
Do not repeat that recreate unless rolling forward the Hub rename above.

Smoke (after a future deploy of this denomination, still not this pass):

- Community Keycloak user: app lists only community projects; Lab tree is
  `codes-hackathon` (or `codes-community`) only.
- Project user: app hides community tenant; Lab is `codes/`; file browser
  omits others' private UUIDs; `ls` in the terminal still sees them
  (expected).
- Owner Open in one tenant does not appear in the other tenant's palette.
- Review Approve does not open the node.

## Rollback

- Revert the git deploy.
- Hub `allowed_users` and volume maps stay on the live names
  (`internal` / `hackathon`) if env defaults were not flipped.
- DB slugs `project` / `community` are the canonical values after migrate;
  code still **reads** leftover `internal` / `hackathon` rows.

## Endpoints

`/api/box/files/<uuid>/submit|approve|publish|unpublish|reject/` and
`/api/box/review-queue/`.
