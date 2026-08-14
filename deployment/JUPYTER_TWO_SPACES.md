# Two Jupyter spaces (internal vs hackathon)

This is the operator runbook for the two shared Labs. It is **not** per-user
container isolation.

## Honesty bound

| Boundary | Mechanism | Strength |
|----------|-----------|----------|
| Hackathon vs internal | Separate bind-mounts (`codes/` vs `codes-hackathon/`) | **Real** |
| Alice vs Bob inside one Lab | ContentsManager hides dirs using a viewer token | **Visual only** |
| App project/node lists | Postgres `tenant` + existing visibility | **Real** for GUI/API |

The kernel and terminal in a Lab can still `ls` every path **mounted in that
Lab**. Do not treat JupyterLab as a security boundary inside a group.

## Hub users

FirstUse (or Dummy in local compose). Two accounts, operator-managed passwords:

| Hub user | Container name | Host tree |
|----------|----------------|-----------|
| `internal` (legacy `user1` still allowed during cutover) | `jupyter-internal` | `codes/projects`, `codes/nodes` |
| `hackathon` | `jupyter-hackathon` | `codes-hackathon/projects`, `codes-hackathon/nodes` |

The app stays on Keycloak. Frontend/backend pick `/user/internal/` vs
`/user/hackathon/` from the user's tenant.

## Keycloak groups

Create realm groups (or roles) and a mapper that puts them in the access token
(`groups` claim or `realm_access.roles`):

- `nw-internal` — project members (default for existing users)
- `nw-hackathon` — temporary / outside users
- `node-reviewers` — can approve submitted nodes **in their own tenant**

On login Django syncs `nw-internal` / `nw-hackathon` onto Django Groups.
If the token has no tenant claim, existing membership is left as-is; users with
neither group are assigned `nw-internal`.

`internal` wins if a user is in both groups.

## Env (compose)

```
JUPYTERHUB_ALLOWED_USERS=internal,hackathon,user1
JUPYTER_GRANT_SUDO=no
JUPYTER_MEM_LIMIT=8G          # tune: ~half remaining RAM per Lab
JUPYTER_CPU_LIMIT=4
HOST_PROJECT_PATH=.../django-project
# HOST_HACKATHON_PATH defaults to $HOST_PROJECT_PATH/codes-hackathon
```

Do not publish Jupyter/Docker ports on `0.0.0.0`. Hub stays behind nginx
`/jupyter`.

## Cutover (needs explicit OK — this recreates Labs)

1. `mkdir -p gui/workflow_backend/django-project/codes-hackathon/{projects,nodes}`
2. Deploy this branch; **backend migrate** applies `tenant` + node governance.
3. Create Hub users `internal` and `hackathon` (FirstUse: first login sets
   password). Keep `user1` until internal users have moved.
4. Recreate JupyterHub so spawners pick up volume maps. **Warn:** this drops
   running kernels; ssh-agent on the backend is unrelated unless backend also
   restarts.
5. Smoke:
   - Guest Keycloak user: app lists only hackathon projects; Lab tree is
     `codes-hackathon` only (`ls /home/jovyan/codes/projects` has no internal
     UUIDs).
   - Internal user: app hides hackathon tenant; Lab is the internal tree;
     file browser omits others' private UUIDs; `ls` in the terminal still sees
     them (expected).
   - Approve a node in one tenant; it does not appear in the other tenant's
     palette.
6. Rotate the two Lab passwords; document them in the operator secret store,
   not git.

## Rollback

- Revert the git deploy.
- Hub `allowed_users=user1` and the previous volume map (all of `codes/`).
- DB columns `tenant` / node `status` are backward compatible (defaults
  `internal` / catalog `public`).

## Node governance

`private → submitted → approved → public`

- New uploads: `private` in the caller's tenant.
- Catalog files (`uploaded_by` null): `public` + `tenant=internal` after
  migrate.
- Palette: same-tenant `public`, plus the owner's own non-public nodes.
  Reviewers also see `submitted` in their tenant.
- Endpoints under `/api/box/files/<uuid>/submit|approve|publish|reject/` and
  `/api/box/review-queue/`.
