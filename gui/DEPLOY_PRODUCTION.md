# Production Deployment (Domain + Let's Encrypt HTTPS)

Runbook for deploying the NeuroWorkflow web app to a public server reachable by a
domain name over HTTPS. The worked example uses `neuro-workflow.izbrain.info` on
host `57.182.155.250`; substitute your own domain where shown.

> **Why HTTPS is mandatory.** The frontend uses `keycloak-js`, which calls
> `crypto.randomUUID()` / `crypto.subtle` (Web Crypto API). Browsers only expose
> these in a **Secure Context** — i.e. `https://`, or `http://localhost`. Plain
> HTTP on a remote IP/host is *not* a secure context, so login fails with
> `Web Crypto API is not available`. Relaxing Keycloak's `sslRequired` does **not**
> help: that is a server-side check and has no effect on the browser requirement.
> Serving the whole app over HTTPS (this runbook) is the supported fix.

## Architecture

Everything is served same-origin under `https://<domain>` so the browser runs in a
Secure Context and `keycloak-js` works. A **host-level nginx** terminates TLS and
reverse-proxies to the Docker services, which bind to `127.0.0.1` only.

```
Browser ──HTTPS──► host nginx (TLS, Let's Encrypt) ──► 127.0.0.1 Docker services
                    ├ /         → frontend   127.0.0.1:5173  (static nginx, Dockerfile.prod)
                    ├ /api/      → backend    127.0.0.1:3000  (gunicorn)
                    ├ /auth/     → keycloak   127.0.0.1:8080
                    ├ /jupyter/  → jupyterhub 127.0.0.1:8000
                    └ /mcp/      → mcp        127.0.0.1:8001
```

The production frontend build bakes relative base URLs (`/api`, `/auth`, `/jupyter`,
`/mcp`) via `docker-compose.prod.yml` build args, so the SPA always talks to the
same origin and the host nginx routes each prefix.

## Repo config already prepared for this deployment

These files are committed for `neuro-workflow.izbrain.info` — adapt for another domain:

- `gui/keycloak/realm-export.json` — `sslRequired: external`; client `neuroworkflow-app`
  allows `https://<domain>/*` in `redirectUris` / `webOrigins` / post-logout.
- `gui/keycloak/setup.sh` — `DEPLOY_URL` defaults to `https://<domain>/`; the realm-wide
  SSL level is `SSL_REQUIRED` (default `external`). Both overridable via env.
- `gui/nginx/neuro-workflow.conf` — `server_name <domain>`; the `/auth/` reverse-proxy
  block is enabled (Keycloak runs same-origin so keycloak-js gets a Secure Context).
- `gui/workflow_frontend/vite.config.docker.ts` — the dev `/auth` proxy preserves the
  browser Host (`changeOrigin: false`, `xfwd: true`) so Keycloak builds external URLs.
  (Dev stack only; production routes `/auth` through the host nginx above.)

## Prerequisites

- Server `57.182.155.250` with ports **80 and 443** open to the internet.
- A domain you control (here `neuro-workflow.izbrain.info`).
- Docker + Docker Compose **v2.24.4+** (the prod override uses `!override` / `!reset`).
- The three `.env` files in place (see below).

## Steps (run on the server)

### 1. DNS

Point an `A` record for `<domain>` at the server IP, and confirm it resolves:

```bash
dig +short neuro-workflow.izbrain.info     # → 57.182.155.250
```

### 2. Environment variables

`docker compose` reads `gui/.env` for `${VAR}` substitution in the compose files.

**`gui/.env`**

```bash
BIND_HOST=127.0.0.1                                            # only host nginx is public
ALLOWED_HOSTS_EXTRA=neuro-workflow.izbrain.info
CORS_ALLOWED_ORIGINS_EXTRA=https://neuro-workflow.izbrain.info
CSRF_TRUSTED_ORIGINS_EXTRA=https://neuro-workflow.izbrain.info
JUPYTERHUB_FRAME_ORIGIN=https://neuro-workflow.izbrain.info
DJANGO_DEBUG=false
KEYCLOAK_DB_PASSWORD=<strong-password>
KEYCLOAK_ADMIN_PASSWORD=<strong-password>
JUPYTERHUB_API_TOKEN=<random-token>
```

**`gui/workflow_backend/.env`** — the backend reads Keycloak + security settings here.
`KEYCLOAK_ISSUER` is **critical**: the backend fetches JWKS internally
(`http://keycloak:8080/auth`) but the token `iss` claim is the **public** URL the
browser logged in through. If they don't match, every login fails with
`Untrusted token issuer`.

```bash
DJANGO_SECRET_KEY=<random, 50+ chars>
SECURE_SSL_REDIRECT=true
SECURE_HSTS_SECONDS=31536000
SESSION_COOKIE_SECURE=true
CSRF_COOKIE_SECURE=true
KEYCLOAK_URL=http://keycloak:8080/auth                         # internal (JWKS fetch)
KEYCLOAK_REALM=neuroworkflow
KEYCLOAK_CLIENT_ID=neuroworkflow-app
KEYCLOAK_ISSUER=https://neuro-workflow.izbrain.info/auth/realms/neuroworkflow   # public (iss)
```

### 3. Install the host nginx site

```bash
sudo cp gui/nginx/neuro-workflow.conf /etc/nginx/sites-available/neuro-workflow
sudo ln -sf /etc/nginx/sites-available/neuro-workflow /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 4. Obtain the Let's Encrypt certificate

The certbot nginx plugin auto-adds the `listen 443 ssl` block, certificate paths,
and the 80→443 redirect to the installed site:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d neuro-workflow.izbrain.info
sudo certbot renew --dry-run                                   # verify auto-renewal
```

### 5. Build and start the production stack

```bash
cd gui
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

`--build` is required: the frontend image bakes the relative base URLs at build time.

### 6. Register the domain in the Keycloak realm

Idempotent; preserves existing users. Defaults already target the prod domain, but
the values are shown explicitly:

```bash
KEYCLOAK_URL=http://localhost:8080/auth \
DEPLOY_URL=https://neuro-workflow.izbrain.info/ \
./keycloak/setup.sh
```

### 7. Verify

1. `https://neuro-workflow.izbrain.info/login` → **Continue to Sign In** → the
   **Keycloak login page** loads (no certificate warning, no `Web Crypto` error).
2. After login you return to the app with no `Untrusted token issuer` or
   `redirect_uri` rejection.
3. The JupyterHub panel renders in its iframe (CSP `frame-ancestors` =
   `JUPYTERHUB_FRAME_ORIGIN`).
4. `http://neuro-workflow.izbrain.info` redirects to `https://`.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Web Crypto API is not available` in the browser console | Page not served over HTTPS (Secure Context). Complete steps 3–4; confirm the URL is `https://`. |
| Login bounces back, backend logs `Untrusted token issuer` | `KEYCLOAK_ISSUER` (step 2) missing or not matching `https://<domain>/auth/realms/neuroworkflow`. |
| Keycloak redirects show an internal host (`keycloak:8080`) | Host nginx `/auth/` block must forward `Host $host` + `X-Forwarded-Proto $scheme` (already in the committed config). |
| `redirect_uri` rejected by Keycloak | Realm client missing `https://<domain>/*`. Re-run step 6 with the correct `DEPLOY_URL`. |
| `nginx -t` fails after certbot | Inspect the auto-generated 443 block; ensure the cert paths exist under `/etc/letsencrypt/live/<domain>/`. |

## Notes

- Keep `BIND_HOST=127.0.0.1`: only the host nginx is public. Keycloak (8080),
  backend (3000), JupyterHub (8000), MCP (8001) stay loopback-only.
- `KC_HOSTNAME` is intentionally **not** pinned (`docker-compose.prod.yml` sets
  `KC_HOSTNAME_STRICT=false`): Keycloak derives the public hostname per-request from
  the forwarded headers, so the same stack serves multiple domains. For multiple
  public issuers, add them to the backend `KEYCLOAK_ISSUERS` (comma-separated).
- Dev vs prod: the dev stack (`docker-compose.yml` alone, Vite dev server on :5173)
  is unaffected by this runbook. Production uses `Dockerfile.prod` static nginx +
  the host nginx, so the Vite dev `/auth` proxy is not on the production path.
