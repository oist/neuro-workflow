#!/usr/bin/env bash
#
# Import the NeuroWorkflow realm into a running Keycloak instance.
#
# Usage:
#   ./setup.sh                          # defaults: admin/admin @ http://localhost:8080
#   KEYCLOAK_ADMIN_PASSWORD=secret ./setup.sh
#
# Prerequisites:
#   - Keycloak container is running and healthy
#   - curl is installed
#
set -euo pipefail

KEYCLOAK_URL="${KEYCLOAK_URL:-http://localhost:8080/auth}"
ADMIN_USER="${KEYCLOAK_ADMIN:-admin}"
ADMIN_PASS="${KEYCLOAK_ADMIN_PASSWORD:-admin}"
REALM_FILE="$(dirname "$0")/realm-export.json"
DEPLOY_URL="${DEPLOY_URL:-https://neuro-workflow.izbrain.info/}"
SSL_REQUIRED="${SSL_REQUIRED:-external}"

echo "Waiting for Keycloak at ${KEYCLOAK_URL} ..."
for i in $(seq 1 30); do
  if curl -sf "${KEYCLOAK_URL}/realms/master" > /dev/null 2>&1; then
    echo "Keycloak is ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "ERROR: Keycloak not reachable after 30 attempts." >&2
    exit 1
  fi
  sleep 2
done

echo "Obtaining admin token ..."
TOKEN=$(curl -sf -X POST "${KEYCLOAK_URL}/realms/master/protocol/openid-connect/token" \
  -d "client_id=admin-cli" \
  -d "username=${ADMIN_USER}" \
  -d "password=${ADMIN_PASS}" \
  -d "grant_type=password" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Checking if realm 'neuroworkflow' already exists ..."
STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer ${TOKEN}" \
  "${KEYCLOAK_URL}/admin/realms/neuroworkflow" || true)

if [ "$STATUS" = "200" ]; then
  echo "Realm 'neuroworkflow' already exists — skipping import."
  echo "To re-import, delete the realm first via the Keycloak admin console."
else
  echo "Importing realm from ${REALM_FILE} ..."
  curl -sf -X POST "${KEYCLOAK_URL}/admin/realms" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d @"${REALM_FILE}"
  echo "Realm 'neuroworkflow' imported successfully."
fi

# Apply deployment settings (idempotent — also updates an already-existing
# realm without re-importing, so registered users are preserved).
echo "Applying deployment settings (sslRequired=${SSL_REQUIRED} + ${DEPLOY_URL}) ..."

# 1. Set the realm-wide SSL requirement (external = require HTTPS for non-local
#    requests; override with SSL_REQUIRED=none only for plain-HTTP setups).
curl -sf -X PUT "${KEYCLOAK_URL}/admin/realms/neuroworkflow" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"sslRequired\":\"${SSL_REQUIRED}\"}"

# 2. Allow the deployment address on the frontend client (redirect URIs,
#    web origins, and post-logout redirect URIs).
CLIENT_UUID=$(curl -sf -H "Authorization: Bearer ${TOKEN}" \
  "${KEYCLOAK_URL}/admin/realms/neuroworkflow/clients?clientId=neuroworkflow-app" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")

CLIENT_JSON=$(curl -sf -H "Authorization: Bearer ${TOKEN}" \
  "${KEYCLOAK_URL}/admin/realms/neuroworkflow/clients/${CLIENT_UUID}" \
  | DEPLOY_URL="${DEPLOY_URL}" python3 -c "
import os, sys, json
c = json.load(sys.stdin)
base = os.environ['DEPLOY_URL'].rstrip('/')
redirect = base + '/*'
for key, val in (('redirectUris', redirect), ('webOrigins', base)):
    lst = c.setdefault(key, [])
    if val not in lst:
        lst.append(val)
attrs = c.setdefault('attributes', {})
parts = [p for p in attrs.get('post.logout.redirect.uris', '').split('##') if p]
if redirect not in parts:
    parts.append(redirect)
attrs['post.logout.redirect.uris'] = '##'.join(parts)
json.dump(c, sys.stdout)")

curl -sf -X PUT "${KEYCLOAK_URL}/admin/realms/neuroworkflow/clients/${CLIENT_UUID}" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "${CLIENT_JSON}"

echo "Deployment settings applied."

echo ""
echo "Done. You can now access Keycloak at:"
echo "  Admin console: ${KEYCLOAK_URL}/admin/"
echo "  Account page:  ${KEYCLOAK_URL}/realms/neuroworkflow/account/"
