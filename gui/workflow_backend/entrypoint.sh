#!/bin/bash
# Container entrypoint for the NeuroWorkflow Django backend.
#
# Starts a long-lived ssh-agent on a fixed socket so the backend can submit
# jobs to the RIKEN compute server over SSH using a passphrase-protected key
# that an admin unlocks ONCE per container start:
#
#   docker exec -e SSH_AUTH_SOCK=/tmp/ssh-agent.sock -it <backend> \
#       ssh-add /root/.ssh/id_ed25519
#
# This is additive and inert by default: the agent simply holds no keys until
# an admin adds one, and nothing here runs unless a workflow is submitted with
# backend=slurm. The live (JupyterHub) execution path is unaffected.
set -e

export SSH_AUTH_SOCK=/tmp/ssh-agent.sock

# A container restart keeps this socket file and kills the agent. ssh-add
# then returns 2 (connection refused). Exit 1 means the agent is up and
# simply has no keys yet, so it must be left alone.
set +e
ssh-add -l >/dev/null 2>&1
agent_rc=$?
set -e
if [ "$agent_rc" -eq 2 ] || [ ! -S "$SSH_AUTH_SOCK" ]; then
    rm -f "$SSH_AUTH_SOCK"
    ssh-agent -a "$SSH_AUTH_SOCK" >/dev/null
fi

exec "$@"
