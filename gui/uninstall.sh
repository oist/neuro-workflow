#!/bin/bash

docker compose -f docker-compose.yml down --rmi all --volumes --remove-orphans
rm ./workflow_backend/nodes.db
