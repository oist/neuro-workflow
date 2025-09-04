#!/bin/bash

# Build NEST simulator enabled JupyterLab user server image
echo "Building NEST simulator enabled JupyterLab image..."

docker build -f Dockerfile.nest -t nest-jupyterlab:latest .

if [ $? -eq 0 ]; then
    echo "✅ Successfully built nest-jupyterlab:latest"
    echo "This image will be used by JupyterHub for user servers"
else
    echo "❌ Failed to build nest-jupyterlab:latest"
    exit 1
fi