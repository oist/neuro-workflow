## Installation

Create the Docker image for the NEST JupyterLab environment

```bash
cd ./gui/workflow_backend/django-project/neuroworkflow
docker build -t nest-jupyterlab -f Dockerfile.nest .
```

Edit the .env files to set environment variables.
rename env.template to .env and set environment variables

- ./gui/.env
- ./gui/workflow_backend/.env
- ./gui/workflow_frontend/.env

Start the Docker containers using docker-compose

```bash
cd ./gui
docker-compose build
docker-compose up
```


Open your web browser

```
http://localhost:5173
```
