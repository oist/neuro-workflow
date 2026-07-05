FROM python:3.11.11-slim

# Set working directory
WORKDIR /app

# Install required packages
#   openssh-client + rsync: needed to submit jobs to / sync files with the
#   RIKEN compute server over SSH (remote Slurm execution backend).
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    openssh-client \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.5

# Setting up Poetry (without creating a virtual environment)
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Fail the build loudly if poetry.lock is out of sync with pyproject.toml
RUN poetry check --lock

# Install the locked dependencies
RUN poetry install --no-interaction --no-ansi --no-root --only main

# Copy project file
COPY . .

# Entrypoint starts an ssh-agent (for remote Slurm submission) then execs the
# container command. The compose `command:` is preserved as arguments to it.
RUN cp /app/entrypoint.sh /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# Port 3000 is open
EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# run command
CMD ["python", "django-project/manage.py", "runserver", "0.0.0.0:3000"]
