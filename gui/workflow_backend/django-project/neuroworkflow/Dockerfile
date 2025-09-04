FROM jupyterhub/jupyterhub:4.0

# Install Docker Spawner
RUN pip install dockerspawner

# Create users
RUN useradd -m user1
RUN useradd -m user2

# Copy configuration
COPY jupyterhub_config.py /srv/jupyterhub/
COPY custom_handlers.py /srv/jupyterhub/

WORKDIR /srv/jupyterhub

# Expose port
EXPOSE 8000

CMD ["jupyterhub", "-f", "jupyterhub_config.py"]
