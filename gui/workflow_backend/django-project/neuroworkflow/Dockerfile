FROM jupyterhub/jupyterhub:4.0

# Install DockerSpawner and a production-suitable first-use authenticator
RUN pip install dockerspawner jupyterhub-firstuseauthenticator

# Create users
RUN useradd -m user1
RUN useradd -m user2

# Copy configuration
COPY jupyterhub_config.py /srv/jupyterhub/
COPY custom_handlers.py /srv/jupyterhub/
COPY space_auth.py /srv/jupyterhub/

WORKDIR /srv/jupyterhub

# Expose port
EXPOSE 8000

CMD ["jupyterhub", "-f", "jupyterhub_config.py"]
