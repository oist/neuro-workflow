# Jupyter server config mounted into both Labs.
# Enables the NeuroWorkflow contents filter (visual hide of others' private
# project folders). Kernel/terminal are not filtered.
c = get_config()  # noqa: F821
c.ServerApp.jpserver_extensions = {"jupyter_tenant_filter": True}
