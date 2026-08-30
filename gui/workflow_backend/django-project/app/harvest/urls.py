from django.urls import path

from .views import OAIPMHFileDownloadView, OAIPMHProxyView

urlpatterns = [
    # Allowlisted OAI-PMH passthrough for workflow kernels (key stays here).
    path("oai/", OAIPMHProxyView.as_view(), name="harvest-oai"),
    path(
        "oai/files/<uuid:file_id>/download/",
        OAIPMHFileDownloadView.as_view(),
        name="harvest-oai-file-download",
    ),
]
