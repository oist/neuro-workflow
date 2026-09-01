from django.urls import path

from .views import OAIPMHFileDownloadView, OAIPMHRecordsView, OAIPMHSearchView

urlpatterns = [
    # Kernel plane (service token): harvested records and file downloads.
    path("records/", OAIPMHRecordsView.as_view(), name="harvest-records"),
    path(
        "oai/files/<uuid:file_id>/download/",
        OAIPMHFileDownloadView.as_view(),
        name="harvest-oai-file-download",
    ),
    # Browser plane (Keycloak): keyword search over the harvested records.
    path("oai/search/", OAIPMHSearchView.as_view(), name="harvest-oai-search"),
]
