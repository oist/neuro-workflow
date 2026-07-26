from django.urls import path

from .views import CatalogLocalView, CatalogReadView, CatalogSyncView

app_name = "catalog"

urlpatterns = [
    # Health / per-source counts. Doubles as the "mdb available" indicator.
    path(
        "statistics/",
        CatalogReadView.as_view(route="statistics"),
        name="statistics",
    ),
    # Catalog listing and full-text search.
    path("datasets/", CatalogReadView.as_view(route="datasets"), name="datasets"),
    path("search/", CatalogReadView.as_view(route="search"), name="search"),
    # Single dataset by ID.
    path("lookup/", CatalogReadView.as_view(route="lookup"), name="lookup"),
    # Local BIDS catalog: dataset index, and its participants/sessions/sites.
    path(
        "local/<str:source>/<str:dataset_id>/",
        CatalogLocalView.as_view(),
        name="local-index",
    ),
    path(
        "local/<str:source>/<str:dataset_id>/<str:view>/",
        CatalogLocalView.as_view(),
        name="local-view",
    ),
    # The only write forwarded to mdb.
    path("sync/", CatalogSyncView.as_view(), name="sync"),
]
