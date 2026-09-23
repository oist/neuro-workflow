from django.urls import path

from .views import (
    CatalogDatasetsView,
    CatalogLookupView,
    CatalogSearchView,
    CatalogStatisticsView,
)

app_name = "catalog"

urlpatterns = [
    path("statistics/", CatalogStatisticsView.as_view(), name="catalog-statistics"),
    path("search/", CatalogSearchView.as_view(), name="catalog-search"),
    path("lookup/", CatalogLookupView.as_view(), name="catalog-lookup"),
    path("datasets/", CatalogDatasetsView.as_view(), name="catalog-datasets"),
]
