from django.urls import path

from .views import UserSecretDetailView, UserSecretListCreateView

app_name = "secrets"

urlpatterns = [
    path("", UserSecretListCreateView.as_view(), name="secret-list-create"),
    path("<uuid:secret_id>/", UserSecretDetailView.as_view(), name="secret-detail"),
]
