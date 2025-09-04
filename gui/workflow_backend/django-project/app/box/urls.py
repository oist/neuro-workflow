from django.urls import path
from .views import (
    PythonFileUploadView,
    PythonFileListView,
    UploadedNodesView,
    PythonFileCodeManagementView,
    PythonFileCopyView,
    PythonFileParameterUpdateView,
)

app_name = "box"

urlpatterns = [
    # ファイルアップロード
    path("upload/", PythonFileUploadView.as_view(), name="python-file-upload"),
    # ファイル一覧
    path("files/", PythonFileListView.as_view(), name="python-file-list"),
    # ファイル詳細・削除
    path("files/<uuid:pk>/", PythonFileListView.as_view(), name="python-file-detail"),
    # アップロードされたノード一覧（フロントエンド用）
    path("uploaded-nodes/", UploadedNodesView.as_view(), name="uploaded-nodes"),
    path(
        "files/<str:filename>/code/",
        PythonFileCodeManagementView.as_view(),
        name="python-file-code",
    ),
    # ファイルコピー
    path("copy/", PythonFileCopyView.as_view(), name="python-file-copy"),
    # パラメーター更新
    path("parameters/update/", PythonFileParameterUpdateView.as_view(), name="python-file-parameter-update"),
]
