from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.auth.authentication import KeycloakAuthentication

from .client import (
    CatalogError,
    MDB_PATH_DATASETS,
    MDB_PATH_LOOKUP,
    MDB_PATH_SEARCH,
    MDB_PATH_STATISTICS,
    clamp_limit,
    mdb_request,
    validate_lookup_table,
    validate_source,
)


def _error_response(exc):
    return Response(
        {"status": "error", "code": exc.code, "error": exc.error},
        status=exc.status_code,
    )


@method_decorator(csrf_exempt, name="dispatch")
class CatalogStatisticsView(APIView):
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            data = mdb_request("GET", MDB_PATH_STATISTICS)
        except CatalogError as exc:
            return _error_response(exc)
        return Response(data)


@method_decorator(csrf_exempt, name="dispatch")
class CatalogSearchView(APIView):
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        body = request.data if isinstance(request.data, dict) else {}
        query = body.get("query")
        if not isinstance(query, str) or not query.strip():
            return Response(
                {
                    "status": "error",
                    "code": "invalid_query",
                    "error": "query is required",
                },
                status=400,
            )

        mode = body.get("mode")
        if mode is not None and mode != "keyword":
            return Response(
                {
                    "status": "error",
                    "code": "invalid_mode",
                    "error": "Only keyword search is supported",
                },
                status=400,
            )

        try:
            source = validate_source(body.get("source"))
            limit = clamp_limit(body.get("limit"))
            upstream = {
                "query": query.strip(),
                "mode": "keyword",
                "limit": limit,
            }
            if source is not None:
                upstream["source"] = source
            data = mdb_request("POST", MDB_PATH_SEARCH, json_body=upstream)
        except CatalogError as exc:
            return _error_response(exc)
        return Response(data)


@method_decorator(csrf_exempt, name="dispatch")
class CatalogLookupView(APIView):
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        record_id = request.query_params.get("id")
        if record_id is None or not str(record_id).strip():
            return Response(
                {
                    "status": "error",
                    "code": "invalid_id",
                    "error": "id is required",
                },
                status=400,
            )

        try:
            table = validate_lookup_table(request.query_params.get("table"))
            source_raw = request.query_params.get("source")
            if source_raw is None or not str(source_raw).strip():
                source = "dandi"
            else:
                source = validate_source(source_raw)
            params = {
                "table": table,
                "source": source,
                "id": str(record_id).strip(),
            }
            data = mdb_request("GET", MDB_PATH_LOOKUP, params=params)
        except CatalogError as exc:
            return _error_response(exc)
        return Response(data)


@method_decorator(csrf_exempt, name="dispatch")
class CatalogDatasetsView(APIView):
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            source = validate_source(request.query_params.get("source"))
            params = {}
            if source is not None:
                params["source"] = source
            if "limit" in request.query_params:
                params["limit"] = clamp_limit(request.query_params.get("limit"))
            data = mdb_request(
                "GET",
                MDB_PATH_DATASETS,
                params=params or None,
            )
        except CatalogError as exc:
            return _error_response(exc)
        return Response(data)
