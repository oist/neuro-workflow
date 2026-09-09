"""
API views for parameter metadata service.

This module provides REST API endpoints for querying parameter metadata
from external databases (Allen Brain Atlas, NeuroMorpho, etc.).
"""

import sys
import os
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.db.models import Q
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging

from app.auth.authentication import KeycloakAuthentication

from .serializers import (
    ParameterSuggestionRequestSerializer,
    ParameterSuggestionResponseSerializer,
    ParameterSuggestionSerializer,
    CustomDatabaseSerializer,
    CustomDatabaseCreateSerializer,
    DatabaseConnectionTestSerializer
)
from .models import CustomDatabase
from django.utils import timezone

logger = logging.getLogger(__name__)


def _visible_custom_databases(user):
    return CustomDatabase.objects.filter(is_active=True).filter(
        Q(created_by=user) | Q(created_by__isnull=True)
    )


def _manageable_custom_databases(user):
    """Return records a user may inspect for management actions.

    Listing and suggestion use should stay active-only, but owners need to be
    able to update/reactivate/delete their inactive databases.
    """
    if not user or not user.is_authenticated:
        return CustomDatabase.objects.none()
    if user and user.is_staff:
        return CustomDatabase.objects.all()
    return CustomDatabase.objects.filter(created_by=user)


def _can_modify_custom_database(user, database):
    if not user or not user.is_authenticated:
        return False
    if user.is_staff:
        return True
    return database.created_by_id == user.id

def _custom_databases_for_suggestions(user):
    if not user or not getattr(user, "is_authenticated", False):
        return CustomDatabase.objects.none()
    return _visible_custom_databases(user).filter(is_verified=True)


# Lazy import function for ParameterMetadataService
def get_metadata_service_instance(user=None):
    """
    Get an instance of ParameterMetadataService.
    Uses lazy loading to avoid import issues at module load time.
    """
    # Always try Docker path first
    docker_src_path = "/django-app/src"
    if docker_src_path not in sys.path:
        sys.path.insert(0, docker_src_path)
    
    # Read OpenAI API key from environment
    # First try: Direct environment variable (from docker-compose env_file)
    openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
    
    # Second try: Load from .env file using python-dotenv (same as config.py does)
    if not openai_api_key:
        try:
            from dotenv import load_dotenv
            # Load .env file (same way config.py does it)
            # This should work because config.py already loads it, but just in case...
            load_dotenv()  # Loads from current directory or parent directories
            openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
            if openai_api_key:
                logger.info("Loaded OpenAI API key from .env file via dotenv")
        except Exception as e:
            logger.debug(f"Could not load .env file: {e}")
    
    # Third try: Explicit path to .env file
    if not openai_api_key:
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            # Try explicit path in Docker container
            env_path = Path('/django-app/.env')
            if env_path.exists():
                load_dotenv(env_path, override=False)
                openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
                if openai_api_key:
                    logger.info(f"Loaded OpenAI API key from explicit path: {env_path}")
        except Exception as e:
            logger.debug(f"Could not load .env from explicit path: {e}")
    
    # Build config for the service
    config = {}
    if openai_api_key:
        config['openai_api_key'] = openai_api_key
        config['use_openai'] = True
        logger.info(f"OpenAI API key found (length: {len(openai_api_key)}), will use OpenAI for suggestions")
    else:
        logger.info("No OpenAI API key found, will use stub implementation")

    # Local RAG / semantic search (e.g. HyperRag) – optional; only active when BASE_URL is set
    local_rag_url = os.getenv('LOCAL_RAG_BASE_URL', '').strip()
    local_rag_enabled = os.getenv('LOCAL_RAG_ENABLED', 'true').lower() in ('true', '1', 'yes')
    if local_rag_url:
        config['local_rag'] = {
            'base_url': local_rag_url,
            'enabled': local_rag_enabled,
            'query_endpoint': os.getenv('LOCAL_RAG_QUERY_ENDPOINT', '/query').strip() or '/query',
            'timeout': int(os.getenv('LOCAL_RAG_TIMEOUT', '90') or '90'),
            'max_chunks': int(os.getenv('LOCAL_RAG_MAX_CHUNKS', '10') or '10'),
            'source_name': os.getenv('LOCAL_RAG_SOURCE_NAME', 'Local RAG').strip() or 'Local RAG',
            'username': os.getenv('LOCAL_RAG_USERNAME', '').strip(),
            'password': os.getenv('LOCAL_RAG_PASSWORD', ''),
            'login_endpoint': (os.getenv('LOCAL_RAG_LOGIN_ENDPOINT') or '').strip(),
            'use_username_as_bearer': os.getenv('LOCAL_RAG_USE_USERNAME_AS_BEARER', 'true').lower() in ('true', '1', 'yes'),
        }
        logger.info("Local RAG configured: base_url=%s (use host.docker.internal when backend runs in Docker)", local_rag_url)
        # Allow enough time for RAG /global_query (often 50–70s); overall adapter wait uses this
        rag_timeout = int(os.getenv('LOCAL_RAG_TIMEOUT', '90') or '90')
        config['database_query_timeout_sec'] = max(rag_timeout + 30, 120)
    else:
        logger.info("Local RAG not configured (set LOCAL_RAG_BASE_URL in backend .env to enable; use http://host.docker.internal:8006 when using Docker)")
    if 'database_query_timeout_sec' not in config:
        config['database_query_timeout_sec'] = 10
    
    try:
        from neuroworkflow.utils.parameter_metadata_service import (
            ParameterMetadataService,
            get_metadata_service
        )
        logger.info(f"Successfully imported ParameterMetadataService from {docker_src_path}")
        service = get_metadata_service(config=config)
        
        # Load custom databases from Django database
        try:
            custom_dbs = _custom_databases_for_suggestions(user)
            for db in custom_dbs:
                db_config = db.to_adapter_config(openai_client=service.openai_client)
                db_config['source_name'] = db.name.lower().replace(' ', '_')
                service.add_custom_database(db_config)
                logger.info(f"Loaded custom database: {db.name}")
        except Exception as e:
            logger.debug(f"Could not load custom databases (this is OK if models aren't migrated yet): {e}")
        
        logger.info(f"Got metadata service instance: {service}")
        return service
    except Exception as e:
        logger.error(f"Failed to import/get metadata service: {e}", exc_info=True)
        # Try other paths as fallback
        backend_dir = Path(__file__).parent.parent.parent.parent
        src_paths = [
            backend_dir / "src",
            backend_dir.parent.parent / "src",
        ]
        
        for src_path in src_paths:
            if src_path.exists():
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                try:
                    from neuroworkflow.utils.parameter_metadata_service import (
                        ParameterMetadataService,
                        get_metadata_service
                    )
                    logger.info(f"Successfully imported ParameterMetadataService from {src_path}")
                    # Read OpenAI API key from environment
                    openai_api_key = os.getenv('OPENAI_API_KEY', '').strip()
                    config = {'openai_api_key': openai_api_key, 'use_openai': True} if openai_api_key else {}
                    return get_metadata_service(config=config)
                except Exception as e2:
                    logger.debug(f"Could not import from {src_path}: {e2}")
                    continue
    
    logger.warning("Could not import ParameterMetadataService from any path")
    return None


@method_decorator(csrf_exempt, name="dispatch")
class ParameterSuggestionView(APIView):
    """
    API endpoint for getting parameter value suggestions.
    
    GET /api/metadata/parameters/suggest/
    
    Query Parameters:
        - parameter_name: str (required) - Name of the parameter
        - parameter_description: str (required) - Description of the parameter
        - node_type: str (optional) - Type of node this parameter belongs to
        - species: str (optional) - Species to query for (mouse, monkey, human, etc.)
    """
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Get parameter value suggestions.
        
        Query parameters:
            - parameter_name: str (required)
            - parameter_description: str (required)
            - node_type: str (optional)
            - species: str (optional)
        """
        # Validate query parameters
        serializer = ParameterSuggestionRequestSerializer(data=request.query_params)
        
        if not serializer.is_valid():
            return Response(
                {
                    "error": "Invalid request parameters",
                    "details": serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        validated_data = serializer.validated_data
        
        try:
            # Get metadata service instance (lazy import)
            metadata_service = get_metadata_service_instance(user=request.user)
            
            if metadata_service is None:
                logger.error("ParameterMetadataService is not available")
                return Response(
                    {
                        "error": "Parameter metadata service is not available",
                        "message": "The service could not be imported. Please check the installation."
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Query for suggestions
            suggestions = metadata_service.suggest_parameter_values(
                parameter_name=validated_data['parameter_name'],
                parameter_description=validated_data['parameter_description'],
                node_type=validated_data.get('node_type'),
                species=validated_data.get('species'),
                context=validated_data.get('context')
            )
            
            # Serialize suggestions
            suggestion_data = []
            for suggestion in suggestions:
                suggestion_data.append({
                    "value": suggestion.value,
                    "source": suggestion.source,
                    "confidence": suggestion.confidence,
                    "description": suggestion.description,
                    "species": suggestion.species,
                    "citation": suggestion.citation,
                    "metadata": suggestion.metadata or {}
                })
            
            # Build response
            response_data = {
                "suggestions": suggestion_data,
                "parameter_name": validated_data['parameter_name'],
                "parameter_description": validated_data['parameter_description'],
                "species": validated_data.get('species')
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting parameter suggestions: {e}", exc_info=True)
            return Response(
                {
                    "error": "Internal server error",
                    "message": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@method_decorator(csrf_exempt, name="dispatch")
class SpeciesSpecificParametersView(APIView):
    """
    API endpoint for getting species-specific parameter values for a node type.
    
    GET /api/metadata/parameters/species-specific/
    
    Query Parameters:
        - node_type: str (required) - Type of node
        - species: str (required) - Species (mouse, monkey, human, etc.)
        - parameter_names: str (optional, comma-separated) - Specific parameters to query
    """
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Get species-specific parameter values."""
        node_type = request.query_params.get('node_type')
        species = request.query_params.get('species')
        parameter_names_str = request.query_params.get('parameter_names')
        
        if not node_type or not species:
            return Response(
                {
                    "error": "Missing required parameters",
                    "message": "Both 'node_type' and 'species' are required"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Parse parameter names if provided
        parameter_names = None
        if parameter_names_str:
            parameter_names = [name.strip() for name in parameter_names_str.split(',')]
        
        try:
            # Get metadata service instance (lazy import)
            metadata_service = get_metadata_service_instance(user=request.user)
            
            if metadata_service is None:
                return Response(
                    {
                        "error": "Parameter metadata service is not available"
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Get species-specific parameters
            parameters = metadata_service.get_species_specific_parameters(
                node_type=node_type,
                species=species,
                parameter_names=parameter_names
            )
            
            return Response(
                {
                    "node_type": node_type,
                    "species": species,
                    "parameters": parameters
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            logger.error(f"Error getting species-specific parameters: {e}", exc_info=True)
            return Response(
                {
                    "error": "Internal server error",
                    "message": str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Built-in database sources that the parameter metadata service always
# initializes (see neuroworkflow.utils.parameter_metadata_service). They are
# configured in code rather than stored as CustomDatabase records, so the UI
# shows them as read-only entries alongside user-added databases.
BUILTIN_DATABASE_SOURCES = [
    {
        "source": "allen_brain",
        "name": "Allen Brain Atlas",
        "description": "Cell Types Database (electrophysiology & morphology) via allensdk.",
        "base_url": "https://api.brain-map.org/api/v2",
        "requires_api_key": False,
    },
    {
        "source": "neuromorpho",
        "name": "NeuroMorpho.org",
        "description": "Digitally reconstructed neuron morphologies.",
        "base_url": "https://neuromorpho.org/api",
        "requires_api_key": False,
    },
    {
        "source": "pubmed",
        "name": "PubMed / NCBI",
        "description": "Parameter values extracted from research abstracts (NCBI E-utilities).",
        "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "requires_api_key": False,
    },
    {
        "source": "neuroml_db",
        "name": "NeuroML-DB",
        "description": "Curated NeuroML model database.",
        "base_url": "https://neuroml-db.org/api",
        "requires_api_key": False,
    },
]


@method_decorator(csrf_exempt, name="dispatch")
class BuiltinDatabaseListView(APIView):
    """List the built-in, always-on database sources used for parameter
    suggestions.

    These are managed in code (not the CustomDatabase table), so they are
    returned as read-only entries. Local RAG is included only when a base URL
    is configured, mirroring get_metadata_service_instance().

    GET /api/metadata/builtin-databases/
    """
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        sources = [
            dict(source, is_active=True, is_builtin=True)
            for source in BUILTIN_DATABASE_SOURCES
        ]

        # Local RAG is only registered when a base URL is configured.
        local_rag_url = os.getenv("LOCAL_RAG_BASE_URL", "").strip()
        if local_rag_url:
            local_rag_enabled = os.getenv("LOCAL_RAG_ENABLED", "true").lower() in ("true", "1", "yes")
            sources.append({
                "source": "local_rag",
                "name": os.getenv("LOCAL_RAG_SOURCE_NAME", "Local RAG").strip() or "Local RAG",
                "description": "Local semantic search over your own documents (RAG).",
                "base_url": local_rag_url,
                "requires_api_key": False,
                "is_active": local_rag_enabled,
                "is_builtin": True,
            })

        return Response(sources, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name="dispatch")
class CustomDatabaseListView(APIView):
    """
    API endpoint for listing and creating custom databases.
    
    GET /api/metadata/custom-databases/ - List all custom databases
    POST /api/metadata/custom-databases/ - Create a new custom database
    """
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """List all custom databases."""
        try:
            databases = _visible_custom_databases(request.user)
            serializer = CustomDatabaseSerializer(databases, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error listing custom databases: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request):
        """Create a new custom database."""
        serializer = CustomDatabaseCreateSerializer(data=request.data, context={"request": request})
        
        if not serializer.is_valid():
            return Response(
                {"error": "Invalid data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Create database instance
            database = serializer.save(created_by=request.user)
            
            # Try to test connection
            test_result = self._test_connection(database)
            database.is_verified = test_result.get('success', False)
            database.last_tested = timezone.now()
            database.test_result = test_result.get('message', '')
            database.test_error = test_result.get('error', '')
            
            # Update adapter config if working pattern was found
            if test_result.get('working_pattern'):
                database.adapter_type = test_result['working_pattern'].get('adapter_type', 'rest_api')
                if 'config' not in database.config:
                    database.config = {}
                database.config.update(test_result['working_pattern'])
            
            database.save()
            
            response_serializer = CustomDatabaseSerializer(database)
            return Response(response_serializer.data, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            logger.error(f"Error creating custom database: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _test_connection(self, database: CustomDatabase) -> dict:
        """Test connection to a custom database."""
        try:
            # Import connection tester
            docker_src_path = "/django-app/src"
            if docker_src_path not in sys.path:
                sys.path.insert(0, docker_src_path)
            
            from neuroworkflow.utils.database_adapters.connection_tester import DatabaseConnectionTester
            
            # Get OpenAI client if available
            openai_client = None
            try:
                metadata_service = get_metadata_service_instance(user=database.created_by)
                if metadata_service:
                    openai_client = metadata_service.openai_client
            except:
                pass
            
            # Test connection
            tester = DatabaseConnectionTester(openai_client=openai_client)
            result = tester.test_adapter_patterns(
                base_url=database.base_url,
                api_key=database.resolve_api_key(),
                config=database.config
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error testing connection: {e}", exc_info=True)
            return {
                'success': False,
                'error': 'Test failed',
                'message': f'Could not test connection: {str(e)}'
            }


@method_decorator(csrf_exempt, name="dispatch")
class CustomDatabaseDetailView(APIView):
    """
    API endpoint for retrieving, updating, and deleting custom databases.
    
    GET /api/metadata/custom-databases/{id}/ - Get database details
    PUT /api/metadata/custom-databases/{id}/ - Update database
    DELETE /api/metadata/custom-databases/{id}/ - Delete database
    """
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, db_id):
        """Get custom database details."""
        try:
            database = _manageable_custom_databases(request.user).get(id=db_id)
            serializer = CustomDatabaseSerializer(database)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except CustomDatabase.DoesNotExist:
            return Response(
                {"error": "Database not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error getting custom database: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def put(self, request, db_id):
        """Update custom database."""
        try:
            database = _manageable_custom_databases(request.user).get(id=db_id)
            if not _can_modify_custom_database(request.user, database):
                return Response(
                    {"error": "You don't have permission to update this database"},
                    status=status.HTTP_403_FORBIDDEN,
                )
            serializer = CustomDatabaseSerializer(database, data=request.data, partial=True, context={"request": request})
            
            if not serializer.is_valid():
                return Response(
                    {"error": "Invalid data", "details": serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            database = serializer.save()
            
            # Re-test connection if URL or API key changed
            if 'base_url' in request.data or 'api_key' in request.data:
                test_result = self._test_connection(database)
                database.is_verified = test_result.get('success', False)
                database.last_tested = timezone.now()
                database.test_result = test_result.get('message', '')
                database.test_error = test_result.get('error', '')
                database.save()
            
            response_serializer = CustomDatabaseSerializer(database)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
        
        except CustomDatabase.DoesNotExist:
            return Response(
                {"error": "Database not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error updating custom database: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def delete(self, request, db_id):
        """Delete custom database (soft delete by setting is_active=False)."""
        try:
            database = _manageable_custom_databases(request.user).get(id=db_id)
            if not _can_modify_custom_database(request.user, database):
                return Response(
                    {"error": "You don't have permission to delete this database"},
                    status=status.HTTP_403_FORBIDDEN,
                )
            database.is_active = False
            database.save()
            return Response(
                {"message": "Database deleted successfully"},
                status=status.HTTP_200_OK
            )
        except CustomDatabase.DoesNotExist:
            return Response(
                {"error": "Database not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error deleting custom database: {e}", exc_info=True)
            return Response(
                {"error": "Internal server error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _test_connection(self, database: CustomDatabase) -> dict:
        """Test connection to a custom database."""
        try:
            docker_src_path = "/django-app/src"
            if docker_src_path not in sys.path:
                sys.path.insert(0, docker_src_path)
            
            from neuroworkflow.utils.database_adapters.connection_tester import DatabaseConnectionTester
            
            openai_client = None
            try:
                metadata_service = get_metadata_service_instance(user=database.created_by)
                if metadata_service:
                    openai_client = metadata_service.openai_client
            except:
                pass
            
            tester = DatabaseConnectionTester(openai_client=openai_client)
            result = tester.test_adapter_patterns(
                base_url=database.base_url,
                api_key=database.resolve_api_key(),
                config=database.config
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error testing connection: {e}", exc_info=True)
            return {
                'success': False,
                'error': 'Test failed',
                'message': f'Could not test connection: {str(e)}'
            }


@method_decorator(csrf_exempt, name="dispatch")
class DatabaseConnectionTestView(APIView):
    """
    API endpoint for testing database connections before creating.
    
    POST /api/metadata/custom-databases/test-connection/
    """
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Test connection to a database."""
        serializer = DatabaseConnectionTestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                {"error": "Invalid data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Import connection tester
            docker_src_path = "/django-app/src"
            if docker_src_path not in sys.path:
                sys.path.insert(0, docker_src_path)
            
            from neuroworkflow.utils.database_adapters.connection_tester import DatabaseConnectionTester
            
            # Get OpenAI client if available
            openai_client = None
            try:
                metadata_service = get_metadata_service_instance(user=request.user)
                if metadata_service:
                    openai_client = metadata_service.openai_client
            except:
                pass
            
            # Test connection
            tester = DatabaseConnectionTester(openai_client=openai_client)
            api_key = serializer.validated_data.get('api_key') or ''
            secret_name = serializer.validated_data.get('api_key_secret_name') or ''
            if secret_name:
                from app.secrets.services import materialize_named_secrets

                mapping = materialize_named_secrets(
                    request.user, [secret_name], actor=request.user, audit=False
                )
                if secret_name not in mapping:
                    return Response(
                        {"error": "Secret not found.", "success": False},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
                api_key = mapping[secret_name]
            result = tester.test_adapter_patterns(
                base_url=serializer.validated_data['base_url'],
                api_key=api_key or None,
                config=serializer.validated_data.get('config', {})
            )
            
            return Response(result, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error testing connection: {e}", exc_info=True)
            return Response(
                {
                    "error": "Internal server error",
                    "message": str(e),
                    "success": False
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

