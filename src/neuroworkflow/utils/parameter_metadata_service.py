"""
Parameter metadata service for connecting node parameters to external databases.

This service provides an interface for querying external databases (Allen Brain Atlas,
NeuroMorpho, custom databases, etc.) to retrieve parameter values based on parameter
descriptions and context.

Enhanced with OpenAI integration for intelligent parameter suggestions.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Try to import OpenAI (optional dependency)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

logger = logging.getLogger(__name__)


class MetadataSource(Enum):
    """Supported metadata sources."""
    ALLEN_BRAIN_ATLAS = "allen_brain"
    NEUROMORPHO = "neuromorpho"
    CUSTOM_DB = "custom_db"
    PUBMED = "pubmed"
    NEUROML_DB = "neuroml_db"
    LOCAL_RAG = "local_rag"


@dataclass
class ParameterSuggestion:
    """A suggested parameter value from a metadata source."""
    value: Any
    source: str  # MetadataSource name
    confidence: float  # 0.0 to 1.0
    description: str  # Explanation of the suggestion
    species: Optional[str] = None  # Species this value applies to (mouse, monkey, human, etc.)
    citation: Optional[str] = None  # Paper or source citation
    metadata: Dict[str, Any] = None  # Additional metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ParameterMetadataService:
    """
    Service for querying parameter metadata from external sources.
    
    Enhanced with OpenAI integration for intelligent parameter suggestions.
    Falls back to stub implementation if OpenAI is not available.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata service.
        
        Args:
            config: Configuration dictionary with database connection info
                     - openai_api_key: OpenAI API key (optional, can also use OPENAI_API_KEY env var)
                     - openai_model: Model to use (default: "gpt-4o-mini")
                     - use_openai: Whether to use OpenAI (default: True if key available)
        """
        self.config = config or {}
        self._sources_enabled = self.config.get('sources', [])
        
        # OpenAI configuration
        self.openai_api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY', '').strip()
        self.openai_model = self.config.get('openai_model', 'gpt-4o-mini')
        self.use_openai = self.config.get('use_openai', True) if self.openai_api_key else False
        self.use_web_search = self.config.get('use_web_search', True)  # Enable web search by default
        
        # Initialize OpenAI client FIRST (before database adapters need it)
        self.openai_client = None
        if OPENAI_AVAILABLE and self.openai_api_key and self.use_openai:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
                self.use_openai = False
        elif not OPENAI_AVAILABLE:
            logger.debug("OpenAI package not available, using stub implementation")
        elif not self.openai_api_key:
            logger.debug("OpenAI API key not provided, using stub implementation")
        
        # How long to wait for all DB adapters (e.g. 10s without RAG; 120s+ when Local RAG is used)
        self.database_query_timeout_sec = float(self.config.get('database_query_timeout_sec', 10))
        # Database adapters configuration (after OpenAI client is ready)
        self.database_adapters = []
        self._initialize_database_adapters()
    
    def _initialize_database_adapters(self):
        """Initialize database adapters from config."""
        try:
            # Try to import database adapters
            from .database_adapters import (
                AllenBrainAdapter,
                NeuroMorphoAdapter,
                PubMedAdapter,
                NeuroMLDBAdapter,
                LocalRAGAdapter,
            )
            
            # Initialize Allen Brain Atlas adapter (no API key needed - it's free!)
            allen_config = self.config.get('allen_brain', {})
            allen_adapter = AllenBrainAdapter({
                'api_key': allen_config.get('api_key', ''),  # Not required, but can be set
                'base_url': allen_config.get('base_url'),
                'enabled': allen_config.get('enabled', True),
                'manifest_file': allen_config.get('manifest_file', 'cell_types/manifest.json'),
                'openai_client': self.openai_client  # Pass OpenAI client for AI-powered mapping
            })
            if allen_adapter.is_available():
                self.database_adapters.append(allen_adapter)
                logger.info("Allen Brain Atlas adapter initialized")
            
            # Initialize NeuroMorpho adapter (no API key needed - it's free!)
            neuromorpho_config = self.config.get('neuromorpho', {})
            neuromorpho_adapter = NeuroMorphoAdapter({
                'api_key': neuromorpho_config.get('api_key', ''),  # Not required
                'base_url': neuromorpho_config.get('base_url'),
                'enabled': neuromorpho_config.get('enabled', True),
                'openai_client': self.openai_client  # Pass OpenAI client for AI-powered mapping
            })
            if neuromorpho_adapter.is_available():
                self.database_adapters.append(neuromorpho_adapter)
                logger.info("NeuroMorpho adapter initialized")

            # Initialize PubMed adapter (API key optional but recommended for higher rate limits)
            pubmed_config = self.config.get('pubmed', {})
            pubmed_adapter = PubMedAdapter({
                'api_key': pubmed_config.get('api_key', ''),  # Optional: 3 req/sec without, 10 req/sec with
                'base_url': pubmed_config.get('base_url'),
                'enabled': pubmed_config.get('enabled', True),
                'max_results': pubmed_config.get('max_results', 10),
                'openai_client': self.openai_client  # Pass OpenAI client for AI-powered query optimization and extraction
            })
            if pubmed_adapter.is_available():
                self.database_adapters.append(pubmed_adapter)
                logger.info("PubMed adapter initialized")

            # Initialize NeuroML-DB adapter (no API key needed - it's free!)
            neuroml_config = self.config.get('neuroml_db', {})
            neuroml_adapter = NeuroMLDBAdapter({
                'base_url': neuroml_config.get('base_url'),
                'enabled': neuroml_config.get('enabled', True),
                'max_results': neuroml_config.get('max_results', 10),
                'openai_client': self.openai_client  # Pass OpenAI client for AI-powered query optimization
            })
            if neuroml_adapter.is_available():
                self.database_adapters.append(neuroml_adapter)
                logger.info("NeuroML-DB adapter initialized")

            # Initialize Local RAG adapter (local semantic search / HyperRag-style API)
            local_rag_config = self.config.get("local_rag", {})
            local_rag_adapter = LocalRAGAdapter({
                "base_url": local_rag_config.get("base_url", ""),
                "api_key": local_rag_config.get("api_key", ""),
                "username": local_rag_config.get("username", ""),
                "password": local_rag_config.get("password", ""),
                "login_endpoint": local_rag_config.get("login_endpoint", ""),
                "use_username_as_bearer": local_rag_config.get("use_username_as_bearer", True),
                "enabled": local_rag_config.get("enabled", True),
                "query_endpoint": local_rag_config.get("query_endpoint", "/query"),
                "timeout": local_rag_config.get("timeout", 15),
                "max_chunks": local_rag_config.get("max_chunks", 10),
                "source_name": local_rag_config.get("source_name", "Local RAG"),
                "openai_client": self.openai_client,
            })
            if local_rag_adapter.is_available():
                self.database_adapters.append(local_rag_adapter)
                logger.info("Local RAG adapter initialized")

            # Load custom databases from config (if provided)
            custom_databases = self.config.get('custom_databases', [])
            self._load_custom_databases(custom_databases)

        except ImportError as e:
            logger.debug(f"Database adapters not available: {e}")
        except Exception as e:
            logger.warning(f"Error initializing database adapters: {e}")
    
    def _load_custom_databases(self, custom_databases: List[Dict[str, Any]]):
        """Load custom database adapters from configuration."""
        try:
            from .database_adapters.generic import GenericDatabaseAdapter
            
            for db_config in custom_databases:
                try:
                    # Ensure source_name is set
                    if 'source_name' not in db_config:
                        db_config['source_name'] = db_config.get('name', 'custom_db')
                    
                    # Add OpenAI client
                    db_config['openai_client'] = self.openai_client
                    
                    # Create adapter
                    adapter = GenericDatabaseAdapter(db_config)
                    
                    if adapter.is_available():
                        self.database_adapters.append(adapter)
                        logger.info(f"Custom database adapter '{db_config.get('source_name')}' initialized")
                    else:
                        logger.debug(f"Custom database adapter '{db_config.get('source_name')}' not available")
                
                except Exception as e:
                    logger.warning(f"Error loading custom database adapter: {e}")
                    continue
        
        except ImportError as e:
            logger.debug(f"Generic database adapter not available: {e}")
        except Exception as e:
            logger.warning(f"Error loading custom databases: {e}")
    
    def add_custom_database(self, db_config: Dict[str, Any]):
        """
        Add a custom database adapter dynamically.
        
        Args:
            db_config: Configuration dictionary for the custom database
                      - base_url: Base URL of the database
                      - api_key: API key if required
                      - source_name: Name for this database source
                      - adapter_type: Type of adapter (rest_api, graphql, sdk)
                      - config: Additional configuration
        """
        try:
            from .database_adapters.generic import GenericDatabaseAdapter
            
            # Ensure source_name is set
            if 'source_name' not in db_config:
                db_config['source_name'] = db_config.get('name', 'custom_db')
            
            # Add OpenAI client
            db_config['openai_client'] = self.openai_client
            
            # Create adapter
            adapter = GenericDatabaseAdapter(db_config)
            
            if adapter.is_available():
                self.database_adapters.append(adapter)
                logger.info(f"Custom database adapter '{db_config.get('source_name')}' added successfully")
                return True
            else:
                logger.warning(f"Custom database adapter '{db_config.get('source_name')}' not available")
                return False
        
        except Exception as e:
            logger.error(f"Error adding custom database: {e}", exc_info=True)
            return False
    
    def suggest_parameter_values(
        self,
        parameter_name: str,
        parameter_description: str,
        node_type: Optional[str] = None,
        species: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ParameterSuggestion]:
        """
        Query external databases and return suggested parameter values.
        
        Uses a hybrid approach:
        1. Query real databases (if configured)
        2. Use OpenAI with web search (if available)
        3. Fall back to stub implementation
        
        Args:
            parameter_name: Name of the parameter
            parameter_description: Description of the parameter (acts as a prompt)
            node_type: Type of node this parameter belongs to
            species: Species to query for (mouse, monkey, human, etc.)
            context: Additional context (e.g., brain region, cell type)
        
        Returns:
            List of parameter suggestions with confidence scores and source information
        """
        all_suggestions = []
        
        # Step 1: Query real databases FIRST (priority)
        # Use threading to query databases in parallel for better performance
        import threading
        from queue import Queue
        
        database_suggestions = []
        results_queue = Queue()
        
        def query_adapter(adapter):
            """Query a single adapter and put results in queue"""
            try:
                db_suggestions = adapter.query_parameter(
                    parameter_name, parameter_description, species, context
                )
                results_queue.put((adapter.get_source_name(), db_suggestions, None))
                logger.info(f"Got {len(db_suggestions)} suggestions from {adapter.get_source_name()}")
            except Exception as e:
                logger.error(f"Database adapter {adapter.get_source_name()} failed: {e}", exc_info=True)
                results_queue.put((adapter.get_source_name(), [], e))
        
        # Start all adapter queries in parallel
        threads = []
        for adapter in self.database_adapters:
            thread = threading.Thread(target=query_adapter, args=(adapter,))
            thread.daemon = True  # Don't wait for threads if main thread exits
            thread.start()
            threads.append(thread)
        
        # Wait for all threads with timeout (configurable; long when Local RAG is used)
        import time
        start_time = time.time()
        timeout = self.database_query_timeout_sec
        
        for thread in threads:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                logger.warning("Database query timeout reached, returning partial results")
                break
            thread.join(timeout=remaining_time)
        
        # Collect results from queue (wait for all threads to finish or timeout)
        # Collect all results that are ready, even if some threads are still running
        collected_sources = set()
        max_wait = timeout + 5.0  # Buffer after thread join to drain queue
        queue_start = time.time()
        
        while (time.time() - queue_start) < max_wait:
            try:
                # Try to get result with short timeout
                source_name, suggestions, error = results_queue.get(timeout=0.5)
                collected_sources.add(source_name)
                
                if error:
                    logger.warning(f"Database adapter {source_name} had error: {error}")
                
                if suggestions:
                    database_suggestions.extend(suggestions)
                    logger.info(f"Collected {len(suggestions)} suggestions from {source_name}")
                
                # If we've collected from all adapters, we're done
                if len(collected_sources) >= len(self.database_adapters):
                    break
                    
            except:
                # Queue empty or timeout - check if all threads are done
                active_threads = [t for t in threads if t.is_alive()]
                if not active_threads:
                    # All threads finished, try one more time to get remaining results
                    try:
                        while not results_queue.empty():
                            source_name, suggestions, error = results_queue.get_nowait()
                            if error:
                                logger.warning(f"Database adapter {source_name} had error: {error}")
                            if suggestions:
                                database_suggestions.extend(suggestions)
                                logger.info(f"Collected {len(suggestions)} suggestions from {source_name}")
                    except:
                        pass
                    break
        
        # Step 2: Use AI to validate, explain, and contextualize database results
        if database_suggestions and self.openai_client and self.use_openai:
            try:
                # AI validates and enhances database results
                enhanced_suggestions = self._validate_and_explain_with_ai(
                    database_suggestions,
                    parameter_name, parameter_description, node_type, species, context
                )
                all_suggestions.extend(enhanced_suggestions)
            except Exception as e:
                logger.warning(f"AI validation failed, using raw database results: {e}")
                # If AI fails, use database results as-is
                all_suggestions.extend(database_suggestions)
        elif database_suggestions:
            # No AI available, use database results directly
            all_suggestions.extend(database_suggestions)
        
        # Step 3: If no database results, use AI to generate suggestions (fallback)
        logger.debug(f"Checking OpenAI fallback - database_suggestions: {len(database_suggestions)}, openai_client: {self.openai_client is not None}, use_openai: {self.use_openai}")
        if not database_suggestions and self.openai_client and self.use_openai:
            try:
                logger.info(f"Attempting OpenAI generation for parameter: {parameter_name}")
                openai_suggestions = self._suggest_with_openai(
                    parameter_name, parameter_description, node_type, species, context,
                    existing_suggestions=[]
                )
                all_suggestions.extend(openai_suggestions)
                logger.info(f"Generated {len(openai_suggestions)} suggestions using OpenAI")
            except Exception as e:
                logger.warning(f"OpenAI suggestion failed: {e}", exc_info=True)
        elif not database_suggestions:
            logger.info(f"OpenAI not available for fallback - client: {self.openai_client is not None}, use_openai: {self.use_openai}, openai_api_key length: {len(self.openai_api_key) if self.openai_api_key else 0}")
        
        # Step 4: Final fallback to stub
        if not all_suggestions:
            all_suggestions = self._suggest_with_stub(parameter_name, parameter_description, species)
        
        return all_suggestions
    
    def _validate_and_explain_with_ai(
        self,
        database_suggestions: List[ParameterSuggestion],
        parameter_name: str,
        parameter_description: str,
        node_type: Optional[str] = None,
        species: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ParameterSuggestion]:
        """
        Use AI to validate, explain, and contextualize real database results.
        
        This prevents hallucinations by only working with real data from databases.
        AI's role is to:
        1. Validate if values make sense
        2. Explain why they're relevant
        3. Check applicability to specific node/context
        4. Enhance descriptions with context
        """
        # Build context string
        context_parts = []
        if node_type:
            context_parts.append(f"Node type: {node_type}")
        if species:
            context_parts.append(f"Species: {species}")
        if context:
            context_parts.append(f"Additional context: {json.dumps(context)}")
        context_str = "\n".join(context_parts) if context_parts else "No additional context"
        
        # Format database suggestions for AI
        db_data = []
        for sug in database_suggestions:
            db_data.append({
                "value": sug.value,
                "source": sug.source,
                "confidence": sug.confidence,
                "description": sug.description,
                "species": sug.species,
                "citation": sug.citation
            })
        
        prompt = f"""You are an expert neuroscientist validating and explaining real parameter values from neuroscience databases.

Parameter Name: {parameter_name}
Parameter Description: {parameter_description}
{context_str}

REAL DATABASE RESULTS (these are verified values from actual databases):
{json.dumps(db_data, indent=2)}

Your task is to:
1. VALIDATE: Check if these values make sense for the parameter and context
2. EXPLAIN: Provide clear explanations of why these values are appropriate
3. CONTEXTUALIZE: Assess how applicable they are for the specific node type and context
4. ENHANCE: Improve descriptions to be more helpful for the user

IMPORTANT:
- DO NOT generate new values or citations
- ONLY work with the real database values provided above
- DO NOT hallucinate citations - only use what's provided or mark as null
- Focus on validation, explanation, and context

Provide your response as a JSON object with a "suggestions" array, where each suggestion:
- Keeps the original value from database
- Keeps the original source
- Has an enhanced description explaining why it's valid and applicable
- Has confidence adjusted based on context relevance
- Keeps original citation (or null if not provided)

Return ONLY valid JSON, no other text."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that validates and explains real neuroscience data. You NEVER generate fake citations or values - only work with provided data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Lower temperature for more factual, validation-focused responses
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            
            # Extract suggestions
            suggestions_data = data.get("suggestions", [])
            
            # Convert to ParameterSuggestion objects
            enhanced_suggestions = []
            for item in suggestions_data:
                if isinstance(item, dict):
                    try:
                        # Keep original value from database (don't let AI change it)
                        original_value = item.get('value')
                        # Find matching original suggestion to preserve value
                        matching_original = next(
                            (s for s in database_suggestions if s.value == original_value),
                            None
                        )
                        if matching_original:
                            original_value = matching_original.value
                        
                        suggestion = ParameterSuggestion(
                            value=original_value,  # Always use original database value
                            source=item.get('source', 'database'),
                            confidence=float(item.get('confidence', 0.7)),
                            description=item.get('description', 'AI-validated database value'),
                            species=item.get('species', species),
                            citation=item.get('citation')  # Keep original citation
                        )
                        enhanced_suggestions.append(suggestion)
                    except Exception as e:
                        logger.warning(f"Error parsing enhanced suggestion: {e}")
                        continue
            
            logger.info(f"AI validated and enhanced {len(enhanced_suggestions)} database suggestions")
            return enhanced_suggestions if enhanced_suggestions else database_suggestions
            
        except Exception as e:
            logger.error(f"Error in AI validation: {e}", exc_info=True)
            # Return original database suggestions if AI fails
            return database_suggestions
    
    def _suggest_with_openai(
        self,
        parameter_name: str,
        parameter_description: str,
        node_type: Optional[str] = None,
        species: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        existing_suggestions: Optional[List[ParameterSuggestion]] = None
    ) -> List[ParameterSuggestion]:
        """
        Use OpenAI to generate intelligent parameter suggestions.
        
        Args:
            parameter_name: Name of the parameter
            parameter_description: Description of the parameter
            node_type: Type of node
            species: Species filter
            context: Additional context
        
        Returns:
            List of parameter suggestions
        """
        # Build context string
        context_parts = []
        if node_type:
            context_parts.append(f"Node type: {node_type}")
        if species:
            context_parts.append(f"Species: {species}")
        if context:
            context_parts.append(f"Additional context: {json.dumps(context)}")
        context_str = "\n".join(context_parts) if context_parts else "No additional context"
        
        # Build context about existing database results
        existing_context = ""
        if existing_suggestions:
            existing_context = "\n\nExisting database results:\n"
            for i, sug in enumerate(existing_suggestions[:3], 1):  # Show first 3
                existing_context += f"{i}. {sug.value} ({sug.source}, confidence: {sug.confidence})\n"
            existing_context += "\nYou can use these as reference, but also search for additional or more recent information."
        
        # Create prompt for OpenAI
        # Use JSON object format for better compatibility with response_format
        web_search_instruction = ""
        if self.use_web_search:
            web_search_instruction = "\n\nIMPORTANT: Use web search to find current, real information from neuroscience databases and recent papers. Search for specific values from Allen Brain Atlas, NeuroMorpho.org, or recent research papers."
        
        prompt = f"""You are an expert neuroscientist helping to suggest parameter values for brain simulation models.

Parameter Name: {parameter_name}
Parameter Description: {parameter_description}
{context_str}{existing_context}

CRITICAL RULES FOR SOURCES AND CITATIONS:
- Since NO real database data was found, you are generating ESTIMATES based on general neuroscience knowledge
- Use source "expert_knowledge" or "openai" ONLY - DO NOT use "allen_brain", "neuromorpho", or "pubmed"
- DO NOT generate fake citations - set citation to null
- Be honest in descriptions that these are estimates, not verified database values
- If you mention a general concept (e.g., "typical values in literature"), do NOT cite a specific paper unless you are 100% certain it exists

Provide your response as a JSON object with a "suggestions" array:
{{
  "suggestions": [
    {{
      "value": <numeric value or null>,
      "source": "expert_knowledge" or "openai",
      "confidence": <0.0 to 1.0>,
      "description": "<explanation - be clear this is an estimate, not verified database data>",
      "species": "{species or "mouse"}",
      "citation": null
    }}
  ]
}}

Guidelines:
- Provide 1-3 suggestions
- Values should be realistic for neuroscience parameters
- Confidence should reflect how certain you are (0.6-0.8 for typical estimates, 0.4-0.6 for general suggestions)
- Include units in description if relevant
- In descriptions, use phrases like "typically observed", "commonly reported", "estimated range" - NOT "from database" or "according to [specific paper]"
- ALWAYS set citation to null when generating estimates

Return ONLY valid JSON, no other text."""

        # Note: OpenAI's web_search tool may not be available for all models
        # For now, we rely on the enhanced prompt to guide the LLM
        # Future: Could implement custom function calling for web search APIs
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides JSON responses for neuroscience parameter suggestions. Base your suggestions on verified neuroscience knowledge and current best practices."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent, factual responses
                max_tokens=500,
                response_format={"type": "json_object"}  # Use JSON mode for structured output
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Handle JSON object vs array
            try:
                data = json.loads(content)
                # If response is wrapped in an object, extract suggestions array
                if isinstance(data, dict) and "suggestions" in data:
                    suggestions_data = data["suggestions"]
                elif isinstance(data, dict) and len(data) == 1 and isinstance(list(data.values())[0], list):
                    suggestions_data = list(data.values())[0]
                elif isinstance(data, list):
                    suggestions_data = data
                else:
                    # Try to find any array in the response
                    suggestions_data = [data] if isinstance(data, dict) else []
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
                if json_match:
                    suggestions_data = json.loads(json_match.group(1))
                else:
                    logger.warning(f"Could not parse OpenAI response as JSON: {content}")
                    return []
            
            # Convert to ParameterSuggestion objects
            suggestions = []
            for item in suggestions_data:
                if isinstance(item, dict):
                    try:
                        # Parse value (handle strings that should be numbers)
                        value = item.get('value')
                        if value is not None and isinstance(value, str):
                            try:
                                # Try to parse as number
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                pass  # Keep as string
                        
                        suggestion = ParameterSuggestion(
                            value=value,
                            source=item.get('source', 'expert_knowledge'),
                            confidence=float(item.get('confidence', 0.7)),
                            description=item.get('description', 'AI-generated suggestion'),
                            species=item.get('species', species),
                            citation=item.get('citation')
                        )
                        suggestions.append(suggestion)
                    except Exception as e:
                        logger.warning(f"Error parsing suggestion item: {e}, item: {item}")
                        continue
            
            logger.info(f"Generated {len(suggestions)} suggestions using OpenAI")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
            raise
    
    def _suggest_with_stub(
        self,
        parameter_name: str,
        parameter_description: str,
        species: Optional[str] = None
    ) -> List[ParameterSuggestion]:
        """
        Stub implementation - returns example suggestions based on keyword matching.
        
        This is used as a fallback when OpenAI is not available.
        """
        suggestions = []
        
        # Example: If parameter description mentions "firing rate" or "spike rate"
        if any(term in parameter_description.lower() for term in ['firing rate', 'spike rate', 'rate']):
            suggestions.append(ParameterSuggestion(
                value=5.0,  # Hz
                source=MetadataSource.ALLEN_BRAIN_ATLAS.value,
                confidence=0.7,
                description="Typical firing rate for cortical neurons",
                species=species or "mouse",
                citation="Allen Brain Atlas - Cell Types Database"
            ))
        
        # Example: If parameter description mentions "membrane potential" or "voltage"
        if any(term in parameter_description.lower() for term in ['membrane potential', 'voltage', 'resting']):
            suggestions.append(ParameterSuggestion(
                value=-65.0,  # mV
                source=MetadataSource.NEUROMORPHO.value,
                confidence=0.8,
                description="Typical resting membrane potential",
                species=species or "mouse"
            ))
        
        # Example: If parameter description mentions "synaptic weight" or "connection strength"
        if any(term in parameter_description.lower() for term in ['synaptic', 'weight', 'strength', 'connection']):
            suggestions.append(ParameterSuggestion(
                value=1.0,
                source=MetadataSource.CUSTOM_DB.value,
                confidence=0.6,
                description="Default synaptic weight",
                species=species
            ))
        
        return suggestions
    
    def get_species_specific_parameters(
        self,
        node_type: str,
        species: str,
        parameter_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get species-specific parameter values for a node type.
        
        Args:
            node_type: Type of node (e.g., 'NW_Population')
            species: Species (mouse, monkey, human, etc.)
            parameter_names: Optional list of specific parameters to query
        
        Returns:
            Dictionary mapping parameter names to species-specific values
        """
        # Stub implementation
        result = {}
        
        # Example: Species-specific firing rates
        if species == "mouse":
            result['firing_rate'] = 5.0
            result['membrane_capacitance'] = 100.0  # pF
        elif species == "monkey":
            result['firing_rate'] = 3.0
            result['membrane_capacitance'] = 150.0  # pF
        elif species == "human":
            result['firing_rate'] = 2.0
            result['membrane_capacitance'] = 200.0  # pF
        
        # Filter by parameter_names if provided
        if parameter_names:
            result = {k: v for k, v in result.items() if k in parameter_names}
        
        return result
    
    def query_parameter_by_description(
        self,
        description: str,
        sources: Optional[List[str]] = None,
        species: Optional[str] = None
    ) -> List[ParameterSuggestion]:
        """
        Query parameters by description text (semantic search).
        
        Args:
            description: Parameter description text
            sources: List of source names to query (None = all enabled sources)
            species: Species filter
        
        Returns:
            List of parameter suggestions
        """
        # This would use semantic search/embedding models in a real implementation
        # For now, delegate to suggest_parameter_values
        return self.suggest_parameter_values(
            parameter_name="",  # Not needed for description-based search
            parameter_description=description,
            species=species
        )
    
    def get_parameter_metadata(
        self,
        parameter_name: str,
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed metadata for a parameter from a specific source.
        
        Args:
            parameter_name: Name of the parameter
            source: Source name (from MetadataSource enum)
        
        Returns:
            Metadata dictionary, or None if not found
        """
        # Stub implementation
        return {
            'parameter_name': parameter_name,
            'source': source,
            'available_species': ['mouse', 'monkey', 'human'],
            'data_points': 100,  # Example
            'last_updated': '2025-01-01'
        }
    
    def register_custom_source(
        self,
        source_name: str,
        query_function: callable
    ) -> None:
        """
        Register a custom metadata source.
        
        Args:
            source_name: Name for the custom source
            query_function: Function that takes (parameter_name, description, species)
                           and returns List[ParameterSuggestion]
        """
        # In a real implementation, would store this in a registry
        if not hasattr(self, '_custom_sources'):
            self._custom_sources = {}
        self._custom_sources[source_name] = query_function


# Convenience function for quick access
def get_metadata_service(config: Optional[Dict[str, Any]] = None) -> ParameterMetadataService:
    """
    Get a configured ParameterMetadataService instance.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        ParameterMetadataService instance
    """
    return ParameterMetadataService(config)

