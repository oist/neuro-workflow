"""
Parameter metadata service for connecting node parameters to external databases.

This service provides an interface for querying external databases (Allen Brain Atlas,
NeuroMorpho, custom databases, etc.) to retrieve parameter values based on parameter
descriptions and context.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MetadataSource(Enum):
    """Supported metadata sources."""
    ALLEN_BRAIN_ATLAS = "allen_brain"
    NEUROMORPHO = "neuromorpho"
    CUSTOM_DB = "custom_db"
    PUBMED = "pubmed"
    NEUROML_DB = "neuroml_db"


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
    
    This is a stub implementation that can be extended with real database connections.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata service.
        
        Args:
            config: Configuration dictionary with database connection info
        """
        self.config = config or {}
        self._sources_enabled = self.config.get('sources', [])
    
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
        
        Args:
            parameter_name: Name of the parameter
            parameter_description: Description of the parameter (acts as a prompt)
            node_type: Type of node this parameter belongs to
            species: Species to query for (mouse, monkey, human, etc.)
            context: Additional context (e.g., brain region, cell type)
        
        Returns:
            List of parameter suggestions with confidence scores and source information
        """
        suggestions = []
        
        # Stub implementation - returns example suggestions
        # In a real implementation, this would query actual databases
        
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
        
        # If no specific matches, return empty list
        # In real implementation, would do more sophisticated querying
        
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

