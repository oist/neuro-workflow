"""Remote neuroscience dataset catalog clients (DANDI, CBS, Brain/MINDS, BMB Human).

Ported from the bm_mindsdb ingestion stack. Each client exposes a listing
method returning a common envelope ``{status, count, total, datasets, ...}``.
Optional enrichment (:mod:`.enrichment`) attaches download URLs, DOI, and
related-publication columns to each record.
"""

from . import enrichment
from .clients import (
    BMBHumanAPIClient,
    BrainMINDSAPIClient,
    CBSAPIClient,
    DANDIAPIClient,
)

__all__ = [
    "DANDIAPIClient",
    "CBSAPIClient",
    "BrainMINDSAPIClient",
    "BMBHumanAPIClient",
    "enrichment",
]
