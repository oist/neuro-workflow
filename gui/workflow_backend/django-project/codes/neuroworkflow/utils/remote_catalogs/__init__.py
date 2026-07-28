"""Remote neuroscience dataset catalog clients (CBS, BMB Human).

Ported from the bm_mindsdb ingestion stack. Each client exposes a listing
method returning a common envelope ``{status, count, total, datasets, ...}``.
Optional enrichment (:mod:`.enrichment`) attaches download URLs, DOI, and
related-publication columns to each record.
"""

from . import enrichment
from .clients import (
    BMBHumanAPIClient,
    CBSAPIClient,
)

__all__ = [
    "CBSAPIClient",
    "BMBHumanAPIClient",
    "enrichment",
]
