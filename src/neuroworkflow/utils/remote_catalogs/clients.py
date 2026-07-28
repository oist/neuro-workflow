#!/usr/bin/env python3
"""HTTP clients for external neuroscience dataset catalogs.

Ported from the bm_mindsdb ingestion stack (``real_api_integrations.py``).
Each client wraps one remote catalog and exposes a ``get_datasets`` listing
method that returns a common envelope::

    {status, count, total, datasets: [...], api_version, timestamp, ...}

and ``{status: "error", error, ...}`` on failure. Per-record failures are
caught and skipped so one bad record never aborts a whole listing.

Only ``requests`` is required; CBS XML parsing uses stdlib ElementTree and
BMB Human landing-page scraping uses plain regex.
"""

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

try:  # advertise a real User-Agent instead of the default requests one
    from importlib.metadata import version as _pkg_version

    _NW_VERSION = _pkg_version("neuroworkflow")
except Exception:  # pragma: no cover - fallback when metadata is unavailable
    _NW_VERSION = "0.1.0"
USER_AGENT = f"neuroworkflow/{_NW_VERSION} (+https://github.com/oist/neuro-workflow)"

DEFAULT_TIMEOUT = 30

CBS_XML_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "rs": "http://www.openarchives.org/rs/terms/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "datacite": "https://schema.datacite.org/meta/kernel-4/",
    "jpcoar": "https://github.com/JPCOAR/schema/blob/master/1.0/",
}
CBS_DEFAULT_RESOURCELIST_URL = "https://neurodata.riken.jp/rs/resourcelist-0001.xml"
NEURODATA_API_BASE = "https://neurodata.riken.jp/api"
NEURODATA_V3_BASE = f"{NEURODATA_API_BASE}/v3"


class CBSAPIClient:
    """CBS (RIKEN neurodata) client fetching metadata via ResourceSync XML."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = "https://neurodata.riken.jp/rs"
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/xml, text/xml, */*", "User-Agent": USER_AGENT}
        )
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self._doi_catalog: Optional[Dict[str, str]] = None

    def _fetch_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def get_doi_catalog(self, force_refresh: bool = False) -> Dict[str, str]:
        """Resolve CBS dataset IDs to neurodata API v3 folder UUIDs."""
        if self._doi_catalog is not None and not force_refresh:
            return self._doi_catalog

        data = self._fetch_json(
            f"{NEURODATA_V3_BASE}/doi/",
            params={"doi": "10.60178/cbs.20230511-001"},
        )
        catalog: Dict[str, str] = {}
        for item in data:
            cbs_id = item.get("id")
            folder_id = (item.get("folder") or {}).get("id")
            if cbs_id and folder_id:
                catalog[str(cbs_id)] = str(folder_id)

        self._doi_catalog = catalog
        logger.info("✅ Loaded CBS neurodata DOI catalog (%s datasets)", len(catalog))
        return catalog

    def resolve_neurodata_folder_id(
        self, identifier: str, doi: str = "", catalog: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        mapping = catalog if catalog is not None else self.get_doi_catalog()
        if identifier in mapping:
            return mapping[identifier]
        doi_norm = doi.replace("https://doi.org/", "").strip("/")
        if doi_norm:
            for cbs_id, folder_id in mapping.items():
                if doi_norm.endswith(cbs_id):
                    return folder_id
        return None

    def _list_files_in_folder(
        self, folder_id: str, remaining: int
    ) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        url: Optional[str] = f"{NEURODATA_V3_BASE}/files/"
        params: Optional[Dict[str, Any]] = {
            "folder_id": folder_id,
            "limit": min(100, remaining),
        }

        while url and len(files) < remaining:
            if params:
                data = self._fetch_json(url, params=params)
                params = None
            else:
                data = self._fetch_json(url)

            batch = data.get("results", [])
            if not batch:
                break
            files.extend(batch[: remaining - len(files)])
            url = data.get("next")
        return files

    def collect_folder_files(
        self, folder_id: str, file_limit: int = 100
    ) -> Dict[str, Any]:
        """Walk neurodata folder tree and collect downloadable files."""
        collected: List[Dict[str, Any]] = []
        queue: List[str] = [folder_id]
        visited: set = set()
        truncated = False

        while queue and len(collected) < file_limit:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            remaining = file_limit - len(collected)
            collected.extend(self._list_files_in_folder(current, remaining))
            if len(collected) >= file_limit:
                truncated = bool(queue)
                break

            try:
                folder = self._fetch_json(f"{NEURODATA_V3_BASE}/folders/{current}/")
            except requests.exceptions.RequestException as exc:
                logger.warning("⚠️ CBS folder skip %s: %s", current, exc)
                continue

            for subfolder in folder.get("sub_folders", []):
                sub_id = subfolder.get("id")
                if sub_id and sub_id not in visited:
                    queue.append(str(sub_id))

        if len(collected) >= file_limit and queue:
            truncated = True

        return {
            "status": "success",
            "files": collected[:file_limit],
            "total_count": len(collected),
            "truncated": truncated,
        }

    def _fetch_xml(self, url: str) -> ET.Element:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return ET.fromstring(response.content)

    def _resolve_resourcelist_url(self) -> str:
        capability_url = f"{self.base_url}/capabilitylist.xml"
        logger.info(f"🔍 Resolving CBS resourcelist from {capability_url}")
        root = self._fetch_xml(capability_url)
        for url_elem in root.findall(".//sm:url", CBS_XML_NS):
            md = url_elem.find("rs:md", CBS_XML_NS)
            loc = url_elem.find("sm:loc", CBS_XML_NS)
            if (
                md is not None
                and loc is not None
                and loc.text
                and md.get("capability") == "resourcelist"
            ):
                return loc.text.strip()
        logger.warning(
            "⚠️ CBS resourcelist not found in capabilitylist; using default URL"
        )
        return CBS_DEFAULT_RESOURCELIST_URL

    def _list_resource_urls(self, resourcelist_url: str) -> List[str]:
        root = self._fetch_xml(resourcelist_url)
        urls = []
        for url_elem in root.findall(".//sm:url", CBS_XML_NS):
            loc = url_elem.find("sm:loc", CBS_XML_NS)
            if loc is not None and loc.text and "resource-" in loc.text:
                urls.append(loc.text.strip())
        return urls

    @staticmethod
    def _identifier_from_resource_url(resource_url: str) -> str:
        match = re.search(r"resource-(\d{8}-\d{3})\.xml", resource_url)
        if match:
            return match.group(1)
        return resource_url.rsplit("/", 1)[-1].replace(".xml", "")

    def _parse_resource_xml(
        self, resource_url: str, root: ET.Element
    ) -> Dict[str, Any]:
        title_elem = root.find(".//dc:title", CBS_XML_NS)
        description_elem = root.find(".//datacite:description", CBS_XML_NS)
        title = (title_elem.text or "").strip() if title_elem is not None else ""
        description = (
            (description_elem.text or "").strip()
            if description_elem is not None
            else ""
        )
        issued = ""
        modified = ""
        for date_elem in root.findall(".//datacite:date", CBS_XML_NS):
            date_type = date_elem.get("dateType", "")
            text = (date_elem.text or "").strip()
            if date_type == "Issued" and not issued:
                issued = text
            elif date_type == "Updated" and not modified:
                modified = text
        if not modified:
            modified = issued

        doi = ""
        landing_page = ""
        for ident in root.findall(".//jpcoar:identifier", CBS_XML_NS):
            id_type = ident.get("identifierType", "")
            text = (ident.text or "").strip()
            if id_type == "DOI" and text:
                doi = text.replace("https://doi.org/", "")
            elif id_type == "URI" and text:
                landing_page = text

        file_urls = []
        for uri_elem in root.findall(".//jpcoar:file/jpcoar:URI", CBS_XML_NS):
            if uri_elem.text:
                file_urls.append(uri_elem.text.strip())

        identifier = self._identifier_from_resource_url(resource_url)
        if landing_page and landing_page.rstrip("/").endswith(identifier):
            pass
        elif landing_page:
            tail = landing_page.rstrip("/").rsplit("/", 1)[-1]
            if tail:
                identifier = tail

        return {
            "identifier": identifier,
            "name": title or identifier,
            "title": title,
            "description": description,
            "doi": doi,
            "landing_page": landing_page,
            "issued": issued,
            "modified": modified,
            "file_urls": file_urls,
            "resource_url": resource_url,
        }

    def _fetch_and_parse_resource(self, resource_url: str) -> Dict[str, Any]:
        root = self._fetch_xml(resource_url)
        return self._parse_resource_xml(resource_url, root)

    def get_datasets(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Fetch CBS datasets using the ResourceSync protocol."""
        try:
            resourcelist_url = self._resolve_resourcelist_url()
            resource_urls = self._list_resource_urls(resourcelist_url)
            if not resource_urls:
                return {
                    "status": "error",
                    "error": "No CBS resource URLs found in resourcelist",
                    "api_version": "real",
                    "timestamp": datetime.now().isoformat(),
                }

            selected_urls = resource_urls[offset : offset + limit]
            datasets: List[Dict[str, Any]] = []
            for resource_url in selected_urls:
                try:
                    datasets.append(self._fetch_and_parse_resource(resource_url))
                except Exception as e:
                    logger.warning("⚠️ CBS resource skip %s: %s", resource_url, e)

            logger.info(
                "✅ Successfully fetched %s/%s CBS datasets via ResourceSync",
                len(datasets),
                len(resource_urls),
            )
            return {
                "status": "success",
                "count": len(datasets),
                "total": len(resource_urls),
                "datasets": datasets,
                "api_version": "real",
                "protocol": "ResourceSync",
                "resourcelist_url": resourcelist_url,
                "timestamp": datetime.now().isoformat(),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ CBS ResourceSync error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "api_version": "real",
                "timestamp": datetime.now().isoformat(),
            }

    def get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get metadata for a specific CBS dataset."""
        try:
            resourcelist_url = self._resolve_resourcelist_url()
            resource_urls = self._list_resource_urls(resourcelist_url)
            normalized_id = dataset_id.strip()

            target_url = None
            for resource_url in resource_urls:
                if (
                    normalized_id in resource_url
                    or self._identifier_from_resource_url(resource_url) == normalized_id
                ):
                    target_url = resource_url
                    break

            if not target_url:
                return {
                    "status": "error",
                    "error": f"Dataset {dataset_id} not found",
                    "api_version": "real",
                    "timestamp": datetime.now().isoformat(),
                }

            metadata = self._fetch_and_parse_resource(target_url)
            logger.info(
                f"✅ Successfully fetched metadata for CBS dataset {dataset_id}"
            )
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "metadata": metadata,
                "api_version": "real",
                "protocol": "ResourceSync",
                "timestamp": datetime.now().isoformat(),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ CBS API error for dataset {dataset_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "api_version": "real",
                "timestamp": datetime.now().isoformat(),
            }


BMB_HUMAN_BASE_URL = "https://mridata-brainminds-beyond.atr.jp"
BMB_HUMAN_DEFAULT_SLUGS = ("bmbts", "bmbpt")

# Best-effort: capture a citation string that ends at a DOI reference. Anchored
# on a citation-style author token ("Surname X") rather than a specific author,
# and accepts any DOI registrant (generalized from the original "Koike S" /
# "10.1016" hardcode).
_PAGE_CITATION_RE = re.compile(
    r"([A-Z][A-Za-z'’\-]+\s+[A-Z][A-Za-z.,;&\s]*?doi:\s*10\.\d{4,9}/[^\s<\"']+)",
    re.DOTALL,
)


class BMBHumanAPIClient:
    """Brain/MINDS Beyond Human MRI portal (mridata-brainminds-beyond.atr.jp).

    The portal has no API: this scrapes the index page for dataset slugs, reads
    a per-slug ``BMB_META.json`` metadata file, and scrapes the landing page for
    DOI/citation. No authentication.
    """

    INDEX_URL = f"{BMB_HUMAN_BASE_URL}/"
    DATASET_LINK_RE = re.compile(
        r'href="(?:https://mridata-brainminds-beyond\.atr\.jp)?/dataset/([^"/]+)/?"',
        re.IGNORECASE,
    )

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "text/html,application/json,*/*", "User-Agent": USER_AGENT}
        )

    def _fetch_text(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _fetch_json(self, url: str) -> Any:
        response = self.session.get(
            url, timeout=self.timeout, headers={"Accept": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_portal_page(html: str) -> Dict[str, str]:
        """Extract DOI and citation text from a dataset landing page."""
        extras: Dict[str, str] = {}
        doi_match = re.search(
            r"https?://(?:dx\.)?doi\.org/([^\s\"'<>]+)",
            html,
            re.IGNORECASE,
        )
        if doi_match:
            extras["doi"] = doi_match.group(1).strip().rstrip(".")

        paper_match = _PAGE_CITATION_RE.search(html)
        if paper_match:
            citation = re.sub(r"\s+", " ", paper_match.group(1)).strip()
            extras["page_citation"] = citation
        return extras

    @staticmethod
    def _dataset_id_from_schema(slug: str, schema: Dict[str, Any]) -> str:
        alternate_names = schema.get("alternateName") or []
        if alternate_names and alternate_names[0]:
            return str(alternate_names[0]).strip()
        at_id = str(schema.get("@id") or "").strip()
        if at_id.startswith("#dataset-"):
            return at_id.replace("#dataset-", "", 1)
        return slug

    def _normalize_dataset(
        self,
        slug: str,
        schema: Dict[str, Any],
        *,
        page_extras: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        page_extras = page_extras or {}
        dataset_id = self._dataset_id_from_schema(slug, schema)
        landing_page = f"{BMB_HUMAN_BASE_URL}/dataset/{slug}/"
        doi = page_extras.get("doi") or ""
        if not doi:
            for ident in schema.get("identifier") or []:
                if ident:
                    text = str(ident)
                    if "doi" in text.lower():
                        doi = text.replace("https://doi.org/", "").strip("/")
                        break

        return {
            "id": dataset_id,
            "identifier": dataset_id,
            "portal_slug": slug,
            "name": schema.get("name", dataset_id),
            "description": schema.get("description", ""),
            "doi": doi,
            "url": landing_page,
            "landing_page": landing_page,
            "portal_url": landing_page,
            "date_created": schema.get("dateCreated", ""),
            "date_published": schema.get("datePublished", ""),
            "date_modified": schema.get("dateModified", ""),
            "keywords": schema.get("keywords", []),
            "page_citation": page_extras.get("page_citation", ""),
            "schema_org": schema,
        }

    def list_dataset_slugs(self) -> List[str]:
        """Discover dataset slugs from the portal index page."""
        try:
            html = self._fetch_text(self.INDEX_URL)
            slugs = self.DATASET_LINK_RE.findall(html)
            unique = []
            seen: set = set()
            for slug in slugs:
                slug = slug.strip().lower()
                if slug and slug not in seen:
                    seen.add(slug)
                    unique.append(slug)
            if unique:
                return unique
        except requests.exceptions.RequestException as exc:
            logger.warning("⚠️ BMB Human index scrape failed: %s", exc)
        return list(BMB_HUMAN_DEFAULT_SLUGS)

    def fetch_dataset(self, slug: str) -> Dict[str, Any]:
        slug = slug.strip().lower()
        portal_url = f"{BMB_HUMAN_BASE_URL}/dataset/{slug}/"
        meta_url = f"{portal_url}json/BMB_META.json"
        schema = self._fetch_json(meta_url)
        if not isinstance(schema, dict):
            raise ValueError(f"Unexpected metadata JSON for slug {slug!r}")
        page_html = self._fetch_text(portal_url)
        page_extras = self._parse_portal_page(page_html)
        return self._normalize_dataset(slug, schema, page_extras=page_extras)

    def get_datasets(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Fetch Brain/MINDS Beyond Human MRI datasets."""
        try:
            slugs = self.list_dataset_slugs()
            selected = slugs[offset : offset + limit]
            datasets: List[Dict[str, Any]] = []
            errors: List[str] = []
            for slug in selected:
                try:
                    datasets.append(self.fetch_dataset(slug))
                except Exception as exc:
                    logger.warning("⚠️ BMB Human skip %s: %s", slug, exc)
                    errors.append(f"{slug}: {exc}")

            if not datasets:
                return {
                    "status": "error",
                    "error": errors[0] if errors else "No BMB Human datasets fetched",
                    "api_version": "real",
                    "timestamp": datetime.now().isoformat(),
                }

            logger.info(
                "✅ Successfully fetched %s/%s BMB Human datasets",
                len(datasets),
                len(slugs),
            )
            return {
                "status": "success",
                "count": len(datasets),
                "total": len(slugs),
                "datasets": datasets,
                "api_version": "real",
                "endpoint": BMB_HUMAN_BASE_URL,
                "errors": errors,
                "timestamp": datetime.now().isoformat(),
            }
        except requests.exceptions.RequestException as exc:
            logger.error("❌ BMB Human API error: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "api_version": "real",
                "timestamp": datetime.now().isoformat(),
            }

    def get_dataset_details(self, dataset_id: str) -> Dict[str, Any]:
        """Get one dataset by portal slug or alternateName id."""
        normalized = dataset_id.strip().lower()
        for slug in self.list_dataset_slugs():
            try:
                dataset = self.fetch_dataset(slug)
            except Exception:
                continue
            if (
                slug == normalized
                or str(dataset.get("identifier", "")).lower() == normalized
                or str(dataset.get("id", "")).lower() == normalized
            ):
                return {
                    "status": "success",
                    "dataset": dataset,
                    "api_version": "real",
                    "timestamp": datetime.now().isoformat(),
                }
        return {
            "status": "error",
            "error": f"Dataset {dataset_id} not found on BMB Human portal",
            "api_version": "real",
            "timestamp": datetime.now().isoformat(),
        }
