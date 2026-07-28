#!/usr/bin/env python3
"""Enrichment helpers for remote catalog dataset records.

Ported from the bm_mindsdb ``data_url_utils.py``. Operates in place on the
per-source dataset dicts returned by the clients in :mod:`.clients`, attaching:

- download URLs (``data_urls`` / ``data_url_summary``)
- DOI / related publications columns (``build_publication_columns``)

CBS enrichment makes extra network calls and takes a client instance; BMB Human
parses the already-present ``schema_org`` tree with no extra network.

Stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_DOI_URL_RE = re.compile(r"https?://(?:dx\.)?doi\.org/([^\s\])>\"']+)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\])>\"']+")


def build_data_url_summary(
    data_urls: List[Dict[str, Any]], total_count: Optional[int] = None
) -> Dict[str, Any]:
    """Build summary block for a data_urls list."""
    summary: Dict[str, Any] = {"count": len(data_urls)}
    if total_count is not None:
        summary["total_count"] = total_count
        if total_count > len(data_urls):
            summary["truncated"] = True
    return summary


def attach_data_urls(
    dataset: Dict[str, Any],
    data_urls: List[Dict[str, Any]],
    total_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Attach normalized data_urls and summary to a dataset dict."""
    dataset["data_urls"] = data_urls
    dataset["data_url_summary"] = build_data_url_summary(data_urls, total_count)
    return dataset


def normalize_dataset_doi(value: Any) -> Optional[str]:
    """Normalize DOI strings to bare DOI form (no URL prefix)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "doi.org/" in text.lower():
        text = text.lower().split("doi.org/", 1)[-1]
    return text.strip().strip("/") or None


def _doi_from_text(text: str) -> Optional[str]:
    match = _DOI_URL_RE.search(text or "")
    if not match:
        return None
    return normalize_dataset_doi(match.group(1))


def _publication_sort_key(publication: Dict[str, Any]) -> Tuple[int, str]:
    relation = str(publication.get("relation") or "")
    if "IsDescribedBy" in relation:
        priority = 0
    elif "Cites" in relation or "References" in relation:
        priority = 1
    else:
        priority = 2
    return priority, str(publication.get("title") or publication.get("url") or "")


def _append_publication(
    publications: List[Dict[str, Any]],
    *,
    title: str,
    url: str,
    relation: str,
    source: str,
    source_field: str,
    doi: Optional[str] = None,
) -> None:
    title = (title or "").strip()
    url = (url or "").strip().rstrip(".,)")
    if not url and not title:
        return
    if not title:
        title = url
    entry = {
        "title": title,
        "url": url or None,
        "doi": normalize_dataset_doi(doi) if doi else _doi_from_text(url or title),
        "relation": relation or None,
        "source": source,
        "source_field": source_field,
    }
    publications.append(entry)


def extract_schema_org_related_publications(
    dataset: Dict[str, Any], source: str
) -> List[Dict[str, Any]]:
    """Parse ``schema_org.citation`` into publication entries."""
    schema_org = dataset.get("schema_org")
    if not isinstance(schema_org, dict):
        schema_org = dataset
    publications: List[Dict[str, Any]] = []
    citation = str(schema_org.get("citation") or "").strip()
    if citation:
        urls = _URL_RE.findall(citation)
        title = citation.split("(")[0].strip() or str(schema_org.get("name") or "")
        if urls:
            for index, url in enumerate(urls):
                _append_publication(
                    publications,
                    title=title if index == 0 else url,
                    url=url,
                    relation="citation",
                    source=source,
                    source_field="schema_org.citation",
                )
        else:
            _append_publication(
                publications,
                title=title,
                url="",
                relation="citation",
                source=source,
                source_field="schema_org.citation",
            )
    return publications


def extract_cbs_related_publications(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    return []


def extract_bmb_human_related_publications(
    dataset: Dict[str, Any]
) -> List[Dict[str, Any]]:
    citation = str(dataset.get("page_citation") or "").strip()
    if not citation:
        schema_org = dataset.get("schema_org")
        if isinstance(schema_org, dict):
            citation = str(schema_org.get("citation") or "").strip()
    if not citation:
        return []
    wrapped = dict(dataset)
    if "schema_org" not in wrapped or not isinstance(wrapped.get("schema_org"), dict):
        wrapped["schema_org"] = {"citation": citation, "name": dataset.get("name", "")}
    else:
        wrapped["schema_org"] = {**wrapped["schema_org"], "citation": citation}
    return extract_schema_org_related_publications(wrapped, "bmb_human")


def extract_related_publications(
    source: str, dataset: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if source == "bmb_human":
        return extract_bmb_human_related_publications(dataset)
    if source == "cbs":
        return extract_cbs_related_publications(dataset)
    return []


def extract_dataset_doi(source: str, dataset: Dict[str, Any]) -> Optional[str]:
    if source in {"cbs", "bmb_human"}:
        return normalize_dataset_doi(dataset.get("doi"))
    return None


def primary_paper_fields(
    publications: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    if not publications:
        return None, None
    primary = publications[0]
    title = (primary.get("title") or "").strip() or None
    url = (primary.get("url") or "").strip() or None
    return title, url


def build_publication_columns(
    source: str, dataset: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    """Build publication column values from enriched dataset metadata."""
    related_publications = extract_related_publications(source, dataset)
    primary_title, primary_url = primary_paper_fields(related_publications)
    return {
        "related_publications": json.dumps(related_publications, ensure_ascii=False),
        "primary_paper_title": primary_title,
        "primary_paper_url": primary_url,
        "dataset_doi": extract_dataset_doi(source, dataset),
    }


def _schema_org_entry(
    node: Dict[str, Any], *, dataset_browse_url: str = ""
) -> Dict[str, Any]:
    encoding = node.get("encodingFormat", [])
    if isinstance(encoding, str):
        encoding = [encoding]

    label = node.get("name") or node.get("description") or node.get("url") or ""
    entry: Dict[str, Any] = {
        "url": node["contentUrl"],
        "label": str(label)[:200],
        "type": "download",
    }
    if dataset_browse_url:
        entry["browse_url"] = dataset_browse_url
    if encoding:
        entry["encoding_format"] = encoding
    if node.get("contentSize"):
        entry["content_size"] = node["contentSize"]
    if node.get("identifier") is not None:
        entry["file_id"] = node["identifier"]
    if node.get("additionalType"):
        entry["additional_type"] = node["additionalType"]
    return entry


def extract_schema_org_data_urls(
    schema_org: Dict[str, Any], *, dataset_browse_url: str = ""
) -> List[Dict[str, Any]]:
    """Extract download URLs from a schema.org distribution tree."""
    seen: set = set()
    data_urls: List[Dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            content_url = node.get("contentUrl")
            if isinstance(content_url, str) and content_url and content_url not in seen:
                seen.add(content_url)
                data_urls.append(
                    _schema_org_entry(node, dataset_browse_url=dataset_browse_url)
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema_org.get("distribution", []))
    return data_urls


def enrich_bmb_human_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Add landing_page and schema.org distribution URLs for a BMB Human dataset."""
    schema_org = dataset.get("schema_org")
    if not isinstance(schema_org, dict):
        schema_org = dataset
    landing_page = str(
        dataset.get("landing_page")
        or dataset.get("portal_url")
        or dataset.get("url")
        or ""
    ).strip()
    if landing_page:
        dataset["landing_page"] = landing_page
    urls = extract_schema_org_data_urls(schema_org, dataset_browse_url=landing_page)
    return attach_data_urls(dataset, urls)


NEURODATA_DOWNLOAD_BASE = "https://neurodata.riken.jp/api/"
NEURODATA_V3_BASE = "https://neurodata.riken.jp/api/v3"


def normalize_neurodata_download_url(download_url: str, file_id: str = "") -> str:
    """Convert neurodata relative download paths to absolute URLs."""
    if download_url.startswith("http"):
        return download_url
    if download_url.startswith("v3/"):
        return f"{NEURODATA_DOWNLOAD_BASE}{download_url}"
    if file_id:
        return f"{NEURODATA_V3_BASE}/files/{file_id}/download/"
    return download_url


def build_cbs_data_urls(
    files: List[Dict[str, Any]], *, landing_page: str = ""
) -> List[Dict[str, Any]]:
    """Convert neurodata API v3 file records to normalized data_urls entries."""
    data_urls: List[Dict[str, Any]] = []
    landing_page = (landing_page or "").strip()
    for file_info in files:
        file_id = file_info.get("id")
        download_url = normalize_neurodata_download_url(
            file_info.get("download_url", ""), str(file_id or "")
        )
        if not download_url:
            continue
        entry: Dict[str, Any] = {
            "url": download_url,
            "label": file_info.get("name", str(file_id)),
            "type": "download",
        }
        if landing_page:
            entry["browse_url"] = landing_page
        if file_id:
            entry["file_id"] = file_id
        if file_info.get("mime_type"):
            entry["mime_type"] = file_info["mime_type"]
        if file_info.get("size") is not None:
            entry["size"] = file_info["size"]
        data_urls.append(entry)
    return data_urls


def enrich_cbs_dataset(
    dataset: Dict[str, Any],
    cbs_client: Any,
    file_limit: int = 100,
    doi_catalog: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Enrich a CBS dataset with neurodata API v3 download URLs."""
    identifier = str(dataset.get("identifier", ""))
    doi = dataset.get("doi", "")
    folder_id = cbs_client.resolve_neurodata_folder_id(
        identifier, doi, catalog=doi_catalog
    )
    if not folder_id:
        return attach_data_urls(dataset, [])

    dataset["neurodata_folder_id"] = folder_id
    result = cbs_client.collect_folder_files(folder_id, file_limit=file_limit)
    if result.get("status") != "success":
        return attach_data_urls(dataset, [])

    landing_page = str(dataset.get("landing_page") or "").strip()
    if landing_page:
        dataset["landing_page"] = landing_page

    files = result.get("files", [])
    data_urls = build_cbs_data_urls(files, landing_page=landing_page)
    total_count = result.get("total_count", len(data_urls))
    if result.get("truncated") and total_count <= len(data_urls):
        total_count = len(data_urls) + 1
    return attach_data_urls(dataset, data_urls, total_count=total_count)
