import { createAuthHeaders } from "./authHeaders";
import { API_BASE_URL } from "../config/urls";

export type CatalogErrorCode =
  | "catalog_unconfigured"
  | "catalog_unavailable"
  | "catalog_auth"
  | "catalog_not_found"
  | string;

export interface CatalogErrorBody {
  status: "error";
  code: CatalogErrorCode;
  error: string;
}

export interface CatalogSyncStatus {
  last_sync?: string | null;
  status?: string | null;
  datasets_count?: number;
  error_message?: string | null;
}

export interface CatalogStatistics {
  total_datasets: number;
  source_counts: Record<string, number>;
  sync_status?: Record<string, CatalogSyncStatus>;
  timestamp?: string;
}

export interface CatalogHit {
  dataset_doi: string | null;
  dataset_id: string;
  description: string | null;
  is_draft: boolean;
  name: string;
  primary_paper_title: string | null;
  primary_paper_url: string | null;
  source: string;
  source_display: string | null;
  synced_at: string | null;
  table: string | null;
}

export interface CatalogSearchResponse {
  status: string;
  mode?: string;
  query?: string;
  source?: string | null;
  results: CatalogHit[];
  count: number;
  timestamp?: string;
}

export interface DatasetRow {
  dataset_id: string;
  name?: string | null;
  description?: string | null;
  is_draft?: boolean;
  source: string;
  source_display?: string | null;
  dataset_doi?: string | null;
  primary_paper_title?: string | null;
  primary_paper_url?: string | null;
  synced_at?: string | null;
  [key: string]: unknown;
}

export interface CatalogDatasetsResponse {
  count: number;
  datasets: DatasetRow[];
  timestamp?: string;
}

export interface CatalogLookupResponse {
  record: Record<string, unknown>;
  requested_id?: string;
  normalized_id?: string;
  [key: string]: unknown;
}

export class CatalogApiError extends Error {
  status: number;
  code: CatalogErrorCode;

  constructor(status: number, code: CatalogErrorCode, message: string) {
    super(message);
    this.name = "CatalogApiError";
    this.status = status;
    this.code = code;
  }
}

export function sourceDisplayName(
  source: string,
  source_display?: string | null
): string {
  if (source === "aws") return "SRPBS_TS";
  if (source_display) return source_display;
  return source;
}

export function dandiUrl(id: string): string {
  return `https://dandiarchive.org/dandiset/${id}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function asString(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return fallback;
}

function asOptionalString(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const text = asString(value);
  return text || null;
}

function asBoolean(value: unknown): boolean {
  return value === true || value === "true";
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

export function normalizeHit(raw: unknown): CatalogHit {
  const row = asRecord(raw);
  const datasetId = asString(row.dataset_id);
  return {
    dataset_doi: asOptionalString(row.dataset_doi),
    dataset_id: datasetId,
    description: asOptionalString(row.description),
    is_draft: asBoolean(row.is_draft),
    name: asOptionalString(row.name) || datasetId,
    primary_paper_title: asOptionalString(row.primary_paper_title),
    primary_paper_url: asOptionalString(row.primary_paper_url),
    source: asString(row.source),
    source_display: asOptionalString(row.source_display),
    synced_at: asOptionalString(row.synced_at),
    table: asOptionalString(row.table),
  };
}

/** Normalize browse (`datasets[]`) or search (`results[]`) into CatalogHit[]. */
export function toCatalogHits(data: unknown): CatalogHit[] {
  const body = asRecord(data);
  if (Array.isArray(body.results)) {
    return body.results.map(normalizeHit);
  }
  if (Array.isArray(body.datasets)) {
    return body.datasets.map(normalizeHit);
  }
  return [];
}

export async function parseError(res: Response): Promise<void> {
  if (res.ok) return;

  let code: CatalogErrorCode = "catalog_error";
  let message = `Catalog request failed (${res.status})`;
  try {
    const body = (await res.json()) as Partial<CatalogErrorBody>;
    if (typeof body.code === "string" && body.code) {
      code = body.code;
    }
    if (typeof body.error === "string" && body.error) {
      message = body.error;
    }
  } catch {
    // keep defaults when the body is not JSON
  }
  throw new CatalogApiError(res.status, code, message);
}

async function catalogFetch(
  path: string,
  init: RequestInit = {}
): Promise<unknown> {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { ...headers, ...(init.headers as Record<string, string> | undefined) },
  });
  await parseError(res);
  return res.json();
}

export async function fetchCatalogStatistics(): Promise<CatalogStatistics> {
  const data = asRecord(await catalogFetch("/catalog/statistics/"));
  const sourceCountsRaw = asRecord(data.source_counts);
  const source_counts: Record<string, number> = {};
  for (const [key, value] of Object.entries(sourceCountsRaw)) {
    source_counts[key] = asNumber(value);
  }

  let sync_status: Record<string, CatalogSyncStatus> | undefined;
  if (data.sync_status && typeof data.sync_status === "object") {
    sync_status = {};
    for (const [key, value] of Object.entries(asRecord(data.sync_status))) {
      const row = asRecord(value);
      sync_status[key] = {
        last_sync: asOptionalString(row.last_sync),
        status: asOptionalString(row.status),
        datasets_count:
          row.datasets_count === undefined
            ? undefined
            : asNumber(row.datasets_count),
        error_message: asOptionalString(row.error_message),
      };
    }
  }

  return {
    total_datasets: asNumber(data.total_datasets),
    source_counts,
    sync_status,
    timestamp: asOptionalString(data.timestamp) ?? undefined,
  };
}

export async function searchCatalog(options: {
  query: string;
  source?: string;
  limit?: number;
}): Promise<CatalogSearchResponse> {
  const body: { query: string; source?: string; limit?: number } = {
    query: options.query,
  };
  if (options.source) body.source = options.source;
  if (options.limit !== undefined) body.limit = options.limit;

  const data = asRecord(
    await catalogFetch("/catalog/search/", {
      method: "POST",
      body: JSON.stringify(body),
    })
  );
  const results = toCatalogHits(data);
  return {
    status: asString(data.status, "ok"),
    mode: asOptionalString(data.mode) ?? undefined,
    query: asOptionalString(data.query) ?? undefined,
    source: asOptionalString(data.source),
    results,
    count: asNumber(data.count, results.length),
    timestamp: asOptionalString(data.timestamp) ?? undefined,
  };
}

export async function lookupCatalog(options: {
  id: string;
  source?: string;
}): Promise<CatalogLookupResponse> {
  const params = new URLSearchParams({ id: options.id });
  if (options.source) params.set("source", options.source);
  const data = asRecord(
    await catalogFetch(`/catalog/lookup/?${params.toString()}`)
  );
  return {
    ...data,
    record: asRecord(data.record),
    requested_id: asOptionalString(data.requested_id) ?? undefined,
    normalized_id: asOptionalString(data.normalized_id) ?? undefined,
  };
}

export async function listCatalogDatasets(options?: {
  source?: string;
  limit?: number;
}): Promise<CatalogDatasetsResponse> {
  const params = new URLSearchParams();
  if (options?.source) params.set("source", options.source);
  if (options?.limit !== undefined) params.set("limit", String(options.limit));
  const qs = params.toString();
  const data = asRecord(
    await catalogFetch(`/catalog/datasets/${qs ? `?${qs}` : ""}`)
  );
  const datasets = Array.isArray(data.datasets)
    ? (data.datasets as DatasetRow[])
    : [];
  return {
    count: asNumber(data.count, datasets.length),
    datasets,
    timestamp: asOptionalString(data.timestamp) ?? undefined,
  };
}
