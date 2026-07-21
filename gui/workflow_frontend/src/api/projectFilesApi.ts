import { createAuthHeaders } from "./authHeaders";

export const PROJECT_UPLOAD_MAX_BYTES = 50 * 1024 * 1024;

export type ProjectFileInfo = {
  filename: string;
  size_bytes: number;
  modified_at?: number;
};

export type ProjectFilesListResponse = {
  status: string;
  project_id: string;
  max_bytes: number;
  files: ProjectFileInfo[];
};

export type ProjectFileUploadResult = {
  filename: string;
  size_bytes: number;
  overwritten: boolean;
};

export type ProjectFilesUploadResponse = {
  status: string;
  max_bytes: number;
  uploaded: ProjectFileUploadResult[];
  errors: { filename: string | null; error: string }[];
};

async function authHeadersForMultipart(): Promise<Record<string, string>> {
  const headers = await createAuthHeaders();
  // Browser must set multipart boundary; JSON Content-Type breaks FormData.
  delete headers["Content-Type"];
  return headers;
}

export async function listProjectFiles(
  projectId: string
): Promise<ProjectFilesListResponse> {
  const headers = await createAuthHeaders();
  const response = await fetch(`/api/workflow/${projectId}/files/`, {
    method: "GET",
    credentials: "include",
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Failed to list files (${response.status})`);
  }
  return response.json();
}

export async function uploadProjectFiles(
  projectId: string,
  files: File[],
  options: { overwrite?: boolean } = {}
): Promise<ProjectFilesUploadResponse> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("file", file, file.name);
  }
  if (options.overwrite) {
    formData.append("overwrite", "true");
  }

  const headers = await authHeadersForMultipart();
  const response = await fetch(`/api/workflow/${projectId}/files/`, {
    method: "POST",
    credentials: "include",
    headers,
    body: formData,
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const firstError = body.errors?.[0]?.error;
    throw new Error(
      firstError || body.error || `Upload failed (${response.status})`
    );
  }
  return body as ProjectFilesUploadResponse;
}

export async function deleteProjectFile(
  projectId: string,
  filename: string
): Promise<void> {
  const headers = await createAuthHeaders();
  const url = `/api/workflow/${projectId}/files/?filename=${encodeURIComponent(
    filename
  )}`;
  const response = await fetch(url, {
    method: "DELETE",
    credentials: "include",
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Delete failed (${response.status})`);
  }
}

export function formatBytes(sizeInBytes: number): string {
  if (sizeInBytes < 1024) return `${sizeInBytes} B`;
  if (sizeInBytes < 1024 * 1024) {
    return `${(sizeInBytes / 1024).toFixed(1)} KB`;
  }
  return `${(sizeInBytes / (1024 * 1024)).toFixed(1)} MB`;
}
