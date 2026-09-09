import { SchemaFields } from "../views/home/type";
import { useState, useEffect, useCallback } from "react";
import { createAuthHeaders } from "../api/authHeaders";
import { useAuth } from "../auth/authContext";

// Backend response type definition
interface UploadedNodesResponse {
  categories: any;
  nodes: BackendNodeType[];
  total_files: number;
  total_nodes: number;
}

interface BackendNodeType {
  id: string;
  type: string;
  label: string;
  description: string;
  category: string;
  file_id: string;
  class_name: string;
  file_name: string;
  schema: SchemaFields;
  color: string;
}

// interface SchemaField {
//   title: string;
//   type: string;
//   description: string;
//   port_direction: "input" | "output" | "parameter";
//   default_value?: string;
// }

interface UseUploadedNodesReturn {
  data: UploadedNodesResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

interface UseUploadedNodesOptions {
  enabled?: boolean;
}

/**
 * A custom hook to get the list of uploaded nodes
 */
export const useUploadedNodes = (
  options?: UseUploadedNodesOptions
): UseUploadedNodesReturn => {
  const enabled = options?.enabled ?? true;
  const { user, loading: authLoading } = useAuth();
  const [data, setData] = useState<UploadedNodesResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(enabled);
  const [error, setError] = useState<string | null>(null);

  const fetchUploadedNodes = useCallback(async () => {
    if (!enabled || authLoading) {
      return;
    }

    if (!user) {
      setData(null);
      setError(null);
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const headers = await createAuthHeaders();
      if (!headers.Authorization) {
        throw new Error("Authentication token is not available");
      }

      const response = await fetch("/api/box/uploaded-nodes/", {
        credentials: "include",
        headers,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: UploadedNodesResponse = await response.json();

      setData(result);
    } catch (err) {
      console.error("Failed to fetch uploaded nodes:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch nodes");
      setData(null);
    } finally {
      setIsLoading(false);
    }
  }, [authLoading, user, enabled]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    void fetchUploadedNodes();
  }, [fetchUploadedNodes, enabled]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchUploadedNodes,
  };
};



/**
 * Python custom hook to get file list
 */
interface PythonFile {
  id: string;
  name: string;
  description: string;
  file: string;
  uploaded_by: string | null;
  uploaded_by_name: string | null;
  file_size: number;
  is_analyzed: boolean;
  analysis_error: string | null;
  node_classes_count: number;
  created_at: string;
  updated_at: string;
}

interface UsePythonFilesReturn {
  files: PythonFile[];
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  deleteFile: (id: string) => Promise<boolean>;
}

export const usePythonFiles = (params?: {
  name?: string;
  analyzed_only?: boolean;
}): UsePythonFilesReturn => {
  const { user, loading: authLoading } = useAuth();
  const [files, setFiles] = useState<PythonFile[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPythonFiles = useCallback(async () => {
    if (authLoading) {
      return;
    }

    if (!user) {
      setFiles([]);
      setError(null);
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Build query parameters
      const searchParams = new URLSearchParams();
      if (params?.name) {
        searchParams.append("name", params.name);
      }
      if (params?.analyzed_only) {
        searchParams.append("analyzed_only", "true");
      }

      const url = `/api/box/files/${
        searchParams.toString() ? `?${searchParams.toString()}` : ""
      }`;
      const headers = await createAuthHeaders();
      if (!headers.Authorization) {
        throw new Error("Authentication token is not available");
      }

      const response = await fetch(url, {
        credentials: "include",
        headers,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: PythonFile[] = await response.json();
      setFiles(result);
    } catch (err) {
      console.error("Failed to fetch Python files:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch files");
      setFiles([]);
    } finally {
      setIsLoading(false);
    }
  }, [params?.name, params?.analyzed_only, authLoading, user]);

  const deleteFile = useCallback(async (id: string): Promise<boolean> => {
    try {
      const headers = await createAuthHeaders();
      if (!headers.Authorization) {
        throw new Error("Authentication token is not available");
      }

      const response = await fetch(`/api/box/files/${id}/`, {
        method: "DELETE",
        credentials: "include",
        headers,
      });

      if (!response.ok) {
        throw new Error(`Failed to delete file: ${response.status}`);
      }

      // Remove from file list
      setFiles((prev) => prev.filter((file) => file.id !== id));
      return true;
    } catch (err) {
      console.error("Failed to delete file:", err);
      setError(err instanceof Error ? err.message : "Failed to delete file");
      return false;
    }
  }, []);

  useEffect(() => {
    void fetchPythonFiles();
  }, [fetchPythonFiles]);

  return {
    files,
    isLoading,
    error,
    refetch: fetchPythonFiles,
    deleteFile,
  };
};
