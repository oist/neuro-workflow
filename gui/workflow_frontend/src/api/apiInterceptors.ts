import { reAuthBus } from "../auth/reAuthBus";
import { createAuthHeaders } from "./authHeaders";
import { isDebugMode } from "./config";

export const onRequest = async (
  url: string,
  requestConfig: RequestInit
): Promise<RequestInit> => {
  const freshHeaders = await createAuthHeaders();
  requestConfig.headers = {
    ...requestConfig.headers,
    ...freshHeaders,
  };

  if (isDebugMode()) {
    console.log(`API Request: ${requestConfig.method || "GET"} ${url}`);
    console.log("Headers:", requestConfig.headers);
  }

  return requestConfig;
};

export const onResponse = async (response: Response): Promise<Response> => {
  if (isDebugMode()) {
    console.log(`API Response: ${response.status} ${response.url}`);
  }

  if (response.status === 401) {
    console.warn("Unauthorized - token may be expired");
    reAuthBus.emit("api-401");
  }

  if (response.status === 403) {
    console.warn("Forbidden - insufficient permissions");
  }

  if (response.status >= 500) {
    console.error("Server Error:", response.status);
    if (isDebugMode()) {
      console.error("Response details:", {
        status: response.status,
        statusText: response.statusText,
        url: response.url,
      });
    }
  }

  return response;
};

export const onError = (error: Error): never => {
  if (isDebugMode()) {
    console.error("API Error:", error);
  }

  if (error.name === "TypeError" && error.message.includes("fetch")) {
    console.error("Network error - server may be down");
    const networkError = new Error(
      "Network connection failed. Please check your internet connection and try again."
    );
    networkError.name = "NetworkError";
    throw networkError;
  }

  if (error.message.includes("timeout")) {
    const timeoutError = new Error("Request timed out. Please try again.");
    timeoutError.name = "TimeoutError";
    throw timeoutError;
  }

  throw error;
};

export const parseResponse = async (response: Response): Promise<any> => {
  const contentType = response.headers.get("content-type");
  try {
    if (contentType?.includes("application/json")) {
      return await response.json();
    } else if (contentType?.includes("text/")) {
      return await response.text();
    } else {
      return await response.blob();
    }
  } catch (error) {
    console.error("Error parsing response:", error);
    throw new Error("Failed to parse response data");
  }
};
