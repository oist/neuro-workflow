import { createAuthHeaders } from "../../../../api/authHeaders";

export type ParameterFieldName =
  | "default_value"
  | "optimizable"
  | "optimization_range"
  | "unit";

/** Write one field of one parameter of a canvas node, the same request the
 *  node modal makes. Throws with the server's message on failure. */
export async function putParameterField(
  workflowId: string,
  nodeId: string,
  parameterKey: string,
  parameterField: ParameterFieldName,
  parameterValue: unknown
): Promise<void> {
  const headers = await createAuthHeaders();
  const res = await fetch(`/api/workflow/${workflowId}/nodes/${nodeId}/parameters/`, {
    method: "PUT",
    headers,
    body: JSON.stringify({
      parameter_key: parameterKey,
      parameter_field: parameterField,
      parameter_value: parameterValue,
    }),
  });
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.error) message = String(body.error);
    } catch {
      // keep the status line
    }
    throw new Error(message);
  }
}
