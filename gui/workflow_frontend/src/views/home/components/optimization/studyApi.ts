import { createAuthHeaders } from "../../../../api/authHeaders";
import { StudyObjective } from "../../type";

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

/** Replace the study's objectives through the endpoint that owns them. Every
 *  entry is validated (an objective measures an output port); a general node
 *  save keeps the stored study, so this is the only way to change it. Returns
 *  the list as stored; throws with the server's message on failure. */
export async function putStudyObjectives(
  workflowId: string,
  objectives: StudyObjective[]
): Promise<StudyObjective[]> {
  const headers = await createAuthHeaders();
  const res = await fetch(`/api/workflow/${workflowId}/study/objectives/`, {
    method: "PUT",
    headers,
    body: JSON.stringify({ objectives }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error ? String(body.error) : `HTTP ${res.status}`);
  return body.objectives as StudyObjective[];
}
