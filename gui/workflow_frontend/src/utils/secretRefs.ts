export const SECRET_REF_KEY = '__nw_secret';

export const BLOCKED_CONTEXT_TOKENS = [
  'password',
  'secret',
  'token',
  'api_key',
  'apikey',
  'aspera_pass',
  'authorization',
];

export type SecretRef = {
  [SECRET_REF_KEY]: { id?: string; name?: string };
};

export function isSecretRef(value: unknown): value is SecretRef {
  return (
    typeof value === 'object' &&
    value !== null &&
    SECRET_REF_KEY in value &&
    typeof (value as SecretRef)[SECRET_REF_KEY] === 'object'
  );
}

export function secretRefName(value: unknown): string {
  if (!isSecretRef(value)) return '';
  return value[SECRET_REF_KEY]?.name || '';
}

export function makeSecretRef(id: string, name: string): SecretRef {
  return { [SECRET_REF_KEY]: { id, name } };
}

export function blockedContextKeys(context: Record<string, unknown> | null | undefined): string[] {
  if (!context || typeof context !== 'object') return [];
  return Object.keys(context).filter((key) => {
    const lowered = key.toLowerCase().replace(/-/g, '_');
    return BLOCKED_CONTEXT_TOKENS.some((token) => lowered.includes(token));
  });
}

function redactNodeData(data: Record<string, unknown> | undefined): Record<string, unknown> | undefined {
  if (!data || typeof data !== 'object') return data;
  const out = structuredClone(data);
  const schema = (out.schema || {}) as Record<string, unknown>;
  const params = (schema.parameters || {}) as Record<string, Record<string, unknown>>;
  for (const param of Object.values(params)) {
    if (!param || typeof param !== 'object') continue;
    if (param.secret || isSecretRef(param.default_value)) {
      const name = secretRefName(param.default_value);
      if (isSecretRef(param.default_value) || param.default_value) {
        param.default_value = makeSecretRef(
          isSecretRef(param.default_value) ? String(param.default_value[SECRET_REF_KEY]?.id || '') : '',
          name,
        );
      }
    }
  }
  const mods = out.parameter_modifications as Record<string, Record<string, unknown>> | undefined;
  if (mods) {
    for (const info of Object.values(mods)) {
      if (!info || typeof info !== 'object') continue;
      for (const field of ['original_value', 'current_value'] as const) {
        if (isSecretRef(info[field])) {
          info[field] = makeSecretRef(
            String((info[field] as SecretRef)[SECRET_REF_KEY]?.id || ''),
            secretRefName(info[field]),
          );
        } else if (typeof info[field] === 'string' && info[field]) {
          const paramsMap = params;
          const match = Object.values(paramsMap).find((p) => p.secret);
          if (match) info[field] = makeSecretRef('', secretRefName(match.default_value) || 'REDACTED');
        }
      }
      const fieldMods = info.field_modifications as Record<string, unknown> | undefined;
      if (fieldMods && typeof fieldMods === 'object') {
        for (const [fk, fv] of Object.entries(fieldMods)) {
          if (isSecretRef(fv)) {
            fieldMods[fk] = makeSecretRef(
              String(fv[SECRET_REF_KEY]?.id || ''),
              secretRefName(fv),
            );
          } else if (typeof fv === 'string' && fv) {
            const match = Object.values(params).find((p) => p.secret);
            if (match) fieldMods[fk] = makeSecretRef('', secretRefName(match.default_value) || 'REDACTED');
          }
        }
      }
    }
  }
  out.schema = { ...schema, parameters: params };
  return out;
}

export function stripSecretValuesFromExport(exportData: {
  project?: unknown;
  flow?: { nodes?: Array<{ data?: Record<string, unknown> }> };
}): typeof exportData {
  const cloned = structuredClone(exportData);
  const nodes = cloned.flow?.nodes;
  if (Array.isArray(nodes)) {
    for (const node of nodes) {
      if (node && typeof node === 'object' && node.data) {
        node.data = redactNodeData(node.data) as Record<string, unknown>;
      }
    }
  }
  return cloned;
}
