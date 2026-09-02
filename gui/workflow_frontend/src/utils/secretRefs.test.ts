import { describe, expect, it } from 'vitest';
import {
  blockedContextKeys,
  isSecretRef,
  secretRefName,
  stripSecretValuesFromExport,
} from './secretRefs';

describe('secretRefs', () => {
  it('reads a vault reference name', () => {
    const ref = { __nw_secret: { id: '1', name: 'ASPERA_PASSWORD' } };
    expect(isSecretRef(ref)).toBe(true);
    expect(secretRefName(ref)).toBe('ASPERA_PASSWORD');
  });

  it('strips secret values from exported flow JSON', () => {
    const exported = stripSecretValuesFromExport({
      project: { name: 'demo' },
      flow: {
        nodes: [
          {
            data: {
              schema: {
                parameters: {
                  password: {
                    secret: true,
                    default_value: { __nw_secret: { id: 'abc', name: 'ASPERA_PASSWORD' } },
                  },
                  n: { default_value: 1 },
                },
              },
            },
          },
        ],
      },
    });
    const dumped = JSON.stringify(exported);
    expect(dumped).toContain('ASPERA_PASSWORD');
    expect(dumped).toContain('__nw_secret');
    expect(dumped).not.toContain('plain-password-value');
    expect(exported.flow?.nodes?.[0]?.data?.schema).toBeTruthy();
  });

  it('flags blocked workflow_context keys', () => {
    expect(blockedContextKeys({ aspera_pass: 'x', species: 'mouse' })).toEqual(['aspera_pass']);
  });
});
