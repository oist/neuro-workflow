import { Node, Edge } from "@xyflow/react";

export interface InputField {
  type: string;
  description?: string;
  required?: boolean;
  default_value?: any;
  constraints?: any;
  optional?: boolean;
  fan_in?: boolean;
}

export interface OutputField {
  type: string;
  description?: string;
  optional?: boolean;
}

export interface ParameterField {
  type?: string;
  description?: string;
  default_value?: any;
  constraints?: {
    min?: number;
    max?: number;
    options?: any[];
    [key: string]: any;
  };
  optional?: boolean;
  widget_type?: string;
  // Optimization metadata
  optimizable?: boolean;
  optimization_range?: [number, number] | number[];
  is_objective?: boolean;
  objective_range?: [number, number] | number[];
  secret?: boolean;
}

export interface Method {
  description?: string;
  inputs: string[];
  outputs: string[];
}

export interface SchemaFields {
  inputs: {
    [key: string]: InputField;
  };
  outputs: {
    [key: string]: OutputField;
  };
  parameters: {
    [key: string]: ParameterField;
  };
  methods: {
    [key: string]: Method;
  };
}

export interface CalculationNodeData {
  [key: string]: unknown;
  file_name: string;
  label: string;
  instanceName: string;
  schema: SchemaFields;
  nodeType?: string;
  operation?: string;
  // Node-specific parameter values (overrides the default_value in the schema)
  nodeParameters?: {
    [key: string]: any;
  };
  isParamExpand?: boolean;
  color: string;
}

export type Visibility = "private" | "public";

export type HpcTarget = "" | "riken" | "fugaku";

export interface ProjectOwner {
  id: number;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
}

export interface Contributor {
  name: string;
  affiliation?: string;
  orcid?: string;
  researchmap?: string;
  role?: string;
}

export interface ProjectLink {
  label: string;
  url: string;
}

export interface AttributionDraft {
  doi: string;
  data_source: string;
  license: string;
  funding: string;
  contact_email: string;
  links: ProjectLink[];
  contributors: Contributor[];
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  workflow_context?: Record<string, any>;
  visibility: Visibility;
  reference?: string;
  hpc_target?: HpcTarget;
  doi?: string;
  data_source?: string;
  license?: string;
  funding?: string;
  contact_email?: string;
  links?: ProjectLink[];
  contributors?: Contributor[];
  owner?: ProjectOwner;
  is_owned_by_me: boolean;
  can_edit: boolean;
  can_delete: boolean;
  can_change_visibility: boolean;
  created_at: string;
  updated_at: string;
}

export interface FlowData {
  nodes: Node[];
  edges: Edge[];
}
