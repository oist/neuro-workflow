import { Node, Edge } from "@xyflow/react";

export interface InputField {
  type: string;
  description?: string;
  required?: boolean;
  default_value?: any;
  constraints?: any;
}

export interface OutputField {
  type: string;
  description?: string;
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
  widget_type?: string;
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
  file_name: string;
  label: string;
  schema: SchemaFields;
  nodeType?: string;
  operation?: string;
  // ノード固有のパラメーター値（スキーマのdefault_valueを上書き）
  nodeParameters?: {
    [key: string]: any;
  };
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface FlowData {
  nodes: Node[];
  edges: Edge[];
}
