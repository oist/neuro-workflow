import json
import os
from typing import Dict, Any, List

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition,
)
from neuroworkflow.core.port import PortType


class NW_Connectivity(Node):
    """
    Defines all synaptic projections between BMTK populations.

    Receives any number of NW_Population outputs through a single fan-in port,
    then applies the connections matrix (list of source→target specs) using
    BMTK's add_edges() API.

    Node-level parameters (model_template, syn_weight, etc.) are defaults.
    Each entry in connections can override any of them individually.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_connectivity",
        stage="connectivity",
        tool="BMTK",
        model_source="https://alleninstitute.github.io/bmtk/",
        description=(
            "Receives all population builders via a fan-in port and wires directed "
            "projections according to the connections list. Each connection spec can "
            "override the node-level synapse defaults (model, weight, delay, etc.), "
            "allowing mixed synapse types in a single connectivity node."
        ),
        parameters={
            "connections": ParameterDefinition(
                default_value=[
                    {"source": "popA", "target": "popB"},
                ],
                description=(
                    "List of directed projections. Each entry must have 'source' and 'target' "
                    "(pop_name strings matching NW_Population pop_name values). "
                    "Every other field is optional and falls back to the node-level default. "
                    "Overridable per connection: connection_rule, model_template, dynamics_params, "
                    "dynamics_params_dict, syn_weight, delay, allow_autapses, allow_multapses."
                    "\n\nExample — heterogeneous connectome with per-connection overrides:"
                    "\n  {"
                    "\n    \"source\": \"exc\", \"target\": \"inh\","
                    "\n    \"connection_rule\": \"lambda src, tgt: 1 if np.random.rand() < 0.1 else 0\","
                    "\n    \"syn_weight\": 8.0"
                    "\n  },"
                    "\n  {"
                    "\n    \"source\": \"inh\", \"target\": \"exc\","
                    "\n    \"connection_rule\": \"lambda src, tgt: 1 if np.random.rand() < 0.2 else 0\","
                    "\n    \"model_template\": \"tsodyks_synapse\","
                    "\n    \"dynamics_params\": \"inh_tsodyks.json\","
                    "\n    \"dynamics_params_dict\": {\"U\": 0.5, \"tau_rec\": 800.0, \"tau_fac\": 0.0},"
                    "\n    \"syn_weight\": 4.0"
                    "\n  },"
                    "\n  {"
                    "\n    \"source\": \"exc\", \"target\": \"exc\","
                    "\n    \"connection_rule\": \"lambda src, tgt: 1 if src['node_id'] != tgt['node_id'] else 0\","
                    "\n    \"allow_autapses\": false"
                    "\n  }"
                ),
            ),
            "connection_rule": ParameterDefinition(
                default_value=1,
                description=(
                    "Accepts an integer OR a Python lambda (callable). "
                    "Do NOT pass a float (e.g. 0.1) — BMTK silently produces 0 connections. "
                    "BMTK calls this once per source→target neuron pair; the return value is the synapse count. "
                    "The lambda receives src and tgt node objects with keys: "
                    "node_id (int), ei_type ('exc'/'inh'), pop_name (str). "
                    "Return 1 to form a synapse, 0 to skip."
                    "\n\nExamples:"
                    "\n  # All-to-all (1 synapse per pair):"
                    "\n  connection_rule = 1"
                    "\n"
                    "\n  # Fixed fanout (3 synapses from each source to randomly chosen targets):"
                    "\n  connection_rule = 3"
                    "\n"
                    "\n  # Sparse random, 10% probability (Erdos-Renyi):"
                    "\n  connection_rule = lambda src, tgt: 1 if np.random.rand() < 0.1 else 0"
                    "\n"
                    "\n  # All-to-all without self-connections:"
                    "\n  connection_rule = lambda src, tgt: 1 if src['node_id'] != tgt['node_id'] else 0"
                    "\n"
                    "\n  # Ring — each neuron connects only to its next neighbour (replace 100 with N):"
                    "\n  connection_rule = lambda src, tgt: 1 if (src['node_id'] + 1) % 100 == tgt['node_id'] else 0"
                    "\n"
                    "\n  # One-to-one (neuron i → neuron i, requires equal population sizes):"
                    "\n  connection_rule = lambda src, tgt: 1 if src['node_id'] == tgt['node_id'] else 0"
                    "\n"
                    "\n  # Local neighbourhood — within 2 neighbours on each side:"
                    "\n  connection_rule = lambda src, tgt: 1 if abs(src['node_id'] - tgt['node_id']) <= 2 else 0"
                    "\n"
                    "\n  # Distance-dependent Gaussian decay (sigma = 10 node-ids):"
                    "\n  connection_rule = lambda src, tgt: 1 if np.random.rand() < np.exp(-abs(src['node_id'] - tgt['node_id']) / 10.0) else 0"
                ),
            ),
            "model_template": ParameterDefinition(
                default_value="static_synapse",
                description=(
                    "NEST synapse model name. Can be overridden per connection. "
                    "Any model from nest.synapse_models is valid. "
                    "Representative examples: 'static_synapse' (fixed weight, default for PointNet), "
                    "'tsodyks_synapse' (short-term plasticity), "
                    "'stdp_synapse' (spike-timing dependent plasticity), "
                    "'bernoulli_synapse' (probabilistic release). "
                    "Note: 'Exp2Syn' is a BioNet/NEURON receptor — not a valid NEST synapse model."
                ),
            ),
            "dynamics_params": ParameterDefinition(
                default_value="conn_syn.json",
                description=(
                    "Synapse dynamics JSON filename in components/synaptic_models/. "
                    "Only written and referenced when dynamics_params_dict is non-empty. "
                    "Can be overridden per connection."
                ),
            ),
            "dynamics_params_dict": ParameterDefinition(
                default_value={},
                description=(
                    "Synapse model parameters written to components/synaptic_models/<dynamics_params>. "
                    "Empty dict ({}) = no file written, dynamics_params not passed to NEST (correct for static_synapse). "
                    "tsodyks_synapse example: {\"U\": 0.5, \"tau_rec\": 800.0, \"tau_fac\": 0.0}. "
                    "stdp_synapse example: {\"tau_plus\": 20.0, \"lambda\": 0.01, \"Wmax\": 1000.0}. "
                    "Can be overridden per connection."
                ),
            ),
            "delay": ParameterDefinition(
                default_value=2.0,
                description="Default synaptic delay in ms. Can be overridden per connection.",
                constraints={"min": 0.1},
            ),
            "syn_weight": ParameterDefinition(
                default_value=5.0,
                description=(
                    "Default synaptic weight. Can be overridden per connection. "
                    "Units depend on model: nS for conductance-based (Exp2Syn), "
                    "pA for current-based (static_synapse with iaf_psc_alpha). "
                    "Negative values = inhibitory for current-based models."
                ),
            ),
            "allow_autapses": ParameterDefinition(
                default_value=False,
                description="Default: prevent a neuron from synapsing onto itself. Can be overridden per connection.",
            ),
            "allow_multapses": ParameterDefinition(
                default_value=True,
                description="Default: allow multiple synapses between the same pair. Can be overridden per connection.",
            ),
        },
        inputs={
            "populations": PortDefinition(
                type=PortType.OBJECT,
                fan_in=True,
                description=(
                    "Fan-in port: connect any number of NW_Population outputs here. "
                    "Each carries {builder, pop_name, network_dir}. "
                    "Populations are identified by their pop_name parameter."
                ),
            ),
        },
        outputs={
            "network": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "Dict keyed by pop_name containing all population dicts with edges added. "
                    "Pass directly to NW_SimConfig.populations."
                ),
            ),
        },
        methods={
            "connect": MethodDefinition(
                description=(
                    "Build pop_name lookup from fan-in list, write synapse JSON files, "
                    "call source_builder.add_edges() for each connection spec, "
                    "merging per-connection overrides with node-level defaults."
                ),
                inputs=["populations"],
                outputs=["network"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("connect", self.connect, method_key="connect")

    def _write_syn_json(self, params_dict: dict, filename: str) -> None:
        base_dir = self._context.get("results_path", "results")
        syn_dir  = os.path.join(base_dir, "components", "synaptic_models")
        os.makedirs(syn_dir, exist_ok=True)
        float_params = {
            k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in params_dict.items()
        }
        with open(os.path.join(syn_dir, filename), "w") as f:
            json.dump(float_params, f, indent=2)

    @staticmethod
    def _coerce_connection_rule(value):
        """Return what BMTK's add_edges() expects: an int or a callable.

        connection_rule can arrive as:
          - an int (e.g. 1, 3)                              -> passed through
          - a real callable (notebook lambda)              -> passed through
          - a numeric string ("3")                          -> int
          - a lambda string ("lambda src, tgt: ...")        -> compiled to a callable

        The last case is the GUI path: a lambda cannot be JSON-encoded, and the
        code generator only un-quotes a lambda at the TOP LEVEL of configure(),
        not one nested inside the `connections` list. So a per-connection rule
        reaches here as a string. Only a single lambda expression is accepted —
        not an arbitrary statement or ``eval`` payload. numpy is available as
        ``np`` for rules that use np.random / np.exp etc.
        """
        if callable(value) or isinstance(value, (int, bool)):
            return value
        if isinstance(value, str):
            s = value.strip()
            if s.lstrip("+-").isdigit():
                return int(s)
            import ast
            import numpy as np
            try:
                tree = ast.parse(s, mode="eval")
            except SyntaxError as exc:
                raise ValueError(
                    f"connection_rule string is not valid Python: {value!r}"
                ) from exc
            if not isinstance(tree.body, ast.Lambda):
                raise ValueError(
                    "connection_rule strings must be a lambda expression "
                    f"(e.g. 'lambda src, tgt: 1'); got {value!r}"
                )
            safe_builtins = {
                "True": True,
                "False": False,
                "None": None,
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "int": int,
                "float": float,
                "len": len,
                "range": range,
            }
            fn = eval(
                compile(tree, "<connection_rule>", "eval"),
                {"__builtins__": safe_builtins, "np": np},
            )
            if not callable(fn):
                raise ValueError(
                    f"connection_rule string did not evaluate to a callable: {value!r}"
                )
            return fn
        raise ValueError(
            f"connection_rule must be an int, a callable, or a lambda string; "
            f"got {type(value).__name__}"
        )

    def connect(self, populations: List[Dict]) -> Dict[str, Any]:
        p = self._parameters

        # Fan-in delivers a list — convert to lookup by pop_name
        pop_lookup = {pop["pop_name"]: pop for pop in populations}

        # Node-level defaults
        defaults = {
            "connection_rule":      p["connection_rule"],
            "model_template":       str(p["model_template"]),
            "dynamics_params":      str(p["dynamics_params"]),
            "dynamics_params_dict": dict(p["dynamics_params_dict"]) if p["dynamics_params_dict"] else {},
            "delay":                float(p["delay"]),
            "syn_weight":           float(p["syn_weight"]),
            "allow_autapses":       bool(p["allow_autapses"]),
            "allow_multapses":      bool(p["allow_multapses"]),
        }

        # Write default synapse JSON (always — BMTK requires dynamics_params in every edge type)
        self._write_syn_json(defaults["dynamics_params_dict"], defaults["dynamics_params"])

        for conn_spec in list(p["connections"]):
            # Merge: connection spec overrides node defaults
            spec = {**defaults, **conn_spec}

            src_name = spec.pop("source")
            tgt_name = spec.pop("target")

            if src_name not in pop_lookup:
                raise ValueError(
                    f"Source population '{src_name}' not found. "
                    f"Available: {list(pop_lookup.keys())}"
                )
            if tgt_name not in pop_lookup:
                raise ValueError(
                    f"Target population '{tgt_name}' not found. "
                    f"Available: {list(pop_lookup.keys())}"
                )

            # Always write the synapse JSON — BMTK's PointNet requires dynamics_params
            # in every edge type row of the SONATA file, even for static_synapse (empty {}).
            per_conn_params = spec.pop("dynamics_params_dict", {})
            self._write_syn_json(per_conn_params, spec["dynamics_params"])

            src_builder = pop_lookup[src_name]["builder"]
            tgt_builder = pop_lookup[tgt_name]["builder"]

            src_builder.add_edges(
                source          = src_builder.nodes(),
                target          = tgt_builder.nodes(),
                connection_rule = self._coerce_connection_rule(spec["connection_rule"]),
                model_template  = str(spec["model_template"]),
                dynamics_params = str(spec["dynamics_params"]),
                delay           = float(spec["delay"]),
                syn_weight      = float(spec["syn_weight"]),
                allow_autapses  = bool(spec["allow_autapses"]),
                allow_multapses = bool(spec["allow_multapses"]),
            )

        # Structural parameters of this node, carried inside the payload that already
        # flows to NW_SimConfig — no new port, so existing workflows are untouched.
        # dynamics_params_dict is excluded: it is written to a JSON that edge_types.csv
        # references by name, so its values are read at simulation time.
        conn_signature = {k: v for k, v in p.items() if k != "dynamics_params_dict"}
        conn_signature["connections"] = [
            {k: v for k, v in spec.items() if k != "dynamics_params_dict"}
            for spec in p["connections"]
        ]
        for pop in pop_lookup.values():
            pop.setdefault("_signature", {})["connectivity"] = conn_signature

        return {"network": pop_lookup}
