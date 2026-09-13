from typing import Dict, Any

import pandas as pd

# BMTK 1.1.4 reads SONATA node-type tables via pandas. Under pandas >= 3.0 the
# default string dtype became StringDtype(na_value=nan), which BMTK then hands to
# np.empty() (sonata/group.py) and which NumPy cannot interpret as a dtype:
#   "Cannot interpret '<StringDtype(...)>' as a data type"
# Disabling the new string inference keeps those columns as object dtype, exactly
# as pre-3.0. set_option is process-global, so setting it once at import — before
# any BMTK SONATA read in workflow.execute() — covers the whole run.
pd.set_option("future.infer_string", False)

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition,
)
from neuroworkflow.core.port import PortType


class NW_SimConfig(Node):
    """
    BMTK simulation node — builds, configures, and runs.

    Receives the live NetworkBuilder OBJECT from NW_Population (or from the last
    NW_Connectivity in the chain). Follows BMTK's own run sequence:
      net.build() → net.save() → create_environment() → Config → Network → Sim → run()

    For multi-population networks: wire each population (or the last NW_Connectivity
    output for that population) into separate 'populations' and 'extra_populations'
    ports, or collect them in a list via a fan-in node.
    """

    #: Artifacts under results_path that a later run can reuse when nothing
    #: structural changed — read by the optimization engine, which copies them into
    #: each trial so this node's signature check can skip rebuilding. The SONATA
    #: network is expensive to build (connection rules are evaluated here) and
    #: identical whenever only simulation-time values move.
    REUSABLE_PATHS = ("network",)

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_sim_config",
        stage="simulation",
        tool="BMTK",
        model_source="https://alleninstitute.github.io/bmtk/",
        description=(
            "Receives BMTK NetworkBuilder population objects, calls build()+save() "
            "to write SONATA network files, generates the config JSON via create_environment(), "
            "then runs the simulation via PointNet (NEST) or BioNet (NEURON). "
            "Scales from single cell to large networks."
        ),
        parameters={
            "simulator": ParameterDefinition(
                default_value="pointnet",
                description="BMTK simulator backend: 'pointnet' (NEST) or 'bionet' (NEURON).",
            ),
            "config_file": ParameterDefinition(
                default_value="config.json",
                description="Filename for the generated SONATA config JSON (relative to results_path).",
            ),
            "tstop_ms": ParameterDefinition(
                default_value=3000.0,
                description="Simulation end time in milliseconds.",
                constraints={"min": 1.0},
            ),
            "dt_ms": ParameterDefinition(
                default_value=0.1,
                description="Simulation time step in milliseconds.",
                constraints={"min": 0.001},
            ),
            "reports": ParameterDefinition(
                default_value={
                    "v_report": {
                        "variable_name": "V_m",
                        "cells": "all",
                        "module": "membrane_report",
                        "sections": "soma",
                    }
                },
                description=(
                    "SONATA reports dict written to the config 'reports' section. "
                    "Each key is a report name; value is a dict of SONATA report fields. "
                    "Empty dict = spikes only, no membrane recording. "
                    "variable_name: 'V_m' for PointNet/NEST, 'v' for BioNet/NEURON. "
                    "cells: 'all', a population name ('v1'), a filter dict ({'ei_type': 'exc'}), or node id list ([0,1,2]). "
                    "sections: 'soma' (default) or 'all' (all compartments, BioNet only). "
                    "PointNet example: {'exc_Vm': {'variable_name': 'V_m', 'cells': {'ei_type': 'exc'}, 'module': 'membrane_report', 'sections': 'soma'}}. "
                    "BioNet example: {'v_report': {'variable_name': 'v', 'cells': 'all', 'module': 'membrane_report', 'sections': 'soma'}}."
                ),
            ),
            "compile_mechanisms": ParameterDefinition(
                default_value=False,
                description=(
                    "Compile NEURON .mod files before running. "
                    "Required for BioNet on first use; not needed for PointNet."
                ),
            ),
            "overwrite": ParameterDefinition(
                default_value=True,
                description=(
                    "Overwrite existing config files. SONATA network files follow "
                    "rebuild_network: with 'auto' (the default), an unchanged signature "
                    "reuses the network on disk, so a random connection_rule is frozen "
                    "after the first build. Use rebuild_network='always' to draw a "
                    "new realisation each run."
                ),
            ),
            "rebuild_network": ParameterDefinition(
                default_value="auto",
                description=(
                    "When to regenerate the SONATA network files. 'auto' rebuilds only when "
                    "a structural parameter changed (population size, neuron model, "
                    "connectivity, synaptic weights) or the files are missing — values that "
                    "are read at simulation time (nest_params, synapse dynamics_params_dict) "
                    "never trigger a rebuild. A random connection_rule is therefore frozen "
                    "after the first successful build. 'always' rebuilds every run. "
                    "'never' reuses whatever is on disk."
                ),
                constraints={"allowed_values": ["auto", "always", "never"]},
            ),
        },
        inputs={
            "populations": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "Single population from NW_Population, or network dict from NW_Connectivity. "
                    "Single pop keys: builder, pop_name, network_dir, optional _current_clamp. "
                    "Network dict (multi-pop): keyed by pop_name, each value has the same keys."
                ),
            ),
        },
        outputs={
            "results": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict with config_file (str), output_dir (str), simulator (str). "
                    "Consumed by NW_Analysis."
                ),
            ),
        },
        methods={
            "setup": MethodDefinition(
                description=(
                    "Call builder.build() + builder.save(), create directory structure, "
                    "generate SONATA config via create_environment(), inject reports."
                ),
                inputs=["populations"],
                outputs=["config_file", "output_dir", "simulator"],
            ),
            "run": MethodDefinition(
                description="Run the simulation using the generated config (PointNet or BioNet).",
                inputs=["config_file", "output_dir", "simulator"],
                outputs=["results"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("setup", self.setup, method_key="setup")
        self.add_process_step("run",   self.run,   method_key="run")

    @staticmethod
    def _fingerprint(value):
        """Text that changes whenever the value changes — including inside an array.

        repr() cannot be used for arrays: numpy prints only the first and last few
        elements of anything past its print threshold (1000 by default), so two arrays
        differing in the middle print identically. The signature would then match and
        rebuild_network='auto' would reuse a network built from the other array —
        silently, with the wrong connectivity. Hashing the bytes covers every element.
        Containers are walked, since an array reached through a dict or list has the
        same problem.
        """
        import hashlib

        if hasattr(value, "tobytes") and hasattr(value, "shape"):
            try:
                import numpy as np

                flat = np.ascontiguousarray(value)
                digest = hashlib.sha1(flat.tobytes()).hexdigest()
                return f"ndarray{flat.shape}{flat.dtype}:{digest}"
            except Exception:
                return repr(value)          # not an array after all
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda kv: repr(kv[0]))
            inner = ", ".join(f"{k!r}: {NW_SimConfig._fingerprint(v)}" for k, v in items)
            return "{" + inner + "}"
        if isinstance(value, (list, tuple)):
            inner = ", ".join(NW_SimConfig._fingerprint(v) for v in value)
            return f"[{inner}]" if isinstance(value, list) else f"({inner})"
        return repr(value)

    @staticmethod
    def _stable(value):
        """A representation that is identical run to run for unchanged inputs.

        connection_rule may be a callable. repr() of a function embeds its memory
        address, which differs on every run, so the signature would never match and
        the network would be rebuilt every time. Use the source text instead, plus any
        closure values (which the source does not show — changing `eps` in
        `lambda s, t: 1 if rand() < eps else 0` leaves the text identical), and fall
        back to the bytecode when source is unavailable.
        """
        if callable(value):
            import inspect
            if hasattr(value, "func") and hasattr(value, "args"):
                # functools.partial and similar wrappers have no __code__.
                return (
                    "partial:"
                    f"{NW_SimConfig._stable(value.func)}:"
                    f"{NW_SimConfig._stable(getattr(value, 'args', ()))}:"
                    f"{NW_SimConfig._stable(getattr(value, 'keywords', {}) or {})}"
                )
            code = getattr(value, "__code__", None)
            try:
                text = inspect.getsource(value).strip()
            except (OSError, TypeError):
                if code is not None:
                    text = code.co_code.hex() + repr(code.co_consts)
                else:
                    text = repr(value)
            # Values the rule reads but does not show in its source. A lambda written
            # in a notebook usually reads them as globals (`eps`), not as closure
            # cells, so both are captured — otherwise changing eps would leave the
            # signature identical and silently reuse a network built with the old value.
            referenced = []
            for cell in (getattr(value, "__closure__", None) or []):
                try:
                    referenced.append(NW_SimConfig._fingerprint(cell.cell_contents))
                except ValueError:
                    referenced.append("<empty>")
            if code is not None:
                globs = getattr(value, "__globals__", {}) or {}
                for name in sorted(code.co_names):
                    if name not in globs:
                        continue
                    referenced_value = globs[name]
                    if isinstance(
                        referenced_value, (int, float, str, bool, type(None), dict, list, tuple)
                    ) or hasattr(referenced_value, "tolist"):
                        referenced.append(
                            f"{name}={NW_SimConfig._fingerprint(referenced_value)}")
            return f"callable:{text}:{referenced}"
        if isinstance(value, dict):
            return {k: NW_SimConfig._stable(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [NW_SimConfig._stable(v) for v in value]
        return value

    def _network_signature(self, pop_list):
        """Hash of everything that ends up inside the SONATA network files."""
        import hashlib
        import json

        parts = {pop["pop_name"]: self._stable(pop.get("_signature", {}))
                 for pop in pop_list}
        blob = json.dumps(parts, sort_keys=True, default=repr)
        return hashlib.sha256(blob.encode()).hexdigest(), parts

    def _network_is_current(self, network_dir, pop_list, digest):
        """True when the network on disk was built from these same parameters."""
        import json
        import os

        path = os.path.join(network_dir, ".signature.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path) as fh:
                stored = json.load(fh)
        except (ValueError, OSError):
            return False
        if stored.get("hash") != digest:
            return False
        return all(
            os.path.exists(os.path.join(network_dir, f"{pop['pop_name']}_nodes.h5"))
            for pop in pop_list
        )

    def setup(self, populations: Dict) -> Dict[str, Any]:
        import json
        import os
        from bmtk.utils.create_environment import create_environment

        p        = self._parameters
        base_dir = self._context.get("results_path", "results")
        output_dir = os.path.join(base_dir, "output")

        # Single population (direct from NW_Population) or network dict (from NW_Connectivity)
        if "builder" in populations:
            pop_list    = [populations]
            network_dir = populations["network_dir"]
        else:
            pop_list    = list(populations.values())
            network_dir = pop_list[0]["network_dir"]

        for subdir in [
            "components/point_neuron_models",
            "components/biophysical_neuron_models",
            "components/morphologies",
            "components/mechanisms/modfiles",
            "components/synaptic_models",
            "components/templates",
            "inputs",
        ]:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

        mode = str(p["rebuild_network"]).lower()
        digest, parts = self._network_signature(pop_list)
        current = self._network_is_current(network_dir, pop_list, digest)

        if mode == "never":
            if current:
                print(f"[NW_SimConfig] network unchanged, reusing {network_dir}")
            else:
                print("[NW_SimConfig] rebuild_network='never': the network on disk does "
                      f"not match the current parameters, reusing it anyway: {network_dir}")
        elif mode == "always" or not current:
            for pop in pop_list:
                pop["builder"].build()
                pop["builder"].save(output_dir=network_dir)
            with open(os.path.join(network_dir, ".signature.json"), "w") as fh:
                json.dump({"hash": digest, "parameters": parts}, fh, indent=2, default=repr)
            print(f"[NW_SimConfig] network built in {network_dir}")
        else:
            print(f"[NW_SimConfig] network unchanged, reusing {network_dir}")

        kwargs: Dict[str, Any] = dict(
            base_dir=base_dir,
            config_file=str(p["config_file"]),
            network_dir=network_dir,
            tstop=float(p["tstop_ms"]),
            dt=float(p["dt_ms"]),
            overwrite=bool(p["overwrite"]),
            compile_mechanisms=bool(p["compile_mechanisms"]),
        )

        # current_clamp lives in the primary pop (single-pop case only)
        primary = pop_list[0]
        if primary.get("_current_clamp"):
            kwargs["current_clamp"] = primary["_current_clamp"]

        create_environment(str(p["simulator"]), **kwargs)

        reports = dict(p["reports"]) if p["reports"] else {}
        if reports:
            config_path = os.path.join(base_dir, str(p["config_file"]))
            with open(config_path) as f:
                config = json.load(f)
            config["reports"] = reports
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        return {
            "config_file": os.path.join(base_dir, str(p["config_file"])),
            "output_dir":  output_dir,
            "simulator":   str(p["simulator"]),
        }

    def run(self, config_file: str, output_dir: str, simulator: str) -> Dict[str, Any]:
        if simulator == "pointnet":
            from bmtk.simulator import pointnet
            conf = pointnet.Config.from_json(config_file)
            conf.build_env()
            net = pointnet.PointNetwork.from_config(conf)
            sim = pointnet.PointSimulator.from_config(conf, net)
            sim.run()

        elif simulator == "bionet":
            from bmtk.simulator import bionet
            conf = bionet.Config.from_json(config_file)
            conf.build_env()
            net = bionet.BioNetwork.from_config(conf)
            sim = bionet.BioSimulator.from_config(conf, net)
            sim.run()

        else:
            raise ValueError(f"Unknown simulator '{simulator}'. Use 'pointnet' or 'bionet'.")

        return {
            "results": {
                "config_file": config_file,
                "output_dir":  output_dir,
                "simulator":   simulator,
            }
        }
