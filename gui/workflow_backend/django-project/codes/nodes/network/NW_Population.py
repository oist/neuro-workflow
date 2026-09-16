import json
import os
from typing import Dict, Any, Optional

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition,
)
from neuroworkflow.core.port import PortType


NEST_MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "iaf_psc_alpha": {
        "C_m": 250.0, "tau_m": 10.0, "t_ref": 2.0,
        "V_th": -55.0, "V_reset": -70.0, "E_L": -70.0,
    },
    "iaf_psc_exp": {
        "C_m": 250.0, "tau_m": 10.0, "t_ref": 2.0,
        "V_th": -55.0, "V_reset": -70.0, "E_L": -70.0,
        "tau_syn_ex": 2.0, "tau_syn_in": 2.0,
    },
    "iaf_cond_alpha": {
        "C_m": 250.0, "g_L": 16.7, "t_ref": 2.0,
        "V_th": -55.0, "V_reset": -70.0, "E_L": -70.0,
        "E_ex": 0.0, "E_in": -85.0,
        "tau_syn_ex": 0.2, "tau_syn_in": 2.0,
    },
    "izhikevich": {
        "a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0,
    },
    "aeif_cond_alpha": {
        "C_m": 281.0, "g_L": 30.0, "E_L": -70.6,
        "V_th": -50.4, "V_reset": -60.0, "t_ref": 0.0,
        "Delta_T": 2.0, "tau_w": 144.0, "a": 4.0, "b": 80.5,
        "E_ex": 0.0, "E_in": -80.0,
        "tau_syn_ex": 0.2, "tau_syn_in": 2.0,
    },
    "glif_cond": {
        "C_m": 250.0, "G": 25.0, "E_L": -70.0,
        "V_th": -55.0, "V_reset": -70.0, "t_ref": 2.0,
        "tau_syn": [2.0, 2.0], "E_rev": [0.0, -85.0],
    },
    "hh_psc_alpha": {
        "C_m": 100.0,
        "g_Na": 12000.0, "g_K": 3600.0, "g_L": 10.0,
        "E_Na": 50.0, "E_K": -77.0, "E_L": -54.402,
        "V_m": -65.0, "t_ref": 2.0,
    },
}


class NW_Population(Node):
    """
    Builds a BMTK neuron population (point or biophysical).

    Follows BMTK's own build sequence:
      NetworkBuilder → add_nodes() → pass builder forward.
    build() and save() are called downstream by NW_SimConfig (single population)
    or NW_Connectivity + NW_SimConfig (multi-population networks).

    All connections (including recurrent) are defined in NW_Connectivity.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_population",
        stage="population",
        tool="BMTK",
        model_source="https://alleninstitute.github.io/bmtk/",
        description=(
            "Creates a BMTK NetworkBuilder population and passes the live builder object "
            "to NW_Connectivity (for inter-population edges) or directly to NW_SimConfig. "
            "build() and save() are called by NW_SimConfig, following BMTK's own sequence."
        ),
        parameters={
            # --- Population identity ---
            "pop_name": ParameterDefinition(
                default_value="v1",
                description="SONATA population name; prefix for generated network files.",
            ),
            "N": ParameterDefinition(
                default_value=1,
                description="Number of neurons in this population.",
                constraints={"min": 1},
            ),
            "ei_type": ParameterDefinition(
                default_value="exc",
                description="Cell class: 'exc' or 'inh'. Metadata for filtering and analysis.",
            ),
            "location": ParameterDefinition(
                default_value="VISp",
                description="Brain area label. Metadata only.",
            ),
            "layer": ParameterDefinition(
                default_value="L4",
                description="Cortical layer label. Metadata only.",
            ),

            # --- Model type and template ---
            "model_type": ParameterDefinition(
                default_value="point_neuron",
                description=(
                    "BMTK model type: 'point_neuron' for PointNet/NEST, "
                    "'biophysical' for BioNet/NEURON."
                ),
            ),
            "model_template": ParameterDefinition(
                default_value="nest:iaf_psc_alpha",
                description=(
                    "Simulator model identifier. "
                    "Format: 'nest:<model_name>' for NEST, 'ctdb:Biophys1.hoc' for BioNet."
                ),
            ),

            # --- Neuron model parameters ---
            "nest_params": ParameterDefinition(
                default_value={
                    "C_m": 250.0, "tau_m": 10.0, "t_ref": 2.0,
                    "V_th": -55.0, "V_reset": -70.0, "E_L": -70.0,
                },
                description=(
                    "Dict of NEST model parameters written to dynamics JSON at runtime. "
                    "Ignored for biophysical models (use dynamics_params_file instead)."
                ),
            ),
            "dynamics_params_file": ParameterDefinition(
                default_value="",
                description=(
                    "Path or filename of the dynamics parameter JSON. "
                    "If a path (starts with '.' or '/'), the file is copied to "
                    "components/{point_neuron_models|biophysical_neuron_models}/. "
                    "If a bare filename, it must already exist in that folder. "
                    "If empty, JSON is generated from nest_params."
                ),
            ),

            # --- Biophysical-only ---
            "model_processing": ParameterDefinition(
                default_value="",
                description="Post-processing for biophysical models (e.g. 'aibs_perisomatic'). Empty for point neurons.",
            ),
            "morphology": ParameterDefinition(
                default_value="",
                description=(
                    "Path or filename to SWC morphology file. "
                    "If a path, copied to components/morphologies/. Required for biophysical only."
                ),
            ),
            "mechanisms_source": ParameterDefinition(
                default_value="",
                description=(
                    "Path to a directory containing .mod mechanism files. "
                    "Copied to components/mechanisms/modfiles/. "
                    "Required for biophysical models with custom ion channels."
                ),
            ),
            "hoc_template": ParameterDefinition(
                default_value="",
                description=(
                    "Path to a HOC template file. Copied to components/templates/. "
                    "Required for biophysical models (e.g. './Biophys1.hoc')."
                ),
            ),

        },
        inputs={
            "iclamp": PortDefinition(
                type=PortType.DICT,
                optional=True,
                description=(
                    "Current clamp dict from NW_IClamp (keys: amp, delay, duration). "
                    "Passed through to NW_SimConfig via the population output."
                ),
            ),
        },
        outputs={
            "population": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "Dict with: builder (live NetworkBuilder), pop_name (str), "
                    "network_dir (str), and optionally _current_clamp (dict). "
                    "Consumed by NW_Connectivity or NW_SimConfig."
                ),
            ),
        },
        methods={
            "build": MethodDefinition(
                description=(
                    "Create NetworkBuilder, add nodes, resolve dynamics params, "
                    "copy biophysical source files. "
                    "Does not call net.build() or net.save() — that is NW_SimConfig's responsibility."
                ),
                inputs=["iclamp"],
                outputs=["population"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build", self.build, method_key="build")

    @staticmethod
    def _is_source_path(s: str) -> bool:
        return s.startswith('.') or os.sep in s or '/' in s

    def _resolve_dynamics_params(self, components_dir: str) -> str:
        import shutil

        p = self._parameters
        pop_name    = str(p["pop_name"])
        provided    = str(p["dynamics_params_file"]).strip()
        user_params = {
            k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in (p["nest_params"] or {}).items()
        }

        subdir = "biophysical_neuron_models" if str(p["model_type"]) == "biophysical" else "point_neuron_models"
        models_dir = os.path.join(components_dir, subdir)
        os.makedirs(models_dir, exist_ok=True)

        if provided and self._is_source_path(provided):
            dest = os.path.join(models_dir, os.path.basename(provided))
            shutil.copy2(provided, dest)
            return os.path.basename(provided)

        if provided:
            if os.path.isfile(os.path.join(models_dir, provided)):
                return provided

        model_name  = str(p["model_template"]).split(":")[-1]
        base_params = NEST_MODEL_DEFAULTS.get(model_name, {})
        if not base_params and not user_params:
            raise ValueError(
                f"No parameters for model '{model_name}'. "
                f"Set nest_params or provide dynamics_params_file."
            )
        final_params = {**base_params, **user_params}
        filename = f"{pop_name}_params.json"
        with open(os.path.join(models_dir, filename), "w") as f:
            json.dump(final_params, f, indent=2)
        return filename

    def build(self, iclamp: Optional[Dict] = None) -> Dict[str, Any]:
        from bmtk.builder.networks import NetworkBuilder

        p              = self._parameters
        base_dir       = self._context.get("results_path", "results")
        network_dir    = os.path.join(base_dir, "network")
        components_dir = os.path.join(base_dir, "components")

        os.makedirs(network_dir, exist_ok=True)
        os.makedirs(components_dir, exist_ok=True)

        dynamics_params = self._resolve_dynamics_params(components_dir)

        node_kwargs: Dict[str, Any] = {
            "N":               int(p["N"]),
            "model_type":      str(p["model_type"]),
            "model_template":  str(p["model_template"]),
            "dynamics_params": dynamics_params,
            "ei_type":         str(p["ei_type"]),
            "location":        str(p["location"]),
            "layer":           str(p["layer"]),
        }
        if str(p["model_processing"]):
            node_kwargs["model_processing"] = str(p["model_processing"])

        morphology = str(p["morphology"]).strip()
        if morphology:
            if self._is_source_path(morphology):
                import shutil
                morphologies_dir = os.path.join(components_dir, "morphologies")
                os.makedirs(morphologies_dir, exist_ok=True)
                shutil.copy2(morphology, os.path.join(morphologies_dir, os.path.basename(morphology)))
                morphology = os.path.basename(morphology)
            node_kwargs["morphology"] = morphology

        mechanisms_source = str(p["mechanisms_source"]).strip()
        if mechanisms_source:
            import shutil
            modfiles_dir = os.path.join(components_dir, "mechanisms", "modfiles")
            os.makedirs(modfiles_dir, exist_ok=True)
            for f in os.listdir(mechanisms_source):
                if f.endswith(".mod"):
                    shutil.copy2(os.path.join(mechanisms_source, f), os.path.join(modfiles_dir, f))

        hoc_template = str(p["hoc_template"]).strip()
        if hoc_template:
            import shutil
            templates_dir = os.path.join(components_dir, "templates")
            os.makedirs(templates_dir, exist_ok=True)
            shutil.copy2(hoc_template, os.path.join(templates_dir, os.path.basename(hoc_template)))

        net = NetworkBuilder(str(p["pop_name"]))
        net.add_nodes(**node_kwargs)

        out: Dict[str, Any] = {
            "builder":     net,
            "pop_name":    str(p["pop_name"]),
            "network_dir": network_dir,
            # Parameters that end up inside the SONATA network files. Changing any of
            # them invalidates the built network, so NW_SimConfig hashes this to decide
            # whether to rebuild. nest_params is excluded: it is written to
            # <pop_name>_params.json, which node_types.csv references by name, so its
            # values are read at simulation time rather than baked into the h5.
            "_signature": {k: v for k, v in p.items() if k not in ("nest_params",)},
        }
        if iclamp:
            # Underscored: internal plumbing for NW_SimConfig, not a measurable
            # value. Without the prefix it shows up in discover_measurables() as a
            # candidate target, echoing back a parameter the user set.
            out["_current_clamp"] = iclamp

        return {"population": out}
