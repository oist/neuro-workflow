from typing import Dict, Any

import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType

from tvb.simulator import models


# Curated TVB neural mass models. Each entry is a validated starting point:
#   params      - literature parameters passed to the TVB model constructor
#   voi         - default variables_of_interest (what the monitors record)
#   coupling    - recommended coupling (type + params) for NW_TVB_Coupling
#   dt, nsig    - recommended integration step (ms) and noise per state variable
#                 for NW_TVB_Integrator (nsig="auto")
#   state_variable_range (optional) - initial-condition ranges overriding TVB defaults
# Coupling strengths assume SC weights normalised to max=1 (tested on human connectivity_76);
# they are starting points - sweep 'a' for each connectome.
# To add a model: add an entry here AND its name to model_type allowed_values.
TVB_MODEL_PRESETS = {
    "Generic2dOscillator": {
        "params": {"a": 1.74},
        "voi": ["V"],
        "coupling": {"type": "Scaling", "params": {"a": 0.0075}},
        "dt": 0.1,
        "nsig": {"V": 0.001, "W": 0.0},
        "reference": "Sanz-Leon et al. 2015, Neuroimage",
        "regime": "Limit-cycle oscillator (a=1.74); V2 human/marmoset default.",
    },
    "MontbrioPazoRoxin": {
        "params": {"eta": -4.6, "Delta": 0.7, "J": 14.5, "tau": 1.0},
        "voi": ["r", "V"],
        # a=0.1 maximises up/down switching on max-normalised human SC (0.45 in the tutorial,
        # Allen mouse SC). Noise on r drives it below 0 and diverges in TVB 2.10: noise on V only.
        "coupling": {"type": "Scaling", "params": {"a": 0.1}},
        # 0.025 as in the tutorial's BOLD run: same switching statistics as 0.01, 2.6x faster.
        "dt": 0.025,
        "nsig": {"V": 0.02},
        "reference": "Montbrio, Pazo & Roxin 2015, PRX; Rabuffo et al. 2021, eNeuro",
        "regime": "Bistable up/down states; noise-driven bursts and rsFC (Rabuffo tutorial).",
    },
    "ReducedWongWangExcInh": {
        # G is fixed to 1 so the global coupling lives only in NW_TVB_Coupling 'a'.
        "params": {"G": 1.0},
        "voi": ["S_e", "S_i"],
        # No feedback-inhibition control in TVB: S_e rises with G (0.17 at G=0, 0.58 at G=0.05).
        "coupling": {"type": "Linear", "params": {"a": 0.02, "b": 0.0}},
        "dt": 0.1,
        "nsig": {"S_e": 5e-5, "S_i": 5e-5},
        "reference": "Deco et al. 2014, J Neurosci",
        "regime": "Dynamic mean field E/I (synaptic gating); rsFC/BOLD fitting via G.",
    },
    "JansenRit": {
        "params": {},
        "voi": ["y0", "y1", "y2", "y3"],
        "coupling": {"type": "SigmoidalJansenRit", "params": {"a": 10.0}},
        "dt": 0.0625,
        "nsig": {"y4": 1e-5},
        "reference": "Jansen & Rit 1995, Biol Cybern",
        "regime": "Cortical column, ~10 Hz alpha; EEG-like signal is y1 - y2.",
    },
    "SupHopf": {
        "params": {"a": -0.01, "omega": 0.0628},
        "voi": ["x", "y"],
        # TVB default initial range (+-5) is stiff for the cubic term and diverges at dt=0.1.
        "state_variable_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
        "coupling": {"type": "Difference", "params": {"a": 0.1}},
        "dt": 0.1,
        "nsig": {"x": 2e-4, "y": 2e-4},
        "reference": "Deco et al. 2017, Sci Rep",
        "regime": "Stuart-Landau near Hopf bifurcation (a<0, noise-driven), 10 Hz carrier.",
    },
    "WilsonCowan": {
        "params": {
            "c_ee": 16.0, "c_ei": 12.0, "c_ie": 15.0, "c_ii": 3.0,
            "tau_e": 8.0, "tau_i": 8.0, "a_e": 1.3, "b_e": 4.0,
            "a_i": 2.0, "b_i": 3.7, "P": 1.25, "Q": 0.0,
        },
        "voi": ["E", "I"],
        "coupling": {"type": "Linear", "params": {"a": 0.1, "b": 0.0}},
        "dt": 0.1,
        "nsig": {"E": 1e-5, "I": 1e-5},
        "reference": "Wilson & Cowan 1972, Biophys J",
        "regime": "E/I limit-cycle oscillations.",
    },
    "Epileptor": {
        "params": {"x0": -1.6},
        "voi": ["x2 - x1", "z"],
        "coupling": {"type": "Difference", "params": {"a": 1.0}},
        "dt": 0.05,
        "nsig": {"x1": 0.0, "y1": 0.0, "z": 0.0, "x2": 3e-4, "y2": 3e-4, "g": 0.0},
        "reference": "Jirsa et al. 2014, Brain",
        "regime": "Seizure dynamics; set epileptogenic zones via region_params on x0.",
    },
    "EpileptorRestingState": {
        "params": {"Ks": -1.0, "K_rs": 1.0, "tau": 1000.0, "r": 0.000015},
        "voi": ["x2 - x1", "z", "x_rs"],
        "coupling": {"type": "Difference", "params": {"a": 1.0}},
        "dt": 0.1,
        "nsig": {"x1": 0.0, "y1": 0.0, "z": 0.0, "x2": 0.00025, "y2": 0.00025,
                 "g": 0.0, "x_rs": 0.001, "y_rs": 0.0},
        "reference": "Courtiol et al. 2020, J Neurosci",
        "regime": "Epileptor + resting-state oscillator; x0 per region via region_params.",
    },
}


class NW_TVB_Model(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_model",
        stage="neuron",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Builds any curated TVB neural mass model from a preset (literature parameters, "
            "recommended coupling, dt and noise) with user overrides, per-region parameters, "
            "selectable recorded variables and initial-condition ranges."
        ),
        parameters={
            "model_type": ParameterDefinition(
                default_value="Generic2dOscillator",
                description=(
                    "Neural mass model that generates the activity of every brain region (the "
                    "local dynamics of each network node). Each choice loads a tested preset. "
                    "Options: 'Generic2dOscillator' - generic 2-variable oscillator (V, W), "
                    "~9 Hz limit cycle, good default for testing. 'MontbrioPazoRoxin' - exact "
                    "mean field of spiking QIF neurons (firing rate r, potential V); bistable "
                    "up/down states, used for resting-state FC and bursts (Rabuffo 2021). "
                    "'ReducedWongWangExcInh' - Deco 2014 dynamic mean field of excitatory/"
                    "inhibitory synaptic gating (S_e, S_i); standard for BOLD rsFC fitting. "
                    "'JansenRit' - cortical column of pyramidal/excitatory/inhibitory "
                    "populations (y0..y5); ~10 Hz alpha, EEG-like signal y1-y2. 'SupHopf' - "
                    "Stuart-Landau oscillator at a Hopf bifurcation (x, y); Deco 2017 whole-brain "
                    "model. 'WilsonCowan' - classic E/I rate model (E, I); gamma-range "
                    "oscillations. 'Epileptor' - 6-variable seizure model (Jirsa 2014). "
                    "'EpileptorRestingState' - Epileptor plus a resting-state oscillator "
                    "(8 variables, Courtiol 2020). Example: 'MontbrioPazoRoxin'."
                ),
                constraints={
                    "allowed_values": [
                        "Generic2dOscillator",
                        "MontbrioPazoRoxin",
                        "ReducedWongWangExcInh",
                        "JansenRit",
                        "SupHopf",
                        "WilsonCowan",
                        "Epileptor",
                        "EpileptorRestingState",
                    ]
                },
            ),
            "use_preset": ParameterDefinition(
                default_value=True,
                description=(
                    "True: start from the preset's literature parameters and initial ranges for "
                    "model_type, then apply model_params on top (recommended). False: start "
                    "from the raw TVB class defaults, which are not always in a useful regime "
                    "(e.g. SupHopf diverges with its default initial range at dt=0.1)."
                ),
            ),
            "model_params": ParameterDefinition(
                default_value={},
                description=(
                    "Model parameters to change, as {'name': value}. A number applies to all "
                    "regions; a list gives one value per region (its length must equal the "
                    "number of regions, and tvb_connectivity must be connected). Names are the "
                    "TVB parameter names of the chosen model. Examples: Generic2dOscillator "
                    "{'a': 1.74}; MontbrioPazoRoxin {'eta': -4.6, 'J': 14.5, 'Delta': 0.7}; "
                    "ReducedWongWangExcInh {'J_i': 1.0, 'w_p': 1.4}; JansenRit {'mu': 0.22}; "
                    "SupHopf {'a': -0.01} (a<0 damped, a>0 oscillating); Epileptor "
                    "{'x0': -1.6}. Empty {} keeps the preset values."
                ),
            ),
            "region_params": ParameterDefinition(
                default_value={},
                description=(
                    "Different parameter values for a few chosen regions, everything else at a "
                    "common value; typical for epilepsy (epileptogenic / propagation zones). "
                    "Format: {'param': {'all': value_elsewhere, 'regions': [index, ...], "
                    "'values': [value, ...]}}. Example: {'x0': {'all': -2.3, 'regions': [40, 47], "
                    "'values': [-1.4, -1.6]}} makes regions 40 and 47 epileptogenic. Requires "
                    "tvb_connectivity. Empty {} disables it."
                ),
            ),
            "variables_of_interest": ParameterDefinition(
                default_value=[],
                description=(
                    "Which model variables the monitors record (and in which order). Use the "
                    "model's state-variable names, or expressions the model offers. Examples: "
                    "MontbrioPazoRoxin ['r', 'V']; ReducedWongWangExcInh ['S_e']; JansenRit "
                    "['y0', 'y1', 'y2']; Epileptor ['x2 - x1', 'z']. Empty [] uses the preset "
                    "default (listed in tvb_model_info)."
                ),
            ),
            "state_variable_range": ParameterDefinition(
                default_value={},
                description=(
                    "Range from which the random initial state of each variable is drawn, "
                    "{'state_var': [low, high]}. Narrow ranges help stiff models start stably. "
                    "Example: SupHopf {'x': [-0.5, 0.5], 'y': [-0.5, 0.5]}; MontbrioPazoRoxin "
                    "{'r': [0.0, 0.5]} to start in the down state. Empty {} keeps preset/TVB ranges."
                ),
            ),
            "stimulus_variables": ParameterDefinition(
                default_value=[],
                description=(
                    "State variables that receive the external stimulus from NW_TVB_Stimulus. "
                    "Example: MontbrioPazoRoxin ['V'] (input current); Generic2dOscillator "
                    "['V']; JansenRit ['y1']. Empty [] uses the TVB default (the variables that "
                    "receive long-range coupling)."
                ),
            ),
        },
        inputs={
            "tvb_connectivity": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "TVB Connectivity object; required only when region_params or per-region "
                    "(list) model_params are used, to know the number of regions."
                ),
                optional=True,
            ),
        },
        outputs={
            "tvb_model": PortDefinition(
                type=PortType.OBJECT,
                description="Configured TVB neural mass model object, ready for NW_TVB_Simulator.",
            ),
            "tvb_model_info": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Model metadata: model_type, state_variables, variables_of_interest, cvar, "
                    "and preset recommendations (coupling, dt, nsig) used by NW_TVB_Integrator "
                    "(nsig='auto') and NW_TVB_Simulator (compatibility checks)."
                ),
            ),
        },
        methods={
            "build_model": MethodDefinition(
                description=(
                    "Merge preset and user parameters, instantiate the TVB model class, apply "
                    "per-region overrides, and export model metadata."
                ),
                inputs=["tvb_connectivity"],
                outputs=["tvb_model", "tvb_model_info"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build_model", self.build_model, method_key="build_model")

    def build_model(self, tvb_connectivity=None) -> Dict[str, Any]:
        model_type = self._parameters["model_type"]
        preset = TVB_MODEL_PRESETS.get(model_type, {})
        nregions = (
            int(tvb_connectivity.number_of_regions) if tvb_connectivity is not None else None
        )

        params = dict(preset.get("params", {})) if self._parameters["use_preset"] else {}
        params.update(self._parameters["model_params"] or {})

        model_class = self._resolve_model_class(model_type)
        kwargs = {k: self._to_array(k, v, nregions) for k, v in params.items()}

        voi = list(self._parameters["variables_of_interest"] or preset.get("voi", []))
        if voi:
            kwargs["variables_of_interest"] = tuple(voi)

        mod = model_class(**kwargs)

        svr = dict(preset.get("state_variable_range", {})) if self._parameters["use_preset"] else {}
        svr.update(self._parameters["state_variable_range"] or {})
        if svr:
            ranges = dict(mod.state_variable_range)
            unknown = set(svr) - set(ranges)
            if unknown:
                raise ValueError(
                    f"state_variable_range: unknown state variables {sorted(unknown)}; "
                    f"{model_type} has {list(ranges)}"
                )
            ranges.update({k: np.array(v, dtype=float) for k, v in svr.items()})
            # Final trait whose default dict is shared by all instances of the class:
            # store a private copy on this instance instead of mutating the shared one.
            mod.__dict__["state_variable_range"] = ranges

        region_params = self._parameters["region_params"] or {}
        if region_params:
            if nregions is None:
                raise ValueError("tvb_connectivity input is required when region_params is non-empty.")
            for param_name, config in region_params.items():
                arr = np.ones(nregions) * config["all"]
                for idx, val in zip(config.get("regions", []), config.get("values", [])):
                    arr[idx] = val
                setattr(mod, param_name, arr)

        stim_vars = self._parameters["stimulus_variables"] or []
        if stim_vars:
            svars = list(mod.state_variables)
            unknown = set(stim_vars) - set(svars)
            if unknown:
                raise ValueError(f"stimulus_variables {sorted(unknown)} not in state variables {svars}")
            mod.stvar = np.array([svars.index(v) for v in stim_vars], dtype=np.int32)

        mod.configure()

        info = {
            "model_type": model_type,
            "state_variables": list(mod.state_variables),
            "variables_of_interest": list(mod.variables_of_interest),
            "cvar": [int(c) for c in mod.cvar],
            "stimulus_variables": [mod.state_variables[int(i)] for i in mod.stvar],
            "parameters": {k: np.asarray(getattr(mod, k)).tolist() for k in params},
            "recommended": {
                "coupling": preset.get("coupling"),
                "dt": preset.get("dt"),
                "nsig": preset.get("nsig"),
            },
            "reference": preset.get("reference", ""),
            "regime": preset.get("regime", ""),
        }
        print(f"[{self.name}] {model_type}: state_variables={info['state_variables']}, "
              f"recording={info['variables_of_interest']}")
        return {"tvb_model": mod, "tvb_model_info": info}

    @staticmethod
    def _to_array(name: str, value, nregions):
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.size > 1 and nregions is not None and arr.size != nregions:
            raise ValueError(
                f"model_params['{name}'] has {arr.size} values but connectivity has {nregions} regions."
            )
        return arr

    @staticmethod
    def _resolve_model_class(model_type: str):
        cls = getattr(models, model_type, None)
        if cls is not None:
            return cls
        if model_type == "EpileptorRestingState":
            from tvb.simulator.models.epileptor_rs import EpileptorRestingState
            return EpileptorRestingState
        raise ValueError(f"Unknown TVB model type: '{model_type}'.")
