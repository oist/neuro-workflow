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

from tvb.simulator import coupling


class NW_TVB_Coupling(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_coupling",
        stage="connectivity",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Builds any TVB long-range coupling function (Linear, Scaling, Difference, Sigmoidal, "
            "SigmoidalJansenRit, HyperbolicTangent, Kuramoto, PreSigmoidal) with a global "
            "coupling strength 'a' and optional extra parameters."
        ),
        parameters={
            "coupling_type": ParameterDefinition(
                default_value="Scaling",
                description=(
                    "How activity from connected regions j is combined into the input of region "
                    "i (summed over j, weighted by the structural connectivity w_ij). Options: "
                    "'Scaling' - a*sum(w_ij*x_j); for Generic2dOscillator, MontbrioPazoRoxin. "
                    "'Linear' - a*sum(w_ij*x_j) + b; for ReducedWongWangExcInh, WilsonCowan. "
                    "'Difference' - a*sum(w_ij*(x_j - x_i)), diffusive; for Epileptor, SupHopf. "
                    "'SigmoidalJansenRit' - sigmoid of the source's y1-y2 (firing rate); "
                    "required by JansenRit. 'Sigmoidal' - sigmoid of the summed input. "
                    "'HyperbolicTangent' - tanh-shaped input. 'Kuramoto' - a*sum(w_ij*sin(x_j - "
                    "x_i)) for phase oscillators. 'PreSigmoidal' - sigmoid applied before "
                    "summation (thalamocortical models). The model preset's recommended choice "
                    "is in tvb_model_info['recommended']['coupling']. Example: 'Scaling'."
                ),
                constraints={
                    "allowed_values": [
                        "Scaling",
                        "Linear",
                        "Difference",
                        "Sigmoidal",
                        "SigmoidalJansenRit",
                        "HyperbolicTangent",
                        "Kuramoto",
                        "PreSigmoidal",
                    ]
                },
            ),
            "a": ParameterDefinition(
                default_value=0.0075,
                description=(
                    "Global coupling strength G: how much the structural network drives each "
                    "region compared to its own local dynamics. G=0 disconnects the regions; "
                    "larger G synchronises them more. It is the main parameter to sweep or "
                    "optimise against empirical FC. Starting points with SC weights normalised to "
                    "max=1: Generic2dOscillator 0.0075, MontbrioPazoRoxin 0.1, "
                    "ReducedWongWangExcInh 0.02, JansenRit 10, SupHopf 0.1, WilsonCowan 0.1, "
                    "Epileptor 1.0. Ignored by PreSigmoidal (set 'G' in coupling_params)."
                ),
                constraints={"min": 0.0, "max": 1000.0},
                optimizable=True,
                optimization_range=[0.001, 1.0],
            ),
            "coupling_params": ParameterDefinition(
                default_value={},
                description=(
                    "Other parameters of the chosen coupling, besides 'a' (unknown names raise "
                    "an error that lists the valid ones). Examples: Linear {'b': 0.0} (constant "
                    "offset); Sigmoidal {'cmin': -1, 'cmax': 1, 'midpoint': 0, 'sigma': 230}; "
                    "SigmoidalJansenRit {'cmin': 0, 'cmax': 0.005, 'midpoint': 6, 'r': 0.56}; "
                    "HyperbolicTangent {'b': 1, 'midpoint': 0, 'sigma': 1}; PreSigmoidal "
                    "{'G': 60, 'P': 1, 'Q': 1, 'theta': 0.5}. Empty {} keeps TVB defaults."
                ),
            ),
        },
        inputs={},
        outputs={
            "tvb_coupling": PortDefinition(
                type=PortType.OBJECT,
                description="Configured TVB coupling function object, ready for NW_TVB_Simulator.",
            ),
        },
        methods={
            "build_coupling": MethodDefinition(
                description="Instantiate the selected TVB coupling class with 'a' and coupling_params.",
                inputs=[],
                outputs=["tvb_coupling"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build_coupling", self.build_coupling, method_key="build_coupling")

    def build_coupling(self) -> Dict[str, Any]:
        coupling_type = self._parameters["coupling_type"]
        cls = getattr(coupling, coupling_type)
        allowed = set(cls.declarative_attrs) - {"gid", "title", "tags"}

        params = dict(self._parameters["coupling_params"] or {})
        if "a" in allowed:
            params.setdefault("a", self._parameters["a"])

        unknown = set(params) - allowed
        if unknown:
            raise ValueError(
                f"coupling_params {sorted(unknown)} not valid for {coupling_type}; "
                f"valid: {sorted(allowed)}"
            )

        kwargs = {}
        for k, v in params.items():
            kwargs[k] = v if isinstance(v, (bool, str)) else np.atleast_1d(np.asarray(v, dtype=float))
        con_coupling = cls(**kwargs)

        print(f"[{self.name}] {coupling_type}: "
              + ", ".join(f"{k}={np.asarray(v).tolist()}" for k, v in kwargs.items()))
        return {"tvb_coupling": con_coupling}
