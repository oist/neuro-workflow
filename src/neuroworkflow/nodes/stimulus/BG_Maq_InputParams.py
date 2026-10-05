from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


class BG_Maq_InputParams(Node):
    """Background (resting-state) inputs of the macaque BG model.

    The three afferent populations CSN (cortico-striatal), PTN (pyramidal tract)
    and CMPf (thalamic centromedian/parafascicular) are layers of parrot neurons,
    each driven by independent Poisson trains. In the original script their rate
    is the lower bound of bgParams['normalrate'][<input>].
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_input_params",
        stage="stimulus",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — top_BG_nest3/bgParams.py + ini_all.py",
        description=(
            "Poisson background rates and population sizes of the CSN, PTN and CMPf input layers "
            "that drive the BG at rest. Defaults: CSN 2 Hz, PTN 15 Hz, CMPf 4 Hz, 36000 neurons each."
        ),
        parameters={
            "CSN_rate_Hz": ParameterDefinition(
                default_value=2.0,
                description="Rate of the cortico-striatal neurons (normalrate CSN min; physiological range 2–19.7 Hz).",
                constraints={"min": 0.0}, unit="Hz",
                optimizable=True, optimization_range=[0.0, 19.7],
            ),
            "PTN_rate_Hz": ParameterDefinition(
                default_value=15.0,
                description="Rate of the pyramidal tract neurons (normalrate PTN min; physiological range 15–46.3 Hz).",
                constraints={"min": 0.0}, unit="Hz",
                optimizable=True, optimization_range=[0.0, 46.3],
            ),
            "CMPf_rate_Hz": ParameterDefinition(
                default_value=4.0,
                description="Rate of the thalamic CM/Pf neurons (normalrate CMPf min; physiological range 4–34 Hz).",
                constraints={"min": 0.0}, unit="Hz",
                optimizable=True, optimization_range=[0.0, 34.0],
            ),
            "nb_CSN": ParameterDefinition(
                default_value=36000, description="nbCSN — number of CSN parrot neurons.", constraints={"min": 1},
            ),
            "nb_PTN": ParameterDefinition(
                default_value=36000, description="nbPTN — number of PTN parrot neurons.", constraints={"min": 1},
            ),
            "nb_CMPf": ParameterDefinition(
                default_value=36000, description="nbCMPf — number of CMPf parrot neurons.", constraints={"min": 1},
            ),
        },
        inputs={},
        outputs={
            "input_params": PortDefinition(
                type=PortType.DICT,
                description="Dict with input_rates {CSN, PTN, CMPf} in Hz and nbCSN/nbPTN/nbCMPf.",
            ),
        },
        methods={
            "build_params": MethodDefinition(
                description="Assemble the input-layer settings.",
                inputs=[],
                outputs=["input_params"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build_params", self.build_params, method_key="build_params")

    def build_params(self) -> Dict[str, Any]:
        p = self._parameters
        input_params = {
            "input_rates": {
                "CSN": float(p["CSN_rate_Hz"]),
                "PTN": float(p["PTN_rate_Hz"]),
                "CMPf": float(p["CMPf_rate_Hz"]),
            },
            "nbCSN": float(p["nb_CSN"]),
            "nbPTN": float(p["nb_PTN"]),
            "nbCMPf": float(p["nb_CMPf"]),
        }
        print(f"[{self.name}] input rates (Hz): {input_params['input_rates']}")
        return {"input_params": input_params}
