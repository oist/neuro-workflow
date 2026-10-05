from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


class BG_Maq_StriatumParams(Node):
    """Striatal pathway parameters of the macaque BG model.

    Controls how the MSN population is split into the direct (D1) and indirect (D2)
    pathways, the D1/D2 collateral asymmetries, and the dopamine-modulated STDP
    synapses used on cortex->MSN projections.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_striatum_params",
        stage="setup",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — top_BG_nest3/bgParams.py + ini_all.py",
        description=(
            "D1/D2 pathway overlap (lambda), MSN collateral structural asymmetries, strength "
            "asymmetry (kappa) and the cortico-striatal dopamine-STDP synapse settings."
        ),
        parameters={
            "overlap_d1d2": ParameterDefinition(
                default_value=0.1,
                description=(
                    "lambda. Fraction of MSN->GPi inputs coming from MSN-D2 and of MSN->GPe inputs coming "
                    "from MSN-D1. 0 = fully segregated direct/indirect pathways."
                ),
                constraints={"min": 0.0, "max": 1.0},
                optimizable=True, optimization_range=[0.0, 0.5],
            ),
            "asymmetry_1": ParameterDefinition(
                default_value=0.5,
                description=(
                    "Structural asymmetry: fraction of the MSN->MSN in-degree used for D1->D1, D2->D1 "
                    "and D2->D2 collaterals."
                ),
                constraints={"min": 0.0, "max": 1.0},
            ),
            "asymmetry_2": ParameterDefinition(
                default_value=0.1,
                description="Structural asymmetry: fraction of the MSN->MSN in-degree used for the sparse D1->D2 collaterals.",
                constraints={"min": 0.0, "max": 1.0},
                optimizable=True, optimization_range=[0.0, 0.5],
            ),
            "syn_asymm": ParameterDefinition(
                default_value=2.0,
                description=(
                    "kappa. Weight multiplier of MSN-D2->MSN collaterals (D2 PSPs 2–4x larger than D1, "
                    "Taverna et al. 2008)."
                ),
                constraints={"min": 0.0},
                optimizable=True, optimization_range=[1.0, 4.0],
            ),
            "plastic_syn": ParameterDefinition(
                default_value=True,
                description=(
                    "Use dopamine-STDP synapses (syn_d1 / syn_d2) for CSN->MSN and PTN->MSN. "
                    "Also switches MSN->GPe/GPi to the G_MSN_GPx gain. With no dopamine release (resting "
                    "state) the STDP weights stay constant, but cortico-striatal weights are scaled by plast_gain."
                ),
            ),
            "plast_gain": ParameterDefinition(
                default_value=0.65,
                description="Multiplier of CSN/PTN->MSN weights when plastic_syn is True.",
                constraints={"min": 0.0},
                optimizable=True, optimization_range=[0.3, 1.0],
            ),
            "stdp_d1": ParameterDefinition(
                default_value={
                    "A_plus": 0.013, "A_minus": 0.00325, "Wmax": 4.0,
                    "b": 0.0, "n": 0.0, "c": 0.0,
                    "tau_plus": 20.0, "tau_n": 100.0, "tau_c": 700.0,
                },
                description="Initial stdp_dopamine_synapse_lbl defaults of syn_d1 (cortex->MSN-D1).",
            ),
            "stdp_d2": ParameterDefinition(
                default_value={
                    "A_plus": 0.013, "A_minus": -0.013, "Wmax": 4.0,
                    "b": 0.0, "n": 0.0, "c": 0.0,
                    "tau_plus": 20.0, "tau_n": 100.0, "tau_c": 700.0,
                },
                description="Initial stdp_dopamine_synapse_lbl defaults of syn_d2 (cortex->MSN-D2).",
            ),
        },
        inputs={},
        outputs={
            "striatum_params": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict with overlap_d1d2, asymmetry_1, asymmetry_2, syn_asymm, plastic_syn, "
                    "plast_gain, stdp_d1, stdp_d2."
                ),
            ),
        },
        methods={
            "build_params": MethodDefinition(
                description="Assemble the striatal part of bgParams.",
                inputs=[],
                outputs=["striatum_params"],
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
        striatum_params = {
            "overlap_d1d2": float(p["overlap_d1d2"]),
            "asymmetry_1": float(p["asymmetry_1"]),
            "asymmetry_2": float(p["asymmetry_2"]),
            "syn_asymm": float(p["syn_asymm"]),
            "plastic_syn": bool(p["plastic_syn"]),
            "plast_gain": float(p["plast_gain"]),
            "stdp_d1": dict(p["stdp_d1"]),
            "stdp_d2": dict(p["stdp_d2"]),
        }
        print(f"[{self.name}] lambda={striatum_params['overlap_d1d2']}, kappa={striatum_params['syn_asymm']}, "
              f"plastic_syn={striatum_params['plastic_syn']}")
        return {"striatum_params": striatum_params}
