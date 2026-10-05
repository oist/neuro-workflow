from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


_NUCLEI = ["MSN", "FSI", "STN", "GPe", "GPi"]


class BG_Maq_NeuronParams(Node):
    """Neuron model and population-size parameters of the macaque BG model.

    All five BG nuclei use NEST iaf_psc_alpha_multisynapse with three receptors
    (1 = AMPA, 2 = NMDA, 3 = GABA). Defaults are those of top_BG_nest3/bgParams.py
    (common_iaf, <nucleus>_iaf, Ie<nucleus>, nb<nucleus>).
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_neuron_params",
        stage="setup",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — top_BG_nest3/bgParams.py",
        description=(
            "Integrate-and-fire parameters (C_m, tau_m, V_th, DC drive I_e) for MSN, FSI, STN, GPe "
            "and GPi, the shared iaf settings and synaptic time constants, and the simulated "
            "population sizes (1:834 scale). MSN is split 50/50 into D1 and D2 at build time."
        ),
        parameters={
            # ── Shared iaf_psc_alpha_multisynapse settings (common_iaf) ──────────
            "E_L_mV": ParameterDefinition(
                default_value=0.0, description="common_iaf.E_L — resting potential (all nuclei).", unit="mV",
            ),
            "V_reset_mV": ParameterDefinition(
                default_value=0.0, description="common_iaf.V_reset — reset potential after a spike.", unit="mV",
            ),
            "V_min_mV": ParameterDefinition(
                default_value=-20.0, description="common_iaf.V_min — lower bound of the membrane potential.", unit="mV",
            ),
            "V_m_init_mV": ParameterDefinition(
                default_value=0.0, description="common_iaf.V_m — initial membrane potential.", unit="mV",
            ),
            "t_ref_ms": ParameterDefinition(
                default_value=2.0, description="common_iaf.t_ref — absolute refractory period.",
                constraints={"min": 0.0}, unit="ms",
            ),
            "tau_syn_AMPA_ms": ParameterDefinition(
                default_value=1.8393972058572117,
                description="common_iaf.tau_syn[0] — AMPA alpha-synapse time constant (receptor 1).",
                constraints={"min": 0.01}, unit="ms",
            ),
            "tau_syn_NMDA_ms": ParameterDefinition(
                default_value=36.787944117144235,
                description="common_iaf.tau_syn[1] — NMDA alpha-synapse time constant (receptor 2).",
                constraints={"min": 0.01}, unit="ms",
            ),
            "tau_syn_GABA_ms": ParameterDefinition(
                default_value=1.8393972058572117,
                description="common_iaf.tau_syn[2] — GABA alpha-synapse time constant (receptor 3).",
                constraints={"min": 0.01}, unit="ms",
            ),
            # ── MSN ─────────────────────────────────────────────────────────────
            "MSN_C_m_pF": ParameterDefinition(
                default_value=13.0, description="MSN_iaf.C_m — membrane capacitance of MSNs.",
                constraints={"min": 0.1}, unit="pF",
            ),
            "MSN_tau_m_ms": ParameterDefinition(
                default_value=13.0, description="MSN_iaf.tau_m — membrane time constant of MSNs.",
                constraints={"min": 0.1}, unit="ms",
            ),
            "MSN_V_th_mV": ParameterDefinition(
                default_value=30.0, description="MSN_iaf.V_th — spike threshold of MSNs.", unit="mV",
            ),
            "MSN_I_e_pA": ParameterDefinition(
                default_value=26.0,
                description="IeMSN — DC current into MSNs. Target resting rate 0.05–1 Hz.",
                unit="pA", optimizable=True, optimization_range=[20.0, 30.0],
            ),
            # ── FSI ─────────────────────────────────────────────────────────────
            "FSI_C_m_pF": ParameterDefinition(
                default_value=3.1, description="FSI_iaf.C_m — membrane capacitance of FSIs.",
                constraints={"min": 0.1}, unit="pF",
            ),
            "FSI_tau_m_ms": ParameterDefinition(
                default_value=3.1, description="FSI_iaf.tau_m — membrane time constant of FSIs.",
                constraints={"min": 0.1}, unit="ms",
            ),
            "FSI_V_th_mV": ParameterDefinition(
                default_value=16.0, description="FSI_iaf.V_th — spike threshold of FSIs.", unit="mV",
            ),
            "FSI_I_e_pA": ParameterDefinition(
                default_value=8.0,
                description="IeFSI — DC current into FSIs. Target resting rate 7.8–14 Hz.",
                unit="pA", optimizable=True, optimization_range=[0.0, 20.0],
            ),
            # ── STN ─────────────────────────────────────────────────────────────
            "STN_C_m_pF": ParameterDefinition(
                default_value=6.0, description="STN_iaf.C_m — membrane capacitance of STN neurons.",
                constraints={"min": 0.1}, unit="pF",
            ),
            "STN_tau_m_ms": ParameterDefinition(
                default_value=6.0, description="STN_iaf.tau_m — membrane time constant of STN neurons.",
                constraints={"min": 0.1}, unit="ms",
            ),
            "STN_V_th_mV": ParameterDefinition(
                default_value=26.0, description="STN_iaf.V_th — spike threshold of STN neurons.", unit="mV",
            ),
            "STN_I_e_pA": ParameterDefinition(
                default_value=9.0,
                description="IeSTN — DC current into STN. Target resting rate 15.2–22.8 Hz.",
                unit="pA", optimizable=True, optimization_range=[0.0, 20.0],
            ),
            # ── GPe ─────────────────────────────────────────────────────────────
            "GPe_C_m_pF": ParameterDefinition(
                default_value=14.0, description="GPe_iaf.C_m — membrane capacitance of GPe neurons.",
                constraints={"min": 0.1}, unit="pF",
            ),
            "GPe_tau_m_ms": ParameterDefinition(
                default_value=14.0, description="GPe_iaf.tau_m — membrane time constant of GPe neurons.",
                constraints={"min": 0.1}, unit="ms",
            ),
            "GPe_V_th_mV": ParameterDefinition(
                default_value=11.0, description="GPe_iaf.V_th — spike threshold of GPe neurons.", unit="mV",
            ),
            "GPe_I_e_pA": ParameterDefinition(
                default_value=11.0,
                description="IeGPe — DC current into GPe. Target resting rate 55.7–74.5 Hz.",
                unit="pA", optimizable=True, optimization_range=[0.0, 25.0],
            ),
            # ── GPi ─────────────────────────────────────────────────────────────
            "GPi_C_m_pF": ParameterDefinition(
                default_value=14.0, description="GPi_iaf.C_m — membrane capacitance of GPi neurons.",
                constraints={"min": 0.1}, unit="pF",
            ),
            "GPi_tau_m_ms": ParameterDefinition(
                default_value=14.0, description="GPi_iaf.tau_m — membrane time constant of GPi neurons.",
                constraints={"min": 0.1}, unit="ms",
            ),
            "GPi_V_th_mV": ParameterDefinition(
                default_value=6.0, description="GPi_iaf.V_th — spike threshold of GPi neurons.", unit="mV",
            ),
            "GPi_I_e_pA": ParameterDefinition(
                default_value=8.5,
                description="IeGPi — DC current into GPi. Target resting rate 59.1–79.5 Hz.",
                unit="pA", optimizable=True, optimization_range=[0.0, 20.0],
            ),
            # ── Simulated population sizes (1:834 scale) ────────────────────────
            "nb_MSN": ParameterDefinition(
                default_value=31728,
                description="nbMSN — total simulated MSNs (split 50/50 into MSN_d1 and MSN_d2).",
                constraints={"min": 2},
            ),
            "nb_FSI": ParameterDefinition(
                default_value=636, description="nbFSI — simulated FSIs.", constraints={"min": 1},
            ),
            "nb_STN": ParameterDefinition(
                default_value=96, description="nbSTN — simulated STN neurons.", constraints={"min": 1},
            ),
            "nb_GPe": ParameterDefinition(
                default_value=300, description="nbGPe — simulated GPe neurons.", constraints={"min": 1},
            ),
            "nb_GPi": ParameterDefinition(
                default_value=168, description="nbGPi — simulated GPi neurons.", constraints={"min": 1},
            ),
        },
        inputs={},
        outputs={
            "neuron_params": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict in bgParams layout: common_iaf, MSN_iaf/FSI_iaf/STN_iaf/GPe_iaf/GPi_iaf, "
                    "IeMSN/IeFSI/IeSTN/IeGPe/IeGPi and nbMSN/nbFSI/nbSTN/nbGPe/nbGPi."
                ),
            ),
        },
        methods={
            "build_params": MethodDefinition(
                description="Assemble the neuron/population part of bgParams from the node parameters.",
                inputs=[],
                outputs=["neuron_params"],
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
        neuron_params = {
            "common_iaf": {
                "E_L": float(p["E_L_mV"]),
                "I_e": 0.0,
                "V_m": float(p["V_m_init_mV"]),
                "V_min": float(p["V_min_mV"]),
                "V_reset": float(p["V_reset_mV"]),
                "V_th": 10.0,  # overridden by every <nucleus>_iaf.V_th
                "t_ref": float(p["t_ref_ms"]),
                "tau_syn": [float(p["tau_syn_AMPA_ms"]), float(p["tau_syn_NMDA_ms"]), float(p["tau_syn_GABA_ms"])],
            },
        }
        for nuc in _NUCLEI:
            neuron_params[nuc + "_iaf"] = {
                "C_m": float(p[nuc + "_C_m_pF"]),
                "tau_m": float(p[nuc + "_tau_m_ms"]),
                "V_th": float(p[nuc + "_V_th_mV"]),
            }
            neuron_params["Ie" + nuc] = float(p[nuc + "_I_e_pA"])
            neuron_params["nb" + nuc] = float(p["nb_" + nuc])

        print(f"[{self.name}] I_e (pA): " + ", ".join(f"{n}={neuron_params['Ie' + n]}" for n in _NUCLEI))
        return {"neuron_params": neuron_params}
