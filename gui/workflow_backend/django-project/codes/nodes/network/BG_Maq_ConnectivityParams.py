from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


# The 24 projections of the model, "Source->Target" (MSN = both D1 and D2).
PROJECTIONS = [
    "CMPf->FSI", "CMPf->GPe", "CMPf->GPi", "CMPf->MSN", "CMPf->STN",
    "CSN->FSI", "CSN->MSN",
    "FSI->FSI", "FSI->MSN",
    "GPe->FSI", "GPe->GPe", "GPe->GPi", "GPe->MSN", "GPe->STN",
    "MSN->GPe", "MSN->GPi", "MSN->MSN",
    "PTN->FSI", "PTN->MSN", "PTN->STN",
    "STN->FSI", "STN->GPe", "STN->GPi", "STN->MSN",
]


class BG_Maq_ConnectivityParams(Node):
    """Connectivity parameters of the macaque BG model.

    Per-projection tables (keys "Source->Target") for in-degree, delay, projection
    fraction, dendritic contact distance, focused/diffuse type and redundancy, plus
    the global weight-computation constants. Defaults are top_BG_nest3/bgParams.py.

    Weights are not set directly: for each projection the model computes
        w = nu / inDegree * attenuation(distcontact) * wPSP[receptor] * gain
    with nu derived from alpha, ProjPercent and the real macaque neuron counts.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_connectivity_params",
        stage="setup",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — top_BG_nest3/bgParams.py + nest_routine.py",
        description=(
            "All 24 BG projections: in-degree (alpha), delays, ProjPercent, distcontact, "
            "focused/diffuse type, redundancy; pathway gains; PSP amplitudes; cable constants; "
            "spatial spreads; real macaque neuron counts used for weight normalisation."
        ),
        parameters={
            # ── Per-projection tables ───────────────────────────────────────────
            "alpha": ParameterDefinition(
                default_value={
                    "CMPf->FSI": 1053, "CMPf->GPe": 79, "CMPf->GPi": 131, "CMPf->MSN": 4965, "CMPf->STN": 76,
                    "CSN->FSI": 250, "CSN->MSN": 342,
                    "FSI->FSI": 116, "FSI->MSN": 4362,
                    "GPe->FSI": 353, "GPe->GPe": 38, "GPe->GPi": 16, "GPe->MSN": 0, "GPe->STN": 19,
                    "MSN->GPe": 171, "MSN->GPi": 210, "MSN->MSN": 210,
                    "PTN->FSI": 5, "PTN->MSN": 5, "PTN->STN": 259,
                    "STN->FSI": 91, "STN->GPe": 428, "STN->GPi": 233, "STN->MSN": 0,
                },
                description=(
                    "alpha — anatomical number of synapses per target neuron for each projection. "
                    "0 disables the projection."
                ),
            ),
            "delays_ms": ParameterDefinition(
                default_value={
                    "CMPf->FSI": 7.0, "CMPf->GPe": 7.0, "CMPf->GPi": 7.0, "CMPf->MSN": 7.0, "CMPf->STN": 7.0,
                    "CSN->FSI": 7.0, "CSN->MSN": 7.0,
                    "FSI->FSI": 1.0, "FSI->MSN": 1.0,
                    "GPe->FSI": 3.0, "GPe->GPe": 1.0, "GPe->GPi": 3.0, "GPe->MSN": 3.0, "GPe->STN": 10.0,
                    "MSN->GPe": 7.0, "MSN->GPi": 11.0, "MSN->MSN": 1.0,
                    "PTN->FSI": 3.0, "PTN->MSN": 3.0, "PTN->STN": 3.0,
                    "STN->FSI": 3.0, "STN->GPe": 3.0, "STN->GPi": 3.0, "STN->MSN": 3.0,
                },
                description="tau — axonal + synaptic delay of each projection (ms).",
                unit="ms",
            ),
            "proj_percent": ParameterDefinition(
                default_value={
                    "CMPf->FSI": 1.0, "CMPf->GPe": 1.0, "CMPf->GPi": 1.0, "CMPf->MSN": 1.0, "CMPf->STN": 1.0,
                    "CSN->FSI": 1.0, "CSN->MSN": 1.0,
                    "FSI->FSI": 1.0, "FSI->MSN": 1.0,
                    "GPe->FSI": 0.16, "GPe->GPe": 0.84, "GPe->GPi": 0.84, "GPe->MSN": 0.16, "GPe->STN": 1.0,
                    "MSN->GPe": 1.0, "MSN->GPi": 0.82, "MSN->MSN": 1.0,
                    "PTN->FSI": 1.0, "PTN->MSN": 1.0, "PTN->STN": 1.0,
                    "STN->FSI": 0.17, "STN->GPe": 0.83, "STN->GPi": 0.72, "STN->MSN": 0.17,
                },
                description="ProjPercent — fraction of source neurons that project to the target nucleus.",
            ),
            "dist_contact": ParameterDefinition(
                default_value={
                    "CMPf->FSI": 0.06, "CMPf->GPe": 0.0, "CMPf->GPi": 0.48, "CMPf->MSN": 0.27, "CMPf->STN": 0.46,
                    "CSN->FSI": 0.82, "CSN->MSN": 0.95,
                    "FSI->FSI": 0.16, "FSI->MSN": 0.19,
                    "GPe->FSI": 0.58, "GPe->GPe": 0.01, "GPe->GPi": 0.13, "GPe->MSN": 0.06, "GPe->STN": 0.58,
                    "MSN->GPe": 0.48, "MSN->GPi": 0.59, "MSN->MSN": 0.77,
                    "PTN->FSI": 0.7, "PTN->MSN": 0.98, "PTN->STN": 0.97,
                    "STN->FSI": 0.41, "STN->GPe": 0.3, "STN->GPi": 0.59, "STN->MSN": 0.16,
                },
                description=(
                    "distcontact — relative distance of the synaptic contact along the dendrite "
                    "(0 = soma, 1 = tip); sets the cable attenuation of the weight."
                ),
            ),
            "projection_type": ParameterDefinition(
                default_value={
                    "CMPf->FSI": "diffuse", "CMPf->GPe": "diffuse", "CMPf->GPi": "diffuse",
                    "CMPf->MSN": "diffuse", "CMPf->STN": "diffuse",
                    "CSN->FSI": "focused", "CSN->MSN": "focused",
                    "FSI->FSI": "diffuse", "FSI->MSN": "diffuse",
                    "GPe->FSI": "diffuse", "GPe->GPe": "diffuse", "GPe->GPi": "focused",
                    "GPe->MSN": "diffuse", "GPe->STN": "focused",
                    "MSN->GPe": "focused", "MSN->GPi": "focused", "MSN->MSN": "focused",
                    "PTN->FSI": "focused", "PTN->MSN": "focused", "PTN->STN": "focused",
                    "STN->FSI": "diffuse", "STN->GPe": "diffuse", "STN->GPi": "diffuse", "STN->MSN": "diffuse",
                },
                description=(
                    "cType — 'focused' (circular mask of radius spread_focused, channel preserving) or "
                    "'diffuse' (radius spread_diffuse, whole layer)."
                ),
            ),
            "redundancy": ParameterDefinition(
                default_value={
                    "CMPf->FSI": 3, "CMPf->GPe": 3, "CMPf->GPi": 3, "CMPf->MSN": 3, "CMPf->STN": 3,
                    "CSN->FSI": 3, "CSN->MSN": 3,
                    "FSI->FSI": 3, "FSI->MSN": 3,
                    "GPe->FSI": 3, "GPe->GPe": 3, "GPe->GPi": 3, "GPe->MSN": 3, "GPe->STN": 3,
                    "MSN->GPe": 3, "MSN->GPi": 3, "MSN->MSN": 3,
                    "PTN->FSI": 3, "PTN->MSN": 3, "PTN->STN": 3,
                    "STN->FSI": 3, "STN->GPe": 3, "STN->GPi": 3, "STN->MSN": 3,
                },
                description=(
                    "redundancy<Src><Tgt> — number of synaptic contacts each simulated axon makes on the "
                    "same target; interpreted according to redundancy_type."
                ),
            ),
            "redundancy_type": ParameterDefinition(
                default_value="outDegreeAbs",
                description=(
                    "RedundancyType. 'outDegreeAbs': inDegree = nu_max / redundancy. 'outDegreeCons': "
                    "inDegree = nu_min + (nu_max - nu_min) * redundancy. 'inDegreeAbs': inDegree = redundancy."
                ),
                constraints={"allowed_values": ["outDegreeAbs", "outDegreeCons", "inDegreeAbs"]},
            ),
            # ── Pathway gains ──────────────────────────────────────────────────
            "G_GPe_STN": ParameterDefinition(
                default_value=1.0, description="GGPe_STN — weight gain of GPe->STN.",
                constraints={"min": 0.0}, optimizable=True, optimization_range=[0.0, 3.0],
            ),
            "G_STN_GPi": ParameterDefinition(
                default_value=1.0, description="GSTN_GPi — weight gain of STN->GPi.",
                constraints={"min": 0.0}, optimizable=True, optimization_range=[0.0, 3.0],
            ),
            "G_MSN_MSN": ParameterDefinition(
                default_value=1.0, description="GMSN_MSN — weight gain of MSN->MSN collaterals.",
                constraints={"min": 0.0}, optimizable=True, optimization_range=[0.0, 4.0],
            ),
            "G_MSN_GPx": ParameterDefinition(
                default_value=0.3,
                description=(
                    "GMSN_GPx — weight gain of MSN->GPe and MSN->GPi, applied only when plastic_syn is True "
                    "(otherwise the gain is 1)."
                ),
                constraints={"min": 0.0}, optimizable=True, optimization_range=[0.0, 1.0],
            ),
            # ── PSP amplitudes and cable constants ─────────────────────────────
            "wPSP_AMPA": ParameterDefinition(
                default_value=1.0, description="wPSP[0] — base weight of AMPA synapses (receptor 1).",
            ),
            "wPSP_NMDA": ParameterDefinition(
                default_value=0.025, description="wPSP[1] — base weight of NMDA synapses (receptor 2).",
            ),
            "wPSP_GABA": ParameterDefinition(
                default_value=-0.25, description="wPSP[2] — base weight of GABA synapses (receptor 3, negative).",
            ),
            "Ri": ParameterDefinition(
                default_value=2.0, description="Ri — axial resistance used in the dendritic attenuation.",
                constraints={"min": 0.0},
            ),
            "Rm": ParameterDefinition(
                default_value=2.0, description="Rm — membrane resistance used in the dendritic attenuation.",
                constraints={"min": 0.0},
            ),
            "lx": ParameterDefinition(
                default_value={"FSI": 0.000961, "GPe": 0.000865, "GPi": 0.001132, "MSN": 0.000619, "STN": 0.00075},
                description="lx — dendritic length per target nucleus (m).",
                unit="m",
            ),
            "dx": ParameterDefinition(
                default_value={"FSI": 1.5e-06, "GPe": 1.7e-06, "GPi": 1.2e-06, "MSN": 1e-06, "STN": 1.5e-06},
                description="dx — dendritic diameter per target nucleus (m).",
                unit="m",
            ),
            # ── Spatial spread and delays ───────────────────────────────────────
            "spread_focused": ParameterDefinition(
                default_value=0.15,
                description="Mask radius of 'focused' projections (layer units; the layer is 1 x 1).",
                constraints={"min": 0.0},
            ),
            "spread_diffuse": ParameterDefinition(
                default_value=2.0,
                description="Mask radius of 'diffuse' projections, multiplied by max(scalefactor).",
                constraints={"min": 0.0},
            ),
            "stochastic_delays": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Relative SD of delays. 0 = fixed delays (original model, None). If > 0, each delay "
                    "is drawn from Normal(tau, tau*value) clipped to [0.5 tau, 1.5 tau]."
                ),
                constraints={"min": 0.0},
            ),
            # ── Real neuron counts (macaque) ────────────────────────────────────
            "real_counts": ParameterDefinition(
                default_value={
                    "MSN": 26448000.0, "FSI": 532000.0, "STN": 77000.0,
                    "GPe": 251000.0, "GPi": 143000.0, "CMPf": 86000.0,
                },
                description=(
                    "count<Nucleus> — real neuron counts in one macaque hemisphere, used to derive the "
                    "number of distinct inputs (nu) per projection. CSN/PTN use alpha directly."
                ),
            ),
        },
        inputs={},
        outputs={
            "connectivity_params": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict in bgParams layout: alpha, tau, ProjPercent, distcontact, cType<Src><Tgt>, "
                    "redundancy<Src><Tgt>, RedundancyType, G*, wPSP, Ri, Rm, lx, dx, spread_*, "
                    "stochastic_delays and count<Nucleus>."
                ),
            ),
        },
        methods={
            "build_params": MethodDefinition(
                description="Check the projection tables and assemble the connectivity part of bgParams.",
                inputs=[],
                outputs=["connectivity_params"],
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
        tables = ["alpha", "delays_ms", "proj_percent", "dist_contact", "projection_type", "redundancy"]
        for table in tables:
            missing = [k for k in PROJECTIONS if k not in p[table]]
            if missing:
                raise ValueError(f"'{table}' is missing projections: {missing}")
        for k in PROJECTIONS:
            if p["projection_type"][k] not in ("focused", "diffuse"):
                raise ValueError(f"projection_type['{k}'] must be 'focused' or 'diffuse'")

        conn = {
            "alpha": {k: float(p["alpha"][k]) for k in PROJECTIONS},
            "tau": {k: float(p["delays_ms"][k]) for k in PROJECTIONS},
            "ProjPercent": {k: float(p["proj_percent"][k]) for k in PROJECTIONS},
            "distcontact": {k: float(p["dist_contact"][k]) for k in PROJECTIONS},
            "RedundancyType": p["redundancy_type"],
            "GGPe_STN": float(p["G_GPe_STN"]),
            "GSTN_GPi": float(p["G_STN_GPi"]),
            "GMSN_MSN": float(p["G_MSN_MSN"]),
            "GMSN_GPx": float(p["G_MSN_GPx"]),
            "wPSP": [float(p["wPSP_AMPA"]), float(p["wPSP_NMDA"]), float(p["wPSP_GABA"])],
            "Ri": float(p["Ri"]),
            "Rm": float(p["Rm"]),
            "lx": dict(p["lx"]),
            "dx": dict(p["dx"]),
            "spread_focused": float(p["spread_focused"]),
            "spread_diffuse": float(p["spread_diffuse"]),
            "stochastic_delays": float(p["stochastic_delays"]) or None,
            "countCSN": None,
            "countPTN": None,
        }
        for k in PROJECTIONS:
            src, tgt = k.split("->")
            conn["cType" + src + tgt] = p["projection_type"][k]
            conn["redundancy" + src + tgt] = float(p["redundancy"][k])
        for nuc, count in p["real_counts"].items():
            conn["count" + nuc] = float(count)

        n_active = sum(1 for k in PROJECTIONS if conn["alpha"][k] > 0)
        print(f"[{self.name}] {n_active}/{len(PROJECTIONS)} projections active, RedundancyType={conn['RedundancyType']}")
        return {"connectivity_params": conn}
