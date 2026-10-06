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

from tvb.simulator import integrators, noise


class NW_TVB_Integrator(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_integrator",
        stage="simulation",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Builds any TVB integration scheme (Euler, Heun, RK4, Dopri5, Dop853, VODE; "
            "deterministic or stochastic) with additive or multiplicative noise sized "
            "automatically to the model's state variables."
        ),
        parameters={
            "integrator_type": ParameterDefinition(
                default_value="HeunStochastic",
                description=(
                    "Numerical method that advances the equations in time. '...Stochastic' "
                    "methods add noise (needed for resting-state activity and FC); "
                    "deterministic ones give the same trajectory every run. Options: "
                    "'HeunStochastic' / 'HeunDeterministic' - 2nd order, the standard TVB choice. "
                    "'EulerStochastic' / 'EulerDeterministic' - 1st order, fastest, needs a "
                    "smaller dt. 'RungeKutta4thOrderDeterministic' - 4th order, accurate for "
                    "smooth noise-free dynamics. 'Dopri5', 'Dop853', 'VODE' (and their "
                    "Stochastic versions) - SciPy adaptive solvers; VODE suits stiff models; "
                    "all are slower. Example: 'HeunStochastic'."
                ),
                constraints={
                    "allowed_values": [
                        "HeunStochastic",
                        "HeunDeterministic",
                        "EulerStochastic",
                        "EulerDeterministic",
                        "RungeKutta4thOrderDeterministic",
                        "Dopri5",
                        "Dopri5Stochastic",
                        "Dop853",
                        "Dop853Stochastic",
                        "VODE",
                        "VODEStochastic",
                    ]
                },
            ),
            "dt": ParameterDefinition(
                default_value=0.1,
                description=(
                    "Integration time step in milliseconds. Smaller steps are more accurate but "
                    "slower; too large a step makes the simulation diverge (NaN). Recommended: "
                    "Generic2dOscillator 0.1, MontbrioPazoRoxin 0.025, ReducedWongWangExcInh 0.1, "
                    "JansenRit 0.0625, SupHopf 0.1, Epileptor 0.05. 0 = use the model "
                    "preset's recommended dt (requires tvb_model_info)."
                ),
                constraints={"min": 0.0, "max": 10.0},
                unit="ms",
            ),
            "noise_type": ParameterDefinition(
                default_value="Additive",
                description=(
                    "Type of noise added by stochastic integrators. 'Additive' - same "
                    "amplitude whatever the state (standard choice). 'Multiplicative' - "
                    "amplitude grows with the state value (TVB default: linear in the state). "
                    "Ignored by deterministic integrators. Example: 'Additive'."
                ),
                constraints={"allowed_values": ["Additive", "Multiplicative"]},
            ),
            "nsig": ParameterDefinition(
                default_value=[0.001, 0.0],
                description=(
                    "Noise intensity for each state variable (TVB 'nsig'; the noise added per "
                    "step has std sqrt(2*nsig*dt)). Accepted forms: 'auto' - the model "
                    "preset's values (requires tvb_model_info); a number, e.g. 0.001 - the same "
                    "for every state variable; a list, e.g. [0.001, 0.0] - one value per state "
                    "variable in model order; a dict, e.g. {'V': 0.02} - by name, unlisted "
                    "variables get 0 (requires tvb_model_info). Ignored by deterministic "
                    "integrators."
                ),
            ),
            "noise_seed": ParameterDefinition(
                default_value=42,
                description=(
                    "Seed of the random number stream used for noise and for the initial state. "
                    "The same seed reproduces the run exactly; change it (e.g. 1, 2, 3) to get "
                    "independent realisations of the same model."
                ),
            ),
        },
        inputs={
            "tvb_model_info": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Optional model metadata from NW_TVB_Model; used to size nsig, resolve "
                    "state-variable names and apply preset dt/nsig."
                ),
                optional=True,
            ),
        },
        outputs={
            "tvb_integrator": PortDefinition(
                type=PortType.OBJECT,
                description="Configured TVB integrator object, ready for NW_TVB_Simulator.",
            ),
        },
        methods={
            "build_integrator": MethodDefinition(
                description=(
                    "Resolve dt and the per-state-variable noise vector, then instantiate the "
                    "selected integrator (with a noise object if stochastic)."
                ),
                inputs=["tvb_model_info"],
                outputs=["tvb_integrator"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step(
            "build_integrator", self.build_integrator, method_key="build_integrator"
        )

    def build_integrator(self, tvb_model_info=None) -> Dict[str, Any]:
        integrator_type = self._parameters["integrator_type"]
        info = tvb_model_info or {}
        recommended = info.get("recommended") or {}

        dt = float(self._parameters["dt"])
        if dt <= 0:
            dt = recommended.get("dt")
            if not dt:
                raise ValueError("dt=0 requires tvb_model_info with a preset dt.")

        cls = getattr(integrators, integrator_type)
        if not issubclass(cls, integrators.IntegratorStochastic):
            print(f"[{self.name}] {integrator_type}: dt={dt} ms (deterministic)")
            return {"tvb_integrator": cls(dt=dt)}

        nsig = self._resolve_nsig(self._parameters["nsig"], info)
        noise_cls = getattr(noise, self._parameters["noise_type"])
        hiss = noise_cls(nsig=nsig, noise_seed=int(self._parameters["noise_seed"]))
        integ = cls(dt=dt, noise=hiss)

        print(f"[{self.name}] {integrator_type}: dt={dt} ms, "
              f"{self._parameters['noise_type']} nsig={nsig.tolist()}")
        return {"tvb_integrator": integ}

    def _resolve_nsig(self, nsig, info) -> np.ndarray:
        svars = info.get("state_variables")

        if isinstance(nsig, str):
            if nsig != "auto":
                raise ValueError(f"nsig string must be 'auto', got '{nsig}'")
            nsig = (info.get("recommended") or {}).get("nsig")
            if nsig is None:
                raise ValueError("nsig='auto' requires tvb_model_info from a preset model.")

        if isinstance(nsig, dict):
            if not svars:
                raise ValueError("nsig as dict requires tvb_model_info (state variable names).")
            unknown = set(nsig) - set(svars)
            if unknown:
                raise ValueError(f"nsig keys {sorted(unknown)} not in state variables {svars}")
            return np.array([float(nsig.get(sv, 0.0)) for sv in svars])

        arr = np.atleast_1d(np.asarray(nsig, dtype=float))
        if svars:
            if arr.size == 1:
                return np.full(len(svars), arr[0])
            if arr.size != len(svars):
                raise ValueError(
                    f"nsig has {arr.size} values but model has {len(svars)} state variables {svars}"
                )
        return arr
