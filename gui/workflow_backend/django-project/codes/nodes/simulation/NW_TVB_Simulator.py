import time as tm
from typing import Dict, Any, List

import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType

from tvb.simulator.lab import simulator


# Couplings a model's equations are written for; any other choice only triggers a warning.
REQUIRED_COUPLING = {
    "JansenRit": ["SigmoidalJansenRit"],
    "Epileptor": ["Difference"],
    "EpileptorRestingState": ["Difference"],
    "Kuramoto": ["Kuramoto"],
}


class NW_TVB_Simulator(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_simulator",
        stage="simulation",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Assembles and runs a TVB whole-brain simulation from connectivity, model, coupling, "
            "integrator, any number of monitors and an optional stimulus. Returns one labelled "
            "result per monitor (time, data, variable names, region labels), with the initial "
            "transient removed."
        ),
        parameters={
            "simulation_length": ParameterDefinition(
                default_value=10000.0,
                description=(
                    "Total simulated time in ms, including the transient. Examples: 2000 (2 s) "
                    "for quick tests of neural activity; 60000 (1 min) or more for BOLD and "
                    "functional connectivity (FC needs many BOLD samples: 5 min at TR=2 s "
                    "gives 150 samples)."
                ),
                constraints={"min": 1.0, "max": 1e8},
                unit="ms",
            ),
            "transient": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Initial period in ms dropped from every monitor's output, because it "
                    "reflects the random initial state rather than the model's own dynamics. "
                    "Examples: 500-1000 for neural activity; 10000-20000 for BOLD (the "
                    "haemodynamic response needs ~10 s to settle). 0 keeps everything."
                ),
                constraints={"min": 0.0, "max": 1e8},
                unit="ms",
            ),
            "random_seed": ParameterDefinition(
                default_value=42,
                description=(
                    "Seed for the random initial state of deterministic runs, so they are "
                    "reproducible. Stochastic integrators use their own noise_seed "
                    "(NW_TVB_Integrator) for both noise and initial state."
                ),
            ),
        },
        inputs={
            "tvb_connectivity": PortDefinition(
                type=PortType.OBJECT,
                description="Structural connectome from NW_TVB_Connectivity.",
            ),
            "tvb_model": PortDefinition(
                type=PortType.OBJECT,
                description="Neural mass model from NW_TVB_Model.",
            ),
            "tvb_coupling": PortDefinition(
                type=PortType.OBJECT,
                description="Long-range coupling function from NW_TVB_Coupling.",
            ),
            "tvb_integrator": PortDefinition(
                type=PortType.OBJECT,
                description="Integration scheme (and noise) from NW_TVB_Integrator.",
            ),
            "tvb_monitors": PortDefinition(
                type=PortType.DICT,
                fan_in=True,
                description=(
                    "Fan-in port: connect one or more NW_TVB_Monitor outputs (e.g. a 1 ms "
                    "TemporalAverage and a 2000 ms Bold). Each is identified by its label."
                ),
            ),
            "tvb_stimulus": PortDefinition(
                type=PortType.OBJECT,
                description="Optional external stimulus from NW_TVB_Stimulus.",
                optional=True,
            ),
        },
        outputs={
            "tvb_results": PortDefinition(
                type=PortType.DICT,
                description=(
                    "{monitor_label: {'time' (n_t,) ms, 'data' (n_t, n_variables, n_regions, "
                    "n_modes), 'variables', 'region_labels', 'monitor_type', 'period'}}; read "
                    "by NW_TVB_TimeSeriesPlot and NW_TVB_FunctionalConnectivity."
                ),
            ),
        },
        methods={
            "run_simulation": MethodDefinition(
                description=(
                    "Check model/coupling compatibility, configure the TVB Simulator with all "
                    "monitors, run it for simulation_length, drop the transient and package "
                    "the labelled results."
                ),
                inputs=[
                    "tvb_connectivity",
                    "tvb_model",
                    "tvb_coupling",
                    "tvb_integrator",
                    "tvb_monitors",
                    "tvb_stimulus",
                ],
                outputs=["tvb_results"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step(
            "run_simulation", self.run_simulation, method_key="run_simulation"
        )

    def run_simulation(
        self,
        tvb_connectivity,
        tvb_model,
        tvb_coupling,
        tvb_integrator,
        tvb_monitors: List[Dict[str, Any]],
        tvb_stimulus=None,
    ) -> Dict[str, Any]:
        monitor_specs = tvb_monitors if isinstance(tvb_monitors, list) else [tvb_monitors]
        monitor_specs = [m for m in monitor_specs if m is not None]
        if not monitor_specs:
            raise ValueError("Connect at least one NW_TVB_Monitor to tvb_monitors.")
        labels = [m["label"] for m in monitor_specs]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Monitor labels must be unique, got {labels}")

        self._check_coupling(tvb_model, tvb_coupling)

        model_voi = list(tvb_model.variables_of_interest)
        variables_per_monitor = []
        for spec in monitor_specs:
            idx = self._resolve_variables(spec, model_voi)
            if idx is not None:
                spec["monitor"].variables_of_interest = np.array(idx)
            variables_per_monitor.append([model_voi[i] for i in (idx or range(len(model_voi)))])

        sim_kwargs = dict(
            model=tvb_model,
            connectivity=tvb_connectivity,
            conduction_speed=float(np.asarray(tvb_connectivity.speed).flat[0]),
            coupling=tvb_coupling,
            integrator=tvb_integrator,
            monitors=[m["monitor"] for m in monitor_specs],
        )
        if tvb_stimulus is not None:
            sim_kwargs["stimulus"] = tvb_stimulus

        np.random.seed(int(self._parameters["random_seed"]))
        sim = simulator.Simulator(**sim_kwargs)
        sim.configure()

        length = float(self._parameters["simulation_length"])
        times = [[] for _ in monitor_specs]
        data = [[] for _ in monitor_specs]
        tic = tm.time()
        for step_output in sim(simulation_length=length):
            for i, mon_out in enumerate(step_output):
                if mon_out is not None:
                    times[i].append(mon_out[0])
                    data[i].append(mon_out[1])
        print(f"[{self.name}] simulated {length:.0f} ms in {tm.time() - tic:.1f} s")

        transient = float(self._parameters["transient"])
        region_labels = [str(l) for l in tvb_connectivity.region_labels]
        results = {}
        for spec, t, d, variables in zip(monitor_specs, times, data, variables_per_monitor):
            t = np.array(t)
            d = np.array(d)
            keep = t >= transient
            t, d = t[keep], d[keep]
            n_regions = d.shape[2] if d.ndim == 4 else 0
            results[spec["label"]] = {
                "time": t,
                "data": d,
                "variables": variables,
                "region_labels": region_labels if n_regions == len(region_labels) else
                                 [f"{spec['label']}_{k}" for k in range(n_regions)],
                "monitor_type": spec["monitor_type"],
                "period": float(spec["monitor"].period),
            }
            n_bad = int(np.size(d) - np.isfinite(d).sum())
            print(f"[{self.name}]   '{spec['label']}': {d.shape} (time, variables {variables}, "
                  f"regions, modes)" + (f"  WARNING {n_bad} non-finite values" if n_bad else ""))
            if n_bad:
                print(f"[{self.name}]   -> the simulation diverged: reduce dt, noise (nsig) or "
                      f"coupling 'a', or narrow the model's state_variable_range.")

        return {"tvb_results": results}

    def _check_coupling(self, model, coupling) -> None:
        model_name = type(model).__name__
        coupling_name = type(coupling).__name__
        required = REQUIRED_COUPLING.get(model_name)
        if required and coupling_name not in required:
            print(f"[{self.name}] WARNING: {model_name} is written for {required} coupling, "
                  f"got {coupling_name}; results may be meaningless.")

    @staticmethod
    def _resolve_variables(spec, model_voi):
        variables = spec.get("variables") or []
        if not variables:
            return None
        idx = []
        for v in variables:
            if isinstance(v, str):
                if v not in model_voi:
                    raise ValueError(
                        f"Monitor '{spec['label']}': variable '{v}' is not recorded by the model; "
                        f"model variables_of_interest = {model_voi}"
                    )
                idx.append(model_voi.index(v))
            else:
                if not 0 <= int(v) < len(model_voi):
                    raise ValueError(
                        f"Monitor '{spec['label']}': variable index {v} out of range for {model_voi}"
                    )
                idx.append(int(v))
        return idx
