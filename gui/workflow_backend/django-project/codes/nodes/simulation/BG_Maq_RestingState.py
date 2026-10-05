import json
import os
import time
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


class BG_Maq_RestingState(Node):
    """Resting-state simulation of the macaque BG network.

    Port of the 'resting_state' branch of top_BG_nest3/stim_all_model.py: no
    stimulus, the network runs on its DC drives and the CSN/PTN/CMPf background.
    Computes the mean firing rate of every layer over [warmup, simDuration] and
    the instantaneous population rate A(t) (sliding window), and writes
    mean_fr.json, At.json and performance.txt like the original script.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_resting_state",
        stage="simulation",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — top_BG_nest3/stim_all_model.py",
        description=(
            "Run the resting-state protocol on a BG_Maq_Network: simulate simDuration + initial_ignore ms "
            "without stimulation, then compute mean firing rates and instantaneous population rates A(t)."
        ),
        parameters={
            "at_window_ms": ParameterDefinition(
                default_value=1.0,
                description="Width of the sliding window used for A(t). Original: 1 ms.",
                constraints={"min": 0.1}, unit="ms",
            ),
            "at_step_ms": ParameterDefinition(
                default_value=0.1,
                description="Step of the sliding window for A(t). Original: the simulation dt (0.1 ms).",
                constraints={"min": 0.01}, unit="ms",
            ),
            "save_files": ParameterDefinition(
                default_value=True,
                description="Write mean_fr.json, At.json and performance.txt into the network's output_dir.",
            ),
        },
        inputs={
            "bg_network": PortDefinition(
                type=PortType.OBJECT, description="Live network handle from BG_Maq_Network.",
            ),
        },
        outputs={
            "mean_fr": PortDefinition(
                type=PortType.DICT,
                description="Mean firing rate (Hz) per layer, plus 'MSN' for D1+D2 together.",
            ),
            "at_fr": PortDefinition(
                type=PortType.DICT,
                description="{'time_ms': [...], 'rates': {layer: [Hz, ...]}} — instantaneous population rates.",
            ),
            "spikes": PortDefinition(
                type=PortType.OBJECT,
                description="{layer: {'senders': ndarray, 'times': ndarray, 'n_neurons': int, 'first_id': int}}.",
            ),
            "run_info": PortDefinition(
                type=PortType.DICT,
                description="Simulated time, wall-clock time, analysis window and output directory.",
            ),
        },
        methods={
            "run_resting_state": MethodDefinition(
                description="nest.Simulate the resting state, read spikes and compute firing rates.",
                inputs=["bg_network"],
                outputs=["mean_fr", "at_fr", "spikes", "run_info"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("run_resting_state", self.run_resting_state, method_key="run_resting_state")

    def run_resting_state(self, bg_network) -> Dict[str, Any]:
        import nest

        p = self._parameters
        sim = bg_network["sim_config"]
        layers = bg_network["layers"]
        detectors = bg_network["detectors"]
        data_path = bg_network["data_path"]

        t_start = float(sim["start_time_sp"])
        t_end = float(sim["simDuration"])
        sim_time = sim["simDuration"] + sim["initial_ignore"]

        print(f"[{self.name}] simulating {sim_time} ms of resting state ...")
        t0 = time.time()
        nest.Simulate(sim_time)
        elapsed = time.time() - t0
        print(f"[{self.name}] done in {elapsed:.1f} s")

        sizes = {name: len(layer) for name, layer in layers.items()}
        sizes["MSN"] = sizes["MSN_d1"] + sizes["MSN_d2"]
        first_ids = {name: int(layer.tolist()[0]) for name, layer in layers.items()}
        first_ids["MSN"] = first_ids["MSN_d1"]

        grid = np.arange(int(t_start), int(t_end), float(p["at_step_ms"]))
        window = float(p["at_window_ms"])
        mean_fr, rates, spikes = {}, {}, {}
        for name, det in detectors.items():
            n = sizes[name]
            mean_fr[name] = float(det.get("n_events")) / ((t_end - t_start) * n / 1000.0)
            senders, times = self._read_events(det, sim["record_to"])
            spikes[name] = {"senders": senders, "times": times, "n_neurons": n, "first_id": first_ids[name]}
            st = np.sort(times)
            counts = np.searchsorted(st, grid + window, side="left") - np.searchsorted(st, grid, side="left")
            rates[name] = (counts / float(n) / window * 1000.0).tolist()

        at_fr = {"time_ms": grid.tolist(), "rates": rates}
        run_info = {
            "simulated_ms": sim_time,
            "analysis_window_ms": [t_start, t_end],
            "simulation_wall_s": elapsed,
            "build_wall_s": bg_network.get("build_time_s"),
            "n_connections": bg_network.get("n_connections"),
            "data_path": data_path,
        }

        if p["save_files"]:
            with open(os.path.join(data_path, "performance.txt"), "a") as f:
                f.write(f"Simulation_Elapse_Time {elapsed}\n")
            with open(os.path.join(data_path, "mean_fr.json"), "w") as f:
                json.dump(mean_fr, f, indent=1)
            with open(os.path.join(data_path, "At.json"), "w") as f:
                json.dump(rates, f)

        print(f"[{self.name}] mean firing rates (Hz):")
        for name, rate in mean_fr.items():
            print(f"  {name:9s} {rate:8.2f}")
        return {"mean_fr": mean_fr, "at_fr": at_fr, "spikes": spikes, "run_info": run_info}

    @staticmethod
    def _read_events(det, record_to):
        if record_to == "memory":
            ev = det.get("events")
            return np.asarray(ev["senders"], dtype=int), np.asarray(ev["times"], dtype=float)
        filenames = det.get("filenames")
        if isinstance(filenames, str):
            filenames = [filenames]
        senders, times = [], []
        for fname in filenames or []:
            if fname and os.path.isfile(fname):
                with open(fname) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2 and not line.startswith("#"):
                            try:
                                senders.append(int(parts[0]))
                                times.append(float(parts[1]))
                            except ValueError:
                                pass  # column header
        return np.asarray(senders, dtype=int), np.asarray(times, dtype=float)
