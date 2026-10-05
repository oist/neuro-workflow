import json
import math
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


class BG_Maq_Network(Node):
    """Build the macaque topological Basal Ganglia network in NEST 3.

    Self-contained port of top_BG_nest3/ini_all.instantiate_bg and the layer /
    connection routines of nest_routine.py. Receives the parameter dicts from the
    BG_Maq_* parameter nodes, merges them into one bgParams dict, resets the NEST
    kernel, creates every layer, wires the 24 projections and attaches spike
    recorders. The live NEST objects are passed on through `bg_network`.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_network",
        stage="setup",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — top_BG_nest3/ini_all.py + nest_routine.py",
        description=(
            "Merge the BG_Maq parameter dicts into bgParams, reset NEST and instantiate the spatial BG "
            "network: MSN_d1, MSN_d2, FSI, STN, GPe, GPi (+ GPi_fake relay) and the CSN/PTN/CMPf "
            "Poisson input layers, all 24 projections and one spike recorder per layer."
        ),
        parameters={
            "save_positions": ParameterDefinition(
                default_value=True,
                description="Write <layer>.txt files (node id, x, y[, z]) and centers.txt into output_dir, as the original script does.",
            ),
            "verbose": ParameterDefinition(
                default_value=False,
                description="Print in-degree and weight of every projection while connecting.",
            ),
        },
        inputs={
            "sim_config": PortDefinition(
                type=PortType.DICT, description="From BG_Maq_SimConfig.",
            ),
            "neuron_params": PortDefinition(
                type=PortType.DICT, description="From BG_Maq_NeuronParams.",
            ),
            "connectivity_params": PortDefinition(
                type=PortType.DICT, description="From BG_Maq_ConnectivityParams.",
            ),
            "striatum_params": PortDefinition(
                type=PortType.DICT, description="From BG_Maq_StriatumParams.",
            ),
            "input_params": PortDefinition(
                type=PortType.DICT, description="From BG_Maq_InputParams.",
            ),
        },
        outputs={
            "bg_network": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "Live network handle: {'layers': {name: NodeCollection}, 'detectors': {name: spike_recorder} "
                    "(layers + combined 'MSN'), 'bg_params', 'sim_config', 'projections', 'data_path'}."
                ),
            ),
            "bg_params": PortDefinition(
                type=PortType.DICT,
                description="The merged, scaled bgParams dict actually used (also saved as bg_params.json).",
            ),
        },
        methods={
            "build_network": MethodDefinition(
                description="Reset NEST, create all layers, connect all projections and attach spike recorders.",
                inputs=["sim_config", "neuron_params", "connectivity_params", "striatum_params", "input_params"],
                outputs=["bg_network", "bg_params"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build_network", self.build_network, method_key="build_network")

    # ------------------------------------------------------------------ main
    def build_network(self, sim_config, neuron_params, connectivity_params,
                      striatum_params, input_params) -> Dict[str, Any]:
        import nest

        t0 = time.time()
        self._verbose = bool(self._parameters["verbose"])
        self._ampa_counter = 0
        self._projections = {}

        bg = {}
        bg.update(neuron_params)
        bg.update(connectivity_params)
        bg.update(striatum_params)
        bg.update({k: v for k, v in input_params.items() if k.startswith("nb")})
        input_rates = dict(input_params["input_rates"])

        sf = sim_config["scalefactor"]
        for nucleus in ["MSN", "FSI", "STN", "GPe", "GPi", "CSN", "PTN", "CMPf"]:
            bg["nb" + nucleus] = bg["nb" + nucleus] * sf[0] * sf[1] * sim_config["density_scale"]

        data_path = os.path.abspath(sim_config["data_path"])
        os.makedirs(data_path, exist_ok=True)
        self._data_path = data_path
        self._save = bool(self._parameters["save_positions"])

        self._initialize_nest(nest, sim_config, data_path)
        bg["circle_center"] = self._channel_centers(sim_config)

        layers = {}
        print(f"[{self.name}] creating layers ...")
        for nucleus in ["GPi", "MSN", "FSI", "STN", "GPe", "GPi_fake"]:
            result = self._create_layer(nest, bg, nucleus, sf)
            if nucleus == "MSN":
                layers["MSN_d1"], layers["MSN_d2"] = result
            else:
                layers[nucleus] = result

        nest.SetDefaults("static_synapse", {"receptor_type": 0})
        nest.Connect(layers["GPi"], layers["GPi_fake"], conn_spec={"rule": "one_to_one"})

        for nucleus in ["CSN", "PTN", "CMPf"]:
            layers[nucleus] = self._create_layer(nest, bg, nucleus, sf, rate=input_rates[nucleus])

        if bg["plastic_syn"]:
            vt_d1 = nest.Create("volume_transmitter")
            vt_d2 = nest.Create("volume_transmitter")
            nest.CopyModel("stdp_dopamine_synapse_lbl", "syn_d1")
            nest.SetDefaults("syn_d1", dict(bg["stdp_d1"], volume_transmitter=vt_d1))
            nest.CopyModel("stdp_dopamine_synapse_lbl", "syn_d2")
            nest.SetDefaults("syn_d2", dict(bg["stdp_d2"], volume_transmitter=vt_d2))

        print(f"[{self.name}] connecting projections ...")
        for connection in sorted(bg["alpha"].keys()):
            src, tgt = connection.split("->")
            n_type = "in" if src in ["MSN", "FSI", "GPe", "GPi"] else "ex"
            self._connect_layers(nest, bg, n_type, layers, src, tgt, sf)

        detectors = {}
        for layer_name, layer in layers.items():
            det = nest.Create("spike_recorder", params={
                "record_to": sim_config["record_to"], "label": layer_name,
                "start": float(sim_config["start_time_sp"])})
            nest.Connect(layer, det)
            detectors[layer_name] = det
        detectors["MSN"] = nest.Create("spike_recorder", params={
            "record_to": sim_config["record_to"], "label": "MSN",
            "start": float(sim_config["start_time_sp"])})
        nest.Connect(layers["MSN_d1"], detectors["MSN"])
        nest.Connect(layers["MSN_d2"], detectors["MSN"])

        n_conn = nest.GetKernelStatus("num_connections")
        elapsed = time.time() - t0
        sizes = {k: len(v) for k, v in layers.items()}
        print(f"[{self.name}] built in {elapsed:.1f} s — {sum(sizes.values())} nodes, {n_conn} connections")
        print(f"[{self.name}] sizes: {sizes}")

        bg_params_json = self._json_safe(bg)
        bg_params_json["input_rates"] = input_rates
        with open(os.path.join(data_path, "bg_params.json"), "w") as f:
            json.dump(bg_params_json, f, indent=1)
        with open(os.path.join(data_path, "performance.txt"), "a") as f:
            f.write(f"Network_Build_Time {elapsed}\n")

        bg_network = {
            "layers": layers,
            "detectors": detectors,
            "bg_params": bg,
            "sim_config": sim_config,
            "projections": self._projections,
            "data_path": data_path,
            "n_connections": n_conn,
            "build_time_s": elapsed,
        }
        return {"bg_network": bg_network, "bg_params": bg_params_json}

    # ---------------------------------------------------------- NEST kernel
    def _initialize_nest(self, nest, sim_config, data_path):
        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")
        nest.SetKernelStatus({"overwrite_files": True,
                              "local_num_threads": int(sim_config["nbcpu"]),
                              "data_path": data_path,
                              "resolution": float(sim_config["dt"]),
                              "rng_seed": int(sim_config["msd"])})
        # nest_routine kept one RandomState per virtual process and only used pyrngs[0]
        self._rng = np.random.RandomState(int(sim_config["msd"]))

    def _channel_centers(self, sim_config):
        centers = []
        if sim_config["channels"]:
            for i in range(sim_config["channels_nb"]):
                angle = math.pi / 180 * (60 * i - 30)
                centers.append([sim_config["hex_radius"] * math.cos(angle),
                                sim_config["hex_radius"] * math.sin(angle)])
            if self._save:
                np.savetxt(os.path.join(self._data_path, "centers.txt"), centers)
        return centers

    # --------------------------------------------------------------- layers
    def _grid_positions(self, n, a0, a1, b0, b1):
        n_sq = np.ceil(np.sqrt(n))
        coord = [[x / n_sq * a1 - a0, y / n_sq * b1 - b0]
                 for x in np.arange(0, n_sq, dtype=float)
                 for y in np.arange(0, n_sq, dtype=float)]
        if len(coord) > n:
            coord = np.array(coord)[np.sort(self._rng.choice(range(len(coord)), size=n, replace=False))].tolist()
        return coord

    def _save_positions(self, layer_name, layer, positions):
        if self._save:
            data = np.column_stack((np.array(layer.tolist()), np.array(positions)))
            np.savetxt(os.path.join(self._data_path, layer_name + ".txt"), data, fmt="%1.3f")

    def _create_layer(self, nest, bg, nucleus, sf, rate=None):
        extent = [1.0 * int(sf[0]) + 1.0, 1.0 * int(sf[1]) + 1.0]

        if nucleus == "GPi_fake":
            pop_size = int(bg["nbGPi"])
            z = self._rng.uniform(0.0, 0.5, pop_size)
            xy = self._gpi_positions
            positions = [[xy[i][0], xy[i][1], z[i]] for i in range(pop_size)]
            layer = nest.Create("parrot_neuron", positions=nest.spatial.free(positions, extent=extent + [1.0], edge_wrap=True))
            self._save_positions(nucleus, layer, positions)
            return layer

        pop_size = int(bg["nb" + nucleus])
        if self._verbose:
            print(f"  population size for {nucleus}: {pop_size}")

        if nucleus == "MSN":
            positions = [[self._rng.uniform(-0.5, 0.5), self._rng.uniform(-0.5, 0.5)] for _ in range(pop_size)]
            nest.SetDefaults("iaf_psc_alpha_multisynapse", bg["common_iaf"])
            nest.SetDefaults("iaf_psc_alpha_multisynapse", bg["MSN_iaf"])
            nest.SetDefaults("iaf_psc_alpha_multisynapse", {"I_e": bg["IeMSN"]})
            nest.CopyModel("iaf_psc_alpha_multisynapse", "msn_d1")
            nest.CopyModel("iaf_psc_alpha_multisynapse", "msn_d2")
            pos_half = positions[:pop_size // 2]
            layer_d1 = nest.Create("msn_d1", positions=nest.spatial.free(pos_half, extent=extent, edge_wrap=True))
            layer_d2 = nest.Create("msn_d2", positions=nest.spatial.free(pos_half, extent=extent, edge_wrap=True))
            self._save_positions("MSN_d1", layer_d1, pos_half)
            self._save_positions("MSN_d2", layer_d2, pos_half)
            return layer_d1, layer_d2

        if nucleus in ("GPi", "STN"):
            positions = self._grid_positions(pop_size, 0.4 * sf[0], sf[0] - 0.1, 0.4 * sf[1], sf[1] - 0.1)
        else:
            positions = self._grid_positions(pop_size, 0.5 * sf[0], sf[0], 0.5 * sf[1], sf[1])
        spatial = nest.spatial.free(positions, extent=extent, edge_wrap=True)

        if rate is None:
            nest.SetDefaults("iaf_psc_alpha_multisynapse", bg["common_iaf"])
            nest.SetDefaults("iaf_psc_alpha_multisynapse", bg[nucleus + "_iaf"])
            nest.SetDefaults("iaf_psc_alpha_multisynapse", {"I_e": bg["Ie" + nucleus]})
            layer = nest.Create("iaf_psc_alpha_multisynapse", positions=spatial)
            if nucleus == "GPi":
                self._gpi_positions = positions
        else:
            # input layer: parrot neurons, each fed an independent Poisson train
            layer = nest.Create("parrot_neuron", positions=spatial)
            if rate > 0:
                poisson = nest.Create("poisson_generator", params={"rate": float(rate)})
                nest.Connect(poisson, layer, conn_spec={"rule": "all_to_all"})
        self._save_positions(nucleus, layer, positions)
        return layer

    # ---------------------------------------------------------- connectivity
    def _input_range(self, bg, src, tgt):
        key = src + "->" + tgt
        if src in ("CSN", "PTN"):
            return [0.0, bg["alpha"][key]]
        ratio = bg["count" + src] / float(bg["count" + tgt]) * bg["ProjPercent"][key]
        return [ratio, ratio * bg["alpha"][key]]

    def _in_degree(self, bg, src, tgt, redundancy):
        rtype = bg["RedundancyType"]
        nu0, nu = self._input_range(bg, src, tgt)
        if rtype == "inDegreeAbs":
            return float(redundancy)
        if rtype == "outDegreeAbs":
            return nu * (1.0 / redundancy)
        if rtype == "outDegreeCons":
            return (nu - nu0) * redundancy + nu0
        raise KeyError("RedundancyType should be one of inDegreeAbs, outDegreeAbs, outDegreeCons")

    def _weights(self, bg, rec_types, src, tgt, in_degree, gain):
        nu = self._input_range(bg, src, tgt)[1]
        LX = bg["lx"][tgt] * np.sqrt((4.0 * bg["Ri"]) / (bg["dx"][tgt] * bg["Rm"]))
        attenuation = np.cosh(LX * (1 - bg["distcontact"][src + "->" + tgt])) / np.cosh(LX)
        idx = {"AMPA": 0, "NMDA": 1, "GABA": 2}
        return {r: float(nu / float(in_degree) * attenuation * bg["wPSP"][idx[r]] * gain) for r in rec_types}

    def _connect_layers(self, nest, bg, n_type, layers, src, tgt, sf):
        key = src + "->" + tgt
        in_degree = self._in_degree(bg, src, tgt, bg["redundancy" + src + tgt])
        if in_degree > bg["nb" + src]:
            if self._verbose:
                print(f"  /!\\ {key}: in-degree {in_degree:.1f} larger than source population, reduced")
            in_degree = bg["nb" + src]
        if in_degree == 0.0:
            self._projections[key] = {"in_degree": 0.0, "skipped": True}
            return

        rec = {"AMPA": 1, "NMDA": 2, "GABA": 3}
        if n_type == "ex":
            rec_types = ["AMPA", "NMDA"]
            self._ampa_counter += 1
            lbl = self._ampa_counter
        else:
            rec_types = ["GABA"]
            lbl = 0

        if key == "GPe->STN":
            gain = bg["GGPe_STN"]
        elif key == "STN->GPi":
            gain = bg["GSTN_GPi"]
        elif key == "MSN->MSN":
            gain = bg["GMSN_MSN"]
        elif key in ("MSN->GPi", "MSN->GPe") and bg["plastic_syn"]:
            gain = bg["GMSN_GPx"]
        else:
            gain = 1.0

        W = self._weights(bg, rec_types, src, tgt, in_degree, gain)
        delay = bg["tau"][key]
        spread = bg["spread_focused"] if bg["cType" + src + tgt] == "focused" else bg["spread_diffuse"] * max(sf)
        self._projections[key] = {"in_degree": float(in_degree), "weights": W, "gain": gain,
                                  "delay_ms": delay, "spread": spread, "type": bg["cType" + src + tgt]}
        if self._verbose:
            print(f"  {key:10s} {n_type} inDegree={in_degree:8.2f}  W={W}  spread={spread}")

        self._mass_connect(nest, bg, layers, src, tgt, lbl, in_degree, rec[rec_types[0]], W[rec_types[0]],
                           delay, spread,
                           nmda_receptor=(rec["NMDA"] if n_type == "ex" else None),
                           nmda_weight=(W["NMDA"] if n_type == "ex" else None))

    def _mass_connect(self, nest, bg, layers, src, tgt, lbl, in_degree, receptor, weight, delay, spread,
                      nmda_receptor=None, nmda_weight=None):
        if bg["stochastic_delays"] and delay > 0:
            delay_param = nest.math.redraw(nest.random.normal(mean=delay, std=delay * bg["stochastic_delays"]),
                                           min=delay * 0.5, max=delay * 1.5)
        else:
            delay_param = delay

        ampa_spec = {"synapse_model": "static_synapse_lbl", "synapse_label": lbl,
                     "receptor_type": receptor, "weight": weight, "delay": delay_param}
        if nmda_receptor is not None:
            nmda_spec = dict(ampa_spec, receptor_type=nmda_receptor, weight=nmda_weight)
            base_syn = nest.CollocatedSynapses(ampa_spec, nmda_spec)
        else:
            base_syn = ampa_spec

        def connect(pre, post, conn, syn):
            if conn.get("indegree", 1) > 0:
                nest.Connect(pre, post, conn_spec=conn, syn_spec=syn)

        int_deg = int(np.floor(in_degree))
        if int_deg > 0:
            base_conn = {"rule": "fixed_indegree", "indegree": int_deg,
                         "mask": {"circular": {"radius": spread}},
                         "allow_oversized_mask": True, "allow_multapses": True}

            if src in ("CSN", "PTN") and tgt == "MSN":
                if bg["plastic_syn"]:
                    w_p = weight * bg["plast_gain"]
                    d1 = dict(ampa_spec, synapse_model="syn_d1", weight=w_p)
                    d2 = dict(ampa_spec, synapse_model="syn_d2", synapse_label=lbl + 1000, weight=w_p)
                    if nmda_receptor is not None:
                        w_pn = nmda_weight * bg["plast_gain"]
                        nd1 = dict(d1, receptor_type=nmda_receptor, weight=w_pn)
                        nd2 = dict(d2, receptor_type=nmda_receptor, weight=w_pn)
                        d1, d2 = nest.CollocatedSynapses(d1, nd1), nest.CollocatedSynapses(d2, nd2)
                    connect(layers[src], layers["MSN_d1"], base_conn, d1)
                    connect(layers[src], layers["MSN_d2"], base_conn, d2)
                else:
                    connect(layers[src], layers["MSN_d1"], base_conn, base_syn)
                    connect(layers[src], layers["MSN_d2"], base_conn, base_syn)

            elif src == "MSN" and tgt in ("GPe", "GPi"):
                lam = bg["overlap_d1d2"]
                if tgt == "GPi":
                    n_d1, n_d2 = int(int_deg * (1.0 - lam)), int(int_deg * lam)
                else:
                    n_d1, n_d2 = int(int_deg * lam), int(int_deg * (1.0 - lam))
                connect(layers["MSN_d1"], layers[tgt], dict(base_conn, indegree=n_d1), base_syn)
                connect(layers["MSN_d2"], layers[tgt], dict(base_conn, indegree=n_d2), base_syn)

            elif src == "MSN" and tgt == "MSN":
                n_a1 = int(int_deg * bg["asymmetry_1"])
                n_a2 = int(int_deg * bg["asymmetry_2"])
                asym = dict(ampa_spec, weight=weight * bg["syn_asymm"])
                connect(layers["MSN_d2"], layers["MSN_d2"], dict(base_conn, indegree=n_a1), asym)
                connect(layers["MSN_d2"], layers["MSN_d1"], dict(base_conn, indegree=n_a1), asym)
                connect(layers["MSN_d1"], layers["MSN_d1"], dict(base_conn, indegree=n_a1), ampa_spec)
                connect(layers["MSN_d1"], layers["MSN_d2"], dict(base_conn, indegree=n_a2), ampa_spec)

            elif tgt == "MSN":
                connect(layers[src], layers["MSN_d1"], base_conn, base_syn)
                connect(layers[src], layers["MSN_d2"], base_conn, base_syn)

            else:
                connect(layers[src], layers[tgt], base_conn, base_syn)

        # fractional part of the in-degree: pairwise_bernoulli, as in nest_routine.mass_connect_bg
        remaining = np.round((in_degree - np.floor(in_degree)) * bg["nb" + tgt])
        if remaining > 0:
            p = 1.0 / (bg["nb" + src] * float(remaining))
            fconn = {"rule": "pairwise_bernoulli", "p": p,
                     "mask": {"circular": {"radius": spread}},
                     "allow_oversized_mask": True, "allow_multapses": True}
            if src == "MSN" and tgt == "MSN":
                # not reachable with the published alpha (in-degree is integer); the source script
                # would fail here because there is no combined 'MSN' layer
                for pre in ("MSN_d1", "MSN_d2"):
                    for post in ("MSN_d1", "MSN_d2"):
                        connect(layers[pre], layers[post], fconn, base_syn)
            elif tgt == "MSN":
                connect(layers[src], layers["MSN_d1"], fconn, base_syn)
                connect(layers[src], layers["MSN_d2"], fconn, base_syn)
            elif src == "MSN":
                connect(layers["MSN_d1"], layers[tgt], fconn, base_syn)
                connect(layers["MSN_d2"], layers[tgt], fconn, base_syn)
            else:
                connect(layers[src], layers[tgt], fconn, base_syn)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: BG_Maq_Network._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BG_Maq_Network._json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        return str(obj)
