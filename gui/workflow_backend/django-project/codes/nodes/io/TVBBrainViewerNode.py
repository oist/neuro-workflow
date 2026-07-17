import json
import os
import shutil
from typing import Any, Dict

import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_static")

# Per-species bundled companion data (cortical mesh + anatomical fiber tracts).
# All files live in viewer_static/ and ship with the node. Same-atlas contract:
# the incoming TVB connectivity must use the atlas that generated these files.
_SPECIES_FILES = {
    "marmoset": {"mesh": "brain_mesh.json",       "tracts": "brain_tracts.json"},
    "human":    {"mesh": "brain_mesh_human.json", "tracts": "brain_tracts_human.json"},
}
_EXPECTED_REGIONS = {"marmoset": 212, "human": 360}


class TVBBrainViewerNode(Node):
    """Generate the interactive brain connectivity viewer (marmoset or human)."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="tvb_brain_viewer",
        stage="output",
        tool="TVB",
        model_source="Marmoset Brain Atlas (212) / HCP-MMP1.0 (360) / TVB",
        description=(
            "Build an interactive Three.js brain connectivity viewer from a TVB connectivity "
            "object, optional TVB stimulus metadata, and optional simulation timeseries "
            "(BOLD or TemporalAverage). Supports the marmoset (212-region) and human "
            "(HCP-MMP1.0, 360-region) atlases, selected by the 'species' parameter (or the "
            "workflow context 'species' key). Embeds the species-appropriate cortical mesh "
            "and anatomical white-matter fiber tracts. Writes brain_viewer.html, "
            "brain_viewer.js, and connectivity_data.json to output_dir."
        ),
        parameters={
            "species": ParameterDefinition(
                default_value="marmoset",
                description=(
                    "Which atlas the incoming connectivity uses: 'marmoset' (212 regions) or "
                    "'human' (HCP-MMP1.0, 360 regions). Selects the bundled cortical mesh and "
                    "fiber tracts. This node parameter is authoritative; the workflow context "
                    "'species' key is only used as a fallback when this is unset."
                ),
                constraints={"options": ["marmoset", "human"]},
            ),
            "output_dir": ParameterDefinition(
                default_value="./results/viewer",
                description="Directory where viewer files are written.",
            ),
            "display_data": ParameterDefinition(
                default_value="bold",
                description=(
                    "Which timeseries to animate: 'bold' uses bold_file, "
                    "'temporal_average' uses temporal_average_file, 'none' shows connectivity only."
                ),
                constraints={"options": ["bold", "temporal_average", "none"]},
            ),
            "show_tracts": ParameterDefinition(
                default_value=True,
                description=(
                    "Embed the anatomical fiber tracts for the selected species so the viewer "
                    "renders real white-matter paths (and shows the 'Fiber tracts' toggle). "
                    "Disable to keep connectivity_data.json small — connections then fall back "
                    "to computed splines."
                ),
            ),
            "area_spheres": ParameterDefinition(
                default_value=True,
                description="Scale region spheres proportional to surface area.",
            ),
            "hemi_colors": ParameterDefinition(
                default_value=True,
                description="Color left (blue) and right (orange) hemispheres differently.",
            ),
            "weight_threshold": ParameterDefinition(
                default_value=0.0,
                description="Minimum connection weight to include (0 = show all non-zero).",
                constraints={"min": 0.0, "max": 1.0},
            ),
        },
        inputs={
            "tvb_connectivity": PortDefinition(
                type=PortType.OBJECT,
                description="TVB Connectivity object from TVBConnectivitySetUpNode.",
            ),
            "tvb_stimulus": PortDefinition(
                type=PortType.OBJECT,
                optional=True,
                description=(
                    "Optional TVB StimuliRegion object from TVBStimuliRegionNodeV2, used to "
                    "mark stimulated regions in the viewer metadata."
                ),
            ),
            "bold_file": PortDefinition(
                type=PortType.STR,
                optional=True,
                description=(
                    "Path to BOLD .npz file — saved_file_path output of "
                    "TVBTimeSeriesPlotNodeV2 configured with monitor_type='Bold'."
                ),
            ),
            "temporal_average_file": PortDefinition(
                type=PortType.STR,
                optional=True,
                description=(
                    "Path to TemporalAverage .npz file — saved_file_path output of "
                    "TVBTimeSeriesPlotNodeV2 configured with monitor_type='TemporalAverage'."
                ),
            ),
        },
        outputs={
            "viewer_path": PortDefinition(
                type=PortType.STR,
                description="Absolute path to the generated brain_viewer.html file.",
            ),
        },
        methods={
            "generate_viewer": MethodDefinition(
                description=(
                    "Resolve the species, extract connectivity and optional stimulus metadata "
                    "from TVB objects, load optional timeseries plus the species mesh and fiber "
                    "tracts, build connectivity_data.json, copy brain_viewer.js, and write "
                    "brain_viewer.html."
                ),
                inputs=["tvb_connectivity", "tvb_stimulus", "bold_file", "temporal_average_file"],
                outputs=["viewer_path"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step(
            "generate_viewer", self.generate_viewer, method_key="generate_viewer"
        )

    def generate_viewer(
        self,
        tvb_connectivity,
        tvb_stimulus=None,
        bold_file=None,
        temporal_average_file=None,
    ) -> Dict[str, Any]:
        root = os.path.abspath(self._parameters["output_dir"])
        os.makedirs(root, exist_ok=True)

        display_data   = self._parameters["display_data"]
        area_spheres   = self._parameters["area_spheres"]
        hemi_colors    = self._parameters["hemi_colors"]
        threshold      = self._parameters["weight_threshold"]
        show_tracts    = self._parameters["show_tracts"]

        # ── Species: the NODE PARAMETER is authoritative; the workflow context
        #    'species' key is only a fallback. A workflow-level context default
        #    (e.g. {"species": "marmoset"}) must NOT override a node that the user
        #    explicitly set to 'human'.
        species = str(self._parameters.get("species") or self._context.get("species") or "marmoset").lower()
        if species not in _SPECIES_FILES:
            raise ValueError(
                f"Unknown species {species!r}; expected one of {sorted(_SPECIES_FILES)}."
            )
        mesh_name   = _SPECIES_FILES[species]["mesh"]
        tracts_name = _SPECIES_FILES[species]["tracts"]
        print(f"[{self.name}] species={species}")

        # ── Connectivity data from TVB object ────────────────────────────────
        regions = _extract_regions(tvb_connectivity)
        conns   = _extract_connections(tvb_connectivity, threshold)
        print(f"[{self.name}] {len(regions)} regions, {len(conns)} connections")

        expected = _EXPECTED_REGIONS.get(species)
        if expected is not None and len(regions) != expected:
            print(
                f"[{self.name}] WARNING: species='{species}' expects {expected} regions but "
                f"connectivity has {len(regions)} — mesh/tracts may not align. Make sure the "
                f"connectome uses the matching atlas."
            )

        # ── Timeseries (BOLD or TemporalAverage) ─────────────────────────────
        bold = None
        if display_data == "bold":
            if bold_file:
                bold = _load_npz(bold_file)
                print(f"[{self.name}] BOLD: {len(bold['time'])} timepoints × {len(bold['data'][0])} regions")
            else:
                print(f"[{self.name}] display_data='bold' but no bold_file provided — skipping animation.")
        elif display_data == "temporal_average":
            if temporal_average_file:
                bold = _load_npz(temporal_average_file)
                print(f"[{self.name}] TemporalAverage: {len(bold['time'])} timepoints × {len(bold['data'][0])} regions")
            else:
                print(f"[{self.name}] display_data='temporal_average' but no temporal_average_file provided — skipping animation.")

        # ── Brain mesh (pre-computed per species, ships with the viewer) ─────
        mesh = None
        mesh_src = os.path.join(_STATIC_DIR, mesh_name)
        if os.path.isfile(mesh_src):
            with open(mesh_src) as f:
                mesh = json.load(f)
        else:
            print(f"[{self.name}] mesh file not found: {mesh_src} — continuing without a mesh.")

        # ── Fiber tracts (pre-computed per species, ships with the viewer) ───
        tracts = None
        if show_tracts:
            tracts_src = os.path.join(_STATIC_DIR, tracts_name)
            if os.path.isfile(tracts_src):
                with open(tracts_src) as f:
                    raw = json.load(f)
                # The tract cache is either a flat {"i,j": [...]} dict (older) or a
                # versioned {"tract_cache_version": N, "tracts": {...}} wrapper
                # (create_marmoset_connectivity tract-cache v1+). Unwrap to the flat
                # dict the viewer expects at D.tracts.
                tracts = raw["tracts"] if isinstance(raw, dict) and "tracts" in raw else raw
                print(f"[{self.name}] tracts: {len(tracts)} fiber paths ({tracts_name})")
            else:
                print(f"[{self.name}] tracts file not found: {tracts_src} — connections use splines.")

        # ── Assemble JSON payload ─────────────────────────────────────────────
        weights_nz = [c[2] for c in conns] if conns else [0.0]
        areas      = [r["area"] for r in regions]
        stimulated_regions, stimulated_region_weights = _extract_stimulated_regions(tvb_stimulus)
        if stimulated_regions:
            print(
                f"[{self.name}] Stimulated regions: "
                f"{', '.join(regions[i]['name'] for i in stimulated_regions)}"
            )

        payload = {
            "meta": {
                "species":            species,
                "n_regions":          len(regions),
                "n_connections":      len(conns),
                "weight_max":         float(f"{max(weights_nz):.6g}"),
                "weight_min_nz":      float(f"{min(weights_nz):.6g}"),
                "area_min":           round(float(min(areas)), 3),
                "area_max":           round(float(max(areas)), 3),
                "area_spheres":       area_spheres,
                "hemi_colors":        hemi_colors,
                "stimulated_regions": stimulated_regions,
                "stimulated_region_weights": stimulated_region_weights,
            },
            "regions":     regions,
            "connections": conns,
            "mesh":        mesh,
            "tracts":      tracts,
        }
        # The timeseries field is named by its monitor ('bold' or 'temporal_average')
        # so the signal type is self-evident in the JSON. Omitted when no signal
        # was provided. The viewer reads whichever key is present.
        if bold is not None:
            payload[display_data] = bold

        # Dataset-aware filename (matches the upstream viewer convention): marmoset
        # keeps the historical connectivity_data.json; human writes human_data.json.
        # The GUI eye button requests the species-appropriate file, so two viewer
        # nodes sharing an output_dir never overwrite each other.
        json_name = "human_data.json" if species == "human" else "connectivity_data.json"
        json_path = os.path.join(root, json_name)
        with open(json_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"[{self.name}] {json_name}  ({os.path.getsize(json_path)/1e6:.1f} MB)")

        # ── Copy brain_viewer.js ─────────────────────────────────────────────
        shutil.copy2(os.path.join(_STATIC_DIR, "brain_viewer.js"),
                     os.path.join(root, "brain_viewer.js"))

        # ── Write brain_viewer.html (defaults to this run's data file) ───────
        html_path = os.path.join(root, "brain_viewer.html")
        with open(html_path, "w") as f:
            f.write(_HTML_TEMPLATE.replace("'connectivity_data.json'", f"'{json_name}'"))

        print(f"[{self.name}] Viewer ready: {html_path}")

        return {"viewer_path": html_path}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_regions(conn) -> list:
    labels  = conn.region_labels
    centres = conn.centres
    areas   = conn.areas
    result  = []
    for i, name in enumerate(labels):
        name = str(name)
        result.append({
            "name": name,
            "x":    round(float(centres[i, 0]), 4),
            "y":    round(float(centres[i, 1]), 4),
            "z":    round(float(centres[i, 2]), 4),
            "hemi": "L" if name.startswith("L_") else "R",
            "area": round(float(areas[i]), 3),
        })
    return result


def _extract_connections(conn, threshold: float = 0.0) -> list:
    W = conn.weights
    L = conn.tract_lengths
    n = W.shape[0]
    conns = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w > threshold:
                conns.append([i, j, float(f"{w:.6g}"), round(float(L[i, j]), 2)])
    conns.sort(key=lambda c: -c[2])
    return conns


def _load_npz(path: str) -> dict:
    d = np.load(path)
    return {
        "time": [round(float(t), 1) for t in d["time"]],
        "data": [[round(float(v), 5) for v in row] for row in d["data"]],
    }


def _extract_stimulated_regions(stimulus) -> tuple[list, dict]:
    if stimulus is None:
        return [], {}

    raw_weights = None
    for attr_name in ("weight", "weights"):
        if hasattr(stimulus, attr_name):
            raw_weights = getattr(stimulus, attr_name)
            break

    if raw_weights is None:
        return [], {}

    weights = np.asarray(raw_weights, dtype=float).reshape(-1)
    stimulated = np.flatnonzero(np.abs(weights) > 0).tolist()
    weight_map = {
        str(index): round(float(weights[index]), 6)
        for index in stimulated
    }
    return stimulated, weight_map


# ---------------------------------------------------------------------------
# HTML template (mirrors viewer/brain_viewer.html from create_marmoset_connectivity)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain Connectivity Viewer</title>
<style>
html, body { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #111; }
</style>
<script type="importmap">
{
  "imports": {
    "three":         "https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/"
  }
}
</script>
</head>
<body>
<script type="module">
import { initBrainViewer } from './brain_viewer.js';
const dataUrl = new URLSearchParams(location.search).get('data') || 'connectivity_data.json';
initBrainViewer(document.body, dataUrl);
</script>
</body>
</html>
"""
