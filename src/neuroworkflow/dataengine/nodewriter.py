"""Write the raw data node for a downloaded dataset.

The raw node is the starting point of every workflow that uses the dataset:

* **parameters**: where the data comes from (source + reference + chosen items)
  and the metadata given by the source, editable by the data owners;
* **outputs**: the local folder, the file list, the metadata;
* **one method, load**: checks the data is complete on this machine
  (``dataengine.ensure``) and passes it on. It downloads nothing unless the
  ``download`` parameter is set, so a workflow never starts a large transfer.

Preprocessing is not here: it belongs in nodes connected after this one, which
take ``raw_data_path`` (or ``files``) as input.
"""

from __future__ import annotations

import os
import pprint
import re
from typing import List, Optional

from .dataset import Dataset, human_size

TEMPLATE = '''"""Raw data node: {title_doc}

Written by the NeuroWorkflow data engine. The data itself is not in this file: it
lives in the data store and is located by `source` + `reference`.
"""

from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)


class {class_name}(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type={type_!r},
        stage="io",
        tool={tool!r},
        model_source={model_source!r},
        description=(
            {description!r}
        ),
        parameters={{
            "source": ParameterDefinition(
                default_value={source!r},
                description="Data engine source that reaches this dataset "
                "(cbs_public, cbs_project, atr).",
            ),
            "reference": ParameterDefinition(
                default_value={reference},
                description="The source's own reference to the dataset "
                "(CBS folder UUID, or path on the Aspera share).",
            ),
            "include": ParameterDefinition(
                default_value={include},
                description="Sub-folders/files of the dataset used by this node; "
                "empty = the whole dataset.",
            ),
            "metadata": ParameterDefinition(
                default_value={metadata},
                description="Metadata of the dataset as given by the source; data "
                "owners may complete it.",
            ),
            "download": ParameterDefinition(
                default_value=False,
                description="Download the data first if it is missing on this "
                "machine ({size}). Off by default so a workflow run never starts a "
                "large transfer.",
            ),
        }},
        inputs={{}},
        outputs={{
            "raw_data_path": PortDefinition(
                type=PortType.FILE_PATH,
                description="Local folder holding the raw data; the chosen items "
                "are sub-folders/files of it, under their names in the repository.",
            ),
            "files": PortDefinition(
                type=PortType.LIST,
                description="Every raw file: dicts with 'path' (relative to "
                "raw_data_path) and 'size_bytes'.",
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description="Dataset metadata: title, DOI, laboratory, access level, "
                "and the source's own fields.",
            ),
        }},
        methods={{
            "load": MethodDefinition(
                description="Check the raw data is complete in the data store and "
                "output its folder, file list and metadata.",
                inputs=[],
                outputs=["raw_data_path", "files", "metadata"],
            ),
        }},
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("load", self.load, method_key="load")

    def load(self) -> Dict[str, Any]:
        import os

        from neuroworkflow.dataengine import ensure

        include = list(self._parameters["include"])
        path = ensure(
            self._parameters["source"],
            self._parameters["reference"],
            include=include,
            store=self._context.get("data_store"),     # else NW_DATA_STORE, else ./data
            download=bool(self._parameters["download"]),
        )
        files = []
        for top in include or sorted(os.listdir(path)):
            full = os.path.join(path, top)
            if top.startswith(".dataset.json"):
                continue
            walk = [(os.path.dirname(full), [], [top])] if os.path.isfile(full) else os.walk(full)
            for root, _, names in walk:
                for n in sorted(names):
                    f = os.path.join(root, n)
                    files.append({{"path": os.path.relpath(f, path),
                                  "size_bytes": os.path.getsize(f)}})
        return {{
            "raw_data_path": path,
            "files": files,
            "metadata": dict(self._parameters["metadata"]),
        }}
'''


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]", "_", text)).strip("_")


def _literal(obj, indent: int) -> str:
    text = pprint.pformat(obj, width=80, sort_dicts=False)
    return text.replace("\n", "\n" + " " * indent)


def write_node(dataset: Dataset, include: List[str], out_dir: str,
               class_name: Optional[str] = None, size: Optional[int] = None) -> str:
    """Write ``<out_dir>/<class_name>.py`` for the dataset; returns its path."""
    name = dataset.metadata.get("folder_name") or os.path.basename(dataset.ref.get("path", "")) \
        or dataset.key
    repo = dataset.source.split("_")[0]
    class_name = class_name or f"Data_{repo.upper()}_{_slug(name)}"
    if not class_name.isidentifier():
        raise ValueError(f"not a valid class name: {class_name!r}")
    if dataset.doi:
        model_source = f"https://doi.org/{dataset.doi}"
    elif "folder_id" in dataset.ref:
        model_source = f"https://neurodata.riken.jp/api/v3/folders/{dataset.ref['folder_id']}/"
    else:
        model_source = f"{dataset.metadata.get('host', '').rstrip('/')}{dataset.ref.get('path', '')}"
    what = f" ({', '.join(include)})" if include else ""
    description = (
        f"Raw data from {dataset.laboratory or repo.upper()}: {dataset.title}{what}. "
        f"Source {dataset.source}, access {dataset.access or 'unknown'}"
        f"{', DOI ' + dataset.doi if dataset.doi else ''}. Outputs the local folder, the "
        f"file list and the metadata; preprocessing belongs in downstream nodes."
    )
    metadata = {"title": dataset.title, "doi": dataset.doi, "laboratory": dataset.laboratory,
                "access": dataset.access, **dataset.metadata}
    code = TEMPLATE.format(
        title_doc=dataset.title.replace('"""', "'''"),
        class_name=class_name,
        type_=class_name.lower(),
        tool=repo,
        model_source=model_source,
        description=description,
        source=dataset.source,
        reference=_literal(dataset.ref, 16),
        include=_literal(list(include), 16),
        metadata=_literal(metadata, 16),
        size=human_size(size if size is not None else dataset.size),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{class_name}.py")
    with open(path, "w") as f:
        f.write(code)
    return path
