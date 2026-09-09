from typing import Dict, List, Any
from django.db import transaction
from .models import FlowProject, FlowNode, FlowEdge
from app.secrets.sanitize import sanitize_node_data


class FlowService:
    """Flow Management Business Logic"""

    @staticmethod
    def create_project(name: str, description: str, owner) -> FlowProject:
        """Create a new flow project"""
        return FlowProject.objects.create(
            name=name, description=description, owner=owner
        )

    @staticmethod
    def save_flow_data(project_id: str, nodes_data: List[Dict], edges_data: List[Dict]):
        """Save React Flow node and edge data"""
        with transaction.atomic():
            project = FlowProject.objects.get(id=project_id)

            # Delete existing nodes and edges
            FlowNode.objects.filter(project=project).delete()
            FlowEdge.objects.filter(project=project).delete()

            # save node
            nodes = []
            for node_data in nodes_data:
                node = FlowNode(
                    id=node_data["id"],
                    project=project,
                    position_x=node_data["position"]["x"],
                    position_y=node_data["position"]["y"],
                    node_type=node_data.get("type", "default"),
                    data=sanitize_node_data(project.owner, node_data.get("data", {})),
                )
                nodes.append(node)

            FlowNode.objects.bulk_create(nodes)

            # save edge
            edges = []
            for edge_data in edges_data:
                try:
                    source_node = FlowNode.objects.get(
                        id=edge_data["source"], project=project
                    )
                    target_node = FlowNode.objects.get(
                        id=edge_data["target"], project=project
                    )

                    edge = FlowEdge(
                        id=edge_data["id"],
                        project=project,
                        source_node=source_node,
                        target_node=target_node,
                        source_handle=edge_data.get("sourceHandle"),
                        target_handle=edge_data.get("targetHandle"),
                        edge_data=edge_data.get("data", {}),
                    )
                    edges.append(edge)
                except FlowNode.DoesNotExist:
                    continue

            FlowEdge.objects.bulk_create(edges)

    @staticmethod
    def get_flow_data(project_id: str) -> Dict[str, List]:
        """Get project flow data"""
        project = FlowProject.objects.get(id=project_id)

        # Build node data
        nodes = []
        for node in project.nodes.all():
            # Ensure old nodes have instanceName for title display
            node_data_dict = node.data.copy() if node.data else {}
            if not node_data_dict.get("instanceName") and not node_data_dict.get("label"):
                # Set default title from available fields
                node_data_dict["instanceName"] = (
                    node_data_dict.get("file_name") or 
                    node_data_dict.get("nodeType") or 
                    f"Node {node.id[:8]}"
                )
            
            node_data = {
                "id": node.id,
                "position": {"x": node.position_x, "y": node.position_y},
                "type": node.node_type,
                "data": node_data_dict,
            }
            nodes.append(node_data)

        # Build edge data
        edges = []
        for edge in project.edges.all():
            edge_data = {
                "id": edge.id,
                "source": edge.source_node_id,
                "target": edge.target_node_id,
            }

            if edge.source_handle:
                edge_data["sourceHandle"] = edge.source_handle
            if edge.target_handle:
                edge_data["targetHandle"] = edge.target_handle
            if edge.edge_data:
                edge_data["data"] = edge.edge_data

            edges.append(edge_data)

        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def create_node(project_id: str, node_data: Dict) -> FlowNode:
        """Create individual nodes"""
        project = FlowProject.objects.get(id=project_id)

        node = FlowNode.objects.create(
            id=node_data["id"],
            project=project,
            position_x=node_data["position"]["x"],
            position_y=node_data["position"]["y"],
            node_type=node_data.get("type", "default"),
            data=sanitize_node_data(project.owner, node_data.get("data", {})),
        )

        return node

    @staticmethod
    def update_node(node_id: str, project_id: str, node_data: Dict) -> FlowNode:
        """node update"""
        project = FlowProject.objects.get(id=project_id)
        node = FlowNode.objects.get(id=node_id, project=project)

        node.position_x = node_data["position"]["x"]
        node.position_y = node_data["position"]["y"]
        node.node_type = node_data.get("type", node.node_type)

        if "data" in node_data:
            new_data = sanitize_node_data(project.owner, dict(node_data["data"]))
            # The PUT /parameters/ endpoint is the sole owner of parameter_modifications.
            # The general node update (triggered by position/schema changes in the
            # frontend) carries a stale copy — always restore from the DB record.
            for key in ("parameter_modifications", "has_parameter_modifications"):
                if key in node.data:
                    new_data[key] = node.data[key]
                elif key in new_data:
                    del new_data[key]
            node.data = new_data

        node.save()

        return node

    @staticmethod
    def delete_node(node_id: str, project_id: str):
        """Node deletion (associated edges are also automatically deleted)"""
        project = FlowProject.objects.get(id=project_id)
        node = FlowNode.objects.get(id=node_id, project=project)
        node.delete()

    @staticmethod
    def create_edge(project_id: str, edge_data: Dict) -> FlowEdge:
        """Individual edge creation"""
        project = FlowProject.objects.get(id=project_id)
        source_node = FlowNode.objects.get(id=edge_data["source"], project=project)
        target_node = FlowNode.objects.get(id=edge_data["target"], project=project)

        edge = FlowEdge.objects.create(
            id=edge_data["id"],
            project=project,
            source_node=source_node,
            target_node=target_node,
            source_handle=edge_data.get("sourceHandle"),
            target_handle=edge_data.get("targetHandle"),
            edge_data=edge_data.get("data", {}),
        )

        return edge

    @staticmethod
    def delete_edge(edge_id: str, project_id: str):
        """edge delete"""
        project = FlowProject.objects.get(id=project_id)
        edge = FlowEdge.objects.get(id=edge_id, project=project)
        edge.delete()

    @staticmethod
    def get_sample_flow_data() -> Dict:
        """Returns sample flow data for arithmetic operations"""
        return {
            "nodes": [
                {
                    "id": "input-a",
                    "position": {"x": 50, "y": 100},
                    "type": "calculationNode",
                    "data": {
                        "label": "Number A",
                        "schema": [
                            {
                                "title": "value",
                                "type": "number",
                                "description": "Input number A",
                            }
                        ],
                    },
                },
                {
                    "id": "input-b",
                    "position": {"x": 50, "y": 300},
                    "type": "calculationNode",
                    "data": {
                        "label": "Number B",
                        "schema": [
                            {
                                "title": "value",
                                "type": "number",
                                "description": "Input number B",
                            }
                        ],
                    },
                },
                {
                    "id": "addition",
                    "position": {"x": 300, "y": 50},
                    "type": "calculationNode",
                    "data": {
                        "label": "Addition (A + B)",
                        "schema": [
                            {"title": "a", "type": "number", "description": "Number A"},
                            {"title": "b", "type": "number", "description": "Number B"},
                            {
                                "title": "result",
                                "type": "number",
                                "description": "A + B result",
                            },
                        ],
                    },
                },
                {
                    "id": "subtraction",
                    "position": {"x": 300, "y": 150},
                    "type": "calculationNode",
                    "data": {
                        "label": "Subtraction (A - B)",
                        "schema": [
                            {"title": "a", "type": "number", "description": "Number A"},
                            {"title": "b", "type": "number", "description": "Number B"},
                            {
                                "title": "result",
                                "type": "number",
                                "description": "A - B result",
                            },
                        ],
                    },
                },
                {
                    "id": "multiplication",
                    "position": {"x": 300, "y": 250},
                    "type": "calculationNode",
                    "data": {
                        "label": "Multiplication (A × B)",
                        "schema": [
                            {"title": "a", "type": "number", "description": "Number A"},
                            {"title": "b", "type": "number", "description": "Number B"},
                            {
                                "title": "result",
                                "type": "number",
                                "description": "A × B result",
                            },
                        ],
                    },
                },
                {
                    "id": "division",
                    "position": {"x": 300, "y": 350},
                    "type": "calculationNode",
                    "data": {
                        "label": "Division (A ÷ B)",
                        "schema": [
                            {"title": "a", "type": "number", "description": "Number A"},
                            {"title": "b", "type": "number", "description": "Number B"},
                            {
                                "title": "result",
                                "type": "number",
                                "description": "A ÷ B result",
                            },
                        ],
                    },
                },
            ],
            "edges": [
                {
                    "id": "input-a-to-add",
                    "source": "input-a",
                    "target": "addition",
                    "sourceHandle": "value-output",
                    "targetHandle": "a-input",
                },
                {
                    "id": "input-b-to-add",
                    "source": "input-b",
                    "target": "addition",
                    "sourceHandle": "value-output",
                    "targetHandle": "b-input",
                },
                {
                    "id": "input-a-to-sub",
                    "source": "input-a",
                    "target": "subtraction",
                    "sourceHandle": "value-output",
                    "targetHandle": "a-input",
                },
                {
                    "id": "input-b-to-sub",
                    "source": "input-b",
                    "target": "subtraction",
                    "sourceHandle": "value-output",
                    "targetHandle": "b-input",
                },
                {
                    "id": "input-a-to-mul",
                    "source": "input-a",
                    "target": "multiplication",
                    "sourceHandle": "value-output",
                    "targetHandle": "a-input",
                },
                {
                    "id": "input-b-to-mul",
                    "source": "input-b",
                    "target": "multiplication",
                    "sourceHandle": "value-output",
                    "targetHandle": "b-input",
                },
                {
                    "id": "input-a-to-div",
                    "source": "input-a",
                    "target": "division",
                    "sourceHandle": "value-output",
                    "targetHandle": "a-input",
                },
                {
                    "id": "input-b-to-div",
                    "source": "input-b",
                    "target": "division",
                    "sourceHandle": "value-output",
                    "targetHandle": "b-input",
                },
            ],
        }
