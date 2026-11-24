"""A whole-flow save must reload in the order it was given.

Generate Code numbers every saved node from 1, and a later notebook import
uses that same order. The flow save deletes the nodes and inserts them again,
so each recreated node needs a later created_at than the one before it.
"""

from datetime import datetime, timezone as dt_timezone

import pytest
from django.utils import timezone

from app.workflow.models import FlowNode, FlowProject
from app.workflow.services import FlowService

pytestmark = pytest.mark.django_db

FIXED = datetime(2026, 11, 17, 7, 0, tzinfo=dt_timezone.utc)


def node(node_id, index):
    return {
        "id": node_id,
        "position": {"x": index, "y": 0},
        "type": "calculationNode",
        "data": {"label": "NW_Population", "instanceName": "NW_Population"},
    }


def test_flow_save_reloads_nodes_in_the_posted_order(user_alice, monkeypatch):
    monkeypatch.setattr(timezone, "now", lambda: FIXED)
    project = FlowProject.objects.create(name="order", owner=user_alice)
    posted = ["n-c", "n-a", "n-b"]
    FlowService.save_flow_data(
        str(project.id),
        [node(node_id, index) for index, node_id in enumerate(posted)],
        [],
    )
    loaded = [item["id"] for item in FlowService.get_flow_data(str(project.id))["nodes"]]
    saved = list(
        FlowNode.objects.filter(project=project).order_by("created_at", "id")
    )
    assert [item.id for item in saved] == posted
    assert saved[0].created_at < saved[1].created_at < saved[2].created_at
    assert loaded == posted

    reversed_ids = list(reversed(posted))
    FlowService.save_flow_data(
        str(project.id),
        [node(node_id, index) for index, node_id in enumerate(reversed_ids)],
        [],
    )
    reloaded = [item["id"] for item in FlowService.get_flow_data(str(project.id))["nodes"]]
    assert reloaded == reversed_ids
