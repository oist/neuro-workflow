"""Flow PUT and node writes reject plaintext secret parameters."""

import pytest
from django.urls import reverse

from app.secrets.redaction import make_secret_ref
from app.secrets.services import create_user_secret
from app.workflow.models import FlowNode, FlowProject

pytestmark = pytest.mark.django_db

PLAIN = "hunter2"


def _aspera_node(password_value):
    return {
        "id": "aspera-node-1",
        "position": {"x": 0, "y": 0},
        "type": "default",
        "data": {
            "label": "Aspera",
            "nodeType": "io",
            "schema": {
                "parameters": {
                    "password": {
                        "secret": True,
                        "default_value": password_value,
                    }
                }
            },
        },
    }


def test_flow_put_rejects_plaintext_secret_param(auth_client, user_alice):
    project = FlowProject.objects.create(name="secrets-flow", owner=user_alice)
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-flow", kwargs={"workflow_id": project.id})
    resp = client.put(url, {"nodes": [_aspera_node(PLAIN)], "edges": []}, format="json")
    assert resp.status_code == 400
    assert PLAIN not in str(FlowNode.objects.filter(project=project).values_list("data", flat=True))
    dumped = "".join(str(n.data) for n in FlowNode.objects.filter(project=project))
    assert PLAIN not in dumped


def test_flow_put_accepts_owned_secret_ref(auth_client, user_alice):
    project = FlowProject.objects.create(name="secrets-flow-ok", owner=user_alice)
    secret = create_user_secret(user_alice, name="ASPERA_PASSWORD", value=PLAIN)
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-flow", kwargs={"workflow_id": project.id})
    resp = client.put(
        url,
        {"nodes": [_aspera_node(make_secret_ref(secret.id, secret.name))], "edges": []},
        format="json",
    )
    assert resp.status_code == 200
    node = FlowNode.objects.get(project=project)
    value = node.data["schema"]["parameters"]["password"]["default_value"]
    assert value["__nw_secret"]["name"] == "ASPERA_PASSWORD"
    assert PLAIN not in str(node.data)
