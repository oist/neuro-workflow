from pathlib import Path

from report_facts import build_workflow_facts

ROOT = Path(__file__).resolve().parents[1]


def test_facts_keep_a_doi_a_file_and_result_shapes():
    facts = build_workflow_facts(
        {
            "name": "Visual cortex",
            "description": "A small simulation",
            "doi": "10.1000/example",
            "data_source": "",
            "license": "",
            "funding": "",
            "reference": "",
            "contributors": [],
            "links": [],
        },
        {"files": [{"filename": "connectivity.npz", "size_bytes": 12}]},
        [],
        [],
        {
            "results": [
                {
                    "filename": "rates.npz",
                    "arrays": [{"name": "rates", "shape": [1000, 2]}],
                }
            ]
        },
        {"code": "pass", "notebook_outputs": []},
    )
    assert facts["project"]["doi"] == "10.1000/example"
    assert facts["project"]["funding"] == ""
    assert facts["inputs"][0]["filename"] == "connectivity.npz"
    assert facts["results"][0]["arrays"][0]["shape"] == [1000, 2]
    assert "simulation_io" not in facts


def test_a_missing_doi_stays_empty():
    facts = build_workflow_facts(
        {"name": "Untitled", "description": None},
        None,
        None,
        None,
        None,
        None,
    )
    assert facts["project"]["doi"] == ""
    assert facts["inputs"] == []
    assert facts["results"] == []


def test_methods_instructions_are_in_the_prompt_and_the_button():
    prompt = (
        ROOT
        / "workflow_backend/django-project/app/chat/services/chat_orchestrator.py"
    ).read_text()
    button = (
        ROOT / "workflow_frontend/src/views/home/components/chatbotView.tsx"
    ).read_text()
    files_line = (
        "Name the files the simulation reads and the result files it writes, "
        "including array shapes when the facts include them."
    )
    credit_line = (
        "When a fact includes a source, owner, paper, DOI, or acknowledgment, "
        "state that credit in the same sentence as the value, and do not invent a source."
    )
    assert files_line in prompt
    assert credit_line in prompt
    assert files_line in button
    assert credit_line in button
