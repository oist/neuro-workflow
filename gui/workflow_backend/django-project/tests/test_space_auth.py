"""Project and community Jupyter passwords are not interchangeable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neuroworkflow"))

from space_auth import authenticate_space_user

PROJECT = "project-secret"
COMMUNITY = "community-secret"


def _auth(username, password):
    return authenticate_space_user(
        username,
        password,
        project_user="internal",
        community_user="external",
        project_password=PROJECT,
        community_password=COMMUNITY,
    )


def test_project_password_accepts_internal_and_user1():
    assert _auth("internal", PROJECT) == "internal"
    assert _auth("user1", PROJECT) == "internal"


def test_community_password_accepts_external():
    assert _auth("external", COMMUNITY) == "external"


def test_passwords_are_rejected_for_the_other_user():
    assert _auth("internal", COMMUNITY) is None
    assert _auth("external", PROJECT) is None
    assert _auth("user1", COMMUNITY) is None


def test_hackathon_is_rejected():
    assert _auth("hackathon", COMMUNITY) is None
    assert _auth("hackathon", PROJECT) is None


def test_empty_password_is_rejected():
    assert _auth("internal", "") is None
    assert _auth("external", "") is None
    assert _auth("internal", None) is None
