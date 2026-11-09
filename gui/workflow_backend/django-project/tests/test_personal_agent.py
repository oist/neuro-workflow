"""Write-only personal Claude and Codex keys."""

import pytest
from app.chat.models import PersonalAgentCredential
from app.chat.personal_agent import decrypt_key
from django.urls import reverse

pytestmark = pytest.mark.django_db

URL = reverse("chat-personal-agent")
CLAUDE = "sk-ant-personal"
CODEX = "sk-codex-personal"


def test_save_and_read_return_only_booleans(auth_client, user_alice, user_bob):
    alice = auth_client(user_alice)
    saved = alice.put(
        URL,
        {"claude_key": CLAUDE, "codex_key": CODEX},
        format="json",
    )
    assert saved.status_code == 200
    assert saved.json() == {"claude": True, "codex": True}
    assert CLAUDE not in saved.content.decode()
    assert CODEX not in saved.content.decode()

    row = PersonalAgentCredential.objects.get(user=user_alice)
    assert row.anthropic_key != CLAUDE
    assert row.openai_key != CODEX
    assert decrypt_key(row.anthropic_key) == CLAUDE
    assert decrypt_key(row.openai_key) == CODEX

    read = alice.get(URL)
    assert read.json() == {"claude": True, "codex": True}
    assert CLAUDE not in read.content.decode()

    other = auth_client(user_bob).get(URL)
    assert other.json() == {"claude": False, "codex": False}
    assert decrypt_key(row.anthropic_key) == CLAUDE


def test_blank_field_keeps_the_saved_key_and_empty_string_clears_one(
    auth_client, user_alice
):
    client = auth_client(user_alice)
    client.put(URL, {"claude_key": CLAUDE, "codex_key": CODEX}, format="json")

    kept = client.put(URL, {"codex_key": "sk-codex-next"}, format="json")
    assert kept.json() == {"claude": True, "codex": True}
    row = PersonalAgentCredential.objects.get(user=user_alice)
    assert decrypt_key(row.anthropic_key) == CLAUDE
    assert decrypt_key(row.openai_key) == "sk-codex-next"

    cleared = client.put(URL, {"claude_key": ""}, format="json")
    assert cleared.json() == {"claude": False, "codex": True}
    row.refresh_from_db()
    assert row.anthropic_key == ""
    assert decrypt_key(row.openai_key) == "sk-codex-next"


def test_delete_clears_both_keys(auth_client, user_alice):
    client = auth_client(user_alice)
    client.put(URL, {"claude_key": CLAUDE, "codex_key": CODEX}, format="json")

    deleted = client.delete(URL)

    assert deleted.status_code == 200
    assert deleted.json() == {"claude": False, "codex": False}
    assert not PersonalAgentCredential.objects.filter(user=user_alice).exists()


def test_unreadable_ciphertext_counts_as_unset(auth_client, user_alice):
    PersonalAgentCredential.objects.create(
        user=user_alice, anthropic_key="not-fernet", openai_key=""
    )

    read = auth_client(user_alice).get(URL)

    assert read.json() == {"claude": False, "codex": False}
