"""Pure listing-filter tests (no Jupyter server required)."""
import importlib.util
from pathlib import Path

FILTER_PATH = (
    Path(__file__).resolve().parents[1] / "neuroworkflow" / "jupyter_tenant_filter.py"
)
spec = importlib.util.spec_from_file_location("jupyter_tenant_filter", FILTER_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_hides_other_uuid_dirs():
    allowed = ["11111111-1111-1111-1111-111111111111"]
    hidden = "22222222-2222-2222-2222-222222222222"
    entries = [
        {"name": allowed[0], "type": "directory"},
        {"name": hidden, "type": "directory"},
        {"name": "README.md", "type": "file"},
    ]
    out = mod.filter_directory_entries(
        "codes/projects",
        entries,
        project_ids=allowed,
        legacy_names=[],
    )
    names = {e["name"] for e in out}
    assert allowed[0] in names
    assert hidden not in names
    assert "README.md" in names


def test_fail_closed_without_allowlist():
    uuid_name = "11111111-1111-1111-1111-111111111111"
    out = mod.filter_directory_entries(
        "codes/projects",
        [{"name": uuid_name, "type": "directory"}],
        project_ids=[],
        fail_closed=True,
    )
    assert out == []


def test_nested_project_path_denied():
    assert not mod.path_is_allowed(
        "codes/projects/22222222-2222-2222-2222-222222222222/workflow.py",
        project_ids=["11111111-1111-1111-1111-111111111111"],
    )
    assert mod.path_is_allowed(
        "codes/nodes/analysis/Foo.py",
        project_ids=[],
    )


def test_fetch_allowlist_without_token_is_open():
    payload = mod.fetch_allowlist(None)
    assert payload["hide_unlisted_projects"] is False


def test_path_allowed_when_filter_disabled():
    path = "codes/projects/22222222-2222-2222-2222-222222222222/workflow.py"
    assert mod.path_is_allowed(path, project_ids=[], fail_closed=False)


def test_projects_root_is_never_denied():
    assert mod.path_is_allowed("codes/projects", project_ids=[], fail_closed=True)
    assert mod.path_is_allowed("/codes/projects", project_ids=[], fail_closed=True)
    assert mod.path_is_allowed("codes", project_ids=[], fail_closed=True)


def test_allowlist_cache_evicts():
    mod._allowlist_cache.clear()
    old_max = mod._CACHE_MAX
    mod._CACHE_MAX = 2
    try:
        for i in range(3):
            token = f"tok-{i}"
            mod._allowlist_cache[token] = (0.0, {"project_ids": [str(i)]})
            while len(mod._allowlist_cache) > mod._CACHE_MAX:
                mod._allowlist_cache.popitem(last=False)
        assert "tok-0" not in mod._allowlist_cache
        assert "tok-1" in mod._allowlist_cache
        assert "tok-2" in mod._allowlist_cache
    finally:
        mod._CACHE_MAX = old_max
        mod._allowlist_cache.clear()


OWN = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"


def test_owner_can_write_own_project_and_not_another():
    allow = {
        "hide_unlisted_projects": True,
        "project_ids": [OWN, OTHER],
        "legacy_names": ["SharedName"],
        "writable_project_ids": [OWN],
    }
    assert mod.write_is_allowed(f"codes/projects/{OWN}/workflow.py", allow)
    assert not mod.write_is_allowed(f"codes/projects/{OTHER}/workflow.py", allow)
    assert not mod.write_is_allowed("codes/projects/SharedName/notes.txt", allow)
    assert mod.write_is_allowed("codes/nodes/analysis/Foo.py", allow)


def test_write_rules_for_legacy_uuid_and_open_viewer():
    owned = {
        "hide_unlisted_projects": True,
        "project_ids": [OWN],
        "legacy_names": ["Aliceown"],
        "writable_project_ids": [OWN, "Aliceown"],
    }
    assert mod.write_is_allowed("codes/projects/Aliceown/workflow.py", owned)
    foreign = f"codes/projects/{OTHER}/workflow.py"
    closed = {
        "hide_unlisted_projects": True,
        "project_ids": [OWN],
        "legacy_names": [],
        "writable_project_ids": [OWN],
    }
    assert not mod.write_is_allowed(foreign, closed)
    assert mod.write_is_allowed(foreign, {**closed, "hide_unlisted_projects": False})
    assert mod.write_is_allowed("codes/projects/ScratchNotes/file.txt", closed)


def test_copy_and_checkpoints_refuse_before_the_original():
    try:
        import jupyter_server  # noqa: F401
        import tornado.web
    except ImportError:
        return

    foreign = f"codes/projects/{OTHER}/data"
    owned = f"codes/projects/{OWN}/data"
    allow = {
        "hide_unlisted_projects": True,
        "project_ids": [OWN, OTHER],
        "legacy_names": [],
        "writable_project_ids": [OWN],
    }

    class Contents:
        def __init__(self):
            self.calls = []

        def get(self, path, content=True, type=None, format=None, **kwargs):
            return {"type": "directory", "content": []}

        def save(self, model, path=""):
            self.calls.append(("save", path))

        def delete_file(self, path):
            self.calls.append(("delete", path))

        def rename_file(self, old_path, new_path):
            self.calls.append(("rename", old_path, new_path))

        def copy(self, from_path, to_path=None):
            self.calls.append(("copy", from_path, to_path))

        def create_checkpoint(self, path):
            self.calls.append(("create_checkpoint", path))

        def delete_checkpoint(self, checkpoint_id, path):
            self.calls.append(("delete_checkpoint", checkpoint_id, path))

    class Server:
        def __init__(self):
            self.contents_manager = Contents()
            self.base_url = "/"
            self.web_app = self
            self.log = self

        def add_handlers(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    original_fetch = mod.fetch_allowlist
    original_prepare = tornado.web.RequestHandler.prepare
    mod.fetch_allowlist = lambda token: allow
    try:
        app = Server()
        mod._load_jupyter_server_extension(app)
        cm = app.contents_manager
        refused = (
            lambda: cm.copy(foreign),
            lambda: cm.copy(owned, foreign),
            lambda: cm.create_checkpoint(foreign),
            lambda: cm.delete_checkpoint("cp", foreign),
        )
        for call in refused:
            try:
                call()
            except tornado.web.HTTPError as exc:
                assert exc.status_code == 403
            else:
                raise AssertionError("expected 403")
        assert cm.calls == []
        cm.copy(foreign, owned)
        cm.create_checkpoint(owned)
        cm.delete_checkpoint("cp", owned)
        assert cm.calls == [
            ("copy", foreign, owned),
            ("create_checkpoint", owned),
            ("delete_checkpoint", "cp", owned),
        ]
    finally:
        mod.fetch_allowlist = original_fetch
        tornado.web.RequestHandler.prepare = original_prepare
