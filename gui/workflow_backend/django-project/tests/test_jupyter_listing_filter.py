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
