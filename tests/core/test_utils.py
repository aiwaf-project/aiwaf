from aiwaf.core.utils import build_tree, normalize_path


def test_utils_module_contract():
    assert normalize_path("API//Users/") == "/API/Users/"
    assert build_tree(["/api/users"]).children["api"].children["users"].is_endpoint is True
