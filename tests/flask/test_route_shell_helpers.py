from flask import Flask

import builtins

from aiwaf.flask.cli import AIWAFManager, _collect_routes, _resolve_target, _route_shell
from aiwaf.core.utils import build_tree as _build_tree


def test_route_shell_tree_building():
    app = Flask(__name__)

    @app.route('/api/v1/users')
    def users():
        return 'OK'

    @app.route('/health')
    def health():
        return 'OK'

    routes = _collect_routes(app)
    root = _build_tree(routes)

    api_node = _resolve_target(root, 'api')
    assert api_node is not None
    v1_node = _resolve_target(api_node, 'v1')
    assert v1_node is not None
    users_node = _resolve_target(v1_node, 'users')
    assert users_node is not None
    assert users_node.is_endpoint is True

    health_node = _resolve_target(root, 'health')
    assert health_node is not None
    assert health_node.is_endpoint is True


def test_route_shell_exempt_flow(monkeypatch, tmp_path):
    app = Flask(__name__)

    @app.route('/api/')
    def api_root():
        return 'OK'

    manager = AIWAFManager(str(tmp_path))

    inputs = iter([
        "cd api",
        "exempt .",
        "Polling endpoint",
        "exit",
    ])

    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    _route_shell(app, manager)

    exemptions = manager.list_path_exemptions()
    assert "/api/" in exemptions


def test_route_shell_help_and_cd_behavior(monkeypatch, capsys, tmp_path):
    app = Flask(__name__)

    @app.route('/api/v1/users')
    def users():
        return 'OK'

    manager = AIWAFManager(str(tmp_path))

    inputs = iter([
        "help",
        "pwd",
        "cd",
        "pwd",
        "cd api",
        "pwd",
        "cd ..",
        "pwd",
        "exit",
    ])

    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(inputs))

    _route_shell(app, manager)

    output = capsys.readouterr().out
    assert "AIWAF Path Shell Commands:" in output
    assert "ls                     # list routes at current level" in output
    assert "cd <index|name>        # enter a route segment" in output
    assert "up / cd ..             # go up one level" in output
    assert "pwd                    # show current path prefix" in output
    assert "exempt <index|name|.>  # add exemption for selection or current path" in output
    assert "exit                   # quit" in output
    assert "Usage: cd <index|name>" in output

    # pwd outputs should reflect that bare `cd` does not change current path
    assert output.count("/api/") == 1
    assert output.count("/") >= 2
