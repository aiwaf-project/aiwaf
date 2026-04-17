def test_fast_cli_reuses_shared_route_shell_helpers():
    from aiwaf.flask.cli import _collect_routes, _resolve_target, _route_shell

    assert callable(_collect_routes)
    assert callable(_resolve_target)
    assert callable(_route_shell)

