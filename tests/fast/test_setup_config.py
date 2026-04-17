from aiwaf.fast.config import AIWAFConfig


def test_fast_config_constructs_with_defaults():
    config = AIWAFConfig()
    assert config is not None
    assert isinstance(config.get("storage"), dict)
