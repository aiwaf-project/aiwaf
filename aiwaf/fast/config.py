"""FastAPI compatibility exports for shared AIWAF runtime config."""

from aiwaf.core.runtime_config import AIWAFConfig, get_config, initialize_config

__all__ = ["AIWAFConfig", "get_config", "initialize_config"]
