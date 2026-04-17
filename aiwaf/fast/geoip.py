"""FastAPI geoip helpers built on shared core geoip lookup."""

import os

from aiwaf.core import geoip as core_geoip


def get_country_for_ip(ip: str, config: dict | None = None):
    config = config or {}
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "core",
        "geolock",
        "ipinfo_lite.mmdb",
    )
    db_path = config.get("AIWAF_GEOIP_DB_PATH", default_path)
    return core_geoip.lookup_country(ip, db_path)
